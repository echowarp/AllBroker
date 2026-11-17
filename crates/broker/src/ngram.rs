//! N-gram (PLE) table service (design: crates/ngram-store/DESIGN.md).
//!
//! Engines send token runs (hash history + tokens); the broker hashes the table rows exactly as gufo does
//! (ngram-store, bit-identical to gufo's HashNgramRows), serves them from a row cache, reads misses from the table
//! file with O_DIRECT + io_uring (never mmap, never the page cache), and copies the rows into the connection's window
//! of a shared staging file in request order. Engines map that file read-only and decode the rows themselves.
//!
//! Dynamic:
//! - Tables load lazily on the first NGRAM_OPEN; nothing is read or allocated for a configured table before that.
//! - The cache grows in segments only as rows are inserted, up to its cap.
//! - A table nobody has open for `idle` is unloaded: cache and reader dropped, memory returned. A short model
//!   eviction keeps the warm cache; a model that is gone for good costs nothing.
//! - Memory pressure: when MemAvailable drops below the reserve (the broker's --mem-reserve-gib signal), the cap
//!   shrinks by the deficit (+256 MiB) at once, before the KV pool has to give up entries (an n-gram miss costs a disk
//!   read, a KV miss a re-prefill). After `regrow` of calm with 3 GiB spare it grows back 512 MiB at a time.
//!
//! Fixed pin locations (the "pinned staging ring" idea): each connection owns one staging window at
//! a fixed offset of a fixed-size file. A window is never moved, resized or hole-punched while its connection holds
//! it, so an engine can register it with its GPU runtime once and DMA from it; it is released (and its pages punched)
//! on disconnect. Prefetches are queued to one background thread and dropped when the queue is full.
//!
//! Bins (optional, `--ngram-bins NAME=PATH`, ngram-store `bins.rs`): bundles of the 8 rows of frequent bigrams and
//! trigrams, packed along common token chains into 16 KiB / 4 KiB bins. A miss on a bundled n-gram reads its whole
//! bin, and every bundle in it goes into the row cache: one read serves this token and usually the next few. Rows of
//! n-grams without a bundle are read from the table as before, in parallel with the bin reads. The cache stays
//! row-granular, so engines see no difference (rows are exact copies).

use kvcore::ConnId;
use kvproto::{NgramSeq, Status};
use ngram_store::bins::{self, BinFile};
use ngram_store::codec::Codec;
use ngram_store::gguf::Gguf;
use ngram_store::hash::{self, History, PleParams};
use ngram_store::layout::TRI;
use ngram_store::reader::RowFile;
use ngram_store::TABLE_TENSOR;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Arc;
use tracing_mutex::stdsync::{Mutex, RwLock};
use std::time::{Duration, Instant};

pub const GATHER_BUCKETS_US: [u64; 10] = [10, 25, 50, 100, 200, 400, 800, 1600, 5000, 50000];
/// Rows per arena segment (~6 MiB with 96-byte slots): the cache allocates one segment at a time as it fills.
const SEG_SLOTS: usize = 64 * 1024;
const GIB: u64 = 1 << 30;

/// Row cache with CLOCK eviction (an LRU approximation) over fixed slots, in lazily allocated segments.
struct RowCache {
    slot_bytes: usize,
    segs: Vec<Box<[u8]>>,
    index: HashMap<u32, u32>,
    rows: Vec<u32>,
    refd: Vec<bool>,
    hand: usize,
    /// current cap in slots (memory pressure moves it); allocated slots = segs.len() * SEG_SLOTS
    cap: usize,
    /// empty allocated slots (after a grow)
    free: Vec<u32>,
}

impl RowCache {
    fn new(row_bytes: usize) -> RowCache {
        RowCache { slot_bytes: row_bytes.div_ceil(16) * 16, segs: Vec::new(), index: HashMap::new(), rows: Vec::new(),
                   refd: Vec::new(), hand: 0, cap: 0, free: Vec::new() }
    }
    /// Bytes per cached row including bookkeeping (index entry ~16 B, row id 4 B, ref bit 1 B).
    fn cost(row_bytes: usize) -> u64 {
        (row_bytes.div_ceil(16) * 16 + 21) as u64
    }
    fn allocated(&self) -> usize {
        self.rows.len()
    }
    fn bytes(&self) -> u64 {
        self.allocated() as u64 * Self::cost(self.slot_bytes)
    }
    fn get(&mut self, row: u32) -> Option<usize> {
        let s = *self.index.get(&row)? as usize;
        self.refd[s] = true;
        Some(s)
    }
    fn slot(&self, s: usize, n: usize) -> &[u8] {
        let (g, i) = (s / SEG_SLOTS, s % SEG_SLOTS);
        &self.segs[g][i * self.slot_bytes..i * self.slot_bytes + n]
    }
    fn write(&mut self, s: usize, data: &[u8]) {
        let (g, i) = (s / SEG_SLOTS, s % SEG_SLOTS);
        let sb = self.slot_bytes;
        self.segs[g][i * sb..i * sb + data.len()].copy_from_slice(data);
    }
    fn insert(&mut self, row: u32, data: &[u8]) {
        if self.cap == 0 || self.index.contains_key(&row) {
            return;
        }
        // Grow one segment when no slot is free and the cap allows it.
        if self.free.is_empty() && self.allocated() + SEG_SLOTS <= self.cap {
            let start = self.allocated();
            self.segs.push(vec![0u8; SEG_SLOTS * self.slot_bytes].into_boxed_slice());
            self.rows.resize(start + SEG_SLOTS, u32::MAX);
            self.refd.resize(start + SEG_SLOTS, false);
            self.free.extend((start..start + SEG_SLOTS).rev().map(|s| s as u32));
        }
        if let Some(s) = self.free.pop() {
            let s = s as usize;
            self.rows[s] = row;
            self.index.insert(row, s as u32);
            self.write(s, data);
            return;
        }
        let n = self.allocated();
        if n == 0 {
            return;
        }
        loop {
            let h = self.hand;
            self.hand = (self.hand + 1) % n;
            if self.refd[h] {
                self.refd[h] = false;
                continue;
            }
            self.index.remove(&self.rows[h]);
            self.rows[h] = row;
            self.index.insert(row, h as u32);
            self.write(h, data);
            return;
        }
    }
    /// New cap in slots; shrinking drops whole tail segments (their rows are evicted, their memory freed).
    fn set_cap(&mut self, cap: usize) {
        self.cap = cap;
        let keep_segs = cap / SEG_SLOTS;
        while self.segs.len() > keep_segs {
            let start = (self.segs.len() - 1) * SEG_SLOTS;
            for s in start..self.rows.len() {
                if self.rows[s] != u32::MAX {
                    self.index.remove(&self.rows[s]);
                }
            }
            self.segs.pop();
            self.rows.truncate(start);
            self.refd.truncate(start);
            self.free.retain(|&s| (s as usize) < start);
        }
        if self.hand >= self.rows.len() {
            self.hand = 0;
        }
    }
}

pub struct Table {
    p: PleParams,
    pub row_bytes: usize,
    pub rows: u64,
    reader: Mutex<RowFile>,
    bins: Option<Mutex<BinFile>>,
    cache: Mutex<RowCache>,
    /// rows file and its byte offset (the GGUF tensor, or `--ngram-rows`)
    src: (PathBuf, u64),
    /// `--ngram-resident`: the whole table in RAM once loaded (no cache, no disk reads); dropped under memory pressure
    resident: RwLock<Option<Box<[u8]>>>,
}

#[derive(Default)]
pub struct Counters {
    pub gathers: AtomicU64,
    pub prefetches: AtomicU64,
    pub prefetch_dropped: AtomicU64,
    pub rows: AtomicU64,
    pub hits: AtomicU64,
    pub disk_rows: AtomicU64,
    pub disk_errors: AtomicU64,
    /// rows read one by one from the table (n-grams without a bundle)
    pub row_reads: AtomicU64,
    pub bin_reads_16k: AtomicU64,
    pub bin_reads_4k: AtomicU64,
    /// rows put into the cache from bins (requested ones and their neighbours)
    pub bin_rows: AtomicU64,
    pub loads: AtomicU64,
    pub unloads: AtomicU64,
    pub shrinks: AtomicU64,
    pub gather_us_sum: AtomicU64,
    pub gather_us: [AtomicU64; GATHER_BUCKETS_US.len()],
}

struct Loaded {
    table: Arc<Table>,
    conns: usize,
    idle_since: Option<Instant>,
}

pub struct Ngram {
    configs: HashMap<String, PathBuf>,
    bin_configs: HashMap<String, PathBuf>,
    rows_configs: HashMap<String, RowsSrc>,
    resident_names: HashSet<String>,
    reserve_bytes: u64,
    pub resident_bytes: Arc<AtomicU64>,
    loaded: Mutex<HashMap<String, Loaded>>,
    pub staging_path: PathBuf,
    staging_fd: fs::File,
    staging: *mut u8,
    window_bytes: u64,
    windows: Mutex<Vec<Option<ConnId>>>,
    conns: Mutex<HashMap<ConnId, (String, Arc<Table>, usize)>>,
    prefetch_tx: SyncSender<(Arc<Table>, Vec<NgramSeq>)>,
    pub c: Arc<Counters>,
    /// configured maximum and current cap of the total cache (bytes)
    max_bytes: u64,
    cap_bytes: AtomicU64,
    idle: Duration,
}

// The staging mapping is written only inside the window owned by the requesting connection; everything else is
// behind mutexes or atomics.
unsafe impl Send for Ngram {}
unsafe impl Sync for Ngram {}

/// Rows from another file than the GGUF (`--ngram-rows NAME=FILE@OFFSET/ROW_BYTES`): same hashing and row count (the
/// GGUF's), other codec (EXL3 K6 trellis rows of 122 B, original bf16 rows of 320 B, ...). Opaque bytes to the broker.
/// Bins for such a table are built from the same rows (ngram-binfile --rows; the header's row size is checked).
#[derive(Clone, Debug)]
pub struct RowsSrc {
    pub path: PathBuf,
    pub offset: u64,
    pub row_bytes: usize,
}

impl RowsSrc {
    pub fn parse(v: &str) -> Option<RowsSrc> {
        let (path, rest) = v.rsplit_once('@')?;
        let (off, rb) = rest.split_once('/')?;
        Some(RowsSrc { path: path.into(), offset: off.parse().ok()?, row_bytes: rb.parse().ok()? })
    }
}

fn load_table(name: &str, path: &Path, bins_path: Option<&Path>, rows_src: Option<&RowsSrc>) -> Result<Table, String> {
    let g = Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let p = PleParams::from_gguf(&g)?;
    let t = g.tensors.get(TABLE_TENSOR).ok_or("no n-gram table in the GGUF")?;
    let rows = t.dims[1];
    let (file, offset, row_bytes) = match rows_src {
        Some(r) => {
            let len = fs::metadata(&r.path).map_err(|e| format!("{}: {e}", r.path.display()))?.len();
            if len < r.offset + rows * r.row_bytes as u64 {
                return Err(format!("{}: {len} B, too short for {rows} rows of {} B at {}", r.path.display(), r.row_bytes, r.offset));
            }
            (r.path.clone(), r.offset, r.row_bytes)
        }
        None => {
            let codec = Codec::from_ggml(t.ggml_type).ok_or(format!("unsupported table type {}", t.ggml_type))?;
            (t.path.clone(), t.file_offset, codec.row_bytes(p.head_dim as usize))
        }
    };
    let reader = RowFile::open(&file, offset, row_bytes, rows, 256).map_err(|e| e.to_string())?;
    let bins = match bins_path {
        None => None,
        Some(bp) => {
            let b = BinFile::open(bp, 128).map_err(|e| format!("bins {}: {e}", bp.display()))?;
            b.matches(&p, row_bytes, rows)?;
            eprintln!("ngram: table {name}: bins {} ({} x 16 KiB + {} x 4 KiB, {} n-grams, directory {:.0} MiB)",
                      bp.display(), b.h.n16, b.h.n4, b.keys(), b.dir_bytes() as f64 / (1 << 20) as f64);
            Some(Mutex::new(b))
        }
    };
    Ok(Table { p, row_bytes, rows, reader: Mutex::new(reader), bins, cache: Mutex::new(RowCache::new(row_bytes)),
               src: (file, offset), resident: RwLock::new(None) })
}

fn mem_available() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()))
        .map(|kb| kb << 10)
        .unwrap_or(u64::MAX)
}

impl Table {
    /// Rows of the runs, head-major per token, in request order.
    fn hash(&self, seqs: &[NgramSeq]) -> Vec<u32> {
        let heads = self.p.heads as usize;
        let mut rows = Vec::with_capacity(seqs.iter().map(|s| s.tokens.len()).sum::<usize>() * heads);
        for s in seqs {
            let mut h = History { prev: s.prev };
            let start = rows.len();
            rows.resize(start + s.tokens.len() * heads, 0);
            hash::hash_rows(&self.p, &mut h, &s.tokens, &mut rows[start..]);
        }
        rows
    }

    /// Rows of the runs (like `hash`) and, per group of `heads_per_ngram` rows, its n-gram key (trigram | TRI).
    fn hash_keyed(&self, seqs: &[NgramSeq]) -> (Vec<u32>, Vec<u64>) {
        let hpn = self.p.heads_per_ngram as usize;
        let (mut kb, mut kt) = (Vec::new(), Vec::new());
        let mut keys = Vec::new();
        for s in seqs {
            kb.clear();
            kt.clear();
            hash::ngram_keys(&self.p, &mut History { prev: s.prev }, &s.tokens, &mut kb, &mut kt);
            for i in 0..s.tokens.len() {
                keys.push(kb[i]);
                keys.push(kt[i] | TRI);
            }
        }
        let mut rows = vec![0u32; keys.len() * hpn];
        for (j, &k) in keys.iter().enumerate() {
            let (key, order) = bins::split_key(k);
            hash::rows_of_key(&self.p, key, order, &mut rows[j * hpn..(j + 1) * hpn]);
        }
        (rows, keys)
    }

    /// Fetch the rows of `seqs` (into `dst` in request order if given). Returns (rows, rows that missed, ok).
    fn fetch_seqs(&self, seqs: &[NgramSeq], dst: Option<&mut [u8]>, c: &Counters) -> (usize, u32, bool) {
        {
            let res = self.resident.read().unwrap();
            if let Some(r) = res.as_ref() {
                let rows = self.hash(seqs);
                let rb = self.row_bytes;
                if let Some(d) = dst {
                    if d.len() < rows.len() * rb {
                        return (rows.len(), 0, false);
                    }
                    for (i, &row) in rows.iter().enumerate() {
                        let o = row as usize * rb;
                        d[i * rb..(i + 1) * rb].copy_from_slice(&r[o..o + rb]);
                    }
                }
                c.rows.fetch_add(rows.len() as u64, Relaxed);
                c.hits.fetch_add(rows.len() as u64, Relaxed);
                return (rows.len(), 0, true);
            }
        }
        let Some(bins) = &self.bins else {
            let rows = self.hash(seqs);
            let (d, ok) = self.fetch(&rows, dst, c);
            c.row_reads.fetch_add(d as u64, Relaxed);
            return (rows.len(), d, ok);
        };
        let (rows, keys) = self.hash_keyed(seqs);
        let need = rows.len() * self.row_bytes;
        if let Some(d) = &dst {
            if d.len() < need {
                return (rows.len(), 0, false);
            }
        }
        let (d, ok) = self.fetch_binned(bins, &rows, &keys, dst, c);
        (rows.len(), d, ok)
    }

    /// Like `fetch`, but misses of bundled n-grams read their bin (all its bundles enter the cache); other misses
    /// read single rows, overlapped with the bin reads.
    fn fetch_binned(&self, bins: &Mutex<BinFile>, rows: &[u32], keys: &[u64], mut dst: Option<&mut [u8]>, c: &Counters) -> (u32, bool) {
        let rb = self.row_bytes;
        let hpn = self.p.heads_per_ngram as usize;
        // row -> slot in `got`, for every distinct missing row; positions in the request still to fill
        let mut need: HashMap<u32, usize> = HashMap::new();
        let mut fill: Vec<(usize, usize)> = Vec::new();
        let mut miss_groups: Vec<usize> = Vec::new();
        {
            let mut cache = self.cache.lock().unwrap();
            for (j, grp) in rows.chunks(hpn).enumerate() {
                let mut missed = false;
                for (g, &r) in grp.iter().enumerate() {
                    let i = j * hpn + g;
                    match cache.get(r) {
                        Some(s) => {
                            if let Some(d) = dst.as_deref_mut() {
                                d[i * rb..(i + 1) * rb].copy_from_slice(cache.slot(s, rb));
                            }
                        }
                        None => {
                            missed = true;
                            let n = need.len();
                            let slot = *need.entry(r).or_insert(n);
                            fill.push((i, slot));
                        }
                    }
                }
                if missed {
                    miss_groups.push(j);
                }
            }
        }
        c.rows.fetch_add(rows.len() as u64, Relaxed);
        c.hits.fetch_add((rows.len() - fill.len()) as u64, Relaxed);
        if need.is_empty() {
            return (0, true);
        }
        let mut got = vec![0u8; need.len() * rb];
        let mut have = vec![false; need.len()];
        // Split the missing n-grams: bundled (read the bin) or not (read their missing rows).
        let mut bin_list: Vec<u32> = Vec::new();
        let mut row_list: Vec<u32> = Vec::new();
        {
            let bf = bins.lock().unwrap();
            let mut seen_bins = HashSet::new();
            let mut seen_rows = HashSet::new();
            for &j in &miss_groups {
                match bf.lookup(keys[j]) {
                    Some(b) => {
                        if seen_bins.insert(b) {
                            bin_list.push(b);
                        }
                    }
                    None => {
                        for &r in &rows[j * hpn..(j + 1) * hpn] {
                            if need.contains_key(&r) && seen_rows.insert(r) {
                                row_list.push(r);
                            }
                        }
                    }
                }
            }
        }
        let mut ok = true;
        let mut bf = bins.lock().unwrap();
        let (b16, b4) = (bf.reads_16k, bf.reads_4k);
        if !bin_list.is_empty() && bf.submit(&bin_list).is_err() {
            ok = false;
        }
        // single rows while the bins are in flight
        let mut rbuf = vec![0u8; row_list.len() * rb];
        if !row_list.is_empty() {
            if self.reader.lock().unwrap().read_rows(&row_list, &mut rbuf).is_err() {
                ok = false;
            }
        }
        let mut bin_rows = 0u64;
        if !bin_list.is_empty() && ok {
            let mut cache = self.cache.lock().unwrap();
            let mut br = [0u32; 8];
            let res = bf.wait(|_, ks, data| {
                for (i, &k) in ks.iter().enumerate() {
                    let (key, order) = bins::split_key(k);
                    hash::rows_of_key(&self.p, key, order, &mut br[..hpn]);
                    for (g, &r) in br[..hpn].iter().enumerate() {
                        let bytes = &data[(i * hpn + g) * rb..(i * hpn + g + 1) * rb];
                        cache.insert(r, bytes);
                        bin_rows += 1;
                        if let Some(&s) = need.get(&r) {
                            got[s * rb..(s + 1) * rb].copy_from_slice(bytes);
                            have[s] = true;
                        }
                    }
                }
            });
            if res.is_err() {
                ok = false;
            }
        } else if !bin_list.is_empty() {
            let _ = bf.wait(|_, _, _| {});
        }
        c.bin_reads_16k.fetch_add(bf.reads_16k - b16, Relaxed);
        c.bin_reads_4k.fetch_add(bf.reads_4k - b4, Relaxed);
        drop(bf);
        c.bin_rows.fetch_add(bin_rows, Relaxed);
        if !ok {
            c.disk_errors.fetch_add(1, Relaxed);
            return (0, false);
        }
        {
            let mut cache = self.cache.lock().unwrap();
            for (i, &r) in row_list.iter().enumerate() {
                let bytes = &rbuf[i * rb..(i + 1) * rb];
                cache.insert(r, bytes);
                let s = need[&r];
                got[s * rb..(s + 1) * rb].copy_from_slice(bytes);
                have[s] = true;
            }
        }
        c.row_reads.fetch_add(row_list.len() as u64, Relaxed);
        // Anything a bin should have held but didn't (never expected): read it from the table.
        let rest: Vec<u32> = need.iter().filter(|(_, &s)| !have[s]).map(|(&r, _)| r).collect();
        if !rest.is_empty() {
            eprintln!("ngram: {} rows missing from their bins; read from the table", rest.len());
            let mut buf = vec![0u8; rest.len() * rb];
            if self.reader.lock().unwrap().read_rows(&rest, &mut buf).is_err() {
                c.disk_errors.fetch_add(1, Relaxed);
                return (0, false);
            }
            c.row_reads.fetch_add(rest.len() as u64, Relaxed);
            for (i, r) in rest.iter().enumerate() {
                let s = need[r];
                got[s * rb..(s + 1) * rb].copy_from_slice(&buf[i * rb..(i + 1) * rb]);
            }
        }
        if let Some(d) = dst {
            for &(i, s) in &fill {
                d[i * rb..(i + 1) * rb].copy_from_slice(&got[s * rb..(s + 1) * rb]);
            }
        }
        c.disk_rows.fetch_add(need.len() as u64, Relaxed);
        (need.len() as u32, true)
    }

    /// Ensure `rows` are cached; if `dst` is given, copy them there in order (rows.len() * row_bytes).
    /// Returns (rows read from disk, ok). Disk reads happen outside the cache lock.
    fn fetch(&self, rows: &[u32], mut dst: Option<&mut [u8]>, c: &Counters) -> (u32, bool) {
        let rb = self.row_bytes;
        let mut missing: Vec<u32> = Vec::new();
        {
            let mut cache = self.cache.lock().unwrap();
            let mut seen = HashSet::new();
            for (i, &r) in rows.iter().enumerate() {
                match cache.get(r) {
                    Some(s) => {
                        if let Some(d) = dst.as_deref_mut() {
                            d[i * rb..(i + 1) * rb].copy_from_slice(cache.slot(s, rb));
                        }
                    }
                    None => {
                        if seen.insert(r) {
                            missing.push(r);
                        }
                    }
                }
            }
        }
        c.rows.fetch_add(rows.len() as u64, Relaxed);
        if missing.is_empty() {
            c.hits.fetch_add(rows.len() as u64, Relaxed);
            return (0, true);
        }
        let mut buf = vec![0u8; missing.len() * rb];
        if self.reader.lock().unwrap().read_rows(&missing, &mut buf).is_err() {
            c.disk_errors.fetch_add(1, Relaxed);
            return (0, false);
        }
        c.disk_rows.fetch_add(missing.len() as u64, Relaxed);
        let pos: HashMap<u32, usize> = missing.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        {
            let mut cache = self.cache.lock().unwrap();
            for (i, &r) in missing.iter().enumerate() {
                cache.insert(r, &buf[i * rb..(i + 1) * rb]);
            }
        }
        let mut hit_rows = 0u64;
        match dst {
            Some(d) => {
                for (i, &r) in rows.iter().enumerate() {
                    match pos.get(&r) {
                        Some(&j) => d[i * rb..(i + 1) * rb].copy_from_slice(&buf[j * rb..(j + 1) * rb]),
                        None => hit_rows += 1, // copied under the first lock
                    }
                }
            }
            None => hit_rows = rows.iter().filter(|r| !pos.contains_key(r)).count() as u64,
        }
        c.hits.fetch_add(hit_rows, Relaxed);
        (missing.len() as u32, true)
    }
}

impl Ngram {
    /// `tables`: (name, GGUF path), loaded on first use. `windows` staging windows of `window_bytes` in
    /// `<dir>/ngram-<epoch>` (a sparse file: only windows in use hold memory).
    pub fn new(tables: &[(String, PathBuf)], bin_files: &[(String, PathBuf)], rows_files: &[(String, RowsSrc)], resident: &[String],
               dir: &Path, epoch: u64, max_bytes: u64, windows: usize, window_bytes: u64,
               idle: Duration, reserve_bytes: u64, regrow: Duration) -> Result<Arc<Ngram>, String> {
        for (name, r) in rows_files {
            if !tables.iter().any(|(t, _)| t == name) {
                return Err(format!("--ngram-rows {name}: no --ngram-table of that name"));
            }
            if !r.path.exists() {
                return Err(format!("table {name}: {} does not exist", r.path.display()));
            }
        }
        if let Some(n) = resident.iter().find(|n| !tables.iter().any(|(t, _)| t == *n)) {
            return Err(format!("--ngram-resident {n}: no --ngram-table of that name"));
        }
        for (name, path) in tables.iter().chain(bin_files) {
            if !path.exists() {
                return Err(format!("table {name}: {} does not exist", path.display()));
            }
        }
        if let Some((name, _)) = bin_files.iter().find(|(n, _)| !tables.iter().any(|(t, _)| t == n)) {
            return Err(format!("--ngram-bins {name}: no --ngram-table of that name"));
        }
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        for e in fs::read_dir(dir).map_err(|e| e.to_string())?.flatten() {
            if e.file_name().to_string_lossy().starts_with("ngram-") {
                let _ = fs::remove_file(e.path());
            }
        }
        let staging_path = dir.join(format!("ngram-{epoch:016x}"));
        let total = windows as u64 * window_bytes;
        let f = fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&staging_path)
            .map_err(|e| e.to_string())?;
        f.set_len(total).map_err(|e| e.to_string())?;
        let _ = fs::set_permissions(&staging_path, std::os::unix::fs::PermissionsExt::from_mode(0o644));
        let staging = unsafe {
            libc::mmap(std::ptr::null_mut(), total as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, f.as_raw_fd(), 0)
        };
        if staging == libc::MAP_FAILED {
            return Err("staging mmap failed".into());
        }
        let c = Arc::new(Counters::default());
        let (tx, rx) = sync_channel::<(Arc<Table>, Vec<NgramSeq>)>(64);
        let n = Arc::new(Ngram {
            configs: tables.iter().cloned().collect(),
            bin_configs: bin_files.iter().cloned().collect(),
            rows_configs: rows_files.iter().cloned().collect(),
            resident_names: resident.iter().cloned().collect(),
            reserve_bytes,
            resident_bytes: Arc::new(AtomicU64::new(0)),
            loaded: Mutex::new(HashMap::new()),
            staging_path,
            staging_fd: f,
            staging: staging as *mut u8,
            window_bytes,
            windows: Mutex::new(vec![None; windows]),
            conns: Mutex::new(HashMap::new()),
            prefetch_tx: tx,
            c: c.clone(),
            max_bytes,
            cap_bytes: AtomicU64::new(max_bytes),
            idle,
        });
        std::thread::Builder::new().name("ngram-prefetch".into()).spawn(move || {
            for (t, seqs) in rx {
                t.fetch_seqs(&seqs, None, &c);
            }
        }).map_err(|e| e.to_string())?;
        {
            let n = n.clone();
            std::thread::Builder::new().name("ngram-maint".into()).spawn(move || n.maintain(reserve_bytes, regrow))
                .map_err(|e| e.to_string())?;
        }
        eprintln!("ngram: {} table(s) configured (loaded on first use), cache up to {:.1} GiB, {} staging windows of {} MiB, idle unload {} s",
                  tables.len(), max_bytes as f64 / GIB as f64, windows, window_bytes >> 20, idle.as_secs());
        Ok(n)
    }

    /// Spread the current cap over the loaded tables (by row size; one table is the normal case).
    fn apply_cap(&self) {
        let loaded = self.loaded.lock().unwrap();
        if loaded.is_empty() {
            return;
        }
        // resident tables need no cache: the cap is split over the others
        let cached = loaded.values().filter(|l| l.table.resident.read().unwrap().is_none()).count().max(1);
        let per = self.cap_bytes.load(Relaxed) / cached as u64;
        for l in loaded.values() {
            let t = &l.table;
            let cap = if t.resident.read().unwrap().is_some() { 0 } else { (per / RowCache::cost(t.row_bytes)) as usize };
            let mut c = t.cache.lock().unwrap();
            if c.cap != cap {
                c.set_cap(cap);
            }
        }
    }

    /// Once a second: idle unload, memory-pressure shrink, calm regrow.
    fn maintain(&self, reserve: u64, regrow: Duration) {
        let mut calm_since: Option<Instant> = None;
        loop {
            std::thread::sleep(Duration::from_secs(1));
            self.apply_cap();   // picks up tables that became (or stopped being) resident
            // idle unload
            {
                let mut loaded = self.loaded.lock().unwrap();
                let idle: Vec<String> = loaded.iter()
                    .filter(|(_, l)| l.conns == 0 && l.idle_since.map(|t| t.elapsed() >= self.idle).unwrap_or(false))
                    .map(|(k, _)| k.clone()).collect();
                for k in idle {
                    loaded.remove(&k);
                    self.c.unloads.fetch_add(1, Relaxed);
                    eprintln!("ngram: table {k} unloaded after {} s without users", self.idle.as_secs());
                }
            }
            if reserve == 0 {
                continue;
            }
            let avail = mem_available();
            let cap = self.cap_bytes.load(Relaxed);
            if avail < reserve && self.resident_bytes.load(Relaxed) > 0 {
                // a resident table goes first: its rows are still on disk (cached reads take over)
                for l in self.loaded.lock().unwrap().values() {
                    if let Some(r) = l.table.resident.write().unwrap().take() {
                        self.resident_bytes.fetch_sub(r.len() as u64, Relaxed);
                        self.c.shrinks.fetch_add(1, Relaxed);
                        eprintln!("ngram: MemAvailable {:.1} GiB < reserve {:.1} GiB: resident table dropped ({:.1} GiB)",
                                  avail as f64 / GIB as f64, reserve as f64 / GIB as f64, r.len() as f64 / GIB as f64);
                        break;
                    }
                }
                calm_since = None;
                continue;
            }
            if avail < reserve {
                let cut = (reserve - avail) + (256 << 20);
                let new = cap.saturating_sub(cut);
                if new < cap {
                    self.cap_bytes.store(new, Relaxed);
                    self.apply_cap();
                    self.c.shrinks.fetch_add(1, Relaxed);
                    eprintln!("ngram: MemAvailable {:.1} GiB < reserve {:.1} GiB: cache cap {:.2} -> {:.2} GiB",
                              avail as f64 / GIB as f64, reserve as f64 / GIB as f64, cap as f64 / GIB as f64, new as f64 / GIB as f64);
                }
                calm_since = None;
            } else if avail > reserve + 3 * GIB && cap < self.max_bytes {
                let t = *calm_since.get_or_insert_with(Instant::now);
                if t.elapsed() >= regrow {
                    self.cap_bytes.store((cap + (512 << 20)).min(self.max_bytes), Relaxed);
                    self.apply_cap();
                    calm_since = Some(Instant::now());
                }
            } else {
                calm_since = None;
            }
        }
    }

    pub fn open(&self, conn: ConnId, table: &str, rows: u64, row_bytes: u32) -> Result<(u64, u64, u32, u32), Status> {
        let path = self.configs.get(table).ok_or(Status::NotFound)?;
        let t = {
            let mut loaded = self.loaded.lock().unwrap();
            if !loaded.contains_key(table) {
                let t = load_table(table, path, self.bin_configs.get(table).map(|p| p.as_path()), self.rows_configs.get(table)).map_err(|e| {
                    eprintln!("ngram: table {table}: {e}");
                    Status::NotFound
                })?;
                eprintln!("ngram: table {table} loaded: {} ({} rows x {} B)", path.display(), t.rows, t.row_bytes);
                self.c.loads.fetch_add(1, Relaxed);
                let t = Arc::new(t);
                if self.resident_names.contains(table) {
                    self.load_resident(table, &t);
                }
                loaded.insert(table.to_string(), Loaded { table: t, conns: 0, idle_since: None });
            }
            loaded[table].table.clone()
        };
        self.apply_cap();
        if t.rows != rows || t.row_bytes as u32 != row_bytes {
            eprintln!("ngram: conn {conn} table {table}: client has {rows} x {row_bytes} B, broker {} x {} B", t.rows, t.row_bytes);
            return Err(Status::BadRequest);
        }
        let mut conns = self.conns.lock().unwrap();
        let w = match conns.get(&conn) {
            Some(&(_, _, w)) => w,
            None => {
                let mut ws = self.windows.lock().unwrap();
                let w = ws.iter().position(|x| x.is_none()).ok_or(Status::Full)?;
                ws[w] = Some(conn);
                let mut loaded = self.loaded.lock().unwrap();
                if let Some(l) = loaded.get_mut(table) {
                    l.conns += 1;
                    l.idle_since = None;
                }
                w
            }
        };
        conns.insert(conn, (table.to_string(), t.clone(), w));
        Ok((w as u64 * self.window_bytes, self.window_bytes, t.row_bytes as u32, t.p.heads))
    }

    /// Load the whole table into RAM in the background (gathers use the cache path until it is in). Refused when it
    /// would leave less than the memory reserve + 4 GiB available (GPU jobs on a unified-memory host such as Strix Halo allocate system RAM).
    fn load_resident(&self, name: &str, t: &Arc<Table>) {
        let bytes = t.rows * t.row_bytes as u64;
        let avail = mem_available();
        if avail < bytes + self.reserve_bytes + 4 * GIB {
            eprintln!("ngram: table {name}: not resident ({:.1} GiB + reserve {:.1} + 4 GiB needed, {:.1} GiB available); cached reads",
                      bytes as f64 / GIB as f64, self.reserve_bytes as f64 / GIB as f64, avail as f64 / GIB as f64);
            return;
        }
        let rb = self.resident_bytes.clone();
        let (name, reserve, src) = (name.to_string(), self.reserve_bytes, t.src.clone());
        let weak = Arc::downgrade(t);
        let _ = std::thread::Builder::new().name("ngram-resident".into()).spawn(move || {
            let t0 = Instant::now();
            let mut v = vec![0u8; bytes as usize].into_boxed_slice();
            // stop if the table is unloaded meanwhile or memory runs short while loading
            let ok = ngram_store::reader::read_range(&src.0, src.1, &mut v, &|| weak.strong_count() > 0 && mem_available() > reserve);
            match (ok, weak.upgrade()) {
                (Ok(()), Some(t)) => {
                    *t.resident.write().unwrap() = Some(v);
                    rb.fetch_add(bytes, Relaxed);
                    eprintln!("ngram: table {name} resident: {:.1} GiB in {:.0} s", bytes as f64 / GIB as f64, t0.elapsed().as_secs_f64());
                }
                (Err(e), _) => eprintln!("ngram: table {name}: resident load stopped: {e}"),
                _ => {}
            }
        });
    }

    pub fn gather(&self, conn: ConnId, seqs: &[NgramSeq]) -> Result<(u32, u32), Status> {
        let t0 = Instant::now();
        let (_, t, w) = self.conns.lock().unwrap().get(&conn).cloned().ok_or(Status::NotFound)?;
        let n_rows: usize = seqs.iter().map(|s| s.tokens.len()).sum::<usize>() * t.p.heads as usize;
        let need = (n_rows * t.row_bytes) as u64;
        if need > self.window_bytes {
            return Err(Status::TooBig);
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(self.staging.add(w * self.window_bytes as usize), need as usize) };
        let (n, disk, ok) = t.fetch_seqs(seqs, Some(dst), &self.c);
        if !ok || n != n_rows {
            return Err(Status::Busy);
        }
        let us = t0.elapsed().as_micros() as u64;
        self.c.gathers.fetch_add(1, Relaxed);
        self.c.gather_us_sum.fetch_add(us, Relaxed);
        for (i, b) in GATHER_BUCKETS_US.iter().enumerate() {
            if us <= *b {
                self.c.gather_us[i].fetch_add(1, Relaxed);
            }
        }
        Ok((n_rows as u32, disk))
    }

    pub fn prefetch(&self, conn: ConnId, seqs: Vec<NgramSeq>) -> Status {
        let Some((_, t, _)) = self.conns.lock().unwrap().get(&conn).cloned() else {
            return Status::NotFound;
        };
        match self.prefetch_tx.try_send((t, seqs)) {
            Ok(()) => {
                self.c.prefetches.fetch_add(1, Relaxed);
                Status::Ok
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.c.prefetch_dropped.fetch_add(1, Relaxed);
                Status::Busy
            }
        }
    }

    /// Release the connection's window (pages punched: no memory held by unused windows) and its table reference.
    pub fn disconnect(&self, conn: ConnId) {
        let Some((name, _, w)) = self.conns.lock().unwrap().remove(&conn) else {
            return;
        };
        unsafe {
            libc::fallocate(self.staging_fd.as_raw_fd(), libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                            (w as u64 * self.window_bytes) as libc::off_t, self.window_bytes as libc::off_t);
        }
        self.windows.lock().unwrap()[w] = None;
        if let Some(l) = self.loaded.lock().unwrap().get_mut(&name) {
            l.conns = l.conns.saturating_sub(1);
            if l.conns == 0 {
                l.idle_since = Some(Instant::now());
            }
        }
    }

    pub fn metrics(&self, s: &mut String) {
        use std::fmt::Write as _;
        let c = &self.c;
        for (name, v) in [("gathers", &c.gathers), ("prefetches", &c.prefetches), ("prefetch_dropped", &c.prefetch_dropped),
                          ("rows", &c.rows), ("hit_rows", &c.hits), ("disk_rows", &c.disk_rows), ("disk_errors", &c.disk_errors),
                          ("row_reads", &c.row_reads), ("bin_rows", &c.bin_rows),
                          ("loads", &c.loads), ("unloads", &c.unloads), ("shrinks", &c.shrinks)] {
            let _ = writeln!(s, "# TYPE llmt_kv_ngram_{name}_total counter\nllmt_kv_ngram_{name}_total {}", v.load(Relaxed));
        }
        let _ = writeln!(s, "# TYPE llmt_kv_ngram_bin_reads_total counter");
        let _ = writeln!(s, "llmt_kv_ngram_bin_reads_total{{size=\"16k\"}} {}", c.bin_reads_16k.load(Relaxed));
        let _ = writeln!(s, "llmt_kv_ngram_bin_reads_total{{size=\"4k\"}} {}", c.bin_reads_4k.load(Relaxed));
        let _ = writeln!(s, "# TYPE llmt_kv_ngram_gather_seconds histogram");
        for (i, b) in GATHER_BUCKETS_US.iter().enumerate() {
            let _ = writeln!(s, "llmt_kv_ngram_gather_seconds_bucket{{le=\"{}\"}} {}", *b as f64 / 1e6, c.gather_us[i].load(Relaxed));
        }
        let n = c.gathers.load(Relaxed);
        let _ = writeln!(s, "llmt_kv_ngram_gather_seconds_bucket{{le=\"+Inf\"}} {n}");
        let _ = writeln!(s, "llmt_kv_ngram_gather_seconds_sum {}", c.gather_us_sum.load(Relaxed) as f64 / 1e6);
        let _ = writeln!(s, "llmt_kv_ngram_gather_seconds_count {n}");
        let _ = writeln!(s, "# TYPE llmt_kv_ngram_cache_cap_bytes gauge\nllmt_kv_ngram_cache_cap_bytes {}", self.cap_bytes.load(Relaxed));
        let _ = writeln!(s, "# TYPE llmt_kv_ngram_cache_max_bytes gauge\nllmt_kv_ngram_cache_max_bytes {}", self.max_bytes);
        let _ = writeln!(s, "# TYPE llmt_kv_ngram_resident_bytes gauge\nllmt_kv_ngram_resident_bytes {}", self.resident_bytes.load(Relaxed));
        let _ = writeln!(s, "# TYPE llmt_kv_ngram_cache_bytes gauge");
        let _ = writeln!(s, "# TYPE llmt_kv_ngram_cached_rows gauge");
        let _ = writeln!(s, "# TYPE llmt_kv_ngram_table_users gauge");
        for (name, l) in self.loaded.lock().unwrap().iter() {
            let cache = l.table.cache.lock().unwrap();
            let _ = writeln!(s, "llmt_kv_ngram_cache_bytes{{table=\"{name}\"}} {}", cache.bytes());
            let _ = writeln!(s, "llmt_kv_ngram_cached_rows{{table=\"{name}\"}} {}", cache.index.len());
            let _ = writeln!(s, "llmt_kv_ngram_table_users{{table=\"{name}\"}} {}", l.conns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(i: u32) -> [u8; 90] {
        let mut b = [0u8; 90];
        b[..4].copy_from_slice(&i.to_le_bytes());
        b
    }

    #[test]
    fn grows_lazily_evicts_at_cap_and_shrinks() {
        let mut c = RowCache::new(90);
        assert_eq!(c.allocated(), 0, "nothing allocated before use");
        c.set_cap(2 * SEG_SLOTS);
        for i in 0..10u32 {
            c.insert(i, &row(i));
        }
        assert_eq!(c.allocated(), SEG_SLOTS, "one segment after a few inserts");
        for i in 0..(3 * SEG_SLOTS) as u32 {
            c.insert(i, &row(i));
        }
        assert_eq!(c.allocated(), 2 * SEG_SLOTS, "never beyond the cap");
        assert_eq!(c.index.len(), 2 * SEG_SLOTS);
        // every cached row holds its own bytes
        for (&r, &s) in &c.index {
            assert_eq!(&c.slot(s as usize, 4)[..4], &r.to_le_bytes());
        }
        c.set_cap(SEG_SLOTS);
        assert_eq!(c.allocated(), SEG_SLOTS, "shrink frees whole segments");
        assert!(c.index.values().all(|&s| (s as usize) < SEG_SLOTS));
        for (&r, &s) in &c.index {
            assert_eq!(&c.slot(s as usize, 4)[..4], &r.to_le_bytes());
        }
        // inserts keep working after a shrink, inside the new cap
        for i in 0..1000u32 {
            c.insert(1_000_000 + i, &row(1_000_000 + i));
        }
        assert_eq!(c.allocated(), SEG_SLOTS);
        c.set_cap(0);
        assert_eq!(c.allocated(), 0);
        assert!(c.index.is_empty());
    }
}
