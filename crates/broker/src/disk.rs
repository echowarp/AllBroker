//! SSD tier, inside the broker.
//!
//! Admission-controlled (no write wear for one-off prompts): an entry is written to `<dir>/<ns>-<key>.kvd` only
//! once it has proven reuse: hit at least twice (`--disk-min-hits`, 2; counted across RAM incarnations of the key by a
//! count-min frequency sketch, TinyLFU-style, so a prefix that was evicted and recomputed still accumulates), or it is
//! an ancestor of such an entry (prefix-closed: a chain is only useful from its root). A daily write budget
//! (`--disk-write-gib-day`, 4) caps wear; when it is used up, candidates wait for the next day. Entries are immutable
//! and written once by a background writer (every `--disk-flush-s`); the same rule applies on idle exit / SIGTERM
//! (never a dump of the RAM pool). A GDN / session state entry (kind 1) qualifies after one resume (`--disk-state-min-hits`
//! 1). Two classes share files, writer, checksums and index (built on the persistent hot set, main.rs `HotSet`, which
//! this replaces when `--disk-dir` is set): "core" = the hot set's stable core (every `--hotset-save-s`, the entries with
//! >= `--hotset-min-hits` hits ranked hottest-first within `--hotset-gib`, for `--hotset-stable-saves` passes in a
//! row; listed in `<dir>/CORE`, loaded first at start, evicted from the SSD last) and "spill" (every other admitted
//! entry, LRU). The writer runs at idle CPU and I/O priority. Files are
//! page-aligned and written / read with O_DIRECT: a 4 KiB header (magic, ns, key, kind, format, producer, hits, bytes,
//! last use, meta, data checksum) then the entry's bytes padded to 4 KiB. A budget (`--disk-gib`, default 40 as
//! gufo-kv's) and a per-namespace cap (`--disk-ns-gib`; namespaces are per engine / cache domain) evict the least
//! recently used files. At start the broker reads every header (index), then loads the most recently used entries
//! back into the pool in the background until it is full. A LOOKUP that stops at a key the disk holds queues that run
//! for promotion (read, checksum, insert into the pool evicting by policy), so a later lookup of the same prefix hits
//! (a first-lookup hit needs a HIT_PENDING-capable client: v2). Keys are the engines' (namespace + compat tag /
//! cache domain), so a file can only ever serve the engine configuration that wrote it. No prompt text anywhere:
//! file names are hashes, logs are counts.
//! Weights entries (kind 5, `kvcore::KIND_WEIGHTS`: a vision tower and the like) are never written here, by any pass
//! (admission, strict age-out, hot-set core), and a file of one is deleted at the scan and never loaded: the weights
//! file is on disk already and the engine reloads it on a miss.
use super::Shared;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};
use tracing_mutex::stdsync::{Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAGIC: &[u8; 8] = b"LKVD1\0\0\0";
const HDR: usize = 4096;
type Key = [u8; 32];

pub static DISK: OnceLock<Arc<DiskTier>> = OnceLock::new();

/// One SSD write: everything the copier needs. Planned by the broker (admission, ancestors first, daily
/// budget); executed by the background worker (BG_SSD_JOBS) or, with --bg internal, by the broker; recorded by the
/// broker.
#[derive(Clone, Debug)]
pub struct WriteJob {
    pub ns: u32,
    pub key: Key,
    pub guard: kvproto::Guard,
    pub kind: u8,
    pub format: u16,
    pub producer: u16,
    pub hits: u64,
    pub bytes: u64,
    pub last_used_ms: u64,
    pub meta: Vec<u8>,
    pub path: PathBuf,
    pub codec: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    Written { file_bytes: u64 },
    /// the entry changed while it was copied (evicted / stored again / moved / rewritten): nothing written
    Raced,
    /// the copy failed the entry's RAM checksum: nothing written, the entry evicted
    Corrupt,
    /// the write failed (I/O): nothing left behind
    Failed,
}

impl WriteOutcome {
    pub fn code(self) -> (u8, u64) {
        match self {
            WriteOutcome::Written { file_bytes } => (0, file_bytes),
            WriteOutcome::Raced => (1, 0),
            WriteOutcome::Corrupt => (2, 0),
            WriteOutcome::Failed => (3, 0),
        }
    }
    pub fn from_code(c: u8, file_bytes: u64) -> WriteOutcome {
        match c {
            0 => WriteOutcome::Written { file_bytes },
            1 => WriteOutcome::Raced,
            2 => WriteOutcome::Corrupt,
            _ => WriteOutcome::Failed,
        }
    }
}

/// Queued writes for the worker at most (the next flush plans the rest again).
const QUEUE_MAX: usize = 256;

/// Execute one SSD write: copy the entry's slots (`read(range, out)`: an unleased copy), hash it once (XXH3: the
/// file's checksum, and for raw slots the RAM integrity checksum), let `check(raw_sum)` decide (0 / 1 ok, 3 raced,
/// else corrupt: the broker's checksum_result, which also tells whether the copy still describes the entry), then
/// compress (codec 1) and write the file (O_DIRECT, temporary file, fsync, rename). Never writes a copy that failed
/// the check (a flipped bit must not reach the SSD).
pub fn execute(job: &WriteJob, slot_bytes: u64, read: impl Fn(kvproto::Range, &mut [u8]),
               mut check: impl FnMut(Option<u64>) -> u8) -> WriteOutcome {
    let n = job.bytes as usize;
    let mut b = ABuf::new(HDR + n);
    {
        let s = b.s();
        let mut off_d = HDR;
        for r in &job.guard.ranges {
            let len = ((r.len as u64 * slot_bytes) as usize).min(HDR + n - off_d);
            read(*r, &mut s[off_d..off_d + len]);
            off_d += len;
            if off_d == HDR + n {
                break;
            }
        }
    }
    let sum = xxhash_rust::xxh3::xxh3_64(&b.s()[HDR..HDR + n]);
    let raw_sum = if job.guard.compressed { kvslotz::image_xxh3(&b.s()[HDR..HDR + n], job.bytes) } else { Some(sum) };
    match check(raw_sum) {
        0 | 1 => {}
        3 => return WriteOutcome::Raced,
        _ => return WriteOutcome::Corrupt,
    }
    let h = Header { ns: job.ns, key: job.key, kind: job.kind, format: job.format, producer: job.producer, hits: job.hits,
                     bytes: job.bytes, last_used_ms: job.last_used_ms, meta: job.meta.clone(), sum, algo: SUM_XXH3 };
    build(&h, b.s());
    if job.codec == 1 {
        let z = compress(&b.s()[HDR..HDR + n]);
        let mut c = ABuf::new(HDR + z.len());
        c.s()[..HDR].copy_from_slice(&b.s()[..HDR]);
        c.s()[CODEC_OFF] = 1;
        c.s()[HDR..HDR + z.len()].copy_from_slice(&z);
        b = c;
    }
    let tmp = job.path.with_extension("tmp");
    let ok = fs::OpenOptions::new().write(true).create(true).truncate(true).custom_flags(libc::O_DIRECT).open(&tmp)
        .and_then(|f| {
            if super::fp::hit("disk-write-io") {
                // failpoints build: an I/O error after half the file reached the device
                let half = (b.n / 2) & !4095;
                let _ = f.write_all_at(&b.s()[..half], 0);
                return Err(std::io::Error::from_raw_os_error(libc::EIO));
            }
            f.write_all_at(b.s(), 0).and_then(|_| f.sync_data())
        })
        .and_then(|_| fs::rename(&tmp, &job.path));
    match ok {
        Ok(()) => WriteOutcome::Written { file_bytes: b.n as u64 },
        Err(err) => {
            eprintln!("disk tier: write failed: {err}");
            let _ = fs::remove_file(&tmp);
            WriteOutcome::Failed
        }
    }
}

#[derive(Clone)]
struct Rec {
    bytes: u64,
    file_bytes: u64,
    last_used_ms: u64,
    core: bool,
}

#[derive(Default)]
pub struct Counters {
    pub writes: AtomicU64,
    pub write_bytes: AtomicU64,
    pub promotions: AtomicU64,
    pub promote_bytes: AtomicU64,
    pub restored: AtomicU64,
    pub damaged: AtomicU64,
    pub deleted: AtomicU64,
    /// files removed by INVALIDATE
    pub invalidated: AtomicU64,
    pub promote_queued: AtomicU64,
    /// entries skipped by admission (not enough reuse yet) in write passes, and by the daily budget
    pub not_admitted: AtomicU64,
    pub budget_waits: AtomicU64,
    /// SSD copies discarded because the entry changed while it was copied
    pub copy_raced: AtomicU64,
    /// copies that failed the entry's RAM checksum (not written; the entry was evicted, or waits while leased)
    pub copy_corrupt: AtomicU64,
    /// uncompressed bytes of the entries written compressed (ratio = raw / written)
    pub raw_bytes: AtomicU64,
    /// strict mode: entries handed over by the age sweep (their session proved repeated use)
    pub strict_candidates: AtomicU64,
}

/// Count-min sketch of key reuse (4 rows x 64 Ki saturating u8 counters, halved every AGE increments).
struct Sketch {
    rows: Vec<[u8; 1 << 16]>,
    n: u64,
}
const AGE: u64 = 1 << 20;
impl Sketch {
    fn new() -> Sketch {
        Sketch { rows: (0..4).map(|_| [0u8; 1 << 16]).collect(), n: 0 }
    }
    fn idx(ns: u32, k: &Key, r: usize) -> usize {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for b in ns.to_le_bytes().iter().chain(k.iter()) {
            h = (h ^ *b as u64).wrapping_mul(0x0100_0000_01b3);
        }
        ((h ^ (h >> 29)) & 0xffff) as usize
    }
    fn add(&mut self, ns: u32, k: &Key) {
        for r in 0..4 {
            let i = Self::idx(ns, k, r);
            self.rows[r][i] = self.rows[r][i].saturating_add(1);
        }
        self.n += 1;
        if self.n % AGE == 0 {
            for row in self.rows.iter_mut() {
                for c in row.iter_mut() {
                    *c >>= 1;
                }
            }
        }
    }
    fn est(&self, ns: u32, k: &Key) -> u8 {
        (0..4).map(|r| self.rows[r][Self::idx(ns, k, r)]).min().unwrap()
    }
}

pub struct DiskTier {
    dir: PathBuf,
    pool: super::shm::Shm,
    pool_bytes: u64,
    slot_bytes: u64,
    max_bytes: u64,
    ns_max_bytes: u64,
    index: Mutex<HashMap<(u32, Key), Rec>>,
    queue: Mutex<VecDeque<(u32, Vec<Key>)>>,
    cv: Condvar,
    flush_lock: Mutex<()>,
    /// reuse evidence per key: +1 per lookup hit (stores don't count)
    sketch: Mutex<Sketch>,
    min_hits: u8,
    state_min_hits: u8,
    /// hot set rule for the core class: (budget bytes, min hits, stable passes), streaks
    core_rule: (u64, u64, u32),
    streak: Mutex<HashMap<(u32, Key), u32>>,
    /// daily write budget (bytes) and (UTC day, bytes written that day)
    day_budget: u64,
    day: Mutex<(u64, u64)>,
    /// background worker mode: planned writes waiting for the worker (and their keys), writes it is doing
    /// (key -> (connection, job))
    wqueue: Mutex<(std::collections::VecDeque<WriteJob>, HashSet<(u32, Key)>)>,
    wflight: Mutex<HashMap<(u32, Key), (u64, WriteJob)>>,
    pub c: Counters,
    _lockf: fs::File,
    /// the broker pool (slab class) this tier serves
    pub cls: usize,
    /// 0 raw, 1 = 4-byte planes + zstd 1 per 4 MiB chunk (--disk-codec)
    pub codec: u8,
    /// metric label (the pool class of a unified broker; empty = none)
    pub label: String,
    warmed: std::sync::atomic::AtomicBool,
    /// strict write mode: no write-through; the age sweep hands over what leaves RAM and belongs to
    /// a session resumed >= `sess_resumes` times (after >= the core's gap) over >= `sess_span_ms`; shared class as before
    strict: bool,
    sess_resumes: u32,
    sess_span_ms: u64,
}


fn unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn hex(k: &Key) -> String {
    k.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Option<Key> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let mut k = [0u8; 32];
    for i in 0..32 {
        k[i] = u8::from_str_radix(std::str::from_utf8(&b[2 * i..2 * i + 2]).ok()?, 16).ok()?;
    }
    Some(k)
}

fn checksum(d: &[u8]) -> u64 {
    super::checksum(d)
}

// ------------------------------------------------------------ codec (lossless compression)
//
// Header byte 73 = codec: 0 raw, 1 = 4-byte planes + zstd 1 per 4 MiB chunk (measured on our public-prompt entries:
// 1.18x Arc 27B / gufo, 1.60x Arc 35B; 16-thread decompress 15-21 GB/s vs NVMe 5-7 GB/s). Data of a codec-1 file:
// u32 chunk count, u32 stored length per chunk (bit 31: stored raw, when zstd didn't shrink it), then the chunks.
// The header checksum is over the uncompressed bytes. Chunks are compressed / decompressed in parallel.
const CHUNK: usize = 4 << 20;
const CODEC_OFF: usize = 73;

fn planes4(d: &[u8]) -> Vec<u8> {
    let n = d.len() / 4;
    let mut out = vec![0u8; d.len()];
    for i in 0..n {
        for b in 0..4 {
            out[b * n + i] = d[i * 4 + b];
        }
    }
    out[n * 4..].copy_from_slice(&d[n * 4..]);
    out
}

fn unplanes4(d: &[u8], out: &mut [u8]) {
    let n = d.len() / 4;
    for i in 0..n {
        for b in 0..4 {
            out[i * 4 + b] = d[b * n + i];
        }
    }
    out[n * 4..].copy_from_slice(&d[n * 4..]);
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16)
}

/// Compressed data section (table + chunks).
fn compress(data: &[u8]) -> Vec<u8> {
    let chunks: Vec<&[u8]> = data.chunks(CHUNK).collect();
    let mut out: Vec<Vec<u8>> = vec![Vec::new(); chunks.len()];
    let per = chunks.len().div_ceil(threads()).max(1);
    std::thread::scope(|s| {
        for (ti, part) in out.chunks_mut(per).enumerate() {
            let ch = &chunks;
            s.spawn(move || {
                for (k, o) in part.iter_mut().enumerate() {
                    let c = ch[ti * per + k];
                    let z = zstd::bulk::compress(&planes4(c), 1).unwrap_or_default();
                    *o = if !z.is_empty() && z.len() < c.len() { z } else { Vec::new() };
                }
            });
        }
    });
    let mut v = Vec::with_capacity(data.len() / 2);
    v.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    for (z, c) in out.iter().zip(&chunks) {
        let l = if z.is_empty() { c.len() as u32 | 1 << 31 } else { z.len() as u32 };
        v.extend_from_slice(&l.to_le_bytes());
    }
    for (z, c) in out.iter().zip(&chunks) {
        v.extend_from_slice(if z.is_empty() { c } else { z });
    }
    v
}

/// Decompress a codec-1 data section into `out` (exactly the uncompressed length).
fn decompress(src: &[u8], out: &mut [u8]) -> Option<()> {
    let n = u32::from_le_bytes(src.get(..4)?.try_into().ok()?) as usize;
    if n != out.len().div_ceil(CHUNK) {
        return None;
    }
    let mut parts = Vec::with_capacity(n);
    let mut at = 4 + 4 * n;
    for i in 0..n {
        let l = u32::from_le_bytes(src.get(4 + 4 * i..8 + 4 * i)?.try_into().ok()?);
        let (raw, len) = (l & 1 << 31 != 0, (l & !(1 << 31)) as usize);
        parts.push((raw, src.get(at..at + len)?));
        at += len;
    }
    let outs: Vec<&mut [u8]> = out.chunks_mut(CHUNK).collect();
    let ok = std::sync::atomic::AtomicBool::new(true);
    let per = n.div_ceil(threads()).max(1);
    let mut jobs: Vec<Vec<(&mut [u8], (bool, &[u8]))>> = Vec::new();
    for (k, (o, p)) in outs.into_iter().zip(parts).enumerate() {
        if k % per == 0 {
            jobs.push(Vec::new());
        }
        jobs.last_mut().unwrap().push((o, p));
    }
    std::thread::scope(|s| {
        for job in jobs {
            let ok = &ok;
            s.spawn(move || {
                for (o, (raw, z)) in job {
                    if raw {
                        if z.len() != o.len() {
                            ok.store(false, Relaxed);
                            continue;
                        }
                        o.copy_from_slice(z);
                    } else {
                        match zstd::bulk::decompress(z, o.len()) {
                            Ok(d) if d.len() == o.len() => unplanes4(&d, o),
                            _ => ok.store(false, Relaxed),
                        }
                    }
                }
            });
        }
    });
    ok.load(Relaxed).then_some(())
}

/// Page-aligned buffer for O_DIRECT.
struct ABuf {
    p: *mut u8,
    n: usize,
}
impl ABuf {
    fn new(n: usize) -> ABuf {
        let n = n.div_ceil(4096).max(1) * 4096;
        ABuf { p: unsafe { alloc_zeroed(Layout::from_size_align(n, 4096).unwrap()) }, n }
    }
    fn s(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.p, self.n) }
    }
}
impl Drop for ABuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.p, Layout::from_size_align(self.n, 4096).unwrap()) }
    }
}

struct Header {
    ns: u32,
    key: Key,
    kind: u8,
    format: u16,
    producer: u16,
    hits: u64,
    bytes: u64,
    last_used_ms: u64,
    meta: Vec<u8>,
    sum: u64,
    /// data checksum algorithm (header byte 74): 0 = the old 64-bit mix (files written before), 1 = XXH3-64
    algo: u8,
}

/// Whether a file's data checksum is `sum` under its header's algorithm; an algorithm this build does not know is
/// never a match (the file is dropped as damaged, never trusted).
fn sum_ok(algo: u8, d: &[u8], sum: u64) -> bool {
    match algo {
        SUM_XXH3 => xxhash_rust::xxh3::xxh3_64(d) == sum,
        0 => checksum(d) == sum,
        _ => false,
    }
}

const SUM_XXH3: u8 = 1;

fn parse(h: &[u8]) -> Option<Header> {
    if h.len() < HDR || &h[..8] != MAGIC {
        return None;
    }
    let u16_ = |i: usize| u16::from_le_bytes(h[i..i + 2].try_into().unwrap());
    let u32_ = |i: usize| u32::from_le_bytes(h[i..i + 4].try_into().unwrap());
    let u64_ = |i: usize| u64::from_le_bytes(h[i..i + 8].try_into().unwrap());
    let ml = u16_(77) as usize;
    if 79 + ml + 8 > HDR {
        return None;
    }
    Some(Header {
        ns: u32_(8),
        key: h[12..44].try_into().unwrap(),
        kind: h[44],
        format: u16_(45),
        producer: u16_(47),
        hits: u64_(49),
        bytes: u64_(57),
        last_used_ms: u64_(65),
        // 73..77 reserved
        meta: h[79..79 + ml].to_vec(),
        sum: u64_(79 + ml),
        algo: h[74],
    })
}

fn build(h: &Header, out: &mut [u8]) {
    out[..HDR].fill(0);
    out[..8].copy_from_slice(MAGIC);
    out[8..12].copy_from_slice(&h.ns.to_le_bytes());
    out[12..44].copy_from_slice(&h.key);
    out[44] = h.kind;
    out[45..47].copy_from_slice(&h.format.to_le_bytes());
    out[47..49].copy_from_slice(&h.producer.to_le_bytes());
    out[49..57].copy_from_slice(&h.hits.to_le_bytes());
    out[57..65].copy_from_slice(&h.bytes.to_le_bytes());
    out[65..73].copy_from_slice(&h.last_used_ms.to_le_bytes());
    out[74] = h.algo;
    let ml = h.meta.len().min(HDR - 100);
    out[77..79].copy_from_slice(&(ml as u16).to_le_bytes());
    out[79..79 + ml].copy_from_slice(&h.meta[..ml]);
    out[79 + ml..87 + ml].copy_from_slice(&h.sum.to_le_bytes());
}

impl DiskTier {
    #[allow(clippy::too_many_arguments)]
    pub fn new(dir: &str, pool: &Path, pool_bytes: u64, slot_bytes: u64, gib: f64, ns_gib: f64, min_hits: u8, state_min_hits: u8,
               core_rule: (f64, u64, u32), day_gib: f64) -> Option<DiskTier> {
        use std::io::Write as _;
        use std::os::fd::AsRawFd;
        if fs::create_dir_all(dir).is_err() {
            eprintln!("disk tier off: cannot create {dir}");
            return None;
        }
        let _ = fs::set_permissions(dir, std::os::unix::fs::PermissionsExt::from_mode(0o700));
        let lp = Path::new(dir).join("LOCK");
        let mut lockf = fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&lp).ok()?;
        if unsafe { libc::flock(lockf.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            eprintln!("disk tier off: {dir} is in use by another broker");
            return None;
        }
        let _ = lockf.set_len(0);
        let _ = writeln!(lockf, "pid {}", std::process::id());
        let f = fs::OpenOptions::new().read(true).write(true).open(pool).ok()?;
        let Ok(pmap) = super::shm::Shm::map(&f, pool_bytes as usize, true) else {
            eprintln!("disk tier off: cannot map the pool");
            return None;
        };
        Some(DiskTier {
            dir: dir.into(),
            pool: pmap,
            pool_bytes,
            slot_bytes,
            max_bytes: (gib * (1u64 << 30) as f64) as u64,
            ns_max_bytes: (ns_gib * (1u64 << 30) as f64) as u64,
            index: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            flush_lock: Mutex::new(()),
            sketch: Mutex::new(Sketch::new()),
            min_hits,
            state_min_hits,
            core_rule: ((core_rule.0 * (1u64 << 30) as f64) as u64, core_rule.1, core_rule.2),
            streak: Mutex::new(HashMap::new()),
            day_budget: (day_gib * (1u64 << 30) as f64) as u64,
            day: Mutex::new((0, 0)),
            wqueue: Mutex::new(Default::default()),
            wflight: Mutex::new(HashMap::new()),
            c: Counters::default(),
            _lockf: lockf,
            cls: 0,
            codec: 1,
            label: String::new(),
            warmed: std::sync::atomic::AtomicBool::new(false),
            strict: false,
            sess_resumes: 3,
            sess_span_ms: 3_600_000,
        })
    }

    fn path(&self, ns: u32, key: &Key) -> PathBuf {
        self.dir.join(format!("{ns:08x}-{}.kvd", hex(key)))
    }

    pub fn entries_bytes(&self) -> (u64, u64) {
        let ix = self.index.lock().unwrap();
        (ix.len() as u64, ix.values().map(|r| r.file_bytes).sum())
    }

    /// Read every header into the index (start).
    fn scan(&self) {
        let Ok(rd) = fs::read_dir(&self.dir) else { return };
        let mut hb = ABuf::new(HDR);
        let mut ix = self.index.lock().unwrap();
        for f in rd.flatten() {
            let name = f.file_name().to_string_lossy().into_owned();
            if name.ends_with(".tmp") {
                let _ = fs::remove_file(f.path());
                continue;
            }
            if !name.ends_with(".kvd") {
                continue;
            }
            let ok = fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(f.path())
                .and_then(|fh| fh.read_exact_at(hb.s(), 0).map(|_| fh.metadata().map(|m| m.len()).unwrap_or(0)));
            match (ok, parse(hb.s())) {
                // weights entries never reach the SSD tier; a file of one (older build, or edited) is never restored
                (Ok(_), Some(h)) if h.kind == kvcore::KIND_WEIGHTS => {
                    self.c.deleted.fetch_add(1, Relaxed);
                    let _ = fs::remove_file(f.path());
                }
                (Ok(len), Some(h)) if len >= (HDR as u64 + h.bytes) || (hb.s()[CODEC_OFF] == 1 && len > HDR as u64) => {
                    ix.insert((h.ns, h.key), Rec { bytes: h.bytes, file_bytes: len, last_used_ms: h.last_used_ms, core: false });
                }
                _ => {
                    self.c.damaged.fetch_add(1, Relaxed);
                    let _ = fs::remove_file(f.path());
                }
            }
        }
    }

    /// The core class's manifest: one `<ns>-<key>` per line (hashes only), rewritten atomically.
    fn read_core(&self) {
        let Ok(t) = fs::read_to_string(self.dir.join("CORE")) else { return };
        let mut ix = self.index.lock().unwrap();
        for l in t.lines() {
            let Some((ns, k)) = l.split_once('-') else { continue };
            let (Ok(ns), Some(k)) = (u32::from_str_radix(ns, 16), unhex(k)) else { continue };
            if let Some(r) = ix.get_mut(&(ns, k)) {
                r.core = true;
            }
        }
    }

    fn write_core(&self) {
        use std::io::Write as _;
        let body: String = self.index.lock().unwrap().iter().filter(|(_, r)| r.core)
            .map(|((ns, k), _)| format!("{ns:08x}-{}\n", hex(k))).collect();
        let tmp = self.dir.join("CORE.tmp");
        let ok = fs::File::create(&tmp).and_then(|mut f| f.write_all(body.as_bytes()).and_then(|_| f.sync_all()))
            .and_then(|_| fs::rename(&tmp, self.dir.join("CORE")));
        if let Err(e) = ok {
            eprintln!("disk tier: cannot write the core manifest: {e}");
        }
    }

    /// Hot set pass (every --hotset-save-s): rank the entries with >= min hits hottest first within the core budget;
    /// those ranked for `stable` passes in a row become core (written with their ancestors if not on disk yet, under
    /// the daily budget). Entries that drop out of the ranking stay on disk as spill (no rewrite, LRU later).
    pub fn core_pass(&self, sh: &Shared) {
        if self.strict {
            return; // strict mode: no hot-set writes of entries still in RAM
        }
        let (budget, min_hits, stable) = self.core_rule;
        let hot = sh.cls(self.cls).core.lock().unwrap().hot_entries(min_hits);
        let old = std::mem::take(&mut *self.streak.lock().unwrap());
        let mut new = HashMap::new();
        let mut core = Vec::new();
        let mut total = 0u64;
        for e in hot {
            if total + e.bytes > budget {
                continue;
            }
            total += e.bytes;
            let st = old.get(&(e.ns, e.key)).copied().unwrap_or(0) + 1;
            new.insert((e.ns, e.key), st);
            if st >= stable {
                core.push((e.ns, e.key));
            }
        }
        *self.streak.lock().unwrap() = new;
        self.write_entries(sh, &core);
        {
            let mut ix = self.index.lock().unwrap();
            for k in &core {
                if let Some(r) = ix.get_mut(k) {
                    r.core = true;
                }
            }
        }
        self.write_core();
        eprintln!("disk tier: hot set pass: {} core entries", core.len());
    }

    /// Read one file and check it; (header, data padded).
    fn read(&self, ns: u32, key: &Key) -> Option<(Header, ABuf)> {
        let p = self.path(ns, key);
        let fh = fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&p).ok()?;
        let len = fh.metadata().ok()?.len() as usize;
        let mut b = ABuf::new(len);
        let n = b.n;
        fh.read_exact_at(&mut b.s()[..n.min(len.div_ceil(4096) * 4096)], 0).ok()?;
        let h = parse(&b.s()[..HDR])?;
        if super::fp::hit("disk-read-corrupt") && h.bytes > 0 {
            // codec 1: inside the first chunk's zstd stream (after the chunk table); raw: mid data
            let i = (if b.s()[CODEC_OFF] == 1 { HDR + 16 } else { HDR + h.bytes as usize / 2 }).min(len.saturating_sub(1));
            b.s()[i] ^= 0x5a;   // failpoints build: a bit error on the SSD (the data checksum must catch it)
        }
        if b.s()[CODEC_OFF] == 1 {
            let mut d = ABuf::new(HDR + h.bytes as usize);
            d.s()[..HDR].copy_from_slice(&b.s()[..HDR]);
            let end = len.min(b.n);
            let ok = {
                let src = &b.s()[HDR..end];
                let src: &[u8] = unsafe { std::slice::from_raw_parts(src.as_ptr(), src.len()) };
                decompress(src, &mut d.s()[HDR..HDR + h.bytes as usize]).is_some()
            };
            if !ok || !sum_ok(h.algo, &d.s()[HDR..HDR + h.bytes as usize], h.sum) {
                self.c.damaged.fetch_add(1, Relaxed);
                let _ = fs::remove_file(&p);
                self.index.lock().unwrap().remove(&(ns, *key));
                return None;
            }
            return Some((h, d));
        }
        if HDR + h.bytes as usize > len || !sum_ok(h.algo, &b.s()[HDR..HDR + h.bytes as usize], h.sum) {
            self.c.damaged.fetch_add(1, Relaxed);
            let _ = fs::remove_file(&p);
            self.index.lock().unwrap().remove(&(ns, *key));
            return None;
        }
        Some((h, b))
    }

    /// Insert a file's entry into the pool. `evict`: make room by policy (promotion); else only into free slots
    /// (start-up load). Returns false when it did not fit.
    fn load(&self, sh: &Shared, ns: u32, key: &Key, evict: bool) -> bool {
        let Some((h, mut b)) = self.read(ns, key) else { return true };
        if h.kind == kvcore::KIND_WEIGHTS {
            // never restored (scan drops such files; this covers one written after the scan)
            self.index.lock().unwrap().remove(&(ns, *key));
            let _ = fs::remove_file(self.path(ns, key));
            return true;
        }
        let mut core = sh.cls(self.cls).core.lock().unwrap();
        if core.contains(ns, key) {
            return true;
        }
        if evict && !core.make_room_for(ns, h.bytes) {
            return false;
        }
        let now = sh.now();
        let Some(ranges) = core.insert_committed(ns, *key, h.kind, h.format, h.producer, h.meta.clone(), h.bytes, h.hits.max(1), now) else {
            return false;
        };
        if let Some(bk) = sh.cls(self.cls).backing {
            if bk.lock().unwrap().ensure(&ranges).is_err() {
                core.remove_key(ns, key);
                return false;
            }
        }
        let data = &b.s()[HDR..HDR + h.bytes as usize];
        let mut off_d = 0usize;
        for r in &ranges {
            let off = r.start as u64 * self.slot_bytes;
            let len = (r.len as u64 * self.slot_bytes) as usize;
            let n = len.min(data.len() - off_d);
            if off + n as u64 <= self.pool_bytes {
                // promotion: the slots were just reserved for this entry under the core lock
                unsafe { self.pool.span(off as usize, n) }.write(0, &data[off_d..off_d + n]);
            }
            off_d += n;
        }
        true
    }

    /// Start: index, then the most recently used entries into free pool slots (background).
    pub fn start_load(self: &Arc<Self>, sh: Arc<Shared>) {
        let t = Instant::now();
        self.scan();
        self.read_core();
        let (n, b) = self.entries_bytes();
        eprintln!("disk tier: {} ({n} entries, {:.2} GiB, budget {:.0} GiB, per namespace {:.0} GiB)", self.dir.display(),
                  b as f64 / (1u64 << 30) as f64, self.max_bytes as f64 / (1u64 << 30) as f64, self.ns_max_bytes as f64 / (1u64 << 30) as f64);
        let (me, shl) = (self.clone(), sh.clone());
        std::thread::Builder::new().name("disk-load".into()).spawn(move || {
            let sh = shl;
            let mut v: Vec<((u32, Key), u64, bool)> = me.index.lock().unwrap().iter().map(|(k, r)| (*k, r.last_used_ms, r.core)).collect();
            v.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));   // core class first, then most recently used
            let mut loaded = 0u64;
            for ((ns, key), _, core) in v {
                // on-demand pool: only the hot set's core comes back at start; the rest on demand (promotion)
                if sh.cls(me.cls).backing.is_some() && !core {
                    continue;
                }
                if !me.load(&sh, ns, &key, false) {
                    break;
                }
                loaded += 1;
            }
            me.c.restored.store(loaded, Relaxed);
            eprintln!("disk tier: {loaded} entries loaded into the pool at start ({:.1} s)", t.elapsed().as_secs_f64());
        }).unwrap();
        let me = self.clone();
        std::thread::Builder::new().name("disk-promote".into()).spawn(move || {
            // promotions stay in the broker (an engine waits for them: TTFT) but below the request threads
            unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 10) };
            me.promote_loop(sh)
        }).unwrap();
    }

    /// On-demand pool: the first client of this pool (a model loading) warms it from the SSD tier in the background:
    /// entries used within `within_ms` (the eviction age: what would still be in RAM had the broker kept running),
    /// most recent first, into free slots only, at most `max_bytes`. Once per broker run.
    pub fn warm(self: &Arc<Self>, sh: Arc<Shared>, within_ms: u64, max_bytes: u64) {
        if self.warmed.swap(true, Relaxed) {
            return;
        }
        let me = self.clone();
        std::thread::Builder::new().name("disk-warm".into()).spawn(move || {
            unsafe {
                libc::setpriority(libc::PRIO_PROCESS, 0, 10);
            }
            let t = Instant::now();
            let now = unix_ms();
            let mut v: Vec<((u32, Key), u64, u64)> = me.index.lock().unwrap().iter()
                .filter(|(_, r)| now.saturating_sub(r.last_used_ms) <= within_ms)
                .map(|(k, r)| (*k, r.last_used_ms, r.bytes)).collect();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            let (mut loaded, mut bytes) = (0u64, 0u64);
            for ((ns, key), _, b) in v {
                if bytes + b > max_bytes {
                    break;
                }
                if sh.cls(me.cls).core.lock().unwrap().contains(ns, &key) {
                    continue;
                }
                if !me.load(&sh, ns, &key, false) {
                    break;
                }
                loaded += 1;
                bytes += b;
            }
            me.c.restored.fetch_add(loaded, Relaxed);
            eprintln!("disk tier {}: warmed {loaded} entries ({:.2} GiB) for the first client, {:.1} s", me.label,
                      bytes as f64 / (1u64 << 30) as f64, t.elapsed().as_secs_f64());
        }).unwrap();
    }

    /// Holds off the writer (compaction moves entries).
    pub fn pause(&self) -> tracing_mutex::stdsync::MutexGuard<'_, ()> {
        self.flush_lock.lock().unwrap()
    }

    pub fn set_strict(&mut self, on: bool, resumes: u32, span_ms: u64) {
        self.strict = on;
        self.sess_resumes = resumes;
        self.sess_span_ms = span_ms;
        if on {
            eprintln!("disk tier {}: strict writes (no write-through; at age-out, sessions resumed >= {resumes} times over >= {} s)",
                      self.dir.display(), span_ms / 1000);
        }
    }

    /// Strict mode's session rule (min resumes, min span ms); None = write-through mode.
    pub fn strict_rule(&self) -> Option<(u32, u64)> {
        self.strict.then_some((self.sess_resumes, self.sess_span_ms))
    }

    /// Strict mode: write entries the age sweep is about to evict (with their missing ancestors, prefix-closed),
    /// under the daily budget.
    pub fn write_aged(&self, sh: &Shared, keys: &[(u32, Key)]) {
        if keys.is_empty() {
            return;
        }
        let _g = self.flush_lock.lock().unwrap();
        self.c.strict_candidates.fetch_add(keys.len() as u64, Relaxed);
        self.write_entries(sh, keys);
        self.enforce();
    }

    /// INVALIDATE: the entry's content is known bad: its file goes, and it is never promoted again.
    pub fn forget(&self, ns: u32, key: &Key) {
        if self.index.lock().unwrap().remove(&(ns, *key)).is_some() {
            let _ = fs::remove_file(self.path(ns, key));
            self.c.deleted.fetch_add(1, Relaxed);
            self.c.invalidated.fetch_add(1, Relaxed);
        }
    }

    /// A hot-set core entry (a chat beginning kept on the SSD).
    pub fn is_core(&self, ns: u32, key: &Key) -> bool {
        self.index.lock().unwrap().get(&(ns, *key)).is_some_and(|r| r.core)
    }

    pub fn has(&self, ns: u32, key: &Key) -> bool {
        self.index.lock().unwrap().contains_key(&(ns, *key))
    }

    /// Reuse evidence: the keys a LOOKUP hit.
    pub fn note_hits(&self, ns: u32, keys: &[Key]) {
        if keys.is_empty() {
            return;
        }
        let mut sk = self.sketch.lock().unwrap();
        for k in keys {
            sk.add(ns, k);
        }
    }

    /// LOOKUP stopped at keys[0] (not in RAM): if the disk holds it, promote the run in the background.
    pub fn request(&self, ns: u32, keys: &[Key]) {
        if keys.is_empty() || !self.index.lock().unwrap().contains_key(&(ns, keys[0])) {
            return;
        }
        let mut q = self.queue.lock().unwrap();
        if q.len() < 256 {
            q.push_back((ns, keys.iter().take(64).copied().collect()));
            self.c.promote_queued.fetch_add(1, Relaxed);
            self.cv.notify_one();
        }
    }

    fn promote_loop(&self, sh: Arc<Shared>) {
        loop {
            let (ns, keys) = {
                let mut q = self.queue.lock().unwrap();
                while q.is_empty() {
                    q = self.cv.wait(q).unwrap();
                }
                q.pop_front().unwrap()
            };
            for k in keys {
                let rec = self.index.lock().unwrap().get(&(ns, k)).cloned();
                let Some(rec) = rec else { break };
                if sh.cls(self.cls).core.lock().unwrap().contains(ns, &k) {
                    continue;
                }
                if !self.load(&sh, ns, &k, true) {
                    break;
                }
                self.c.promotions.fetch_add(1, Relaxed);
                self.c.promote_bytes.fetch_add(rec.bytes, Relaxed);
            }
        }
    }

    /// Write the admitted committed entries not on disk yet (and their ancestors first); refresh last-use times;
    /// enforce the budgets.
    pub fn flush(&self, sh: &Shared) {
        if super::fp::hit("disk-flush-hold") {
            return;                                         // failpoints build: the SSD writer waits (test ordering)
        }
        // lock order: never the core lock while holding the index (LOOKUP holds core, then takes the index)
        let _g = self.flush_lock.lock().unwrap();
        let (keys, now_b) = {
            let core = sh.cls(self.cls).core.lock().unwrap();
            let v: Vec<(u32, Key, u64, u64, u8, bool)> = core.committed_keys().into_iter()
                .filter_map(|(ns, k, lu)| core.entry_info(ns, &k).map(|e| (ns, k, lu, e.hits, e.kind, e.shared))).collect();
            (v, sh.now())
        };
        let now_u = unix_ms();
        let to_unix = |b_ms: u64| now_u.saturating_sub(now_b.saturating_sub(b_ms));
        let mut admitted: Vec<(u32, Key)> = Vec::new();
        {
            let mut ix = self.index.lock().unwrap();
            let sk = self.sketch.lock().unwrap();
            let mut skipped = 0u64;
            for (ns, k, lu, hits, kind, shared) in &keys {
                if *kind == kvcore::KIND_WEIGHTS {
                    continue;                               // weights: never on the SSD (the model file is)
                }
                if let Some(r) = ix.get_mut(&(*ns, *k)) {
                    r.last_used_ms = r.last_used_ms.max(to_unix(*lu));
                    continue;
                }
                // admission: proven reuse (hits in this incarnation, or across incarnations by the sketch); a state
                // entry (session state at a prompt / output end) after one resume
                // a partial tail (KIND_TAIL) likewise: it reaches the SSD only once it was resumed
                let need = if *kind == 1 || *kind == kvcore::KIND_TAIL { self.state_min_hits } else { self.min_hits } as u64;
                // shared-prefix class (chat beginnings): admitted at once; strict mode: nothing else is written here
                if !*shared && (self.strict || (*hits).max(sk.est(*ns, k) as u64) < need) {
                    skipped += 1;
                    continue;
                }
                admitted.push((*ns, *k));
            }
            self.c.not_admitted.fetch_add(skipped, Relaxed);
        }
        self.write_entries(sh, &admitted);
        self.enforce();
    }

    /// Write entries not on disk yet with their missing ancestors first (prefix-closed: root to leaf), under the
    /// daily budget.
    fn write_entries(&self, sh: &Shared, keys: &[(u32, Key)]) {
        let (now_b, now_u) = (sh.now(), unix_ms());
        let to_unix = |b_ms: u64| now_u.saturating_sub(now_b.saturating_sub(b_ms));
        let chains: Vec<(u32, Vec<Key>)> = {
            let core = sh.cls(self.cls).core.lock().unwrap();
            keys.iter().map(|(ns, k)| {
                let mut c = core.ancestors(*ns, k);
                c.reverse();
                c.push(*k);
                (*ns, c)
            }).collect()
        };
        let mut todo = Vec::new();
        {
            let ix = self.index.lock().unwrap();
            let mut seen = HashSet::new();
            for (ns, c) in chains {
                for a in c {
                    if !ix.contains_key(&(ns, a)) && seen.insert((ns, a)) {
                        todo.push((ns, a));
                    }
                }
            }
        }
        for (ns, k) in todo {
            if self.index.lock().unwrap().contains_key(&(ns, k)) {
                continue;
            }
            let (e, g0) = {
                let core = sh.cls(self.cls).core.lock().unwrap();
                let (Some(e), Some(g)) = (core.entry_info(ns, &k), core.copy_guard(ns, &k)) else { continue };
                (e, g)
            };
            if e.kind == kvcore::KIND_WEIGHTS {
                continue;                                   // never persisted, whichever pass asked (strict, core)
            }
            if g0.3 {
                continue;                                   // being rewritten in place: next pass
            }
            // daily write budget (wear): candidates wait for the next UTC day
            {
                let mut d = self.day.lock().unwrap();
                let today = now_u / 86_400_000;
                if d.0 != today {
                    *d = (today, 0);
                }
                let need = (HDR as u64 + e.bytes).div_ceil(4096) * 4096;
                // the shared-prefix class is exempt (small: bounded by --shared-gib), it is counted though
                if d.1 + need > self.day_budget && !e.shared {
                    self.c.budget_waits.fetch_add(1, Relaxed);
                    break;
                }
                d.1 += need;
            }
            let job = WriteJob {
                ns, key: k, guard: kvproto::Guard { entry_id: g0.0, ranges: g0.1.clone(), compressed: g0.2, rewriting: g0.3 },
                kind: e.kind, format: e.format, producer: e.producer, hits: e.hits, bytes: e.bytes,
                last_used_ms: to_unix(e.last_used), meta: e.meta.clone(), path: self.path(ns, &k), codec: self.codec,
            };
            if sh.bg_worker {
                // the background worker copies, checks, compresses and writes (BG_SSD_JOBS); the index learns of the
                // file when it reports (BG_SSD_DONE)
                let mut q = self.wqueue.lock().unwrap();
                if q.0.len() < QUEUE_MAX && !q.1.contains(&(ns, k)) {
                    q.1.insert((ns, k));
                    q.0.push_back(job);
                }
                continue;
            }
            let core = &sh.cls(self.cls).core;
            let out = execute(&job, self.slot_bytes,
                              |r, out| {
                                  let off = r.start as u64 * self.slot_bytes;
                                  if off + out.len() as u64 <= self.pool_bytes {
                                      // unleased copy: the checksum check decides whether it is kept
                                      unsafe { self.pool.span(off as usize, out.len()) }.read(0, out);
                                  }
                              },
                              |raw_sum| {
                                  let g = (job.guard.entry_id, job.guard.ranges.clone(), job.guard.compressed, job.guard.rewriting);
                                  match core.lock().unwrap().checksum_result(ns, &k, &g, raw_sum, sh.now()) {
                                      kvcore::ChecksumResult::Stored | kvcore::ChecksumResult::Verified => 0,
                                      kvcore::ChecksumResult::Raced => 3,
                                      _ => 2,
                                  }
                              });
            self.complete(&job, out);
        }
    }

    /// Jobs for the background worker (BG_SSD_JOBS): at most `max`, moved to in flight for connection `conn`.
    pub fn take_jobs(&self, conn: u64, max: usize) -> Vec<WriteJob> {
        let mut q = self.wqueue.lock().unwrap();
        let n = max.min(q.0.len());
        let jobs: Vec<WriteJob> = q.0.drain(..n).collect();
        let mut f = self.wflight.lock().unwrap();
        for j in &jobs {
            q.1.remove(&(j.ns, j.key));
            f.insert((j.ns, j.key), (conn, j.clone()));
        }
        jobs
    }

    /// The worker's report on an in-flight job (BG_SSD_DONE); false if it was not in flight for this connection.
    pub fn job_done(&self, conn: u64, ns: u32, key: &Key, out: WriteOutcome) -> bool {
        let job = {
            let mut f = self.wflight.lock().unwrap();
            match f.get(&(ns, *key)) {
                Some((c, _)) if *c == conn => f.remove(&(ns, *key)).unwrap().1,
                _ => return false,
            }
        };
        self.complete(&job, out);
        true
    }

    /// A worker connection went away: its in-flight jobs are dropped (their temporary files removed; the next flush
    /// plans them again).
    pub fn worker_gone(&self, conn: u64) {
        let mut f = self.wflight.lock().unwrap();
        let gone: Vec<(u32, Key)> = f.iter().filter(|(_, v)| v.0 == conn).map(|(k, _)| *k).collect();
        for k in gone {
            if let Some((_, j)) = f.remove(&k) {
                let _ = fs::remove_file(j.path.with_extension("tmp"));
            }
        }
    }

    /// Record an executed write: the index learns of the file; counters; the daily budget (counted uncompressed when
    /// planned) gets the compression back.
    fn complete(&self, job: &WriteJob, out: WriteOutcome) {
        match out {
            WriteOutcome::Written { file_bytes } => {
                let planned = (HDR as u64 + job.bytes).div_ceil(4096) * 4096;
                if job.codec == 1 {
                    let mut d = self.day.lock().unwrap();
                    d.1 = d.1.saturating_sub(planned.saturating_sub(file_bytes));
                    self.c.raw_bytes.fetch_add(job.bytes, Relaxed);
                }
                self.c.writes.fetch_add(1, Relaxed);
                self.c.write_bytes.fetch_add(file_bytes, Relaxed);
                self.index.lock().unwrap().insert((job.ns, job.key), Rec { bytes: job.bytes, file_bytes,
                                                                            last_used_ms: job.last_used_ms, core: false });
            }
            WriteOutcome::Raced => {
                self.c.copy_raced.fetch_add(1, Relaxed);
            }
            WriteOutcome::Corrupt => {
                self.c.copy_corrupt.fetch_add(1, Relaxed);
                eprintln!("disk tier: an entry of {} bytes failed its RAM checksum while being copied: not written", job.bytes);
            }
            WriteOutcome::Failed => {}
        }
    }

    /// Least recently used files go first: per namespace over its cap, then globally over the budget.
    fn enforce(&self) {
        let mut ix = self.index.lock().unwrap();
        let mut v: Vec<((u32, Key), Rec)> = ix.iter().map(|(k, r)| (*k, r.clone())).collect();
        v.sort_by_key(|(_, r)| (r.core, r.last_used_ms));   // spill class first (LRU), the core last
        let mut per_ns: HashMap<u32, u64> = HashMap::new();
        for ((ns, _), r) in &v {
            *per_ns.entry(*ns).or_default() += r.file_bytes;
        }
        let mut total: u64 = per_ns.values().sum();
        let mut gone = HashSet::new();
        for ((ns, k), r) in &v {
            if total <= self.max_bytes && per_ns[ns] <= self.ns_max_bytes {
                continue;
            }
            if per_ns[ns] > self.ns_max_bytes || total > self.max_bytes {
                let _ = fs::remove_file(self.path(*ns, k));
                total -= r.file_bytes;
                *per_ns.get_mut(ns).unwrap() -= r.file_bytes;
                gone.insert((*ns, *k));
            }
        }
        for g in &gone {
            ix.remove(g);
        }
        self.c.deleted.fetch_add(gone.len() as u64, Relaxed);
    }

    /// Writer thread at idle CPU and I/O priority: admission passes every `every`, hot set passes every `core_every`.
    pub fn flush_loop(self: &Arc<Self>, sh: Arc<Shared>, every: Duration, core_every: Duration) {
        let me = self.clone();
        std::thread::Builder::new().name("disk-flush".into()).spawn(move || {
            unsafe {
                libc::setpriority(libc::PRIO_PROCESS, 0, 19);
                libc::syscall(libc::SYS_ioprio_set, 1, 0, 3 << 13);   // IOPRIO_WHO_PROCESS (this thread), IOPRIO_CLASS_IDLE
            }
            let mut last_core = Instant::now();
            loop {
                std::thread::sleep(every);
                if !core_every.is_zero() && last_core.elapsed() >= core_every {
                    me.core_pass(&sh);
                    last_core = Instant::now();
                }
                me.flush(&sh);
            }
        }).unwrap();
    }

    pub fn metrics(&self, s: &mut String) {
        use std::fmt::Write as _;
        let l = if self.label.is_empty() { String::new() } else { format!("{{class=\"{}\"}}", self.label) };
        let (n, b) = self.entries_bytes();
        let core = self.index.lock().unwrap().values().filter(|r| r.core).count() as u64;
        let today = self.day.lock().unwrap().1;
        for (name, v) in [("entries", n), ("core_entries", core), ("written_today_bytes", today), ("bytes", b)] {
            let _ = writeln!(s, "llmt_kv_disk_{name}{l} {v}");
        }
        for (name, v) in [("writes", &self.c.writes), ("write_bytes", &self.c.write_bytes), ("promotions", &self.c.promotions),
                          ("promote_bytes", &self.c.promote_bytes), ("restored", &self.c.restored), ("damaged", &self.c.damaged),
                          ("deleted", &self.c.deleted), ("invalidated", &self.c.invalidated), ("promote_queued", &self.c.promote_queued),
                          ("not_admitted", &self.c.not_admitted), ("budget_waits", &self.c.budget_waits), ("copy_raced", &self.c.copy_raced),
                          ("copy_corrupt", &self.c.copy_corrupt),
                          ("strict_candidates", &self.c.strict_candidates),
                          ("compressed_raw_bytes", &self.c.raw_bytes)] {
            let _ = writeln!(s, "llmt_kv_disk_{name}_total{l} {}", v.load(Relaxed));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_algorithms_by_version() {
        let d = b"some entry bytes".repeat(100);
        assert!(sum_ok(0, &d, checksum(&d)), "old files: the old mix");
        assert!(sum_ok(SUM_XXH3, &d, xxhash_rust::xxh3::xxh3_64(&d)));
        assert!(!sum_ok(SUM_XXH3, &d, checksum(&d)) && !sum_ok(0, &d, xxhash_rust::xxh3::xxh3_64(&d)));
        for algo in 2..=255u8 {
            assert!(!sum_ok(algo, &d, checksum(&d)) && !sum_ok(algo, &d, xxhash_rust::xxh3::xxh3_64(&d)),
                    "unknown algorithm {algo} trusted");
        }
    }
}
