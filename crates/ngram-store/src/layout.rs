//! Bundle layout: which n-gram bundles exist, how they are packed into 16 KiB bins, and a simulator that replays
//! token streams against a layout and a RAM budget.
//!
//! A bundle is the 8 table rows of one bigram or trigram key, copied side by side (720 B with IQ4_NL rows). A bin is
//! one aligned 16 KiB read: a header with its keys, then its bundles. Bins follow chains of n-grams that usually
//! follow each other (trigram (a,b,c) is the step from bigram (a,b) to bigram (b,c)), so one read tends to serve the
//! next few tokens. Bins are ordered by heat; a prefix of them can be preloaded into RAM.

use crate::hash::{self, PleParams};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const BIN_BYTES: usize = 16 * 1024;
pub const PAGE_BYTES: usize = 4096;
/// Bit 63 tags trigram keys (bigram keys use 36 bits, trigram keys 54, so the tag never collides).
pub const TRI: u64 = 1 << 63;

/// One profile record: key, count, maximum undercount from lossy counting.
#[derive(Clone, Copy, Debug)]
pub struct Rec {
    pub key: u64,
    pub count: u32,
}

pub fn read_profile(path: &Path, limit: usize) -> io::Result<Vec<Rec>> {
    let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut out = Vec::new();
    let mut b = [0u8; 16];
    while out.len() < limit {
        match r.read_exact(&mut b) {
            Ok(()) => out.push(Rec {
                key: u64::from_le_bytes(b[0..8].try_into().unwrap()),
                count: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            }),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

/// Key of the bigram at the same position as a trigram (drop the oldest token).
pub fn bigram_of_trigram(tri: u64) -> u64 {
    tri & ((1u64 << (2 * hash::KEY_BITS)) - 1)
}

/// Bigram that precedes a trigram's position: trigram (ctx0=d, ctx1=c, ctx2=b) steps from bigram (c after b) to
/// bigram (d after c).
pub fn predecessor_bigram(tri: u64) -> u64 {
    tri >> hash::KEY_BITS
}

#[derive(Clone, Debug)]
pub struct BuildOptions {
    pub bigrams: usize,
    pub trigrams: usize,
    pub bundle_bytes: usize,
    /// Trigram bundles placed right after their bigram inside a chain.
    pub tri_per_bigram: usize,
    /// Bundles allowed in a second place (a hot successor already placed elsewhere), as a fraction of all bundles.
    pub dup_fraction: f64,
    /// Bundles only (no chains): bins filled in plain heat order.
    pub no_chains: bool,
    /// Hot tier: chains start from the hottest `chain_bigrams` bigrams and go into 16 KiB bins. Every other selected
    /// bundle goes into 4 KiB bins (rare n-grams gain little from neighbours; 4 KiB reads are faster and the drive
    /// does 2.3x more of them per second), grouped by their shared bigram.
    pub chain_bigrams: usize,
}

pub struct Layout {
    pub bundle_bytes: usize,
    pub per_bin: usize,
    /// Keys of each bin (bigram keys plain, trigram keys | TRI): 16 KiB bins hottest first, then 4 KiB bins.
    pub bins: Vec<Vec<u64>>,
    /// true = 4 KiB bin.
    pub small: Vec<bool>,
    pub per_small: usize,
    /// Primary bin of every key.
    pub dir: HashMap<u64, u32>,
    pub duplicates: usize,
}

impl Layout {
    pub fn bundles_per_bin(bundle_bytes: usize) -> usize {
        Self::bundles_in(BIN_BYTES, bundle_bytes)
    }

    /// Bundles in a block: header (u32 count + u32 reserved) plus one u64 key per bundle.
    pub fn bundles_in(block: usize, bundle_bytes: usize) -> usize {
        (block - 8) / (bundle_bytes + 8)
    }

    pub fn bin_bytes(&self, b: usize) -> usize {
        if self.small[b] { PAGE_BYTES } else { BIN_BYTES }
    }

    pub fn build(bi: &[Rec], tri: &[Rec], o: &BuildOptions) -> Layout {
        let per_bin = Self::bundles_per_bin(o.bundle_bytes);
        let per_small = Self::bundles_in(PAGE_BYTES, o.bundle_bytes);
        let bi = &bi[..bi.len().min(o.bigrams)];
        let tri = &tri[..tri.len().min(o.trigrams)];
        let count: HashMap<u64, u32> = bi.iter().map(|r| (r.key, r.count))
            .chain(tri.iter().map(|r| (r.key | TRI, r.count))).collect();
        // Trigrams attached to their own-position bigram, and successor edges between bigrams (weight = trigram count).
        let mut attached: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut succ: HashMap<u64, Vec<(u32, u64)>> = HashMap::new();
        for r in tri {
            attached.entry(bigram_of_trigram(r.key)).or_default().push(r.key | TRI);
            let (from, to) = (predecessor_bigram(r.key), bigram_of_trigram(r.key));
            if count.contains_key(&from) && count.contains_key(&to) && from != to {
                succ.entry(from).or_default().push((r.count, to));
            }
        }
        for v in succ.values_mut() {
            v.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            v.truncate(8);
        }
        let mut placed: HashMap<u64, u32> = HashMap::with_capacity(count.len());
        let mut bins: Vec<Vec<u64>> = Vec::new();
        let dup_budget = (o.dup_fraction * count.len() as f64) as usize;
        let mut dups = 0usize;
        let mut cur_bin: Vec<u64> = Vec::new();
        let flush = |bins: &mut Vec<Vec<u64>>, cur: &mut Vec<u64>| {
            if !cur.is_empty() {
                bins.push(std::mem::take(cur));
            }
        };
        let place = |k: u64, cur: &mut Vec<u64>, placed: &mut HashMap<u64, u32>, nbins: usize| {
            cur.push(k);
            placed.entry(k).or_insert(nbins as u32);
        };

        if o.no_chains {
            let mut all: Vec<(u32, u64)> = count.iter().map(|(&k, &c)| (c, k)).collect();
            all.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            for (_, k) in all {
                if cur_bin.len() == per_bin {
                    flush(&mut bins, &mut cur_bin);
                }
                let n = bins.len();
                place(k, &mut cur_bin, &mut placed, n);
            }
        } else {
            for r in bi.iter().take(o.chain_bigrams) {
                if placed.contains_key(&r.key) {
                    continue;
                }
                // A chain that ended early leaves room: the next hottest chain continues in the same bin.
                if cur_bin.len() >= per_bin {
                    flush(&mut bins, &mut cur_bin);
                }
                let mut cur = r.key;
                loop {
                    let n = bins.len();
                    if !placed.contains_key(&cur) {
                        place(cur, &mut cur_bin, &mut placed, n);
                    }
                    if let Some(ts) = attached.get(&cur) {
                        let mut ts: Vec<u64> = ts.iter().copied().filter(|t| !placed.contains_key(t)).collect();
                        ts.sort_unstable_by(|a, b| count[b].cmp(&count[a]));
                        for t in ts.into_iter().take(o.tri_per_bigram) {
                            if cur_bin.len() < per_bin {
                                place(t, &mut cur_bin, &mut placed, n);
                            }
                        }
                    }
                    if cur_bin.len() >= per_bin {
                        break;
                    }
                    // Follow the strongest successor; allow a duplicate of an already placed one within budget.
                    let next = succ.get(&cur).and_then(|v| {
                        v.iter().find(|(_, s)| !placed.contains_key(s)).or_else(|| {
                            if dups < dup_budget { v.iter().find(|(_, s)| !cur_bin.contains(s)) } else { None }
                        })
                    });
                    match next {
                        Some(&(_, s)) => {
                            if placed.contains_key(&s) {
                                dups += 1;
                                cur_bin.push(s);
                                if cur_bin.len() >= per_bin {
                                    break;
                                }
                            }
                            cur = s;
                        }
                        None => break,
                    }
                }
            }
            flush(&mut bins, &mut cur_bin);
        }
        let n_hot = bins.len();
        // Rare tier: every selected bundle not yet placed, in 4 KiB bins, grouped by its bigram, hottest group first.
        let group = |k: u64| if k & TRI != 0 { bigram_of_trigram(k & !TRI) } else { k };
        let mut gheat: HashMap<u64, u64> = HashMap::new();
        let mut left: Vec<(u32, u64)> = Vec::new();
        for (&k, &c) in &count {
            if !placed.contains_key(&k) {
                *gheat.entry(group(k)).or_default() += c as u64;
                left.push((c, k));
            }
        }
        left.sort_unstable_by(|a, b| {
            let (ga, gb) = (group(a.1), group(b.1));
            gheat[&gb].cmp(&gheat[&ga]).then(ga.cmp(&gb)).then(b.0.cmp(&a.0)).then(a.1.cmp(&b.1))
        });
        for (_, k) in left {
            if cur_bin.len() == per_small {
                flush(&mut bins, &mut cur_bin);
            }
            let n = bins.len();
            place(k, &mut cur_bin, &mut placed, n);
        }
        flush(&mut bins, &mut cur_bin);

        // Hottest bins first; the directory points every key at the first (hottest) bin holding it.
        let heat = |b: &Vec<u64>| -> u64 { b.iter().map(|k| count.get(k).copied().unwrap_or(0) as u64).sum() };
        // 16 KiB bins first (hottest first), then 4 KiB bins (hottest first): the preloaded prefix favours the hot tier.
        let mut order: Vec<(bool, u64, usize)> = bins.iter().enumerate().map(|(i, b)| (i >= n_hot, heat(b), i)).collect();
        order.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
        let small: Vec<bool> = order.iter().map(|o| o.0).collect();
        let order: Vec<(u64, usize)> = order.into_iter().map(|(_, h, i)| (h, i)).collect();
        let mut slots: Vec<Option<Vec<u64>>> = bins.into_iter().map(Some).collect();
        let bins: Vec<Vec<u64>> = order.into_iter().map(|(_, i)| slots[i].take().unwrap()).collect();
        let mut dir = HashMap::with_capacity(count.len());
        for (i, b) in bins.iter().enumerate() {
            for &k in b {
                dir.entry(k).or_insert(i as u32);
            }
        }
        Layout { bundle_bytes: o.bundle_bytes, per_bin, bins, small, per_small, dir, duplicates: dups }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"NGBL")?;
        w.write_all(&2u32.to_le_bytes())?;
        w.write_all(&(self.bundle_bytes as u32).to_le_bytes())?;
        w.write_all(&(self.bins.len() as u32).to_le_bytes())?;
        for (i, b) in self.bins.iter().enumerate() {
            w.write_all(&[self.small[i] as u8])?;
            w.write_all(&(b.len() as u16).to_le_bytes())?;
            for k in b {
                w.write_all(&k.to_le_bytes())?;
            }
        }
        w.flush()
    }

    pub fn load(path: &Path) -> io::Result<Layout> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut h = [0u8; 16];
        r.read_exact(&mut h)?;
        if &h[0..4] != b"NGBL" {
            return Err(io::Error::other("not a bundle layout"));
        }
        let bundle_bytes = u32::from_le_bytes(h[8..12].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(h[12..16].try_into().unwrap()) as usize;
        let mut bins = Vec::with_capacity(n);
        let mut small = Vec::with_capacity(n);
        let mut dir = HashMap::new();
        for i in 0..n {
            let mut sc = [0u8; 1];
            r.read_exact(&mut sc)?;
            small.push(sc[0] != 0);
            let mut c = [0u8; 2];
            r.read_exact(&mut c)?;
            let m = u16::from_le_bytes(c) as usize;
            let mut b = Vec::with_capacity(m);
            for _ in 0..m {
                let mut k = [0u8; 8];
                r.read_exact(&mut k)?;
                let k = u64::from_le_bytes(k);
                dir.entry(k).or_insert(i as u32);
                b.push(k);
            }
            bins.push(b);
        }
        Ok(Layout { bundle_bytes, per_bin: Self::bundles_per_bin(bundle_bytes), bins, small,
                    per_small: Self::bundles_in(PAGE_BYTES, bundle_bytes), dir, duplicates: 0 })
    }
}

/// CLOCK cache (LRU approximation) over u64 keys with a fixed entry count; `pinned` entries are never evicted.
pub struct Clock {
    slots: Vec<(u64, bool)>,
    index: HashMap<u64, usize>,
    hand: usize,
    cap: usize,
}

impl Clock {
    pub fn new(cap: usize) -> Clock {
        Clock { slots: Vec::with_capacity(cap.min(1 << 26)), index: HashMap::with_capacity(cap.min(1 << 26)), hand: 0, cap }
    }
    pub fn get(&mut self, k: u64) -> bool {
        match self.index.get(&k) {
            Some(&i) => {
                self.slots[i].1 = true;
                true
            }
            None => false,
        }
    }
    pub fn insert(&mut self, k: u64) {
        if self.cap == 0 || self.index.contains_key(&k) {
            return;
        }
        if self.slots.len() < self.cap {
            self.index.insert(k, self.slots.len());
            self.slots.push((k, false));
            return;
        }
        loop {
            let (key, refd) = self.slots[self.hand];
            if refd {
                self.slots[self.hand].1 = false;
                self.hand = (self.hand + 1) % self.cap;
            } else {
                self.index.remove(&key);
                self.slots[self.hand] = (k, false);
                self.index.insert(k, self.hand);
                self.hand = (self.hand + 1) % self.cap;
                return;
            }
        }
    }
}

/// Measured on the 990 PRO (ngram-bench sizes): latency of n reads issued together, interpolated between 1 and 16,
/// extrapolated linearly beyond; deep-queue throughput for prefill estimates.
pub fn batch_latency_us(n4: u64, n16: u64) -> f64 {
    let f = |n: u64, one: f64, sixteen: f64| if n == 0 { 0.0 } else { one + (sixteen - one) * (n - 1) as f64 / 15.0 };
    f(n4, 68.0, 133.0).max(f(n16, 78.0, 235.0))
}
pub const IOPS_4K: f64 = 248_515.0;
pub const IOPS_16K: f64 = 107_513.0;

#[derive(Default, Clone, Debug)]
pub struct ReplayStats {
    pub tokens: u64,
    pub lookups: u64,
    pub hits: u64,
    pub reads_4k: u64,
    pub reads_16k: u64,
    pub exposed_us: f64,
    pub tokens_with_io: u64,
}

impl ReplayStats {
    pub fn add(&mut self, o: &ReplayStats) {
        self.tokens += o.tokens;
        self.lookups += o.lookups;
        self.hits += o.hits;
        self.reads_4k += o.reads_4k;
        self.reads_16k += o.reads_16k;
        self.exposed_us += o.exposed_us;
        self.tokens_with_io += o.tokens_with_io;
    }
    /// Disk time of a 2,048-token prefill chunk at deep queue depth (ms).
    pub fn prefill_ms_per_2k(&self) -> f64 {
        let t = self.tokens.max(1) as f64;
        2048.0 * (self.reads_4k as f64 / IOPS_4K + self.reads_16k as f64 / IOPS_16K) / t * 1e3
    }
}

pub enum Mode<'a> {
    /// Today: every row of every token, row-granular cache, 4 KiB reads.
    Raw,
    /// Bundles in bins: bundle-granular cache, 16 KiB bin reads insert the whole bin; hot bins pinned.
    Bins { layout: &'a Layout, pinned_bytes: u64 },
}

/// Replays documents (token streams) through a cache of `ram_bytes`. Metrics skip the first `warmup` tokens.
pub fn replay(p: &PleParams, docs: &[Vec<i32>], mode: &Mode, ram_bytes: u64, row_bytes: usize, warmup: u64) -> ReplayStats {
    let mut st = ReplayStats::default();
    let mut seen = 0u64;
    let (mut kb, mut kt) = (Vec::new(), Vec::new());
    let mut rows = [0u32; 8];
    match mode {
        Mode::Raw => {
            let mut cache = Clock::new((ram_bytes / (row_bytes as u64 + 16)) as usize);
            for d in docs {
                kb.clear();
                kt.clear();
                hash::ngram_keys(p, &mut hash::History::default(), d, &mut kb, &mut kt);
                for i in 0..d.len() {
                    let mut tok = ReplayStats { tokens: 1, ..Default::default() };
                    for (key, order) in [(kb[i], 2), (kt[i], 3)] {
                        hash::rows_of_key(p, key, order, &mut rows);
                        for &r in &rows {
                            tok.lookups += 1;
                            if cache.get(r as u64) {
                                tok.hits += 1;
                            } else {
                                tok.reads_4k += 1;
                                cache.insert(r as u64);
                            }
                        }
                    }
                    seen += 1;
                    if seen > warmup {
                        tok.exposed_us = batch_latency_us(tok.reads_4k, 0);
                        tok.tokens_with_io = (tok.reads_4k > 0) as u64;
                        st.add(&tok);
                    }
                }
            }
        }
        Mode::Bins { layout, pinned_bytes } => {
            let mut pinned: std::collections::HashSet<u64> = std::collections::HashSet::new();
            let mut used = 0u64;
            for (i, b) in layout.bins.iter().enumerate() {
                let cost = b.len() as u64 * (layout.bundle_bytes as u64 + 16);
                if used + cost > *pinned_bytes {
                    break;
                }
                used += cost;
                pinned.extend(b.iter().copied());
                let _ = i;
            }
            let pinned_bytes = pinned.len() as u64 * (layout.bundle_bytes as u64 + 16);
            let rest = ram_bytes.saturating_sub(pinned_bytes);
            // Dynamic part: bundles (bundle-granular) and fallback rows share the remaining budget half and half.
            let mut bcache = Clock::new((rest / 2 / (layout.bundle_bytes as u64 + 16)) as usize);
            let mut rcache = Clock::new((rest / 2 / (row_bytes as u64 + 16)) as usize);
            for d in docs {
                kb.clear();
                kt.clear();
                hash::ngram_keys(p, &mut hash::History::default(), d, &mut kb, &mut kt);
                for i in 0..d.len() {
                    let mut tok = ReplayStats { tokens: 1, ..Default::default() };
                    let mut bins_read: Vec<u32> = Vec::new();
                    for (key, order) in [(kb[i], 2usize), (kt[i] | TRI, 3usize)] {
                        tok.lookups += 1;
                        if pinned.contains(&key) || bcache.get(key) {
                            tok.hits += 1;
                            continue;
                        }
                        if let Some(&b) = layout.dir.get(&key) {
                            if !bins_read.contains(&b) {
                                bins_read.push(b);
                                if layout.small[b as usize] {
                                    tok.reads_4k += 1;
                                } else {
                                    tok.reads_16k += 1;
                                }
                                for &k in &layout.bins[b as usize] {
                                    bcache.insert(k);
                                }
                            }
                            continue;
                        }
                        // Not bundled: its 8 rows, row-cached, 4 KiB reads.
                        hash::rows_of_key(p, key & !TRI, order, &mut rows);
                        for &r in &rows {
                            if !rcache.get(r as u64) {
                                tok.reads_4k += 1;
                                rcache.insert(r as u64);
                            }
                        }
                    }
                    seen += 1;
                    if seen > warmup {
                        tok.exposed_us = batch_latency_us(tok.reads_4k, tok.reads_16k);
                        tok.tokens_with_io = (tok.reads_4k + tok.reads_16k > 0) as u64;
                        st.add(&tok);
                    }
                }
            }
        }
    }
    st
}

pub fn read_holdout(path: &Path, max_tokens: usize) -> io::Result<Vec<Vec<i32>>> {
    let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut docs = Vec::new();
    let mut cur = Vec::new();
    let mut b = [0u8; 4];
    let mut total = 0usize;
    while total < max_tokens {
        match r.read_exact(&mut b) {
            Ok(()) => {
                let t = u32::from_le_bytes(b);
                if t == u32::MAX {
                    docs.push(std::mem::take(&mut cur));
                } else {
                    cur.push(t as i32);
                    total += 1;
                }
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    if !cur.is_empty() {
        docs.push(cur);
    }
    Ok(docs)
}
