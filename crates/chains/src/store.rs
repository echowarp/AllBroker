//! Global chain store ("token tree"): continuations seen in earlier requests (and corpus seeds), shared by all
//! connections and both engines. Layout: documents (closed requests' outputs) keyed by a document id, plus an index
//! 8-gram hash → up to BRANCHES occurrences (the branches of the tree at that context), each with metadata.
//! Hot occurrences point into a document (id << 32 | offset); permanent ones (corpus seeds, or promoted by accepted
//! tokens) keep their own copy of left context + continuation.
//!
//! What is kept is decided by **heat**: an exponentially decayed sum of usefulness events (seen again, hinted,
//! accepted tokens; rejected tokens subtract), stored as (value, timestamp) and decayed lazily on read/update, so
//! nothing walks the store on the lookup path. Wall-clock seconds, so a snapshot restored later has decayed for the
//! time the broker was down. An occurrence colder than `prune_heat` becomes a prune candidate and is removed only if
//! it stays cold for `prune_grace_s` (any hit / accept in between rescues it); a document is freed when none of its
//! occurrences is left. Over the memory cap, documents are evicted coldest first (heat per token).
//! Inserts and maintenance happen on the idle GC thread, never on the lookup path. Snapshot/restore to a file.
use crate::index::GRAM;
use std::collections::HashMap;
use std::io::Write;

pub const BRANCHES: usize = 4;
/// tokens of left context kept for the suffix check of a permanent entry
pub const LEFT: usize = 64;
pub const MAX_CHAIN: usize = 64;
/// accepted tokens that promote a hot occurrence to permanent
pub const PROMOTE_ACCEPTED: u32 = 256;
/// an occurrence offered this often with an accept rate below DEMOTE_RATE is dropped
pub const DEMOTE_OFFERS: u32 = 256;
pub const DEMOTE_RATE: f32 = 0.10;

pub const SRC_OUTPUT: u8 = 1;
pub const SRC_CORPUS: u8 = 2;

/// Heat event weights. A new occurrence starts at W_SEEN; the break-even accept rate of a hint is
/// W_REJECT / (W_ACCEPT + W_REJECT) = 20% of the offered tokens.
pub const W_SEEN: f32 = 1.0;
pub const W_HIT: f32 = 1.0;
pub const W_ACCEPT: f32 = 4.0;
pub const W_REJECT: f32 = 1.0;
/// defaults (broker flags --chains-heat-half-life-s, --chains-prune-grace-s, --chains-prune-heat)
pub const HALF_LIFE_S: u32 = 6 * 3600;
pub const PRUNE_GRACE_S: u32 = 2 * 3600;
pub const PRUNE_HEAT: f32 = 0.25;
/// full sweeps (candidate marking, pruning, counter halving) at most this often
pub const SWEEP_S: u32 = 60;
/// branch-choice counters (seen / offered / accepted) are halved this often
pub const HALVE_S: u32 = 3600;
/// upper bounds of the heat histogram buckets (last bucket: +Inf)
pub const HEAT_BUCKETS: [f32; 6] = [0.25, 1.0, 4.0, 16.0, 64.0, 256.0];

/// Memory estimate per item (bytes, plus the token arrays).
const DOC_BYTES: usize = 64;
const KEY_BYTES: usize = 48;
const OCC_BYTES: usize = std::mem::size_of::<Occ>();
const COPY_BYTES: usize = 16;

/// The index is split into shards so the idle GC thread can sweep and serialize it one shard per store-lock hold
/// (a lookup only ever `try_lock`s the store: a long hold means missed global hints, never a wait).
pub const SHARDS: usize = 64;

/// 8-gram key → branches, in SHARDS hash maps (by the key's low bits).
pub struct Index {
    pub shards: Vec<HashMap<u64, Vec<Occ>>>,
}

#[inline]
fn shard_of(k: u64) -> usize {
    k as usize & (SHARDS - 1)
}

impl Index {
    fn new() -> Index {
        Index { shards: (0..SHARDS).map(|_| HashMap::new()).collect() }
    }
    pub fn get(&self, k: &u64) -> Option<&Vec<Occ>> {
        self.shards[shard_of(*k)].get(k)
    }
    pub fn get_mut(&mut self, k: &u64) -> Option<&mut Vec<Occ>> {
        self.shards[shard_of(*k)].get_mut(k)
    }
    fn entry(&mut self, k: u64) -> std::collections::hash_map::Entry<'_, u64, Vec<Occ>> {
        self.shards[shard_of(k)].entry(k)
    }
    fn remove(&mut self, k: &u64) {
        self.shards[shard_of(*k)].remove(k);
    }
    pub fn len(&self) -> usize {
        self.shards.iter().map(|m| m.len()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &Vec<Occ>)> {
        self.shards.iter().flatten()
    }
    pub fn values(&self) -> impl Iterator<Item = &Vec<Occ>> {
        self.shards.iter().flat_map(|m| m.values())
    }
}

impl std::ops::Index<&u64> for Index {
    type Output = Vec<Occ>;
    fn index(&self, k: &u64) -> &Vec<Occ> {
        &self.shards[shard_of(*k)][k]
    }
}

/// State of a sweep in progress (it runs shard by shard).
#[derive(Default)]
struct SweepRun {
    halve: bool,
    changed: bool,
    rep: HeatReport,
    us: u64,
}

/// Index keys are 61-bit (so a key + branch index fits in a chain_ref).
pub const KEY_MASK: u64 = (1 << 61) - 1;
/// `Occ::pos` of a permanent copy
pub const PERM: u64 = u64::MAX;

#[inline]
pub fn key_of(t: &[u32]) -> u64 {
    gram_hash(t) & KEY_MASK
}

#[inline]
pub fn gram_hash(t: &[u32]) -> u64 {
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    for &x in t {
        h = (h ^ x as u64).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29);
    }
    h ^ (h >> 31)
}

/// Wall-clock seconds (heat and grace timestamps survive a restart).
pub fn unix_now() -> u32 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as u32)
}

/// Exponentially decayed value, decayed lazily: `v` as of time `t`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Heat {
    pub v: f32,
    pub t: u32,
}

impl Heat {
    pub fn new(v: f32, t: u32) -> Heat {
        Heat { v, t }
    }
    /// value at `now` (`inv_hl` = 1 / half-life in seconds; 0 = no decay)
    #[inline]
    pub fn at(&self, now: u32, inv_hl: f32) -> f32 {
        if self.v == 0.0 {
            return 0.0;
        }
        self.v * (-(now.saturating_sub(self.t) as f32) * inv_hl).exp2()
    }
    /// add an event of weight `w` at `now`; never below 0 (a clock going backwards never makes heat grow)
    #[inline]
    pub fn add(&mut self, now: u32, inv_hl: f32, w: f32) {
        let t = now.max(self.t);
        self.v = (self.at(t, inv_hl) + w).max(0.0);
        self.t = t;
    }
}

#[derive(Clone, Debug)]
pub struct Occ {
    /// hot: document id << 32 | offset of the continuation start in the document; PERM for a permanent copy
    pub pos: u64,
    /// permanent: left context (`left` tokens, ≤ LEFT) + continuation (≤ MAX_CHAIN)
    pub copy: Option<Box<[u32]>>,
    pub left: u16,
    /// continuation tokens available (≤ MAX_CHAIN, bounded by the document's end)
    pub cont_len: u16,
    /// branch-choice counters (halved hourly)
    pub seen: u32,
    pub offered: u32,
    pub accepted: u32,
    pub first_seen: u32,
    pub last_used: u32,
    pub heat: Heat,
    /// prune candidate since (wall-clock s); 0 = not a candidate
    pub cand_since: u32,
    pub source: u8,
    pub engine: u16,
    pub pinned: bool,
}

impl Occ {
    fn score(&self) -> f32 {
        // expected accepted tokens per offer, with a prior from how often the continuation was seen
        (self.accepted as f32 + 4.0 * self.seen as f32) / (self.offered as f32 + 8.0)
    }
    /// corpus seeds are never pruned or evicted
    pub fn is_seed(&self) -> bool {
        self.pinned && self.source == SRC_CORPUS
    }
    fn doc_id(&self) -> Option<u32> {
        self.copy.is_none().then_some((self.pos >> 32) as u32)
    }
    fn copy_len(&self) -> usize {
        self.copy.as_ref().map_or(0, |c| c.len())
    }
}

#[derive(Clone, Debug)]
pub struct Doc {
    pub toks: Box<[u32]>,
    /// decayed sum of its occurrences' heat events (eviction order: heat per token)
    pub heat: Heat,
    /// live occurrences pointing into it; 0 = freed at the next sweep
    pub n_occ: u32,
    pub born: u32,
}

pub struct Store {
    pub docs: HashMap<u32, Doc>,
    next_doc: u32,
    pub index: Index,
    pub mem_cap: usize,
    pub model_tag: String,
    /// set by `load`: the tag came from a snapshot and no engine has confirmed it yet
    pub tag_provisional: bool,
    pub half_life_s: u32,
    pub prune_grace_s: u32,
    pub prune_heat: f32,
    pub sweep_s: u32,
    next_sweep: u32,
    last_halve: u32,
    // memory accounting (kept incrementally: mem_bytes() is O(1))
    arena_tokens: usize,
    n_occ: usize,
    /// occurrences into evicted documents, not yet removed by a sweep
    dead_occ: usize,
    copy_tokens: usize,
    n_copies: usize,
    pub stats: StoreStats,
    /// distribution as of the last sweep
    pub heat: HeatReport,
    sweep_run: Option<SweepRun>,
}

#[derive(Default, Clone, Debug)]
pub struct StoreStats {
    pub docs_in: u64,
    pub tokens_in: u64,
    /// documents evicted over the memory cap
    pub docs_evicted: u64,
    /// documents freed because no occurrence pointed into them any more
    pub docs_freed: u64,
    pub occ_dropped: u64,
    pub promoted: u64,
    pub demoted: u64,
    /// occurrences removed after staying cold for the grace period
    pub pruned: u64,
    /// prune candidates rescued by a hit / accept / repeat during the grace period
    pub rescued: u64,
    /// permanent (promoted) copies evicted over the memory cap
    pub copies_evicted: u64,
    pub lookups: u64,
    pub hits: u64,
    /// counter halvings
    pub decays: u64,
    pub sweeps: u64,
    pub sweep_us: u64,
}

/// Heat distribution as of the last sweep.
#[derive(Default, Clone, Debug)]
pub struct HeatReport {
    /// occurrences per heat bucket (HEAT_BUCKETS upper bounds, then +Inf)
    pub occ: [u64; 7],
    /// documents per heat-per-token bucket
    pub docs: [u64; 7],
    pub candidates: u64,
    pub pinned: u64,
    pub seeds: u64,
    pub heat_sum: f64,
}

pub struct GlobalHit {
    pub match_len: usize,
    pub chain_ref: u64,
    pub tokens: Vec<u32>,
}

pub const GLOBAL_REF: u64 = 1 << 63;

/// chain_ref of a global hint: bit 63 set, 61-bit key << 2, branch index in the low 2 bits.
pub fn global_ref(key: u64, branch: usize) -> u64 {
    GLOBAL_REF | ((key & KEY_MASK) << 2) | (branch as u64 & 3)
}

fn put32(b: &mut Vec<u8>, xs: &[u32]) {
    for &x in xs {
        b.extend_from_slice(&x.to_le_bytes());
    }
}

fn bucket(h: f32) -> usize {
    HEAT_BUCKETS.iter().position(|&b| h <= b).unwrap_or(HEAT_BUCKETS.len())
}

/// The tokens of an occurrence: (array, index of the continuation start); None if its document is gone.
fn occ_tokens<'a>(docs: &'a HashMap<u32, Doc>, o: &'a Occ) -> Option<(&'a [u32], usize)> {
    match &o.copy {
        Some(c) => Some((c, o.left as usize)),
        None => docs.get(&((o.pos >> 32) as u32)).map(|d| (&d.toks[..], (o.pos & 0xFFFF_FFFF) as usize)),
    }
}

impl Store {
    pub fn new(mem_cap: usize) -> Store {
        Store {
            docs: HashMap::new(), next_doc: 0, index: Index::new(), mem_cap, model_tag: String::new(),
            tag_provisional: false, half_life_s: HALF_LIFE_S, prune_grace_s: PRUNE_GRACE_S, prune_heat: PRUNE_HEAT,
            sweep_s: SWEEP_S, next_sweep: 0, last_halve: 0, arena_tokens: 0, n_occ: 0, dead_occ: 0, copy_tokens: 0,
            n_copies: 0, stats: StoreStats::default(), heat: HeatReport::default(), sweep_run: None,
        }
    }

    /// Drop all contents, keep the configuration (memory cap, heat parameters).
    pub fn clear(&mut self) {
        let mut s = Store::new(self.mem_cap);
        (s.half_life_s, s.prune_grace_s, s.prune_heat, s.sweep_s) =
            (self.half_life_s, self.prune_grace_s, self.prune_heat, self.sweep_s);
        *self = s;
    }

    #[inline]
    fn inv_hl(&self) -> f32 {
        if self.half_life_s == 0 { 0.0 } else { 1.0 / self.half_life_s as f32 }
    }

    /// Estimated bytes in use (O(1)); occurrences into evicted documents (and their share of the keys) count as gone.
    pub fn mem_bytes(&self) -> usize {
        let keys = if self.dead_occ == 0 { self.index.len() } else {
            (self.index.len() as f64 * (self.n_occ - self.dead_occ) as f64 / self.n_occ.max(1) as f64) as usize
        };
        self.arena_tokens * 4 + self.docs.len() * DOC_BYTES + keys * KEY_BYTES
            + (self.n_occ - self.dead_occ) * OCC_BYTES + self.copy_tokens * 4 + self.n_copies * COPY_BYTES
    }

    pub fn n_tokens(&self) -> usize {
        self.arena_tokens
    }

    pub fn n_occ(&self) -> usize {
        self.n_occ - self.dead_occ
    }

    pub fn n_pinned(&self) -> usize {
        self.index.values().map(|v| v.iter().filter(|o| o.pinned).count()).sum()
    }

    fn alloc_doc_id(&mut self) -> u32 {
        // ids are not reused while an occurrence could still point at them (4G documents before wrapping)
        loop {
            let id = self.next_doc;
            self.next_doc = self.next_doc.wrapping_add(1);
            if !self.docs.contains_key(&id) {
                return id;
            }
        }
    }

    /// Insert a document (a closed request's output, or a corpus seed when `pinned`: then every occurrence keeps its
    /// own copy and no document is stored).
    pub fn insert_doc(&mut self, toks: &[u32], source: u8, engine: u16, pinned: bool, now: u32) {
        if toks.len() <= GRAM || toks.len() > u32::MAX as usize {
            return;
        }
        let inv = self.inv_hl();
        let id = if pinned { None } else { Some(self.alloc_doc_id()) };
        if let Some(id) = id {
            self.docs.insert(id, Doc { toks: toks.into(), heat: Heat::new(0.0, now), n_occ: 0, born: now });
            self.arena_tokens += toks.len();
        }
        self.stats.docs_in += 1;
        self.stats.tokens_in += toks.len() as u64;
        let copy_at = |e: usize, cont_len: u16| -> (Box<[u32]>, u16) {
            let l0 = e.saturating_sub(LEFT);
            (toks[l0..e + cont_len as usize].into(), (e - l0) as u16)
        };
        let Store { docs, index, stats, n_occ, dead_occ, copy_tokens, n_copies, .. } = self;
        for e in GRAM..toks.len() {
            let key = key_of(&toks[e - GRAM..e]);
            let cont_len = (toks.len() - e).min(MAX_CHAIN) as u16;
            let first = toks[e];
            // most keys keep one branch: start at capacity 1 (a default Vec grows to 4 on the first push)
            let occs = index.entry(key).or_insert_with(|| Vec::with_capacity(1));
            // same continuation start → the same branch: count it (keep the older position, it was offered already)
            let same = occs.iter().position(|o| occ_tokens(docs, o).and_then(|(t, i)| t.get(i)) == Some(&first));
            if let Some(i) = same {
                let o = &mut occs[i];
                o.seen += 1;
                o.last_used = now;
                o.heat.add(now, inv, W_SEEN);
                if o.cand_since != 0 {
                    o.cand_since = 0;
                    stats.rescued += 1;
                }
                // (an occurrence into an evicted document never matches: `same` needs its tokens)
                if let Some(d) = o.doc_id().and_then(|d| docs.get_mut(&d)) {
                    d.heat.add(now, inv, W_SEEN);
                }
                if pinned && !o.is_seed() {
                    // a corpus seed makes the branch permanent: own copy from the seed's tokens
                    if let Some(d) = o.doc_id().and_then(|d| docs.get_mut(&d)) {
                        d.n_occ -= 1;
                    }
                    *copy_tokens -= o.copy_len();
                    *n_copies -= o.copy.is_some() as usize;
                    let (c, l) = copy_at(e, cont_len);
                    *copy_tokens += c.len();
                    *n_copies += 1;
                    (o.copy, o.left, o.cont_len, o.pos, o.pinned, o.source) = (Some(c), l, cont_len, PERM, true, source);
                }
                continue;
            }
            let (copy, left) = if pinned { let (c, l) = copy_at(e, cont_len); (Some(c), l) } else { (None, 0) };
            let pos = match id { Some(id) => (id as u64) << 32 | e as u64, None => PERM };
            let occ = Occ { pos, copy, left, cont_len, seen: 1, offered: 0, accepted: 0, first_seen: now,
                            last_used: now, heat: Heat::new(W_SEEN, now), cand_since: 0, source, engine, pinned };
            let slot = if occs.len() < BRANCHES {
                occs.push(occ);
                occs.len() - 1
            } else {
                // replace a branch into an evicted document first, else the coldest unpinned one
                let victim = occs.iter().enumerate().filter(|(_, o)| !o.pinned)
                    .map(|(i, o)| (occ_tokens(docs, o).is_some(), o.heat.at(now, inv), i))
                    .min_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
                let Some((alive, _, i)) = victim else { continue };
                let old = std::mem::replace(&mut occs[i], occ);
                *n_occ -= 1;
                if !alive {
                    *dead_occ -= 1;
                } else if let Some(d) = old.doc_id().and_then(|d| docs.get_mut(&d)) {
                    d.n_occ -= 1;
                }
                *copy_tokens -= old.copy_len();
                *n_copies -= old.copy.is_some() as usize;
                stats.occ_dropped += 1;
                i
            };
            *n_occ += 1;
            let o = &occs[slot];
            *copy_tokens += o.copy_len();
            *n_copies += o.copy.is_some() as usize;
            if let Some(d) = id.and_then(|id| docs.get_mut(&id)) {
                d.n_occ += 1;
                d.heat.add(now, inv, W_SEEN);
            }
        }
        if let Some(id) = id {
            if self.docs[&id].n_occ == 0 {
                // nothing new in it (all repeats): keep nothing
                let d = self.docs.remove(&id).unwrap();
                self.arena_tokens -= d.toks.len();
                self.stats.docs_freed += 1;
            }
        }
    }

    /// Best continuation for a request whose tokens end with `ctx` (suffix match ≥ gate against the stored left
    /// context). A hit warms the occurrence and rescues a prune candidate.
    pub fn lookup(&mut self, ctx: &[u32], gate: usize, max_chain: usize, now: u32) -> Option<GlobalHit> {
        if ctx.len() < GRAM {
            return None;
        }
        self.stats.lookups += 1;
        let key = key_of(&ctx[ctx.len() - GRAM..]);
        let occs = self.index.get(&key)?;
        let mut best: Option<(f32, usize, usize)> = None; // (score, occ index, match_len)
        for (i, o) in occs.iter().enumerate() {
            let mlen = self.match_len(o, ctx);
            if mlen >= gate && best.is_none_or(|b| o.score() > b.0) {
                best = Some((o.score(), i, mlen));
            }
        }
        let (_, i, mlen) = best?;
        let o = &occs[i];
        let (t, at) = occ_tokens(&self.docs, o)?;
        let n = (o.cont_len as usize).min(max_chain).min(t.len() - at);
        let tokens = t[at..at + n].to_vec();
        if tokens.is_empty() {
            return None;
        }
        let chain_ref = global_ref(key, i);
        let inv = self.inv_hl();
        let o = &mut self.index.get_mut(&key).unwrap()[i];
        o.last_used = now;
        o.heat.add(now, inv, W_HIT);
        if o.cand_since != 0 {
            o.cand_since = 0;
            self.stats.rescued += 1;
        }
        if let Some(d) = o.doc_id().and_then(|d| self.docs.get_mut(&d)) {
            d.heat.add(now, inv, W_HIT);
        }
        self.stats.hits += 1;
        Some(GlobalHit { match_len: mlen, chain_ref, tokens })
    }

    fn match_len(&self, o: &Occ, ctx: &[u32]) -> usize {
        let Some((t, at)) = occ_tokens(&self.docs, o) else { return 0 };
        let cap = ctx.len().min(LEFT).min(at);
        let mut n = 0;
        while n < cap && t[at - 1 - n] == ctx[ctx.len() - 1 - n] {
            n += 1;
        }
        n
    }

    /// Engine feedback for a global hint (chain_ref from `lookup`): heat += W_ACCEPT·accepted − W_REJECT·rejected;
    /// an accept rescues a prune candidate. Promotion to permanent copies the occurrence.
    pub fn feedback(&mut self, chain_ref: u64, offered: u16, accepted: u16, now: u32) {
        if chain_ref & GLOBAL_REF == 0 {
            return;
        }
        let key = (chain_ref >> 2) & KEY_MASK;
        let i = (chain_ref & 3) as usize;
        let inv = self.inv_hl();
        let Store { docs, index, stats, copy_tokens, n_copies, .. } = self;
        let Some(o) = index.get_mut(&key).and_then(|v| v.get_mut(i)) else { return };
        o.offered += offered as u32;
        o.accepted += accepted as u32;
        let w = W_ACCEPT * accepted as f32 - W_REJECT * offered.saturating_sub(accepted) as f32;
        o.heat.add(now, inv, w);
        if accepted > 0 && o.cand_since != 0 {
            o.cand_since = 0;
            stats.rescued += 1;
        }
        let Some(d) = o.doc_id().and_then(|d| docs.get_mut(&d)) else { return };
        d.heat.add(now, inv, w);
        if !o.pinned && o.accepted >= PROMOTE_ACCEPTED {
            let at = (o.pos & 0xFFFF_FFFF) as usize;
            let l0 = at.saturating_sub(LEFT);
            let end = (at + o.cont_len as usize).min(d.toks.len());
            let c: Box<[u32]> = d.toks[l0..end].into();
            d.n_occ -= 1;
            *copy_tokens += c.len();
            *n_copies += 1;
            (o.copy, o.left, o.pos, o.pinned) = (Some(c), (at - l0) as u16, PERM, true);
            stats.promoted += 1;
        }
    }

    /// Idle maintenance: evict over the memory cap (coldest first), and every `sweep_s` (or when many occurrences
    /// point into evicted documents) a full sweep. Returns true if anything was removed or changed state.
    pub fn gc(&mut self, now: u32) -> bool {
        let evicted = self.evict_to_cap(now);
        if self.sweep_due(now) {
            return self.sweep(now) || evicted;
        }
        evicted
    }

    /// A sweep is due every `sweep_s`, or early when many occurrences point into evicted documents.
    pub fn sweep_due(&self, now: u32) -> bool {
        self.sweep_run.is_none() && (now >= self.next_sweep || self.dead_occ * 8 > self.n_occ.max(1024))
    }

    /// Evict documents coldest first (heat per token; older first on ties) until the estimate is under the cap; if
    /// only permanent copies are left, the coldest promoted copies (corpus seeds never).
    pub fn evict_to_cap(&mut self, now: u32) -> bool {
        if self.mem_bytes() <= self.mem_cap {
            return false;
        }
        let inv = self.inv_hl();
        let mut order: Vec<(f32, u32, u32)> =
            self.docs.iter().map(|(&id, d)| (d.heat.at(now, inv) / d.toks.len() as f32, d.born, id)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, _, id) in order {
            if self.mem_bytes() <= self.mem_cap {
                return true;
            }
            let d = self.docs.remove(&id).unwrap();
            self.arena_tokens -= d.toks.len();
            self.dead_occ += d.n_occ as usize;
            self.stats.docs_evicted += 1;
        }
        if self.mem_bytes() > self.mem_cap {
            let mut c: Vec<(f32, u64, usize)> = self.index.iter()
                .flat_map(|(&k, v)| v.iter().enumerate().filter(|(_, o)| o.copy.is_some() && !o.is_seed())
                    .map(move |(i, o)| (o.heat.at(now, inv), k, i)))
                .collect();
            c.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut drop: HashMap<u64, Vec<usize>> = HashMap::new();
            let mut freed = 0;
            let over = self.mem_bytes() - self.mem_cap;
            for (_, k, i) in c {
                if freed >= over {
                    break;
                }
                let o = &self.index[&k][i];
                freed += OCC_BYTES + COPY_BYTES + o.copy_len() * 4;
                drop.entry(k).or_default().push(i);
            }
            for (k, mut ix) in drop {
                ix.sort_unstable_by(|a, b| b.cmp(a));
                let v = self.index.get_mut(&k).unwrap();
                for i in ix {
                    let o = v.remove(i);
                    self.n_occ -= 1;
                    self.copy_tokens -= o.copy_len();
                    self.n_copies -= 1;
                    self.stats.copies_evicted += 1;
                }
                if v.is_empty() {
                    self.index.remove(&k);
                }
            }
        }
        true
    }

    /// Full sweep: drop occurrences into evicted documents and chains that earn nothing; mark occurrences colder than
    /// `prune_heat` as prune candidates and prune those that stayed cold for `prune_grace_s`; free documents without
    /// occurrences; halve the branch counters hourly; refresh the heat report. In one go (`sweep`) or shard by shard
    /// (`sweep_begin`, `sweep_shard` for every shard, `sweep_end`), releasing the store lock in between.
    pub fn sweep(&mut self, now: u32) -> bool {
        self.sweep_begin(now);
        for i in 0..SHARDS {
            self.sweep_shard(i, now);
        }
        self.sweep_end(now)
    }

    pub fn sweep_begin(&mut self, now: u32) {
        if self.last_halve == 0 {
            self.last_halve = now; // a new store: the first halving one HALVE_S after its first sweep
        }
        let halve = now.saturating_sub(self.last_halve) >= HALVE_S;
        if halve {
            self.last_halve = now;
            self.stats.decays += 1;
        }
        self.sweep_run = Some(SweepRun { halve, changed: halve, ..Default::default() });
    }

    pub fn sweep_shard(&mut self, shard: usize, now: u32) {
        let t0 = std::time::Instant::now();
        let inv = self.inv_hl();
        let (grace, thr) = (self.prune_grace_s, self.prune_heat);
        let Store { docs, index, stats, n_occ, dead_occ, copy_tokens, n_copies, sweep_run, .. } = self;
        let Some(run) = sweep_run.as_mut() else { return };
        let SweepRun { halve, changed, rep, .. } = run;
        index.shards[shard].retain(|_, v| {
            v.retain_mut(|o| {
                let doc = match o.doc_id() {
                    Some(id) => match docs.get_mut(&id) {
                        Some(d) => Some(d),
                        None => {
                            *n_occ -= 1;
                            *dead_occ -= 1;
                            stats.occ_dropped += 1;
                            return false;
                        }
                    },
                    None => None,
                };
                // offered/accepted count tokens: drop a chain that keeps costing verify rows and earns little
                let demote = !o.pinned && o.offered >= DEMOTE_OFFERS && (o.accepted as f32) < DEMOTE_RATE * o.offered as f32;
                let h = o.heat.at(now, inv);
                let prune = if o.is_seed() || h >= thr {
                    o.cand_since = 0;
                    false
                } else if o.cand_since == 0 {
                    o.cand_since = now.max(1);
                    *changed = true;
                    false
                } else {
                    now.saturating_sub(o.cand_since) >= grace
                };
                if demote || prune {
                    *n_occ -= 1;
                    match doc {
                        Some(d) => d.n_occ -= 1,
                        None => {
                            *copy_tokens -= o.copy_len();
                            *n_copies -= 1;
                        }
                    }
                    if demote { stats.demoted += 1 } else { stats.pruned += 1 }
                    *changed = true;
                    return false;
                }
                if *halve {
                    o.seen /= 2;
                    o.offered /= 2;
                    o.accepted /= 2;
                }
                rep.occ[bucket(h)] += 1;
                rep.heat_sum += h as f64;
                rep.candidates += (o.cand_since != 0) as u64;
                rep.pinned += o.pinned as u64;
                rep.seeds += o.is_seed() as u64;
                true
            });
            !v.is_empty()
        });
        run.us += t0.elapsed().as_micros() as u64;
    }

    /// Finish a sweep: free documents without occurrences, publish the heat report. Returns true if anything changed.
    pub fn sweep_end(&mut self, now: u32) -> bool {
        let Some(mut run) = self.sweep_run.take() else { return false };
        let t0 = std::time::Instant::now();
        let inv = self.inv_hl();
        let mut freed = 0;
        let Store { docs, stats, .. } = self;
        docs.retain(|_, d| {
            if d.n_occ == 0 {
                freed += d.toks.len();
                stats.docs_freed += 1;
                return false;
            }
            run.rep.docs[bucket(d.heat.at(now, inv) / d.toks.len() as f32)] += 1;
            true
        });
        self.arena_tokens -= freed;
        self.heat = run.rep;
        self.next_sweep = now.saturating_add(self.sweep_s);
        self.stats.sweeps += 1;
        self.stats.sweep_us = run.us + t0.elapsed().as_micros() as u64;
        run.changed || freed > 0
    }

    // ---------------------------------------------------------------- snapshot

    /// Snapshot v2: "LLMTCH2\n" + JSON header line, then (little endian)
    /// - u32 n_docs, n_docs × document: id u32, len u32, heat f32, heat_t u32, born u32, len × u32 tokens;
    /// - per index shard (`shards` in the header): u32 n_keys, n_keys × (key u64, n u8, n × occurrence: pos u64,
    ///   cont_len u16, left u16, copy_len u32, seen, offered, accepted, first_seen, last_used u32, heat f32,
    ///   heat_t u32, cand_since u32, source u8, pinned u8, engine u16, copy_len × u32).
    ///
    /// Serialized into memory under the store lock, in pieces (`serialize_head`, then `serialize_shard` per shard;
    /// the lock can be released in between: occurrences whose document is not in the file are skipped at load), and
    /// written outside the lock by `write_snapshot`.
    pub fn serialize(&self, now: u32) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.mem_bytes() * 2 / 3 + 4096);
        self.serialize_head(now, &mut b);
        for i in 0..SHARDS {
            self.serialize_shard(i, &mut b);
        }
        b
    }

    pub fn serialize_head(&self, now: u32, b: &mut Vec<u8>) {
        let hdr = format!(
            "{{\"version\":2,\"saved_unix\":{now},\"shards\":{SHARDS},\"next_doc\":{},\"last_halve\":{},\
             \"half_life_s\":{},\"model_tag\":{:?}}}\n",
            self.next_doc, self.last_halve, self.half_life_s, self.model_tag);
        b.extend_from_slice(b"LLMTCH2\n");
        b.extend_from_slice(hdr.as_bytes());
        put32(b, &[self.docs.len() as u32]);
        for (&id, d) in &self.docs {
            put32(b, &[id, d.toks.len() as u32, d.heat.v.to_bits(), d.heat.t, d.born]);
            put32(b, &d.toks);
        }
    }

    pub fn serialize_shard(&self, shard: usize, b: &mut Vec<u8>) {
        let m = &self.index.shards[shard];
        put32(b, &[m.len() as u32]);
        for (k, v) in m {
            b.extend_from_slice(&k.to_le_bytes());
            b.push(v.len() as u8);
            for o in v {
                b.extend_from_slice(&o.pos.to_le_bytes());
                b.extend_from_slice(&o.cont_len.to_le_bytes());
                b.extend_from_slice(&o.left.to_le_bytes());
                put32(b, &[o.copy_len() as u32, o.seen, o.offered, o.accepted, o.first_seen, o.last_used,
                           o.heat.v.to_bits(), o.heat.t, o.cand_since]);
                b.extend_from_slice(&[o.source, o.pinned as u8]);
                b.extend_from_slice(&o.engine.to_le_bytes());
                if let Some(c) = &o.copy {
                    put32(b, c);
                }
            }
        }
    }

    /// Write a serialized snapshot: temp file, fsync, mode 0600, atomic rename.
    pub fn write_snapshot(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
        use std::os::unix::fs::OpenOptionsExt;
        let tmp = path.with_extension("tmp");
        {
            let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
        }
        std::fs::set_permissions(&tmp, std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
        std::fs::rename(&tmp, path)
    }

    pub fn save(&self, path: &std::path::Path, now: u32) -> std::io::Result<()> {
        Self::write_snapshot(path, &self.serialize(now))
    }

    /// Restore a snapshot. Heat timestamps are wall clock, so the time the broker was down counts as decay; prune
    /// candidates restart their grace period at `now` (they could not be hit while the broker was down). A v1
    /// (pre-heat) or unknown snapshot is an error (the caller starts empty).
    pub fn load(path: &std::path::Path, mem_cap: usize, now: u32) -> std::io::Result<Store> {
        let b = std::fs::read(path)?;
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        if b.starts_with(b"LLMTCH1\n") {
            return Err(bad("snapshot format v1 (before heat-based pruning) is not supported"));
        }
        if !b.starts_with(b"LLMTCH2\n") {
            return Err(bad("not a chains snapshot (or an unknown format version)"));
        }
        let nl = b[8..].iter().position(|&c| c == b'\n').ok_or_else(|| bad("no header"))? + 8;
        let hdr = std::str::from_utf8(&b[8..nl]).map_err(|_| bad("header"))?;
        let num = |k: &str| -> std::io::Result<u64> {
            let i = hdr.find(&format!("\"{k}\":")).ok_or_else(|| bad(k))? + k.len() + 3;
            hdr[i..].split(|c: char| !c.is_ascii_digit()).next().unwrap_or("").parse().map_err(|_| bad(k))
        };
        if num("version")? != 2 {
            return Err(bad("unknown snapshot version"));
        }
        let nshards = num("shards")? as usize;
        let tag = hdr.split("\"model_tag\":\"").nth(1).and_then(|s| s.split('"').next()).unwrap_or("").to_string();
        let mut s = Store::new(mem_cap);
        s.next_doc = num("next_doc")? as u32;
        s.last_halve = num("last_halve")? as u32;
        s.tag_provisional = !tag.is_empty();
        s.model_tag = tag;
        let mut p = nl + 1;
        let mut take = |n: usize| -> std::io::Result<&[u8]> {
            if p + n > b.len() {
                return Err(bad("truncated"));
            }
            p += n;
            Ok(&b[p - n..p])
        };
        fn u32v(s: &[u8]) -> Vec<u32> {
            s.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
        }
        let nd = u32v(take(4)?)[0];
        for _ in 0..nd {
            let h = u32v(take(20)?);
            let toks: Box<[u32]> = u32v(take(h[1] as usize * 4)?).into();
            s.arena_tokens += toks.len();
            s.docs.insert(h[0], Doc { toks, heat: Heat::new(f32::from_bits(h[2]), h[3]), n_occ: 0, born: h[4] });
        }
        for _ in 0..nshards {
            let nk = u32v(take(4)?)[0];
            for _ in 0..nk {
                let k = u64::from_le_bytes(take(8)?.try_into().unwrap());
                let n = take(1)?[0] as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    let pos = u64::from_le_bytes(take(8)?.try_into().unwrap());
                    let cl = u16::from_le_bytes(take(2)?.try_into().unwrap());
                    let left = u16::from_le_bytes(take(2)?.try_into().unwrap());
                    let f = u32v(take(36)?);
                    let sp = take(2)?.to_vec();
                    let engine = u16::from_le_bytes(take(2)?.try_into().unwrap());
                    let copy = if pos == PERM { Some(u32v(take(f[0] as usize * 4)?).into_boxed_slice()) } else { None };
                    let o = Occ { pos, copy, left, cont_len: cl, seen: f[1], offered: f[2], accepted: f[3], first_seen: f[4],
                                  last_used: f[5], heat: Heat::new(f32::from_bits(f[6]), f[7]),
                                  cand_since: if f[8] != 0 { now.max(1) } else { 0 }, source: sp[0], engine, pinned: sp[1] != 0 };
                    match o.doc_id() {
                        Some(d) => match s.docs.get_mut(&d) {
                            Some(d) => d.n_occ += 1,
                            None => continue, // its document was not saved
                        },
                        None => {
                            s.copy_tokens += o.copy_len();
                            s.n_copies += 1;
                        }
                    }
                    s.n_occ += 1;
                    v.push(o);
                }
                if !v.is_empty() {
                    s.index.entry(k).or_insert(v);
                }
            }
        }
        if p != b.len() {
            return Err(bad("trailing bytes"));
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u32 = 1_800_000_000;

    fn doc(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| seed.wrapping_mul(1_000_003).wrapping_add(i * 7919) % 150_000).collect()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("chains-{name}-{}.bin", std::process::id()))
    }

    /// the O(1) memory estimate matches a full recount
    fn check_accounting(s: &Store) {
        let occ: usize = s.index.values().map(|v| v.iter().filter(|o| occ_tokens(&s.docs, o).is_some()).count()).sum();
        let copies: usize = s.index.values().flatten().map(|o| o.copy_len()).sum();
        let ncopies = s.index.values().flatten().filter(|o| o.copy.is_some()).count();
        let toks: usize = s.docs.values().map(|d| d.toks.len()).sum();
        assert_eq!(s.n_occ - s.dead_occ, occ, "live occurrences");
        assert_eq!((s.copy_tokens, s.n_copies, s.arena_tokens), (copies, ncopies, toks));
        for (id, d) in &s.docs {
            let n = s.index.values().flatten().filter(|o| o.doc_id() == Some(*id)).count();
            assert_eq!(d.n_occ as usize, n, "doc {id} refcount");
        }
    }

    #[test]
    fn chain_from_another_request() {
        let mut s = Store::new(64 << 20);
        let d = doc(1, 300);
        s.insert_doc(&d, SRC_OUTPUT, 1, false, T0);
        // another request ends with 30 tokens of d[100..130]: the store offers d[130..194]
        let mut ctx: Vec<u32> = (0..50).map(|i| 900_000 + i).collect();
        ctx.extend_from_slice(&d[100..130]);
        let h = s.lookup(&ctx, 16, 64, T0 + 1).expect("global hit");
        assert_eq!(h.match_len, 30);
        assert_eq!(h.tokens, d[130..194]);
        assert!(h.chain_ref & GLOBAL_REF != 0);
        assert!(s.lookup(&ctx[..60], 16, 64, T0 + 1).is_none_or(|h| h.match_len >= 16));
        check_accounting(&s);
    }

    #[test]
    fn branches_counted_and_best_chosen_by_feedback() {
        let mut s = Store::new(64 << 20);
        let ctx: Vec<u32> = (0..40).map(|i| 500 + i).collect();
        let mut a = ctx.clone();
        a.extend(1000..1064);
        let mut b = ctx.clone();
        b.extend(2000..2064);
        s.insert_doc(&a, SRC_OUTPUT, 1, false, T0);
        s.insert_doc(&b, SRC_OUTPUT, 2, false, T0);
        s.insert_doc(&b, SRC_OUTPUT, 2, false, T0); // b seen twice (a pure repeat stores no document)
        assert_eq!(s.docs.len(), 2);
        let key = key_of(&ctx[32..40]);
        assert_eq!(s.index[&key].len(), 2, "two branches after the same context");
        let h = s.lookup(&ctx, 16, 64, T0 + 1).unwrap();
        assert_eq!(h.tokens[0], 2000, "more often seen branch first");
        // feedback: b is offered and never accepted, a accepted fully → a wins
        for _ in 0..10 {
            s.feedback(h.chain_ref, 32, 0, T0 + 2);
        }
        let ia = s.index[&key].iter().position(|o| o.seen == 1).unwrap();
        s.feedback(global_ref(key, ia), 32, 32, T0 + 2);
        assert_eq!(s.lookup(&ctx, 16, 64, T0 + 3).unwrap().tokens[0], 1000);
        check_accounting(&s);
    }

    #[test]
    fn heat_decay_math() {
        let inv = 1.0 / 3600.0;
        let mut h = Heat::new(8.0, 1000);
        assert!((h.at(1000 + 3600, inv) - 4.0).abs() < 1e-4, "one half-life halves");
        assert!((h.at(1000 + 3 * 3600, inv) - 1.0).abs() < 1e-4);
        assert_eq!(h.at(500, inv), 8.0, "no growth when the clock goes backwards");
        h.add(1000 + 3600, inv, 2.0);
        assert!((h.v - 6.0).abs() < 1e-4 && h.t == 4600, "decay, then add");
        h.add(4600, inv, -100.0);
        assert_eq!(h.v, 0.0, "clamped at 0");
        assert_eq!(Heat::new(5.0, 0).at(1_000_000, 0.0), 5.0, "half-life 0 = no decay");
        // events: seen 1, hit 1, accept 4/token, reject 1/token
        let mut s = Store::new(64 << 20);
        s.half_life_s = 3600;
        let d = doc(3, 200);
        s.insert_doc(&d, SRC_OUTPUT, 1, false, T0);
        let h = s.lookup(&d[20..60], 16, 64, T0 + 3600).unwrap();
        let key = (h.chain_ref >> 2) & KEY_MASK;
        let o = &s.index[&key][0];
        assert!((o.heat.v - 1.5).abs() < 1e-4, "W_SEEN decayed one half-life + W_HIT: {}", o.heat.v);
        s.feedback(h.chain_ref, 10, 5, T0 + 3600);
        let o = &s.index[&key][0];
        assert!((o.heat.v - (1.5 + 20.0 - 5.0)).abs() < 1e-4, "{}", o.heat.v);
        let dd = s.docs.values().next().unwrap();
        // doc: 192 occurrences × W_SEEN at T0, halved, + hit + feedback
        assert!((dd.heat.v - (96.0 + 1.0 + 15.0)).abs() < 1e-2, "{}", dd.heat.v);
    }

    #[test]
    fn prune_after_grace_and_rescue_during_grace() {
        let mut s = Store::new(64 << 20);
        (s.half_life_s, s.prune_grace_s, s.prune_heat, s.sweep_s) = (3600, 7200, 0.25, 0);
        let a = doc(11, 200);
        let b = doc(12, 200);
        s.insert_doc(&a, SRC_OUTPUT, 1, false, T0);
        s.insert_doc(&b, SRC_OUTPUT, 1, false, T0);
        let seed = doc(13, 120);
        s.insert_doc(&seed, SRC_CORPUS, 0, true, T0);
        s.sweep(T0 + 3600);
        assert_eq!(s.heat.candidates, 0, "0.5 is still warm");
        // 1.0 → 0.125 after 3 half-lives: everything except the seeds becomes a candidate
        s.sweep(T0 + 3 * 3600);
        assert_eq!(s.heat.candidates, 2 * 192, "{:?}", s.heat);
        assert_eq!(s.stats.pruned, 0, "not before the grace period");
        // a hit on b during the grace period rescues that occurrence
        let h = s.lookup(&b[50..90], 16, 64, T0 + 4 * 3600).unwrap();
        assert_eq!(h.tokens[0], b[90]);
        assert_eq!(s.stats.rescued, 1);
        s.sweep(T0 + 3 * 3600 + 7199);
        assert_eq!(s.stats.pruned, 0);
        s.sweep(T0 + 3 * 3600 + 7200);
        // a's 192 and b's other 191 are pruned; the rescued one is hot again (1.0 + 0.125 decayed ~1 h)
        assert_eq!(s.stats.pruned, 2 * 192 - 1);
        assert!(!s.docs.values().any(|d| d.toks[..] == a[..]), "a freed: no occurrence left");
        assert_eq!(s.docs.len(), 1, "b kept for its rescued chain");
        assert_eq!(s.stats.docs_freed, 1);
        assert_eq!(s.lookup(&b[50..90], 16, 64, T0 + 5 * 3600).unwrap().tokens, b[90..154]);
        assert!(s.lookup(&a[50..90], 16, 64, T0 + 5 * 3600).is_none());
        // corpus seeds are never pruned
        s.sweep(T0 + 100 * 3600);
        s.sweep(T0 + 200 * 3600);
        assert_eq!(s.lookup(&seed[10..40], 16, 64, T0 + 200 * 3600).unwrap().tokens, seed[40..104]);
        assert_eq!(s.heat.seeds, 112);
        check_accounting(&s);
    }

    #[test]
    fn accept_rescues_and_rejections_cool() {
        let mut s = Store::new(64 << 20);
        (s.half_life_s, s.prune_grace_s, s.sweep_s) = (3600, 3600, 0);
        let d = doc(21, 100);
        s.insert_doc(&d, SRC_OUTPUT, 1, false, T0);
        let h = s.lookup(&d[10..40], 16, 64, T0).unwrap();
        s.feedback(h.chain_ref, 30, 0, T0); // rejected: 1 + 1 - 30 → 0
        let key = (h.chain_ref >> 2) & KEY_MASK;
        assert_eq!(s.index[&key][0].heat.v, 0.0);
        s.sweep(T0 + 1);
        assert_ne!(s.index[&key][0].cand_since, 0, "cold right away");
        s.feedback(h.chain_ref, 0, 0, T0 + 2); // no accept: no rescue
        assert_ne!(s.index[&key][0].cand_since, 0);
        s.feedback(h.chain_ref, 8, 8, T0 + 3); // accept: rescued
        assert_eq!(s.index[&key][0].cand_since, 0);
        assert_eq!(s.stats.rescued, 1);
    }

    #[test]
    fn coldest_documents_evicted_first_under_the_cap() {
        let mut s = Store::new(64 << 20);
        s.half_life_s = 3600;
        let docs: Vec<Vec<u32>> = (0..6).map(|i| doc(100 + i, 400)).collect();
        for (i, d) in docs.iter().enumerate() {
            s.insert_doc(d, SRC_OUTPUT, 1, false, T0 + i as u32 * 600);
        }
        // the oldest document is in use (accepted tokens): it outlives younger cold ones
        let h = s.lookup(&docs[0][100..140], 16, 64, T0 + 4000).unwrap();
        s.feedback(h.chain_ref, 64, 64, T0 + 4000);
        let full = s.mem_bytes();
        s.mem_cap = full * 3 / 5;
        assert!(s.gc(T0 + 4000));
        assert!(s.mem_bytes() <= s.mem_cap);
        let alive = |s: &Store, d: &[u32]| s.docs.values().any(|x| x.toks[..] == d[..]);
        assert!(alive(&s, &docs[0]), "hot old document kept");
        assert!(!alive(&s, &docs[1]) && !alive(&s, &docs[2]), "coldest (oldest unused) evicted first");
        assert!(alive(&s, &docs[5]), "newest kept");
        assert_eq!(s.lookup(&docs[0][100..140], 16, 64, T0 + 4001).unwrap().tokens, docs[0][140..204]);
        assert!(s.lookup(&docs[1][100..140], 16, 64, T0 + 4001).is_none(), "occurrences into evicted docs are dead");
        s.sweep(T0 + 4002);
        assert_eq!(s.dead_occ, 0);
        check_accounting(&s);
    }

    #[test]
    fn eviction_promotion_gc_and_snapshot() {
        let mut s = Store::new(200_000);
        let keep = doc(7, 200);
        s.insert_doc(&keep, SRC_OUTPUT, 1, false, T0);
        let ctx: Vec<u32> = keep[50..90].to_vec();
        let h = s.lookup(&ctx, 16, 64, T0 + 1).unwrap();
        s.feedback(h.chain_ref, 64, PROMOTE_ACCEPTED as u16, T0 + 1); // promoted: survives eviction
        assert_eq!(s.stats.promoted, 1);
        let seed = doc(9, 120);
        s.insert_doc(&seed, SRC_CORPUS, 0, true, T0); // permanent corpus seed
        for i in 0..40 {
            s.insert_doc(&doc(100 + i, 500), SRC_OUTPUT, 2, false, T0 + 10 + i);
            s.gc(T0 + 10 + i);
        }
        assert!(s.mem_bytes() <= 200_000, "mem {}", s.mem_bytes());
        assert!(s.stats.docs_evicted > 0);
        assert_eq!(s.lookup(&ctx, 16, 64, T0 + 60).unwrap().tokens, keep[90..154], "promoted chain survives eviction");
        assert_eq!(s.lookup(&seed[10..40], 16, 64, T0 + 60).unwrap().tokens, seed[40..104], "corpus seed survives");
        s.sweep(T0 + 61);
        check_accounting(&s);
        let p = tmp("snap");
        s.model_tag = "qwen-test".into();
        s.save(&p, T0 + 61).unwrap();
        let mut r = Store::load(&p, 200_000, T0 + 62).unwrap();
        assert_eq!(r.index.len(), s.index.len());
        assert_eq!(r.docs.len(), s.docs.len());
        assert_eq!((r.model_tag.as_str(), r.tag_provisional), ("qwen-test", true));
        check_accounting(&r);
        assert_eq!(r.mem_bytes(), s.mem_bytes());
        assert_eq!(r.lookup(&seed[10..40], 16, 64, T0 + 63).unwrap().tokens, seed[40..104], "restored");
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn snapshot_round_trip_keeps_heat_and_decays_over_downtime() {
        let mut s = Store::new(64 << 20);
        (s.half_life_s, s.prune_grace_s, s.sweep_s) = (3600, 1800, 0);
        let a = doc(31, 300);
        let b = doc(32, 300);
        s.insert_doc(&a, SRC_OUTPUT, 1, false, T0);
        s.insert_doc(&b, SRC_OUTPUT, 1, false, T0);
        let h = s.lookup(&a[0..40], 16, 64, T0 + 10).unwrap();
        s.feedback(h.chain_ref, 40, 40, T0 + 10);
        s.sweep(T0 + 3 * 3600); // most occurrences become candidates
        let cands = s.heat.candidates;
        assert!(cands > 0);
        let p = tmp("heat");
        s.save(&p, T0 + 3 * 3600).unwrap();
        let meta = std::fs::metadata(&p).unwrap();
        assert_eq!(std::os::unix::fs::PermissionsExt::mode(&meta.permissions()) & 0o777, 0o600);
        // the broker was down for 10 hours
        let back = T0 + 13 * 3600;
        let mut r = Store::load(&p, 64 << 20, back).unwrap();
        (r.half_life_s, r.prune_grace_s, r.sweep_s) = (3600, 1800, 0);
        for (k, v) in s.index.iter() {
            for (o, q) in v.iter().zip(&r.index[k]) {
                assert_eq!((o.heat, o.pos, o.seen, o.offered, o.accepted, o.copy.is_some()),
                           (q.heat, q.pos, q.seen, q.offered, q.accepted, q.copy.is_some()));
                assert_eq!(o.cand_since != 0, q.cand_since != 0, "candidate state kept");
                if q.cand_since != 0 {
                    assert_eq!(q.cand_since, back, "grace restarts at restore");
                }
            }
        }
        let key = (h.chain_ref >> 2) & KEY_MASK;
        let o = &r.index[&key][(h.chain_ref & 3) as usize];
        let before = o.heat.at(T0 + 3 * 3600, 1.0 / 3600.0);
        let after = o.heat.at(back, 1.0 / 3600.0);
        assert!((after / before - 2f32.powi(-10)).abs() < 1e-4, "10 h down = 10 half-lives of decay");
        // after restore: candidates survive the first sweep (grace restarted), and go after the grace period
        r.sweep(back + 1);
        assert_eq!(r.stats.pruned, 0);
        r.sweep(back + 1800);
        assert!(r.stats.pruned >= cands, "{} {}", r.stats.pruned, cands);
        assert!(r.lookup(&b[0..40], 16, 64, back + 1801).is_none(), "cold b is gone");
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn old_or_unknown_snapshot_is_refused() {
        let p = tmp("v1");
        std::fs::write(&p, b"LLMTCH1\n{\"floor\":0,\"arena\":0,\"docs\":0,\"keys\":0,\"model_tag\":\"x\"}\n").unwrap();
        let e = Store::load(&p, 1 << 20, T0).err().expect("v1 refused");
        assert!(e.to_string().contains("v1"), "{e}");
        std::fs::write(&p, b"LLMTCH9\nwhatever").unwrap();
        assert!(Store::load(&p, 1 << 20, T0).is_err());
        // truncated v2
        let mut s = Store::new(1 << 20);
        s.insert_doc(&doc(1, 100), SRC_OUTPUT, 1, false, T0);
        let mut b = s.serialize(T0);
        b.truncate(b.len() - 3);
        std::fs::write(&p, &b).unwrap();
        assert!(Store::load(&p, 1 << 20, T0).is_err());
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn seed_over_existing_branch_gets_a_copy() {
        let mut s = Store::new(64 << 20);
        let d = doc(41, 150);
        s.insert_doc(&d, SRC_OUTPUT, 1, false, T0);
        s.insert_doc(&d, SRC_CORPUS, 0, true, T0); // the same text as a corpus seed
        assert_eq!(s.n_pinned(), 142);
        s.mem_cap = 1;
        s.gc(T0 + 1);
        s.sweep(T0 + 2);
        assert!(s.docs.is_empty());
        assert_eq!(s.lookup(&d[20..60], 16, 64, T0 + 3).unwrap().tokens, d[60..124], "seed copy survives");
        check_accounting(&s);
    }

    /// Lock-hold (serialize) and write time for a realistic store; `cargo test --release -p kvchains -- --ignored
    /// --nocapture snapshot_timing` (CHAINS_MIB=512 by default).
    #[test]
    #[ignore]
    fn snapshot_timing() {
        let mib: usize = std::env::var("CHAINS_MIB").ok().and_then(|x| x.parse().ok()).unwrap_or(512);
        let mut s = Store::new(mib << 20);
        let mut rng = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let t = std::time::Instant::now();
        let mut i = 0u32;
        while s.mem_bytes() < (mib << 20) - (4 << 20) {
            let n = 200 + (next() % 3000) as usize;
            // a quarter of the documents repeat a shared template span (branches, repeats)
            let d: Vec<u32> = (0..n).map(|j| if i % 4 == 0 && j < 300 { j as u32 } else { (next() % 150_000) as u32 }).collect();
            s.insert_doc(&d, SRC_OUTPUT, 1, false, T0 + i / 10);
            i += 1;
        }
        let rss = || std::fs::read_to_string("/proc/self/statm").ok()
            .and_then(|x| x.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok())).unwrap_or(0.0) * 4096.0 / 1048576.0;
        eprintln!("built {} docs, {} tokens, {} keys, {} occ, est {:.0} MiB (process RSS {:.0} MiB) in {:.1} s; Occ {} B",
                  s.docs.len(), s.n_tokens(), s.index.len(), s.n_occ(), s.mem_bytes() as f64 / 1048576.0, rss(),
                  t.elapsed().as_secs_f64(), OCC_BYTES);
        // the GC thread's pattern: one store-lock hold per shard
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        let (t, mut hold) = (std::time::Instant::now(), 0f64);
        s.sweep_begin(T0 + i / 10 + 60);
        for k in 0..SHARDS {
            let t = std::time::Instant::now();
            s.sweep_shard(k, T0 + i / 10 + 60);
            hold = hold.max(ms(t));
        }
        let t1 = std::time::Instant::now();
        s.sweep_end(T0 + i / 10 + 60);
        hold = hold.max(ms(t1));
        eprintln!("sweep {:.0} ms in {} lock holds of <= {:.1} ms", ms(t), SHARDS + 1, hold);
        let (t, mut hold) = (std::time::Instant::now(), 0f64);
        let mut b = Vec::with_capacity(s.mem_bytes() * 2 / 3);
        s.serialize_head(T0, &mut b);
        hold = hold.max(ms(t));
        for k in 0..SHARDS {
            let t = std::time::Instant::now();
            s.serialize_shard(k, &mut b);
            hold = hold.max(ms(t));
        }
        let ser = t.elapsed();
        eprintln!("serialize {:.0} ms in {} lock holds of <= {:.1} ms", ser.as_secs_f64() * 1e3, SHARDS + 1, hold);
        let p = tmp("timing");
        let t = std::time::Instant::now();
        Store::write_snapshot(&p, &b).unwrap();
        let wr = t.elapsed();
        let t = std::time::Instant::now();
        let r = Store::load(&p, mib << 20, T0).unwrap();
        let ld = t.elapsed();
        eprintln!("serialize {:.0} ms, write+fsync {:.0} ms, load {:.0} ms; {:.0} MiB on disk = {:.2} MiB per \
                   MiB of store estimate", ser.as_secs_f64() * 1e3, wr.as_secs_f64() * 1e3, ld.as_secs_f64() * 1e3,
                  b.len() as f64 / 1048576.0, b.len() as f64 / s.mem_bytes() as f64);
        assert_eq!(r.index.len(), s.index.len());
        let _ = std::fs::remove_file(p);
    }
}

#[cfg(test)]
mod demote_tests {
    use super::*;

    #[test]
    fn chain_that_earns_nothing_is_dropped_by_gc() {
        let mut s = Store::new(64 << 20);
        let d: Vec<u32> = (0..100).map(|i| 7_000 + i).collect();
        s.insert_doc(&d, SRC_OUTPUT, 1, false, 1000);
        let h = s.lookup(&d[10..40], 16, 64, 1001).unwrap();
        s.feedback(h.chain_ref, 300, 3, 1001); // 1% accepted
        s.gc(1002);
        assert!(s.lookup(&d[10..40], 16, 64, 1003).is_none(), "demoted");
        assert_eq!(s.stats.demoted, 1);
    }
}
