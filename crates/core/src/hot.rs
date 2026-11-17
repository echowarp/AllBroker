//! Persistent hot set ranked by use (what agents share), not by RAM-tier hits.
//!
//! Engines serve the prompts every agent shares (system prompt + tool definitions) from their own VRAM prefix cache,
//! so those blocks rarely produce RAM-tier lookup hits and never ranked by hits. But every request TOUCHes its whole
//! chain (each KV group, VRAM-served or not), so the broker sees which blocks many requests share.
//!
//! Use counter (per entry, and per non-resident key in a bounded map): +1 per request chain that touches the key,
//! decayed with a half-life (`use_half_life_ms`). Requests, not touches: a request touches its chain many times (each
//! scheduling attempt while it waits, each chunk store); each key keeps the latest end of the last CHAINS chains
//! through it, and a batch containing one of them continues that chain (no count; the end moves to the new batch's).
//! So a shared system prompt counts once per request of every agent, a conversation's own blocks about once per turn
//! (once in all if the next turn repeats the previous request's last chunk; Qwen's template drops earlier turns'
//! thinking, so the last decoded chunk usually is not repeated). The shared prompt is on every request's chain, so
//! it always ranks at least as high as any one conversation's blocks.
//! Lookup hits count only from connections that never TOUCH (vLLM touches after every lookup).
//!
//! Ranking (`hot_entries_by_use`) keeps restored prefixes loadable: in a hybrid namespace (anchored GDN states,
//! seg.rs) a unit is a boundary = every GDN group's state of one block + the attention chain from the root to that
//! block, all committed in RAM (a boundary whose chain or states are not all in RAM is counted as unloadable, not
//! saved); its score is the highest use of its anchor attention block and its states (deeper requests touch the GDN
//! groups only from their hit point, but the anchor block on every request's chain). A boundary is superseded when a
//! deeper candidate boundary on its chain has at least 3/4 of its score (the same requests went on through it). In
//! other namespaces an entry is a unit with its committed ancestors (link map, else metadata parent); kinds other than
//! attention / GDN state stand alone. Units reaching `min_use` go in hottest first, shared members counted once,
//! within the byte budget. Weights (KIND_WEIGHTS) never.
//!
//! Retention: the states of a candidate boundary inherit their anchor block's last use (`hot_refresh`, every
//! REFRESH_MS from tick), so the pool does not age out the state that makes a shared prefix loadable while its
//! attention is in use every request. Counts only: no keys in logs.

use super::*;
use seg::{p16, Gen2, P16};

/// Chains remembered per key (more concurrent chains through one key count again when they return).
pub(crate) const CHAINS: usize = 6;
const MISSING_CAP: usize = 50_000;
const REFRESH_MS: u64 = 30_000;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UseStat {
    score: f32,
    t: u64,
    /// latest end (ghost) of each request chain seen recently; 0 = empty
    ends: [u64; CHAINS],
    next: u8,
}

/// One touch batch: its keys' ghosts.
pub(crate) struct Batch {
    set: HashSet<u64>,
    end: u64,
}

impl Batch {
    fn new(ns: u32, keys: &[Key]) -> Batch {
        let set: HashSet<u64> = keys.iter().map(|k| ghost(ns, k).max(1)).collect();
        Batch { set, end: ghost(ns, keys.last().unwrap()).max(1) }
    }
}

impl UseStat {
    pub(crate) fn at(&self, now: u64, half: u64) -> f64 {
        if self.score <= 0.0 {
            return 0.0;
        }
        if half == 0 {
            return self.score as f64;
        }
        let dt = now.saturating_sub(self.t) as f64;
        self.score as f64 * (-(dt / half as f64) * std::f64::consts::LN_2).exp()
    }

    /// One touch batch through this key; true if it counted. A batch containing the latest end of a chain seen
    /// before continues it (the same request touched again while it waits or stores its next chunk, or a next turn
    /// that repeats the previous request's last chunk) and does not count; the chain's end moves to this batch's.
    fn touch(&mut self, b: &Batch, now: u64, half: u64) -> bool {
        if let Some(e) = self.ends.iter_mut().find(|e| **e != 0 && b.set.contains(*e)) {
            *e = b.end;
            return false;
        }
        self.ends[self.next as usize % CHAINS] = b.end;
        self.next = self.next.wrapping_add(1);
        self.add(1.0, now, half);
        true
    }

    fn add(&mut self, n: f64, now: u64, half: u64) {
        self.score = (self.at(now, half) + n) as f32;
        self.t = self.t.max(now);
    }

    pub(crate) fn seeded(score: f64, now: u64) -> UseStat {
        UseStat { score: score.max(0.0) as f32, t: now, ..Default::default() }
    }
}

/// The relation a hot-set file carries so a restored entry is linked as before (seg.rs maps are not persisted
/// otherwise): an attention block's parent (zeros = a chain root), a GDN state's anchor block and group count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HotRel {
    #[default]
    None,
    Parent([u8; 16]),
    Anchor([u8; 16], u8),
}

/// Result of the last ranking pass (gauges and the save log line).
#[derive(Clone, Copy, Debug, Default)]
pub struct HotStats {
    pub units: u64,
    pub entries: u64,
    pub bytes: u64,
    /// candidate units (use >= min) left out by the byte budget
    pub over_budget: u64,
    /// boundaries / chains with use >= min whose chain or states are not all in RAM (not loadable: not saved)
    pub unloadable: u64,
    /// keys touched by >= min chains that are not in RAM (e.g. a prompt only in an engine's VRAM)
    pub missing_keys: u64,
    /// attention entries of a hybrid namespace with use >= min on no candidate boundary's chain (no GDN state in RAM
    /// at or below them on their chain: not loadable from RAM, not saved)
    pub stateless: u64,
}

#[derive(Default)]
pub(crate) struct Hot {
    /// non-resident keys' use (a block in VRAM only), moved to the entry when it is stored
    pub(crate) missing: Option<Gen2<(u32, P16), UseStat>>,
    /// connections that sent TOUCH / TOUCH_REQUEST (their lookups are not counted again)
    pub(crate) touching: HashSet<ConnId>,
    pub(crate) last_refresh: u64,
    pub(crate) stats: HotStats,
}

struct Boundary {
    ns: u32,
    anchor: EntryId,
    states: Vec<EntryId>,
    /// attention entries root .. anchor (anchor last); empty when not loadable
    path: Vec<EntryId>,
    /// anchor p16s of the chain above the anchor (for supersession)
    above: Vec<P16>,
    loadable: bool,
    score: f64,
}

impl Core {
    fn half(&self) -> u64 {
        self.use_half_life_ms
    }

    /// Use bookkeeping for one TOUCH batch (one KV group of one request, root to end or a suffix).
    pub(crate) fn note_use(&mut self, ns: u32, keys: &[Key], now: u64) {
        if keys.is_empty() {
            return;
        }
        let batch = Batch::new(ns, keys);
        let half = self.half();
        self.c.touch_keys += keys.len() as u64;
        for k in keys {
            match self.index.get(&(ns, *k)) {
                Some(&id) if self.entries[&id].state == State::Committed => {
                    self.entries.get_mut(&id).unwrap().use_.touch(&batch, now, half);
                }
                _ => {
                    self.c.touch_missing_keys += 1;
                    let m = self.hot.missing.get_or_insert_with(|| Gen2::new(MISSING_CAP));
                    let mut u = m.get(&(ns, p16(k))).unwrap_or_default();
                    u.touch(&batch, now, half);
                    m.insert((ns, p16(k)), u);
                }
            }
        }
    }

    /// Lookup hits from a connection that does not TOUCH count as one use each (deduplicated per chain likewise).
    pub(crate) fn note_hit_use(&mut self, conn: ConnId, ns: u32, keys: &[Key], hit: &[EntryId], now: u64) {
        if hit.is_empty() || self.hot.touching.contains(&conn) {
            return;
        }
        let batch = Batch::new(ns, &keys[..hit.len()]);
        let half = self.half();
        for id in hit {
            self.entries.get_mut(id).unwrap().use_.touch(&batch, now, half);
        }
    }

    /// A committed entry takes over the use its key gathered while not in RAM.
    pub(crate) fn hot_on_commit(&mut self, id: EntryId) {
        let (ns, k) = (self.entries[&id].ns, p16(&self.entries[&id].key));
        if let Some(m) = self.hot.missing.as_mut() {
            if let Some(u) = m.get(&(ns, k)) {
                m.remove(&(ns, k));
                self.entries.get_mut(&id).unwrap().use_ = u;
            }
        }
    }

    /// A committed entry left RAM: its key keeps the use it gathered (a later store of the same block takes it).
    pub(crate) fn hot_on_remove(&mut self, e: &Entry) {
        if e.state != State::Committed || e.use_.score <= 0.0 || e.kind == KIND_WEIGHTS {
            return;
        }
        let m = self.hot.missing.get_or_insert_with(|| Gen2::new(MISSING_CAP));
        m.insert((e.ns, p16(&e.key)), e.use_);
    }

    /// Decayed use of a committed entry (0 = none / unknown).
    pub fn use_of(&self, ns: u32, key: &Key) -> f64 {
        self.index.get(&(ns, *key)).map_or(0.0, |id| self.entries[id].use_.at(self.now, self.half()))
    }

    /// Restored hot-set entry: its saved use (as of now) and its chain relation.
    pub fn restore_hot(&mut self, ns: u32, key: &Key, score: f64, rel: HotRel) {
        let Some(&id) = self.index.get(&(ns, *key)) else { return };
        let now = self.now;
        self.entries.get_mut(&id).unwrap().use_ = UseStat::seeded(score, now);
        if !self.cfg.seg_evict {
            return;
        }
        match rel {
            HotRel::None => {}
            HotRel::Parent(p) => {
                if !self.seg.link.contains(&(ns, p16(key))) {
                    self.seg.link.insert((ns, p16(key)), (p, now));
                }
            }
            HotRel::Anchor(a, ng) => {
                self.seg.anchor.insert((ns, p16(key)), (a, ng.max(1)));
                self.seg.linked.insert(ns);
            }
        }
    }

    /// The relation to persist with an entry (see HotRel).
    pub fn hot_rel(&self, ns: u32, key: &Key) -> HotRel {
        let Some(e) = self.index.get(&(ns, *key)).map(|id| &self.entries[id]) else { return HotRel::None };
        match e.kind {
            1 => self.seg.anchor.get(&(ns, p16(key))).map_or(HotRel::None, |(a, n)| HotRel::Anchor(a, n)),
            0 => {
                let strict = self.cfg.seg_evict && self.seg.linked.contains(&ns);
                self.parent16(ns, p16(key), strict).map_or(HotRel::None, HotRel::Parent)
            }
            _ => HotRel::None,
        }
    }

    /// Parent block of an attention key: the link map, else a committed entry's explicit parent, else its metadata
    /// parent. Some(zeros) = a root. `strict` (hybrid namespaces, chains known from touches): a committed entry with
    /// none of these is unknown (None), not a root.
    fn parent16(&self, ns: u32, k: P16, strict: bool) -> Option<P16> {
        if let Some((p, _)) = self.seg.link.get(&(ns, k)) {
            return Some(p);
        }
        let e = self.entries.get(self.pfx.get(&(ns, k))?)?;
        if let Some(p) = e.parent.and_then(|p| self.entries.get(&p)) {
            return Some(p16(&p.key));
        }
        match meta_parent(&e.meta) {
            Some(p) => Some(p),
            None if strict => None,
            None => Some([0u8; 16]),
        }
    }

    /// Committed attention chain root .. `k` (k last); None if a block is not in RAM or the chain is unknown.
    fn resident_chain(&self, ns: u32, k: P16, above: &mut Vec<P16>) -> Option<Vec<EntryId>> {
        let strict = self.cfg.seg_evict && self.seg.linked.contains(&ns);
        let mut path = Vec::new();
        let mut p = k;
        for hops in 0.. {
            let id = *self.pfx.get(&(ns, p))?;
            let e = self.entries.get(&id)?;
            if e.state != State::Committed || e.kind != 0 || e.rewriting {
                return None;
            }
            path.push(id);
            if hops > 0 {
                above.push(p);
            }
            match self.parent16(ns, p, strict) {
                Some(q) if q == [0u8; 16] => break,
                Some(q) if hops < 4096 && q != p => p = q,
                _ => return None,
            }
        }
        path.reverse();
        Some(path)
    }

    fn boundaries(&self) -> Vec<Boundary> {
        let (now, half) = (self.now, self.half());
        let mut bnd: HashMap<(u32, P16), (Vec<EntryId>, u8)> = HashMap::new();
        for (id, e) in &self.entries {
            if e.kind == 1 && e.state == State::Committed && self.seg.linked.contains(&e.ns) {
                if let Some((a, n)) = self.seg.anchor.get(&(e.ns, p16(&e.key))) {
                    let b = bnd.entry((e.ns, a)).or_insert((Vec::new(), n));
                    b.0.push(*id);
                    b.1 = b.1.max(n);
                }
            }
        }
        let mut out = Vec::new();
        for ((ns, a), (mut states, ng)) in bnd {
            states.sort_unstable();
            let anchor = self.pfx.get(&(ns, a)).copied().filter(|id| self.entries[id].kind == 0);
            let st_score = states.iter().map(|i| self.entries[i].use_.at(now, half)).fold(0.0, f64::max);
            let an_score = anchor.map_or_else(
                || self.hot.missing.as_ref().and_then(|m| m.get(&(ns, a))).map_or(0.0, |u| u.at(now, half)),
                |id| self.entries[&id].use_.at(now, half));
            let mut above = Vec::new();
            let path = if states.len() >= ng as usize { self.resident_chain(ns, a, &mut above) } else { None };
            let loadable = path.is_some() && states.iter().all(|i| !self.entries[i].rewriting);
            out.push(Boundary { ns, anchor: anchor.unwrap_or(0), states, path: path.unwrap_or_default(), above,
                                loadable, score: st_score.max(an_score) });
        }
        out
    }

    /// Candidate boundaries that are not superseded (a deeper loadable candidate on the chain has >= 3/4 the score).
    fn live_boundaries(&self, min_use: f64) -> (Vec<Boundary>, u64) {
        let all = self.boundaries();
        let mut deeper: HashMap<(u32, P16), f64> = HashMap::new();
        for b in all.iter().filter(|b| b.loadable && b.score >= min_use) {
            for a in &b.above {
                let x = deeper.entry((b.ns, *a)).or_insert(0.0);
                *x = x.max(b.score);
            }
        }
        let mut unloadable = 0;
        let mut v = Vec::new();
        for b in all {
            if b.score < min_use {
                continue;
            }
            if !b.loadable {
                unloadable += 1;
                continue;
            }
            let a = p16(&self.entries[&b.anchor].key);
            if deeper.get(&(b.ns, a)).is_some_and(|d| *d * 4.0 >= b.score * 3.0) {
                continue;
            }
            v.push(b);
        }
        (v, unloadable)
    }

    /// Hot-set candidates by use (module doc), within `max_bytes`, hottest unit first, each unit root first.
    pub fn hot_entries_by_use(&mut self, min_use: f64, max_bytes: u64) -> Vec<HotEntry> {
        let (now, half) = (self.now, self.half());
        let (bs, mut unloadable) = self.live_boundaries(min_use);
        // (score, last use, ids root first)
        let mut units: Vec<(f64, u64, Vec<EntryId>)> = Vec::new();
        let covered: HashSet<EntryId> = bs.iter().flat_map(|b| b.path.iter().copied()).collect();
        let mut stateless = 0;
        for b in bs {
            let mut ids = b.path;
            ids.extend(b.states);
            let lu = ids.iter().map(|i| self.entries[i].last_used).max().unwrap_or(0);
            units.push((b.score, lu, ids));
        }
        let hybrid = |ns: u32| self.cfg.seg_evict && self.seg.linked.contains(&ns);
        for (id, e) in &self.entries {
            if e.state != State::Committed || e.kind == KIND_WEIGHTS {
                continue;
            }
            let s = e.use_.at(now, half);
            if s < min_use {
                continue;
            }
            match e.kind {
                0 | 1 if hybrid(e.ns) => {
                    // in boundaries only (attention alone / a state without its chain: useless)
                    if e.kind == 0 && !covered.contains(id) {
                        stateless += 1;
                    }
                }
                0 => match self.resident_chain(e.ns, p16(&e.key), &mut Vec::new()) {
                    Some(path) => units.push((s, e.last_used, path)),
                    None => unloadable += 1,
                },
                _ => units.push((s, e.last_used, vec![*id])),
            }
        }
        // class order: units with a normal member first, then low-class units, then lowest (a unit's class = its
        // members' lowest rank) fill what is left of the budget; by use within each
        let low = |ids: &[EntryId]| ids.iter().map(|i| rank_of(self.entries[i].flags)).min().unwrap_or(0);
        units.sort_by(|a, b| low(&a.2).cmp(&low(&b.2)).then(b.0.total_cmp(&a.0)).then(b.1.cmp(&a.1))
            .then(a.2.len().cmp(&b.2.len())));
        let mut st = HotStats { unloadable, stateless, ..Default::default() };
        let mut taken: HashSet<EntryId> = HashSet::new();
        let mut out = Vec::new();
        let mut total = 0u64;
        for (_, _, ids) in units {
            let extra: Vec<EntryId> = ids.into_iter().filter(|i| !taken.contains(i)).collect();
            let cost: u64 = extra.iter().map(|i| self.entries[i].n_slots as u64 * self.cfg.slot_bytes).sum();
            if total + cost > max_bytes {
                st.over_budget += 1;
                continue;
            }
            total += cost;
            st.units += 1;
            for i in extra {
                taken.insert(i);
                let e = &self.entries[&i];
                out.push(HotEntry {
                    ns: e.ns,
                    key: e.key,
                    kind: e.kind,
                    format: e.format,
                    producer: e.producer,
                    meta: e.meta.clone(),
                    bytes: e.bytes,
                    hits: e.hits,
                    last_used: e.last_used,
                    ranges: e.ranges.clone(),
                    shared: e.shared,
                });
            }
        }
        st.entries = out.len() as u64;
        st.bytes = total;
        st.missing_keys = self.missing_used(min_use);
        self.hot.stats = st;
        out
    }

    fn missing_used(&self, min_use: f64) -> u64 {
        let (now, half) = (self.now, self.half());
        self.hot.missing.as_ref().map_or(0, |m| {
            m.iter().filter(|((ns, k), u)| u.at(now, half) >= min_use && !self.pfx.contains_key(&(*ns, *k))).count() as u64
        })
    }

    /// Retention (from tick, every REFRESH_MS): candidate boundaries' states take their anchor block's last use.
    pub(crate) fn hot_refresh(&mut self) {
        if !self.cfg.seg_evict || self.hot_min_use <= 0.0 || self.now.saturating_sub(self.hot.last_refresh) < REFRESH_MS {
            return;
        }
        self.hot.last_refresh = self.now;
        self.hot.touching.retain(|c| self.conns.contains_key(c));
        let (bs, unloadable) = self.live_boundaries(self.hot_min_use);
        self.hot.stats.unloadable = unloadable;
        for b in bs {
            let t = self.entries[&b.anchor].last_used;
            for s in b.states {
                let e = self.entries.get_mut(&s).unwrap();
                if e.last_used < t {
                    e.last_used = t;
                    self.policy.access(s);
                    self.c.hot_state_refreshed += 1;
                }
            }
        }
    }

    pub fn hot_stats(&self) -> HotStats {
        self.hot.stats
    }

    pub(crate) fn hot_gauges(&self) -> Vec<(String, u64)> {
        let s = self.hot.stats;
        vec![
            ("hotset_ranked_units".into(), s.units),
            ("hotset_ranked_entries".into(), s.entries),
            ("hotset_ranked_bytes".into(), s.bytes),
            ("hotset_over_budget_units".into(), s.over_budget),
            ("hotset_unloadable_units".into(), s.unloadable),
            ("hotset_stateless_entries".into(), s.stateless),
            ("hotset_touched_missing_keys".into(), self.missing_used(self.hot_min_use.max(1.0))),
            ("hotset_missing_tracked_keys".into(), self.hot.missing.as_ref().map_or(0, |m| m.len() as u64)),
        ]
    }
}
