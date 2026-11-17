//! Segment-wise eviction: attention blocks and GDN states that can't actually be used are dropped together.
//! A RAM hit of a hybrid model needs the attention chain up to some block
//! AND a GDN state at exactly that block; attention past the deepest state and a state whose chain is gone are useless.
//!
//! Linkage: the Arc's connector keys every KV group per block (blake2b of block hash + group id: opaque here) and
//! stores attention without a parent label (lazy stores). What links them is the TOUCH order: per request the
//! scheduler touches each KV group in group order, one batch per group, all ending at the same block (attention
//! batches from the root, GDN groups from the last hit chunk on: suffixes). Per connection, consecutive TOUCHes with
//! no other request between them are one run; runs are split into requests of G batches (G = the most frequent run
//! length, learned per connection), the batch kinds per position in the request are learned from the entries the
//! batches name (votes), and the batches are end-aligned: a GDN state key's anchor = the attention key of the same
//! block. Attention batches give the chain links (child -> parent). Kind-4 tails cover the chain below their metadata
//! parent if the client sets one (none today: a namespace with live unanchored tails gets no eager orphan cleanup).
//!
//! Plan (computed when needed; a few thousand entries): every anchored state walks its chain (through attention not in
//! RAM: the GPU may hold it) and covers the attention entries on it. Units:
//!   - segment = a state + the attention only it covers (the blocks above the next shallower state on its chain);
//!   - orphan attention = covered by no state (deeper than the deepest state, or the chain lost its states), unless
//!     stored before its chain's last touch and that touch is within seg_grace_ms (a running prompt: its prompt-end
//!     state may follow; a lazy copy stored after the last touch gets no grace);
//!   - orphan states = its chain has a block that left RAM (and is not on the SSD) since the last touch of it, or a
//!     GDN group's state of the same block is gone; after seg_grace_ms.
//! A superseded state (a deeper state of its chain exists) has no attention of its own: it leaves alone in policy
//! order, what remains still ends at a usable state (holding such states until their chain's deep end left was tried:
//! kvsim_seg -4 points of prompt tokens from RAM, the held states crowd out chains). Attention covered by two or more
//! states (a shared trunk), by a tail only, and young uncovered attention are not evictable on their own.
//! Counts and times only: no keys in logs.

use super::*;
use std::collections::VecDeque;

pub(crate) type P16 = [u8; 16];

pub(crate) fn p16(k: &Key) -> P16 {
    let mut p = [0u8; 16];
    p.copy_from_slice(&k[..16]);
    p
}

/// Two-generation bounded map: inserts go to `cur`; at `cap` entries `old` is dropped and `cur` becomes `old`.
pub(crate) struct Gen2<K, V> {
    cur: HashMap<K, V>,
    old: HashMap<K, V>,
    cap: usize,
}

impl<K: std::hash::Hash + Eq + Copy, V: Copy> Gen2<K, V> {
    pub(crate) fn new(cap: usize) -> Self {
        Gen2 { cur: HashMap::new(), old: HashMap::new(), cap }
    }
    pub(crate) fn get(&self, k: &K) -> Option<V> {
        self.cur.get(k).or_else(|| self.old.get(k)).copied()
    }
    pub(crate) fn contains(&self, k: &K) -> bool {
        self.cur.contains_key(k) || self.old.contains_key(k)
    }
    pub(crate) fn insert(&mut self, k: K, v: V) {
        self.old.remove(&k);
        self.cur.insert(k, v);
        if self.cur.len() >= self.cap {
            self.old = std::mem::take(&mut self.cur);
        }
    }
    pub(crate) fn remove(&mut self, k: &K) {
        self.cur.remove(k);
        self.old.remove(k);
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.cur.iter().chain(self.old.iter().filter(|(k, _)| !self.cur.contains_key(k)))
    }
    pub(crate) fn len(&self) -> usize {
        self.cur.len() + self.old.len()
    }
}

/// Touch batches of one connection since its last other request.
struct Run {
    ns: u32,
    batches: Vec<Vec<Key>>,
    over: bool,
}

/// Per connection: recent run lengths (G = the most frequent) and kind votes per batch position.
#[derive(Default)]
struct Learn {
    lens: VecDeque<usize>,
    votes: Vec<[u32; 2]>,
}

const RUN_MAX: usize = 64;
const SCAN_EVERY_MS: u64 = 1000;

pub(crate) struct Seg {
    /// attention chain links (namespace, child) -> (parent (zeros = a root), last touch / store in broker ms), from
    /// touch runs and entry metadata
    pub(crate) link: Gen2<(u32, P16), (P16, u64)>,
    /// GDN state key -> (attention key of the same block, GDN groups per block)
    pub(crate) anchor: Gen2<(u32, P16), (P16, u8)>,
    /// attention keys that left RAM: (full key for the SSD check, broker ms); cleared when stored or touched again
    pub(crate) gone: Gen2<(u32, P16), (Key, u64)>,
    /// states that left RAM while the SSD tier held them: they still cover their chain while it does
    away: Gen2<(u32, Key), (P16, u8)>,
    /// namespaces with at least one anchored state (orphan rules apply only there: never in attention-only models)
    pub(crate) linked: HashSet<u32>,
    runs: HashMap<ConnId, Run>,
    learn: HashMap<ConnId, Learn>,
    last_scan: u64,
}

impl Default for Seg {
    fn default() -> Self {
        Seg {
            link: Gen2::new(200_000),
            anchor: Gen2::new(100_000),
            gone: Gen2::new(100_000),
            away: Gen2::new(20_000),
            linked: HashSet::new(),
            runs: HashMap::new(),
            learn: HashMap::new(),
            last_scan: 0,
        }
    }
}

/// An eviction unit: entries that leave together. `orphan`: 0 = useful, 1 = useless (goes first everywhere),
/// 2 = uncovered attention in a namespace with live unanchored tails (useless once the tails are gone: first in RESERVE /
/// pressure picks, which take every evictable tail before; an ordinary entry in age sweeps).
pub(crate) struct Unit {
    pub(crate) ids: Vec<EntryId>,
    pub(crate) orphan: u8,
}

#[derive(Default)]
pub(crate) struct Plan {
    unit_of: HashMap<EntryId, usize>,
    pub(crate) units: Vec<Unit>,
    /// not evictable alone: shared trunk, tail-covered, young uncovered attention
    hold: HashSet<EntryId>,
    pub(crate) anchored: u64,
    pub(crate) unanchored: u64,
}

impl Plan {
    /// The unit an entry leaves with (itself if it is in none); None = held.
    pub(crate) fn unit(&self, id: EntryId) -> Option<(Vec<EntryId>, u8)> {
        if let Some(&u) = self.unit_of.get(&id) {
            let u = &self.units[u];
            return Some((u.ids.clone(), u.orphan));
        }
        (!self.hold.contains(&id)).then(|| (vec![id], 0))
    }
}

impl Core {
    /// TOUCH from a connection: as `touch`, and the batch joins the connection's run (segment linkage).
    pub fn touch_conn(&mut self, conn: ConnId, ns: u32, keys: &[Key], now: u64) -> Vec<Status> {
        self.now = self.now.max(now);
        self.hot.touching.insert(conn);
        let st = self.touch(ns, keys, now);
        if self.cfg.seg_evict && !keys.is_empty() {
            if self.seg.runs.get(&conn).is_some_and(|r| r.ns != ns) {
                self.end_run(conn);
            }
            let r = self.seg.runs.entry(conn).or_insert_with(|| Run { ns, batches: Vec::new(), over: false });
            if r.batches.len() < RUN_MAX {
                r.batches.push(keys.to_vec());
            } else {
                r.over = true;
            }
        }
        st
    }

    /// Any other request of the connection ends its touch run: link it.
    pub(crate) fn end_run(&mut self, conn: ConnId) {
        let Some(run) = self.seg.runs.remove(&conn) else { return };
        let l = run.batches.len();
        if run.over || l == 0 {
            self.c.seg_runs_skipped += 1;
            self.c.seg_skip_over += 1;
            return;
        }
        let lr = self.seg.learn.entry(conn).or_default();
        lr.lens.push_back(l);
        if lr.lens.len() > 32 {
            lr.lens.pop_front();
        }
        // The batches per request (KV groups): the largest length (at least 2: attention + a state group; nothing
        // links with fewer) dividing at least 75 % of the recent runs (needs 3 runs; odd runs are skipped, they do
        // not drag it down to 1). Several requests touched in one scheduler step arrive as one run of k x groups batches; with many
        // concurrent sessions those merged runs became the most common length, and the old "most common length"
        // rule then skipped every single-request run (production: 43 % skipped). A too large guess
        // only skips (link_request checks every request's kinds), never mislinks.
        let n = lr.lens.len();
        let g = (n >= 3).then(|| {
            let max = *lr.lens.iter().max().unwrap();
            (2..=max).rev().find(|d| lr.lens.iter().filter(|x| **x % d == 0).count() * 4 >= n * 3)
        }).flatten();
        let Some(g) = g else {
            self.c.seg_runs_skipped += 1;
            self.c.seg_skip_shape += 1;
            return;
        };
        if l % g != 0 {
            self.c.seg_runs_skipped += 1;
            self.c.seg_skip_len += 1;
            return;
        }
        for ch in run.batches.chunks(g) {
            self.link_request(conn, run.ns, ch);
        }
    }

    /// One request's batches (one per KV group, in group order): vote kinds, link attention, anchor states.
    fn link_request(&mut self, conn: ConnId, ns: u32, ch: &[Vec<Key>]) {
        let g = ch.len();
        let seen: Vec<Option<u8>> = ch.iter()
            .map(|b| b.iter().find_map(|k| self.index.get(&(ns, *k)).map(|id| self.entries[id].kind).filter(|x| *x <= 1)))
            .collect();
        let lr = self.seg.learn.get_mut(&conn).unwrap();
        if lr.votes.len() != g {
            lr.votes = vec![[0, 0]; g];
        }
        for (p, s) in seen.iter().enumerate() {
            if let Some(kd) = s {
                lr.votes[p][*kd as usize] += 1;
            }
        }
        let mut kinds: Vec<Option<u8>> = lr.votes.iter()
            .map(|v| {
                let t = v[0] + v[1];
                let (w, n) = if v[0] >= v[1] { (0u8, v[0]) } else { (1u8, v[1]) };
                (t >= 3 && n * 10 >= t * 9).then_some(w)
            })
            .collect();
        // all other positions are states: the undecided one is the attention group (lazy stores: attention rarely in RAM)
        if kinds.iter().filter(|k| k.is_none()).count() == 1 && kinds.iter().flatten().all(|k| *k == 1) {
            if let Some(x) = kinds.iter_mut().find(|k| k.is_none()) {
                *x = Some(0);
            }
        }
        // a run cut at the wrong place (another request between a request's batches): a batch naming an entry of the
        // other kind; skipped, never linked
        if seen.iter().zip(&kinds).any(|(s, k)| s.is_some() && k.is_some() && s != k) {
            self.c.seg_runs_skipped += 1;
            self.c.seg_skip_kind += 1;
            return;
        }
        let att: Vec<usize> = (0..g).filter(|p| kinds[*p] == Some(0)).collect();
        let st: Vec<usize> = (0..g).filter(|p| kinds[*p] == Some(1)).collect();
        if att.len() != 1 || st.is_empty() || st.iter().any(|p| ch[*p].len() > ch[att[0]].len()) {
            self.c.seg_runs_skipped += 1;
            self.c.seg_skip_layout += 1;
            return;
        }
        let states: Vec<&[Key]> = st.iter().map(|p| &ch[*p][..]).collect();
        self.link_known(ns, &ch[att[0]], &states);
        self.c.seg_runs_linked += 1;
    }

    /// Link one request whose batch kinds are known: the attention chain from its root (`a`), and each GDN group's
    /// states (suffixes of the chain, at most as long) anchored at the attention block of the same position.
    fn link_known(&mut self, ns: u32, a: &[Key], states: &[&[Key]]) {
        for j in 0..a.len() {
            let parent = if j > 0 { p16(&a[j - 1]) } else { [0u8; 16] };
            self.seg.link.insert((ns, p16(&a[j])), (parent, self.now));
            self.seg.gone.remove(&(ns, p16(&a[j])));
        }
        let ng = states.len() as u8;
        for s in states {
            let off = a.len() - s.len();
            for (j, k) in s.iter().enumerate() {
                // also keys not in RAM yet: a cold prompt is touched at its lookup, its prompt-end state is stored
                // after the run was linked (the batch kinds are known here, from the votes / the tags); the anchor is
                // used once the state commits (seg_on_commit)
                self.seg.anchor.insert((ns, p16(k)), (p16(&a[off + j]), ng));
                if self.index.get(&(ns, *k)).is_some_and(|id| self.entries[id].kind == 1) {
                    self.seg.linked.insert(ns);
                }
            }
        }
    }

    /// TOUCH_REQUEST: one request's keys per KV group with their kinds, from a client that knows them.
    /// Touched like TOUCH, then linked from the tags: no runs, no votes, no inferred group count. Linking needs one
    /// attention group and at least one GDN state group, no longer than the attention batch (hybrid models); other
    /// shapes are only touched (counted seg_tagged_unlinked).
    pub fn touch_request(&mut self, conn: ConnId, ns: u32, groups: &[(u8, Vec<Key>)], now: u64) {
        self.now = self.now.max(now);
        self.end_run(conn);                     // never mixed into TOUCH runs
        self.hot.touching.insert(conn);
        for (_, ks) in groups {
            if !ks.is_empty() {
                self.touch(ns, ks, now);
            }
        }
        if !self.cfg.seg_evict {
            return;
        }
        let att: Vec<&[Key]> = groups.iter().filter(|g| g.0 == 0 && !g.1.is_empty()).map(|g| &g.1[..]).collect();
        let st: Vec<&[Key]> = groups.iter().filter(|g| g.0 == 1 && !g.1.is_empty()).map(|g| &g.1[..]).collect();
        if att.len() != 1 || st.is_empty() || st.iter().any(|s| s.len() > att[0].len()) {
            self.c.seg_tagged_unlinked += 1;
            return;
        }
        self.link_known(ns, att[0], &st);
        self.c.seg_tagged_linked += 1;
    }

    /// A committed (or restored) entry: its key is in RAM again; an attention entry's metadata parent is a link.
    pub(crate) fn seg_on_commit(&mut self, id: EntryId) {
        if !self.cfg.seg_evict {
            return;
        }
        let e = &self.entries[&id];
        let (ns, k) = (e.ns, p16(&e.key));
        let t = self.now.max(e.born);   // COMMIT does not advance the core clock: never stamp a link before its entry
        self.seg.gone.remove(&(ns, k));
        if e.kind == 0 {
            if let Some(pp) = meta_parent(&e.meta) {
                if !self.seg.link.contains(&(ns, k)) {
                    self.seg.link.insert((ns, k), (pp, t));
                }
            }
        } else if e.kind == 1 && self.seg.anchor.contains(&(ns, k)) {
            self.seg.linked.insert(ns);
        }
    }

    /// A committed entry left RAM: remember attention as gone (chain broken unless stored again / touched), keep a
    /// state the SSD holds as still covering.
    pub(crate) fn seg_on_remove(&mut self, e: &Entry) {
        if !self.cfg.seg_evict || e.state != State::Committed {
            return;
        }
        let k = p16(&e.key);
        if e.kind == 0 && self.seg.link.contains(&(e.ns, k)) {
            self.seg.gone.insert((e.ns, k), (e.key, self.now));
        } else if e.kind == 1 {
            if let (Some(a), Some(f)) = (self.seg.anchor.get(&(e.ns, k)), self.elsewhere.as_ref()) {
                if f(e.ns, &e.key) {
                    self.seg.away.insert((e.ns, e.key), a);
                }
            }
        }
    }

    pub(crate) fn on_disk(&self, ns: u32, k: &Key) -> bool {
        self.elsewhere.as_ref().is_some_and(|f| f(ns, k))
    }

    pub(crate) fn seg_plan(&self) -> Plan {
        let now = self.now;
        let grace = self.cfg.seg_grace_ms;
        let mut plan = Plan::default();
        // boundaries: the states of one block (one per GDN group) share an anchor; (ns, anchor) -> (RAM states, on
        // SSD, groups expected)
        let mut bnd: HashMap<(u32, P16), (Vec<EntryId>, u32, u8)> = HashMap::new();
        let mut tails: Vec<(u32, P16)> = Vec::new();
        let mut blind: HashSet<u32> = HashSet::new();
        for (id, e) in &self.entries {
            match e.kind {
                1 => match self.seg.anchor.get(&(e.ns, p16(&e.key))) {
                    Some((a, n)) => {
                        let b = bnd.entry((e.ns, a)).or_insert((Vec::new(), 0, n));
                        b.0.push(*id);
                        b.2 = b.2.max(n);
                        plan.anchored += 1;
                    }
                    None => plan.unanchored += 1,
                },
                KIND_TAIL => match meta_parent(&e.meta) {
                    Some(pp) => tails.push((e.ns, pp)),
                    None => {
                        blind.insert(e.ns);
                    }
                },
                _ => {}
            }
        }
        for ((ns, k), (a, n)) in self.seg.away.iter() {
            if !self.index.contains_key(&(*ns, *k)) && self.on_disk(*ns, k) {
                let b = bnd.entry((*ns, *a)).or_insert((Vec::new(), 0, *n));
                b.1 += 1;
            }
        }
        // walk each chain from its anchor to the root (or the first unknown link); coverers: boundaries, then tails
        let keys: Vec<(u32, P16)> = bnd.keys().copied().collect();
        let nb = keys.len();
        let mut cover: HashMap<EntryId, (u32, usize)> = HashMap::new(); // count, first coverer
        let mut broken: HashMap<usize, u64> = HashMap::new();
        for (ci, (ns, a)) in keys.iter().chain(tails.iter()).enumerate() {
            let (ns, a) = (*ns, *a);
            let mut path = Vec::new();
            let (mut p, mut hops, mut brk) = (a, 0u32, None);
            loop {
                match self.pfx.get(&(ns, p)) {
                    Some(aid) if self.entries[aid].kind == 0 => path.push(*aid),
                    Some(_) => {}
                    None => {
                        if let Some((k, t)) = self.seg.gone.get(&(ns, p)) {
                            if !self.on_disk(ns, &k) {
                                brk = Some(t);
                                break;
                            }
                        }
                    }
                }
                hops += 1;
                match self.seg.link.get(&(ns, p)).map(|x| x.0) {
                    Some(q) if q != [0u8; 16] && hops < 4096 => p = q,
                    _ => break,
                }
            }
            if ci < nb {
                let b = &bnd[&(ns, a)];
                if b.0.len() as u32 + b.1 < b.2 as u32 {
                    // a GDN group's state of this block is gone: the others are useless
                    let t = b.0.iter().map(|i| self.entries[i].last_used.max(self.entries[i].born)).max().unwrap_or(0);
                    brk = Some(brk.unwrap_or(0).max(t));
                }
            }
            if let Some(t) = brk {
                if ci < nb {
                    broken.insert(ci, t);
                }
                continue;
            }
            for aid in path {
                let c = cover.entry(aid).or_insert((0, ci));
                c.0 += 1;
            }
        }
        // segments: a boundary's states + the attention only it covers; shared trunk / tail-covered attention held
        let mut members: HashMap<usize, Vec<EntryId>> = HashMap::new();
        for (aid, (n, ci)) in &cover {
            if *n == 1 && *ci < nb {
                if bnd[&keys[*ci]].0.is_empty() {
                    continue; // only an SSD state covers it: an ordinary entry
                }
                members.entry(*ci).or_default().push(*aid);
            } else {
                plan.hold.insert(*aid);
            }
        }
        for (ci, k) in keys.iter().enumerate() {
            let states = &bnd[k].0;
            if states.is_empty() {
                continue;
            }
            if let Some(t) = broken.get(&ci) {
                let ids: Vec<EntryId> = states.iter().copied().filter(|i| self.entries[i].state == State::Committed).collect();
                let last = states.iter().map(|i| self.entries[i].last_used.max(self.entries[i].born)).max().unwrap_or(0);
                if !ids.is_empty() && now.saturating_sub(last.max(*t)) >= grace {
                    let u = plan.units.len();
                    for x in &ids {
                        plan.unit_of.insert(*x, u);
                    }
                    plan.units.push(Unit { ids, orphan: 1 });
                }
                continue;
            }
            let mut ids = states.clone();
            ids.extend(members.get(&ci).into_iter().flatten());
            ids.sort_unstable();
            let u = plan.units.len();
            for x in &ids {
                plan.unit_of.insert(*x, u);
            }
            plan.units.push(Unit { ids, orphan: 0 });
        }
        for (id, e) in &self.entries {
            if e.state != State::Committed || e.kind != 0 || cover.contains_key(id) || !self.seg.linked.contains(&e.ns) {
                continue;
            }
            // young = stored before its chain's last touch, which is within the grace (a request runs on it: the
            // prompt-end state may follow); a copy stored after the last touch (lazy store at GPU eviction) is not
            if let Some((_, tt)) = self.seg.link.get(&(e.ns, p16(&e.key))) {
                if e.born <= tt && now.saturating_sub(tt) < grace || e.shared {
                    plan.hold.insert(*id);
                } else {
                    plan.unit_of.insert(*id, plan.units.len());
                    plan.units.push(Unit { ids: vec![*id], orphan: if blind.contains(&e.ns) { 2 } else { 1 } });
                }
            }
        }
        plan
    }

    /// Pressure / RESERVE victim unit (the `pick_victim` order on units): partial tails, orphans, stale never-hit,
    /// namespace cap, floor, policy order. Falls back to the plain entry order when no unit is evictable.
    pub(crate) fn pick_unit(&self, for_ns: Option<u32>) -> Option<(Vec<EntryId>, u8)> {
        if !self.cfg.seg_evict {
            return self.pick_victim(for_ns).map(|id| (vec![id], 0));
        }
        let plan = self.seg_plan();
        if self.shared_now > 0 {
            if let Some(u) = self.pick_unit_in(&plan, for_ns, false) {
                return Some(u);
            }
        }
        if let Some(u) = self.pick_unit_in(&plan, for_ns, true) {
            return Some(u);
        }
        self.pick_victim(for_ns).map(|id| (vec![id], 0))
    }

    fn pick_unit_in(&self, plan: &Plan, for_ns: Option<u32>, with_shared: bool) -> Option<(Vec<EntryId>, u8)> {
        let now = self.now;
        let ev = |id: &EntryId| self.entries.get(id).is_some_and(|e| e.evictable() && (with_shared || !e.shared));
        let uev = |u: &(Vec<EntryId>, u8)| u.0.iter().all(ev);
        // (0) partial tails first
        if let Some((id, _)) = self.entries.iter().filter(|(id, e)| e.kind == KIND_TAIL && ev(id))
            .min_by_key(|(id, e)| (e.last_used, **id)) {
            return Some((vec![*id], 0));
        }
        // (0b) orphans (every evictable tail is gone at this point: kind 2 too), least recently used first
        let o = plan.units.iter().filter(|u| u.orphan > 0 && u.ids.iter().all(ev))
            .min_by_key(|u| (self.entries[&u.ids[0]].last_used, u.ids[0]));
        if let Some(u) = o {
            return Some((u.ids.clone(), u.orphan));
        }
        let unit = |id: EntryId| plan.unit(id).filter(uev);
        if self.cfg.stale_unhit_ms > 0 {
            let mut v: Vec<(u64, EntryId)> = self.entries.iter()
                .filter(|(id, e)| ev(id) && e.hits == 0 && now.saturating_sub(e.born) >= self.cfg.stale_unhit_ms)
                .map(|(id, e)| (e.born, *id))
                .collect();
            v.sort_unstable();
            for (_, id) in v {
                if let Some(u) = unit(id).filter(|u| u.0.iter().all(|x| self.entries[x].hits == 0)) {
                    return Some(u);
                }
            }
        }
        if self.cfg.ns_cap_slots > 0 {
            if let Some(ns) = for_ns {
                if self.ns_slots(ns) >= self.cfg.ns_cap_slots {
                    if let Some(u) = self.policy.candidates().filter(|id| self.entries[id].ns == ns).find_map(unit) {
                        return Some(u);
                    }
                }
            }
        }
        // weights entries (KIND_WEIGHTS) are not in the policy: they go once colder than its pick (colder_weights);
        // with none, the order is unchanged
        let wev = |e: &Entry| e.evictable() && (with_shared || !e.shared);
        let with_w = |p: Option<(Vec<EntryId>, u8)>, ok: &dyn Fn(&Entry) -> bool| {
            self.colder_weights(p.as_ref().map(|u| &u.0[..]), ok).map(|w| (vec![w], 0)).or(p)
        };
        if self.cfg.ns_floor_slots > 0 {
            let mut held: HashMap<u32, u64> = HashMap::new();
            for e in self.entries.values() {
                *held.entry(e.ns).or_default() += e.n_slots as u64;
            }
            let p = self.policy.candidates().filter(|id| held[&self.entries[id].ns] > self.cfg.ns_floor_slots)
                .find_map(unit);
            if let Some(u) = with_w(p, &|e| wev(e) && held[&e.ns] > self.cfg.ns_floor_slots) {
                return Some(self.low_first(plan, u, &|e| ev(&e) && held[&self.entries[&e].ns] > self.cfg.ns_floor_slots));
            }
        }
        with_w(self.policy.candidates().find_map(unit), &wev).map(|u| self.low_first(plan, u, &|x| ev(&x)))
    }

    /// Class order on units: the coldest lowest-class unit, else the coldest low-class unit (the class of a unit = its
    /// members' lowest rank; all evictable by `ok`) instead of `victim` when `colder_low` says so.
    fn low_first(&self, plan: &Plan, victim: (Vec<EntryId>, u8), ok: &dyn Fn(EntryId) -> bool) -> (Vec<EntryId>, u8) {
        if self.low_margin_ms == 0 {
            return victim;
        }
        for r in [2u8, 1] {
            let low = self.entries.iter().filter(|(id, e)| rank_of(e.flags) >= r && ok(**id))
                .filter_map(|(id, _)| plan.unit(*id))
                .filter(|u| u.0.iter().all(|i| self.entries.get(i).is_some_and(|e| rank_of(e.flags) >= r) && ok(*i)))
                .map(|u| (u.0.iter().map(|i| self.heat(&self.entries[i])).max().unwrap(), u.0));
            if let Some(ids) = self.colder_low(Some(&victim.0), r, low) {
                return (ids, 0);
            }
        }
        victim
    }

    /// Evict a unit; slots freed. Orphans leave as `Removal::Orphan` (counted per kind), segments count once.
    pub(crate) fn evict_unit(&mut self, ids: &[EntryId], orphan: u8, why: Removal) -> u64 {
        let mut n = 0;
        let before = self.entries.len();
        for id in ids {
            let Some(e) = self.entries.get(id) else { continue };   // gone already: taken by an earlier member's cascade
            if !e.evictable() {
                continue;
            }
            if orphan > 0 {
                if e.kind == 1 {
                    self.c.evict_orphan_state += 1;
                } else {
                    self.c.evict_orphan_attn += 1;
                }
            }
            n += self.evict_one(*id, if orphan > 0 { Removal::Orphan } else { why });
        }
        let k = (before - self.entries.len()) as u64;
        if k > 1 && orphan == 0 {
            self.c.seg_units += 1;
            self.c.seg_unit_entries += k;
        }
        n
    }

    /// Evictable units in heat order (orphans first, then the warmest member's heat), skipping units with a member
    /// in `skip`: (heat, ns, ids, orphan).
    pub(crate) fn units_by_heat(&self, plan: &Plan, skip: Option<&HashSet<EntryId>>) -> Vec<(i64, u32, Vec<EntryId>, u8)> {
        let ok = |id: &EntryId| self.entries.get(id).is_some_and(|e| e.evictable()) && !skip.is_some_and(|s| s.contains(id));
        let mut v: Vec<(i64, u32, Vec<EntryId>, u8)> = Vec::new();
        for u in &plan.units {
            if u.ids.iter().all(ok) {
                let o = if u.orphan == 1 { 1 } else { 0 };
                let h = if o == 1 { i64::MIN } else { u.ids.iter().map(|i| self.heat(&self.entries[i])).max().unwrap() };
                v.push((h, self.entries[&u.ids[0]].ns, u.ids.clone(), o));
            }
        }
        for (id, e) in &self.entries {
            if !plan.unit_of.contains_key(id) && !plan.hold.contains(id) && ok(id) {
                v.push((self.heat(e), e.ns, vec![*id], 0));
            }
        }
        v.sort_unstable_by(|a, b| (a.0, a.2[0]).cmp(&(b.0, b.2[0])));
        v
    }

    /// Age sweep on units: evict units colder than `before(ns)`, coldest first, until `max_slots` (0 = no limit).
    /// Orphan units go regardless of their heat once every member is older than the grace (the `seg_scan` rule);
    /// a younger orphan ages by its heat like any unit (evicting orphans always took a just-stored cold prompt
    /// seconds after its commit, with free slots, and its chain by cascade). Re-planned after each pass (a shared
    /// trunk becomes a segment once its other branches left).
    pub(crate) fn seg_evict_aged_by(&mut self, before: &dyn Fn(u32) -> u64, max_slots: u64) -> u64 {
        let mut freed = 0u64;
        let grace = self.cfg.seg_grace_ms;
        loop {
            let plan = self.seg_plan();
            let v = self.units_by_heat(&plan, None);
            let mut did = false;
            for (h, ns, ids, mut o) in v {
                let mut h = h;
                if o == 1 && ids.iter().any(|id| self.now.saturating_sub(self.entries[id].born) < grace) {
                    o = 0;
                    h = ids.iter().map(|i| self.heat(&self.entries[i])).max().unwrap();
                }
                if h >= i64::try_from(before(ns)).unwrap_or(i64::MAX) {
                    continue;
                }
                if max_slots > 0 && freed >= max_slots {
                    return freed;
                }
                freed += self.evict_unit(&ids, o, Removal::LowWater);
                did = true;
            }
            if !did {
                return freed;
            }
        }
    }

    /// Heat of the next unit `seg_evict_oldest_except` takes.
    pub(crate) fn seg_oldest_except(&self, skip: Option<&HashSet<EntryId>>) -> Option<(i64, Vec<EntryId>, u8)> {
        let plan = self.seg_plan();
        self.units_by_heat(&plan, skip).into_iter().next().map(|(h, _, ids, o)| (h, ids, o))
    }

    /// Eager cleanup (from `tick`, at most once a second): orphan units go once every member is older than the grace.
    /// (a 189k-token prefill stores its attention for minutes after the request's touch and its prompt-end
    /// state only at the end; cleaning those "orphans" at once threw away 99 blocks of a live session with 33 GiB of
    /// RAM free, and its next turn recomputed 164k tokens. When room is needed, RESERVE / pressure still take orphans
    /// first, young or not.)
    pub(crate) fn seg_scan(&mut self) {
        if !self.cfg.seg_evict || self.now.saturating_sub(self.seg.last_scan) < SCAN_EVERY_MS {
            return;
        }
        self.seg.last_scan = self.now;
        let conns: Vec<ConnId> = self.seg.runs.keys().filter(|c| !self.conns.contains_key(c)).copied().collect();
        for c in conns {
            self.seg.runs.remove(&c);
            self.seg.learn.remove(&c);
        }
        let plan = self.seg_plan();
        let grace = self.cfg.seg_grace_ms;
        let now = self.now;
        for u in plan.units.iter().filter(|u| u.orphan == 1) {
            if u.ids.iter().all(|id| self.entries.get(id)
                .is_some_and(|e| e.evictable() && now.saturating_sub(e.born) >= grace)) {
                self.evict_unit(&u.ids, 1, Removal::LowWater);
            }
        }
    }

    /// Gauges of the segment plan: anchored / unanchored states, orphan units now, held entries, links known.
    pub(crate) fn seg_gauges(&self) -> Vec<(String, u64)> {
        if !self.cfg.seg_evict {
            return Vec::new();
        }
        let plan = self.seg_plan();
        vec![
            ("seg_states_anchored".into(), plan.anchored),
            ("seg_states_unanchored".into(), plan.unanchored),
            ("seg_orphan_units".into(), plan.units.iter().filter(|u| u.orphan > 0).count() as u64),
            ("seg_held_entries".into(), plan.hold.len() as u64),
            ("seg_links".into(), self.seg.link.len() as u64),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen2_two_generations() {
        let mut g: Gen2<u32, u32> = Gen2::new(3);
        assert!(!g.contains(&1) && g.len() == 0 && g.iter().count() == 0);
        g.insert(1, 10);
        g.insert(2, 20);
        assert_eq!((g.len(), g.get(&1), g.get(&9)), (2, Some(10), None));
        assert!(g.contains(&2) && !g.contains(&9));
        g.insert(3, 30);                       // cur reaches cap: becomes old
        assert_eq!(g.len(), 3);
        g.insert(1, 11);                       // re-inserted: moves to cur, the old copy goes
        assert_eq!((g.len(), g.get(&1)), (3, Some(11)));
        let mut seen: Vec<(u32, u32)> = g.iter().map(|(k, v)| (*k, *v)).collect();
        seen.sort();
        assert_eq!(seen, vec![(1, 11), (2, 20), (3, 30)], "each key once, the newest value");
        g.insert(4, 40);
        g.insert(5, 50);                       // cur full again: the first generation (2, 3) is dropped
        assert!(!g.contains(&2) && !g.contains(&3) && g.contains(&1) && g.contains(&5));
        assert_eq!(g.len(), 3);
        g.remove(&1);
        g.remove(&5);
        assert!(!g.contains(&1) && !g.contains(&5));
        assert_eq!(g.iter().count(), 1);
    }
}
