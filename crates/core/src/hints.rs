//! Prefix-affinity hints (docs/prefix-hints.md): routers ask WHERE a request's prefix is held, engines tell the broker
//! (HINT) which of their keys a router anchor covers. The side table here is bounded, separate from the entry index,
//! and never changes an entry: WHERE is read-only (no lease, no hit, no heat, no policy access).

use crate::{Core, State};
use kvproto::{Anchor, Key};
use std::collections::{BTreeMap, HashMap};

/// Group kind of GDN / Mamba state (the broker's entry kind 1): a resume needs the state AT the boundary, not a run.
const KIND_STATE: u8 = 1;

/// One engine's hint for one anchor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HintRec {
    pub engine_id: u16,
    /// the engine connection's HELLO name when it hinted (routers map backends by it or by engine id)
    pub name: String,
    /// pool class of the hinting connection (unified broker) and its index namespace (domain-scoped)
    pub class: usize,
    pub ns: u32,
    pub chunk_tokens: u32,
    pub flags: u8,
    /// per KV group in group order: (kind, keys in token order)
    pub groups: Vec<(u8, Vec<Key>)>,
    /// broker ms of the hint
    pub t_ms: u64,
}

impl HintRec {
    fn bytes(&self) -> usize {
        96 + self.name.len() + self.groups.iter().map(|(_, k)| 8 + 32 * k.len()).sum::<usize>()
    }
    /// chunks the hint covers (longest group)
    pub fn n_chunks(&self) -> u32 {
        self.groups.iter().map(|(_, k)| k.len()).max().unwrap_or(0) as u32
    }
    fn who(&self) -> (u16, usize, u32) {
        (self.engine_id, self.class, self.ns)
    }
}

/// Anchor -> hints, at most `max_per_anchor` engines per anchor, at most `max_bytes` in all (oldest hint dropped first).
pub struct HintIndex {
    by_anchor: HashMap<Anchor, Vec<u64>>,
    recs: BTreeMap<u64, (Anchor, HintRec)>,
    seq: u64,
    bytes: usize,
    pub max_bytes: usize,
    pub max_per_anchor: usize,
    pub max_keys: usize,
    /// counters: hints stored, refused (too big / off), dropped for room
    pub stored: u64,
    pub refused: u64,
    pub dropped: u64,
}

impl HintIndex {
    pub fn new(max_bytes: usize) -> HintIndex {
        HintIndex { by_anchor: HashMap::new(), recs: BTreeMap::new(), seq: 0, bytes: 0, max_bytes, max_per_anchor: 8,
                    max_keys: 16384, stored: 0, refused: 0, dropped: 0 }
    }

    pub fn len(&self) -> usize {
        self.recs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.recs.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Store (or replace: same engine, class and namespace) the hint for `anchor`. False = refused (off, too many keys).
    pub fn insert(&mut self, anchor: Anchor, rec: HintRec) -> bool {
        let n_keys: usize = rec.groups.iter().map(|(_, k)| k.len()).sum();
        if self.max_bytes == 0 || n_keys == 0 || n_keys > self.max_keys || rec.bytes() > self.max_bytes {
            self.refused += 1;
            return false;
        }
        let ids = self.by_anchor.get(&anchor).cloned().unwrap_or_default();
        for id in ids {
            if self.recs.get(&id).is_some_and(|(_, r)| r.who() == rec.who()) {
                self.remove(id);
            }
        }
        if self.by_anchor.get(&anchor).is_some_and(|v| v.len() >= self.max_per_anchor) {
            let oldest = self.by_anchor[&anchor][0];
            self.remove(oldest);
            self.dropped += 1;
        }
        self.seq += 1;
        self.bytes += rec.bytes();
        self.recs.insert(self.seq, (anchor, rec));
        self.by_anchor.entry(anchor).or_default().push(self.seq);
        while self.bytes > self.max_bytes {
            let Some((&id, _)) = self.recs.iter().next() else { break };
            self.remove(id);
            self.dropped += 1;
        }
        self.stored += 1;
        true
    }

    fn remove(&mut self, id: u64) {
        if let Some((a, r)) = self.recs.remove(&id) {
            self.bytes -= r.bytes();
            if let Some(v) = self.by_anchor.get_mut(&a) {
                v.retain(|x| *x != id);
                if v.is_empty() {
                    self.by_anchor.remove(&a);
                }
            }
        }
    }

    /// Candidate hints for WHERE, in request order (anchor index, hint); per anchor newest hint first. At most
    /// `per_engine` candidates per (engine, class, namespace): the caller checks residency of these only (the first
    /// anchors are the longest prefixes, and a conversation's earlier anchors cover prefixes of the same keys).
    pub fn candidates(&self, anchors: &[Anchor], per_engine: usize) -> Vec<(u32, HintRec)> {
        let mut n: HashMap<(u16, usize, u32), usize> = HashMap::new();
        let mut out = Vec::new();
        for (i, a) in anchors.iter().enumerate() {
            let Some(ids) = self.by_anchor.get(a) else { continue };
            for id in ids.iter().rev() {
                let r = &self.recs[id].1;
                let c = n.entry(r.who()).or_default();
                if *c < per_engine {
                    *c += 1;
                    out.push((i as u32, r.clone()));
                }
            }
        }
        out
    }
}

impl Core {
    fn readable(&self, ns: u32, k: &Key) -> bool {
        self.index.get(&(ns, *k)).is_some_and(|id| {
            let e = &self.entries[id];
            e.state == State::Committed && !e.rewriting
        })
    }

    /// Read-only: how many leading chunks of a request's KV groups an engine could load from RAM now. Attention (and any
    /// non-state) groups need every key up to the boundary; state groups (kind 1) need their key AT the boundary
    /// (chunk n - 1). The largest n that satisfies every group; 0 if none. No lease, no hit, no policy access.
    pub fn resident_prefix(&self, ns: u32, groups: &[(u8, Vec<Key>)]) -> u32 {
        let state: Vec<&Vec<Key>> = groups.iter().filter(|(k, v)| *k == KIND_STATE && !v.is_empty()).map(|(_, v)| v).collect();
        let runs: Vec<usize> = groups.iter().filter(|(k, v)| *k != KIND_STATE && !v.is_empty())
            .map(|(_, v)| v.iter().take_while(|k| self.readable(ns, k)).count()).collect();
        let mut n = match runs.iter().min() {
            Some(&m) => m,
            None => state.iter().map(|v| v.len()).max().unwrap_or(0),
        };
        if let Some(m) = state.iter().map(|v| v.len()).min() {
            n = n.min(m);
        }
        while n > 0 && !state.iter().all(|v| self.readable(ns, &v[n - 1])) {
            n -= 1;
        }
        n as u32
    }

    /// The engine id a connection sent in HELLO (None before HELLO).
    pub fn engine_id(&self, conn: crate::ConnId) -> Option<u16> {
        self.conns.get(&conn).filter(|c| c.hello).map(|c| c.engine_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Core};
    use crate::policy::PolicyKind;
    use kvproto::ReserveItem;

    fn cfg() -> Config {
        Config { n_slots: 64, slot_bytes: 100, reserve_timeout_ms: 30_000, lease_timeout_ms: 60_000, low_water_slots: 0,
                 policy: PolicyKind::Lru, stale_unhit_ms: 0, ns_floor_slots: 0, ns_cap_slots: 0, tail_age_div: 4, heat_hit_ms: 0,
                 shared_slots: 0, shared_age_mult: 24, shared_min_cont: 3, shared_max_depth: 19, seg_evict: true,
                 seg_grace_ms: 300_000, weights_kv_slot_ms: 0 }
    }
    fn k(n: u32) -> Key {
        let mut x = [0u8; 32];
        x[..4].copy_from_slice(&n.to_le_bytes());
        x
    }
    fn store(c: &mut Core, ns: u32, keys: &[u32], kind: u8) {
        let items: Vec<ReserveItem> = keys.iter().map(|&n| ReserveItem { key: k(n), parent: None, bytes: 100, kind, format: 1,
                                                                          cost_hint: 0, flags: 0, ttl_ms: 0, meta: vec![] }).collect();
        let r = c.reserve(1, ns, &items, 1);
        assert!(r.iter().all(|(s, _)| *s == kvproto::Status::Ok));
        c.commit(1, ns, &keys.iter().map(|&n| k(n)).collect::<Vec<_>>(), 2);
    }
    fn core() -> Core {
        let mut c = Core::new(cfg());
        c.connect(1);
        c.hello(1, 3);
        c
    }
    fn rec(engine_id: u16, groups: Vec<(u8, Vec<Key>)>) -> HintRec {
        HintRec { engine_id, name: format!("e{engine_id}"), class: 0, ns: 1, chunk_tokens: 1664, flags: 0, groups, t_ms: 0 }
    }

    #[test]
    fn resident_prefix_attention_and_state() {
        let mut c = core();
        store(&mut c, 1, &[1, 2, 3], 0);          // attention chunks 0..3 of 5
        store(&mut c, 1, &[12, 13], 1);           // state at chunks 1 and 2 (not at 0, 3, 4)
        let attn: Vec<Key> = (1..=5).map(k).collect();
        let st: Vec<Key> = (11..=15).map(k).collect();
        assert_eq!(c.resident_prefix(1, &[(0, attn.clone())]), 3, "leading attention run");
        assert_eq!(c.resident_prefix(1, &[(0, attn.clone()), (1, st.clone())]), 3, "state at chunk 2 = boundary 3");
        store(&mut c, 1, &[4], 0);
        assert_eq!(c.resident_prefix(1, &[(0, attn.clone()), (1, st.clone())]), 3, "no state at chunk 3: back to 3");
        assert_eq!(c.resident_prefix(2, &[(0, attn.clone())]), 0, "other namespace");
        assert_eq!(c.resident_prefix(1, &[(1, st.clone())]), 3, "state only: latest state boundary");
        assert_eq!(c.resident_prefix(1, &[(0, vec![k(9), k(1)])]), 0, "a hole at chunk 0");
        assert_eq!(c.resident_prefix(1, &[]), 0);
    }

    #[test]
    fn resident_prefix_is_read_only() {
        let mut c = core();
        store(&mut c, 1, &[1, 2], 0);
        let before = c.counters();
        for _ in 0..10 {
            assert_eq!(c.resident_prefix(1, &[(0, vec![k(1), k(2), k(3)])]), 2);
        }
        assert_eq!(c.counters(), before, "no counter (lookups, hits) moved");
        assert_eq!(c.lookup(1, 1, &[k(1)], false, 3).0, 1);
        assert_ne!(c.counters(), before, "while a LOOKUP does move them");
    }

    #[test]
    fn index_replace_per_engine_and_bound() {
        let mut h = HintIndex::new(1 << 20);
        let a = [1u8; 16];
        assert!(h.insert(a, rec(1, vec![(0, vec![k(1)])])));
        assert!(h.insert(a, rec(1, vec![(0, vec![k(1), k(2)])])), "same engine: replaced");
        assert!(h.insert(a, rec(2, vec![(0, vec![k(5)])])));
        assert_eq!(h.len(), 2);
        let c = h.candidates(&[[9u8; 16], a], 4);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].0, 1, "anchor index into the request");
        assert_eq!(c[0].1.engine_id, 2, "newest first");
        assert_eq!(c[1].1.n_chunks(), 2);
        assert!(!h.insert(a, rec(1, vec![(0, vec![])])), "no keys: refused");
        // bound: each hint ~ 96 + 2 + 8 + 32 bytes; room for a few
        let mut small = HintIndex::new(600);
        for i in 0..10u8 {
            assert!(small.insert([i; 16], rec(1, vec![(0, vec![k(i as u32)])])));
        }
        assert!(small.bytes() <= 600 && small.len() < 10 && small.dropped > 0);
        assert!(small.candidates(&[[9u8; 16]], 4).len() == 1, "the newest survived");
        assert!(small.candidates(&[[0u8; 16]], 4).is_empty(), "the oldest went first");
        let mut off = HintIndex::new(0);
        assert!(!off.insert(a, rec(1, vec![(0, vec![k(1)])])), "max_bytes 0 = off");
    }

    #[test]
    fn candidates_capped_per_engine() {
        let mut h = HintIndex::new(1 << 20);
        for i in 0..6u8 {
            h.insert([i; 16], rec(1, vec![(0, vec![k(i as u32)])]));
        }
        let anchors: Vec<Anchor> = (0..6u8).rev().map(|i| [i; 16]).collect();
        let c = h.candidates(&anchors, 2);
        assert_eq!(c.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1], "the longest two");
    }
}
