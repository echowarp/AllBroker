//! Eviction order over committed entries. The index decides what is evictable (unleased leaves); a policy only
//! orders candidates. Sizes are in slots. `ghost` is a stable hash of (namespace, key) so ARC can recognise a
//! re-stored key after its entry id is gone.

use std::collections::{BTreeMap, HashMap};

pub type EntryId = u64;

pub trait Policy: Send {
    fn name(&self) -> &'static str;
    fn insert(&mut self, id: EntryId, ghost: u64, size: u64);
    fn access(&mut self, id: EntryId);
    /// Most evictable from now on (HEAT cold): a dead chain goes before everything else.
    fn demote(&mut self, id: EntryId);
    /// `evicted` = removed to make room (ARC remembers it as a ghost), not aborted or expired.
    fn remove(&mut self, id: EntryId, evicted: bool);
    /// Candidates, most evictable first.
    fn candidates(&self) -> Box<dyn Iterator<Item = EntryId> + '_>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyKind {
    Arc,
    Lru,
}

impl std::str::FromStr for PolicyKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "arc" => Ok(PolicyKind::Arc),
            "lru" => Ok(PolicyKind::Lru),
            _ => Err(format!("unknown policy {s} (arc, lru)")),
        }
    }
}

pub fn make(kind: PolicyKind, capacity: u64) -> Box<dyn Policy> {
    match kind {
        PolicyKind::Arc => Box::new(Arc::new(capacity)),
        PolicyKind::Lru => Box::new(Lru::default()),
    }
}

/// Recency list: O(log n) touch/remove, iteration from least recent.
#[derive(Default)]
struct List<K: Copy + Eq + std::hash::Hash> {
    order: BTreeMap<u64, K>,
    pos: HashMap<K, (u64, u64)>, // key -> (seq, size)
    seq: u64,
    lo: u64,                      // push_front sequence, counting down below SEQ_BASE
    bytes: u64,
}

const SEQ_BASE: u64 = 1 << 62;

impl<K: Copy + Eq + std::hash::Hash> List<K> {
    fn push(&mut self, k: K, size: u64) {
        self.seq = self.seq.max(SEQ_BASE) + 1;
        self.order.insert(self.seq, k);
        self.pos.insert(k, (self.seq, size));
        self.bytes += size;
    }
    fn push_front(&mut self, k: K, size: u64) {
        self.lo = if self.lo == 0 { SEQ_BASE } else { self.lo } - 1;
        self.order.insert(self.lo, k);
        self.pos.insert(k, (self.lo, size));
        self.bytes += size;
    }
    fn take(&mut self, k: &K) -> Option<u64> {
        let (s, size) = self.pos.remove(k)?;
        self.order.remove(&s);
        self.bytes -= size;
        Some(size)
    }
    fn contains(&self, k: &K) -> bool {
        self.pos.contains_key(k)
    }
    fn pop_oldest(&mut self) -> Option<(K, u64)> {
        let (&s, &k) = self.order.iter().next()?;
        self.order.remove(&s);
        let (_, size) = self.pos.remove(&k).unwrap();
        self.bytes -= size;
        Some((k, size))
    }
    fn iter(&self) -> impl Iterator<Item = K> + '_ {
        self.order.values().copied()
    }
}

#[derive(Default)]
pub struct Lru {
    l: List<EntryId>,
}

impl Policy for Lru {
    fn name(&self) -> &'static str {
        "lru"
    }
    fn insert(&mut self, id: EntryId, _ghost: u64, size: u64) {
        self.l.push(id, size);
    }
    fn access(&mut self, id: EntryId) {
        if let Some(size) = self.l.take(&id) {
            self.l.push(id, size);
        }
    }
    fn demote(&mut self, id: EntryId) {
        if let Some(size) = self.l.take(&id) {
            self.l.push_front(id, size);
        }
    }
    fn remove(&mut self, id: EntryId, _evicted: bool) {
        self.l.take(&id);
    }
    fn candidates(&self) -> Box<dyn Iterator<Item = EntryId> + '_> {
        Box::new(self.l.iter())
    }
}

/// Size-weighted ARC (Megiddo & Modha): T1 = seen once, T2 = seen again; ghosts B1/B2 steer the T1 target `p`.
pub struct Arc {
    cap: u64,
    p: u64,
    t1: List<EntryId>,
    t2: List<EntryId>,
    b1: List<u64>,
    b2: List<u64>,
    ghost_of: HashMap<EntryId, u64>,
}

impl Arc {
    pub fn new(cap: u64) -> Self {
        Arc {
            cap,
            p: 0,
            t1: List::default(),
            t2: List::default(),
            b1: List::default(),
            b2: List::default(),
            ghost_of: HashMap::new(),
        }
    }
    pub fn target(&self) -> u64 {
        self.p
    }
    fn trim_ghosts(&mut self) {
        while self.b1.bytes > self.cap {
            self.b1.pop_oldest();
        }
        while self.b2.bytes > self.cap {
            self.b2.pop_oldest();
        }
    }
}

impl Policy for Arc {
    fn name(&self) -> &'static str {
        "arc"
    }
    fn insert(&mut self, id: EntryId, ghost: u64, size: u64) {
        self.ghost_of.insert(id, ghost);
        if self.b1.contains(&ghost) {
            let (g1, g2) = (self.b1.bytes.max(1), self.b2.bytes);
            self.p = (self.p + size.max(g2 * size / g1)).min(self.cap);
            self.b1.take(&ghost);
            self.t2.push(id, size);
        } else if self.b2.contains(&ghost) {
            let (g1, g2) = (self.b1.bytes, self.b2.bytes.max(1));
            self.p = self.p.saturating_sub(size.max(g1 * size / g2));
            self.b2.take(&ghost);
            self.t2.push(id, size);
        } else {
            self.t1.push(id, size);
        }
    }
    fn access(&mut self, id: EntryId) {
        if let Some(size) = self.t1.take(&id).or_else(|| self.t2.take(&id)) {
            self.t2.push(id, size);
        }
    }
    fn demote(&mut self, id: EntryId) {
        // T1's cold end; REPLACE reads T2 first only while T1 is within its target, so also leave T2
        if let Some(size) = self.t1.take(&id).or_else(|| self.t2.take(&id)) {
            self.t1.push_front(id, size);
        }
    }
    fn remove(&mut self, id: EntryId, evicted: bool) {
        let ghost = self.ghost_of.remove(&id);
        let (size, from_t1) = match self.t1.take(&id) {
            Some(s) => (s, true),
            None => match self.t2.take(&id) {
                Some(s) => (s, false),
                None => return,
            },
        };
        if let (true, Some(g)) = (evicted, ghost) {
            if from_t1 {
                self.b1.push(g, size);
            } else {
                self.b2.push(g, size);
            }
            self.trim_ghosts();
        }
    }
    fn candidates(&self) -> Box<dyn Iterator<Item = EntryId> + '_> {
        // REPLACE: take from T1 while it is over its target (or T2 is empty), else from T2; then the other list,
        // because the head of the preferred list may be all leased or interior chain entries.
        if !self.t1.order.is_empty() && (self.t1.bytes > self.p || self.t2.order.is_empty()) {
            Box::new(self.t1.iter().chain(self.t2.iter()))
        } else {
            Box::new(self.t2.iter().chain(self.t1.iter()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_order() {
        let mut l = Lru::default();
        for i in 1..=3 {
            l.insert(i, i, 1);
        }
        l.access(1);
        assert_eq!(l.candidates().collect::<Vec<_>>(), vec![2, 3, 1]);
        l.remove(3, true);
        assert_eq!(l.candidates().collect::<Vec<_>>(), vec![2, 1]);
    }

    #[test]
    fn arc_promotes_and_learns() {
        let mut a = Arc::new(4);
        a.insert(1, 101, 1);
        a.insert(2, 102, 1);
        a.access(1); // 1 -> T2
        // T1 = [2] over target p = 0: T1 first
        assert_eq!(a.candidates().collect::<Vec<_>>(), vec![2, 1]);
        a.remove(2, true); // ghost in B1
        a.insert(3, 102, 1); // same key comes back: B1 hit raises p and goes to T2
        assert!(a.target() >= 1);
        assert!(a.t2.contains(&3));
    }

    #[test]
    fn list_bytes_and_front_order() {
        let mut l: List<u32> = List::default();
        l.push(1, 3);
        l.push(2, 5);
        l.push_front(3, 7);
        l.push_front(4, 11);   // the coldest end: last pushed to the front comes first
        assert_eq!(l.iter().collect::<Vec<_>>(), vec![4, 3, 1, 2]);
        assert_eq!(l.bytes, 26);
        assert_eq!(l.take(&3), Some(7));
        assert_eq!(l.bytes, 19);
        assert_eq!(l.take(&3), None);
        l.push(5, 1);
        assert_eq!(l.iter().collect::<Vec<_>>(), vec![4, 1, 2, 5]);
        assert_eq!(l.pop_oldest(), Some((4, 11)));
        assert_eq!(l.bytes, 9);
    }

    #[test]
    fn demote_goes_to_the_cold_end() {
        for kind in [PolicyKind::Lru, PolicyKind::Arc] {
            let mut p = make(kind, 100);
            for i in 1..=4 {
                p.insert(i, 100 + i, 1);
            }
            p.demote(3);
            p.demote(2);
            let c: Vec<_> = p.candidates().collect();
            assert_eq!(&c[..2], &[2, 3], "{kind:?}: the last demoted first");
            p.insert(5, 105, 1);
            assert_eq!(p.candidates().collect::<Vec<_>>().len(), 5);
        }
    }

    #[test]
    fn policy_names_parse() {
        assert_eq!("arc".parse::<PolicyKind>(), Ok(PolicyKind::Arc));
        assert_eq!("lru".parse::<PolicyKind>(), Ok(PolicyKind::Lru));
        assert!("lfu".parse::<PolicyKind>().is_err());
        assert_eq!(make(PolicyKind::Lru, 1).name(), "lru");
        assert_eq!(make(PolicyKind::Arc, 1).name(), "arc");
    }

    #[test]
    fn arc_adapts_by_ghost_ratios() {
        let mut a = Arc::new(100);
        a.insert(1, 11, 2);
        a.insert(2, 12, 6);
        a.remove(1, true);                 // B1: 2 bytes
        a.access(2);
        a.remove(2, true);                 // B2: 6 bytes
        a.insert(3, 11, 3);                // B1 hit: p += max(3, 6 x 3 / 2) = 9
        assert_eq!(a.target(), 9);
        assert!(!a.b1.contains(&11) && a.t2.contains(&3));
        a.insert(4, 14, 8);
        a.remove(4, true);                 // B1: 8 bytes
        a.insert(5, 12, 3);                // B2 hit: p -= max(3, 8 x 3 / 6) = 4
        assert_eq!(a.target(), 5);
        assert!(!a.b2.contains(&12) && a.t2.contains(&5));
    }

    #[test]
    fn arc_ghost_lists_bounded() {
        let mut a = Arc::new(4);
        for i in 1..=3 {
            a.insert(i, 10 + i, 2);
        }
        for i in 1..=3 {
            a.remove(i, true);             // B1: 2, 4, 6 bytes -> trimmed to 4
        }
        assert_eq!(a.b1.bytes, 4);
        assert!(!a.b1.contains(&11) && a.b1.contains(&12) && a.b1.contains(&13));
        for i in 4..=6 {
            a.insert(i, 10 + i, 2);
            a.access(i);
        }
        for i in 4..=6 {
            a.remove(i, true);             // B2 likewise
        }
        assert_eq!(a.b2.bytes, 4);
        assert!(!a.b2.contains(&14) && a.b2.contains(&15));
        a.remove(99, true);                // unknown: nothing
        assert_eq!((a.b1.bytes, a.b2.bytes), (4, 4));
    }

    #[test]
    fn arc_replace_prefers_t1_only_over_its_target() {
        let mut a = Arc::new(100);
        a.insert(1, 11, 2);
        a.insert(2, 12, 2);
        a.access(2);                       // T1 = [1] (2 bytes), T2 = [2], p = 0
        assert_eq!(a.candidates().collect::<Vec<_>>(), vec![1, 2], "T1 over its target");
        a.p = 2;                           // T1 at its target: T2 first
        assert_eq!(a.candidates().collect::<Vec<_>>(), vec![2, 1]);
        a.p = 3;
        assert_eq!(a.candidates().collect::<Vec<_>>(), vec![2, 1]);
        a.remove(2, false);                // T2 empty: T1 whatever the target
        assert_eq!(a.candidates().collect::<Vec<_>>(), vec![1]);
        a.remove(1, false);
        assert_eq!(a.candidates().count(), 0);
        assert_eq!((a.b1.bytes, a.b2.bytes), (0, 0), "removed, not evicted: no ghosts");
    }
}
