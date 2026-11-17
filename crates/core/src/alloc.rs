//! Slot allocator: free runs kept by start (for merging) and by (len, start) (for best fit).

use kvproto::Range;
use std::collections::{BTreeMap, BTreeSet};

pub struct Alloc {
    by_start: BTreeMap<u32, u32>,
    by_len: BTreeSet<(u32, u32)>,
    free: u64,
    total: u32,
    /// Parked runs (dynamic pool): slots taken out of the allocator so the broker can return their
    /// memory to the kernel (hole-punched); neither free nor occupied until unparked.
    parked: BTreeMap<u32, u32>,
    /// First fit by address instead of best fit (on-demand pools): new entries go into the lowest free
    /// slots, which the broker keeps allocated ahead (its own memory, never swapped), so engines rarely fault in
    /// pages of their own.
    pub low_first: bool,
}

impl Alloc {
    pub fn new(n_slots: u32) -> Self {
        let mut a = Alloc { by_start: BTreeMap::new(), by_len: BTreeSet::new(), free: 0, total: n_slots, parked: BTreeMap::new(), low_first: false };
        if n_slots > 0 {
            a.insert_run(0, n_slots);
        }
        a
    }
    pub fn free(&self) -> u64 {
        self.free
    }
    pub fn total(&self) -> u32 {
        self.total
    }
    pub fn runs(&self) -> usize {
        self.by_start.len()
    }
    pub fn parked(&self) -> u64 {
        self.parked.values().map(|&l| l as u64).sum()
    }
    pub fn parked_runs(&self) -> Vec<Range> {
        self.parked.iter().map(|(&start, &len)| Range { start, len }).collect()
    }
    /// Take up to `n` FREE slots out of the allocator (largest runs first, splitting the last one) and park them.
    /// Returns the newly parked runs (merged with adjacent parked runs in the table, but reported as taken).
    pub fn park(&mut self, n: u32) -> Vec<Range> {
        let mut out = Vec::new();
        let mut left = n;
        while left > 0 {
            let Some(&(len, start)) = self.by_len.iter().next_back() else { break };
            self.remove_run(start, len);
            let take = len.min(left);
            if len > take {
                self.insert_run(start + take, len - take);
            }
            self.park_run(start, take);
            out.push(Range { start, len: take });
            left -= take;
        }
        out.sort_by_key(|r| r.start);
        out
    }
    fn park_run(&mut self, mut start: u32, mut len: u32) {
        if let Some((&ps, &pl)) = self.parked.range(..start).next_back() {
            if ps + pl == start {
                self.parked.remove(&ps);
                start = ps;
                len += pl;
            }
        }
        if let Some(&nl) = self.parked.get(&(start + len)) {
            self.parked.remove(&(start + len));
            len += nl;
        }
        self.parked.insert(start, len);
    }
    /// Return parked slots to the allocator: the given ranges must be parked (sub-ranges of parked runs).
    pub fn unpark(&mut self, ranges: &[Range]) {
        for r in ranges {
            let Some((&ps, &pl)) = self.parked.range(..=r.start).next_back() else { continue };
            if r.start < ps || r.start + r.len > ps + pl {
                continue;   // not parked (double unpark): ignore
            }
            self.parked.remove(&ps);
            if r.start > ps {
                self.parked.insert(ps, r.start - ps);
            }
            if r.start + r.len < ps + pl {
                self.parked.insert(r.start + r.len, ps + pl - (r.start + r.len));
            }
            self.release(&[*r]);
        }
    }
    fn insert_run(&mut self, start: u32, len: u32) {
        self.by_start.insert(start, len);
        self.by_len.insert((len, start));
        self.free += len as u64;
    }
    fn remove_run(&mut self, start: u32, len: u32) {
        self.by_start.remove(&start);
        self.by_len.remove(&(len, start));
        self.free -= len as u64;
    }
    /// Free runs in address order (the broker hole-punches their memory in a lazily backed pool).
    pub fn free_runs(&self) -> Vec<Range> {
        self.by_start.iter().map(|(&start, &len)| Range { start, len }).collect()
    }
    /// Take exactly these slots out of the free runs (they must all be free: one run holds the whole range), for
    /// the broker to hole-punch them without the lock; `release` gives them back. false: nothing taken.
    pub fn take_range(&mut self, r: Range) -> bool {
        let Some((&ps, &pl)) = self.by_start.range(..=r.start).next_back() else { return false };
        if r.len == 0 || r.start + r.len > ps + pl {
            return false;
        }
        self.remove_run(ps, pl);
        if r.start > ps {
            self.insert_run(ps, r.start - ps);
        }
        if r.start + r.len < ps + pl {
            self.insert_run(r.start + r.len, ps + pl - (r.start + r.len));
        }
        true
    }
    /// Compaction: `n` free slots all below slot `below`, lowest addresses first; None (nothing taken) if there are
    /// fewer.
    pub fn alloc_below(&mut self, n: u32, below: u32) -> Option<Vec<Range>> {
        let mut pick = Vec::new();
        let mut left = n;
        for (&start, &len) in self.by_start.range(..below) {
            if left == 0 {
                break;
            }
            let take = len.min(below - start).min(left);
            pick.push((start, len, take));
            left -= take;
        }
        if left > 0 {
            return None;
        }
        let mut out = Vec::new();
        for (start, len, take) in pick {
            self.remove_run(start, len);
            if len > take {
                self.insert_run(start + take, len - take);
            }
            out.push(Range { start, len: take });
        }
        Some(out)
    }
    /// Allocate `n` slots: one best-fit run if possible, else the largest runs first (fewest ranges).
    /// The caller guarantees `free() >= n`.
    pub fn alloc(&mut self, n: u32) -> Vec<Range> {
        debug_assert!(self.free >= n as u64);
        if self.low_first {
            return self.alloc_low(n);
        }
        if let Some(&(len, start)) = self.by_len.range((n, 0)..).next() {
            self.remove_run(start, len);
            if len > n {
                self.insert_run(start + n, len - n);
            }
            return vec![Range { start, len: n }];
        }
        let mut out = Vec::new();
        let mut left = n;
        while left > 0 {
            let &(len, start) = self.by_len.iter().next_back().expect("free count out of sync");
            self.remove_run(start, len);
            let take = len.min(left);
            if len > take {
                self.insert_run(start + take, len - take);
            }
            out.push(Range { start, len: take });
            left -= take;
        }
        out.sort_by_key(|r| r.start);
        out
    }
    /// `low_first`: the lowest run that holds all `n`, else the lowest runs in address order.
    fn alloc_low(&mut self, n: u32) -> Vec<Range> {
        if let Some((&start, &len)) = self.by_start.iter().find(|(_, &l)| l >= n) {
            self.remove_run(start, len);
            if len > n {
                self.insert_run(start + n, len - n);
            }
            return vec![Range { start, len: n }];
        }
        let mut out = Vec::new();
        let mut left = n;
        while left > 0 {
            let (&start, &len) = self.by_start.iter().next().expect("free count out of sync");
            self.remove_run(start, len);
            let take = len.min(left);
            if len > take {
                self.insert_run(start + take, len - take);
            }
            out.push(Range { start, len: take });
            left -= take;
        }
        out
    }
    pub fn release(&mut self, ranges: &[Range]) {
        for r in ranges {
            let (mut start, mut len) = (r.start, r.len);
            if let Some((&ps, &pl)) = self.by_start.range(..start).next_back() {
                debug_assert!(ps + pl <= start, "double free");
                if ps + pl == start {
                    self.remove_run(ps, pl);
                    start = ps;
                    len += pl;
                }
            }
            if let Some(&nl) = self.by_start.get(&(start + len)) {
                self.remove_run(start + len, nl);
                len += nl;
            }
            self.insert_run(start, len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_fit_split_merge() {
        let mut a = Alloc::new(100);
        let x = a.alloc(10);
        let y = a.alloc(20);
        let z = a.alloc(30);
        assert_eq!(a.free(), 40);
        a.release(&y);
        assert_eq!(a.runs(), 2);
        // best fit: 15 goes into the 20-slot hole, not the 40-slot tail
        let w = a.alloc(15);
        assert_eq!(w, vec![Range { start: 10, len: 15 }]);
        a.release(&x);
        a.release(&z);
        a.release(&w);
        assert_eq!((a.free(), a.runs()), (100, 1));
    }

    #[test]
    fn take_range_splits_a_free_run_and_release_merges_it() {
        let mut a = Alloc::new(10);
        assert!(a.take_range(Range { start: 3, len: 2 }));
        assert_eq!((a.free(), a.runs()), (8, 2));
        assert!(!a.take_range(Range { start: 2, len: 2 }), "slot 3 is taken");
        assert!(!a.take_range(Range { start: 9, len: 2 }), "past the end");
        a.release(&[Range { start: 3, len: 2 }]);
        assert_eq!((a.free(), a.runs()), (10, 1));
    }

    #[test]
    fn low_first() {
        let mut a = Alloc::new(100);
        a.low_first = true;
        let x = a.alloc(10);
        let _y = a.alloc(20);
        a.release(&x);
        // first fit by address: 5 goes into the hole at 0, not the 70-slot tail
        assert_eq!(a.alloc(5), vec![Range { start: 0, len: 5 }]);
        // 10 fits nowhere below 30 except the tail
        assert_eq!(a.alloc(10), vec![Range { start: 30, len: 10 }]);
        // 65 > any run: lowest runs in address order
        let z = a.alloc(65);
        assert_eq!(z, vec![Range { start: 5, len: 5 }, Range { start: 40, len: 60 }]);
        assert_eq!(a.free(), 0);
    }

    #[test]
    fn park_unpark() {
        let mut a = Alloc::new(100);
        let x = a.alloc(10);
        let _y = a.alloc(20);
        a.release(&x);                 // free: [0,10) and [30,100)
        let p = a.park(75);            // largest first: 70 from the tail, 5 from the head
        assert_eq!(p.iter().map(|r| r.len).sum::<u32>(), 75);
        assert_eq!((a.free(), a.parked()), (5, 75));
        assert!(a.alloc(5).len() == 1 && a.free() == 0);
        a.unpark(&p);
        assert_eq!((a.free(), a.parked()), (75, 0));
        a.unpark(&p);                  // double unpark is ignored
        assert_eq!((a.free(), a.parked()), (75, 0));
    }

    #[test]
    fn alloc_below_low_first() {
        let mut a = Alloc::new(20);
        let x = a.alloc(4);            // [0,4)
        let y = a.alloc(4);            // [4,8)
        let _z = a.alloc(4);           // [8,12)
        a.release(&x);
        a.release(&y);                 // free: [0,8) and [12,20)
        assert_eq!(a.alloc_below(5, 12), Some(vec![Range { start: 0, len: 5 }]));
        assert_eq!(a.alloc_below(4, 12), None);       // only 3 free below 12
        assert_eq!(a.free(), 11);
        assert_eq!(a.free_runs(), vec![Range { start: 5, len: 3 }, Range { start: 12, len: 8 }]);
    }

    #[test]
    fn fragmented_alloc() {
        let mut a = Alloc::new(10);
        let parts: Vec<_> = (0..10).map(|_| a.alloc(1)).collect();
        for p in parts.iter().step_by(2) {
            a.release(p);
        }
        let r = a.alloc(4);
        assert_eq!(r.iter().map(|r| r.len).sum::<u32>(), 4);
        assert_eq!(a.free(), 1);
    }

    // ------------------------------------------------ model-based property test (after cargo mutants)

    use proptest::prelude::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum S {
        Free,
        Used(u32),
        Parked,
    }

    #[derive(Clone, Debug)]
    enum Op {
        Alloc(u32, bool),
        Release(u8),
        Park(u32),
        Unpark(u8, u8, u8),
        Below(u32, u32),
    }

    const N: u32 = 48;

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            4 => (1u32..12, any::<bool>()).prop_map(|(n, l)| Op::Alloc(n, l)),
            4 => any::<u8>().prop_map(Op::Release),
            1 => (0u32..20).prop_map(Op::Park),
            2 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(a, b, c)| Op::Unpark(a, b, c)),
            1 => (1u32..10, 0u32..=N).prop_map(|(n, b)| Op::Below(n, b)),
        ]
    }

    /// Maximal runs of slots in state `want`, in address order.
    fn runs_of(m: &[S], want: impl Fn(S) -> bool) -> Vec<Range> {
        let mut out: Vec<Range> = Vec::new();
        for (i, s) in m.iter().enumerate() {
            if want(*s) {
                match out.last_mut() {
                    Some(r) if r.start + r.len == i as u32 => r.len += 1,
                    _ => out.push(Range { start: i as u32, len: 1 }),
                }
            }
        }
        out
    }

    fn take(m: &mut [S], rs: &[Range], from: S, to: S) -> Result<(), TestCaseError> {
        for r in rs {
            prop_assert!(r.len > 0 && r.start + r.len <= N, "bad range {:?}", r);
            for i in r.start..r.start + r.len {
                prop_assert_eq!(m[i as usize], from, "slot {} handed out from {:?}", i, m[i as usize]);
                m[i as usize] = to;
            }
        }
        Ok(())
    }

    fn run(ops: Vec<Op>) -> Result<(), TestCaseError> {
        let mut a = Alloc::new(N);
        let mut m = vec![S::Free; N as usize];
        let mut owned: Vec<(u32, Vec<Range>)> = Vec::new();
        let mut next = 0u32;
        for o in ops {
            match o {
                Op::Alloc(n, low) => {
                    if a.free() < n as u64 {
                        continue;
                    }
                    a.low_first = low;
                    let free = runs_of(&m, |s| s == S::Free);
                    let got = a.alloc(n);
                    prop_assert_eq!(got.iter().map(|r| r.len).sum::<u32>(), n);
                    let fits: Vec<&Range> = free.iter().filter(|r| r.len >= n).collect();
                    if low {
                        // lowest run that holds all, else the lowest runs in address order
                        match fits.first() {
                            Some(r) => prop_assert_eq!(&got, &vec![Range { start: r.start, len: n }]),
                            None => {
                                prop_assert!(got.windows(2).all(|w| w[0].start < w[1].start));
                                let mut left = n;
                                for (g, f) in got.iter().zip(&free) {
                                    prop_assert_eq!(g.start, f.start);
                                    prop_assert_eq!(g.len, f.len.min(left));
                                    left -= g.len;
                                }
                            }
                        }
                    } else {
                        // best fit: the smallest run that holds all (lowest of equals), else largest runs first
                        match fits.iter().min_by_key(|r| (r.len, r.start)) {
                            Some(r) => prop_assert_eq!(&got, &vec![Range { start: r.start, len: n }]),
                            None => {
                                let mut by_len: Vec<&Range> = free.iter().collect();
                                by_len.sort_by_key(|r| std::cmp::Reverse((r.len, r.start)));
                                prop_assert_eq!(got.len(), by_len.iter().scan(0u32, |s, r| {
                                    let before = *s;
                                    *s += r.len;
                                    Some(before)
                                }).take_while(|b| *b < n).count());
                            }
                        }
                    }
                    take(&mut m, &got, S::Free, S::Used(next))?;
                    owned.push((next, got));
                    next += 1;
                }
                Op::Release(pick) => {
                    if owned.is_empty() {
                        continue;
                    }
                    let (id, rs) = owned.remove(pick as usize % owned.len());
                    take(&mut m, &rs, S::Used(id), S::Free)?;
                    a.release(&rs);
                }
                Op::Park(n) => {
                    let want = (n as u64).min(a.free()) as u32;
                    let largest = runs_of(&m, |s| s == S::Free).iter().map(|r| r.len).max().unwrap_or(0);
                    let got = a.park(n);
                    prop_assert_eq!(got.iter().map(|r| r.len).sum::<u32>(), want);
                    if want > 0 {
                        prop_assert!(got.iter().any(|r| r.len == largest.min(want)), "largest run first");
                    }
                    prop_assert!(got.windows(2).all(|w| w[0].start < w[1].start));
                    take(&mut m, &got, S::Free, S::Parked)?;
                }
                Op::Unpark(i, off, len) => {
                    let parked = runs_of(&m, |s| s == S::Parked);
                    // a sub-range of a parked run, or (when off/len overshoot) one that is not wholly parked
                    let r = match parked.get(i as usize % parked.len().max(1)) {
                        Some(p) => Range { start: p.start + off as u32 % (p.len + 1), len: 1 + len as u32 % (p.len + 1) },
                        None => Range { start: off as u32 % N, len: 1 },
                    };
                    let all_parked = r.start + r.len <= N && (r.start..r.start + r.len).all(|x| m[x as usize] == S::Parked);
                    a.unpark(&[r]);
                    if all_parked {
                        take(&mut m, &[r], S::Parked, S::Free)?;
                    }
                }
                Op::Below(n, below) => {
                    let free_below: Vec<u32> = (0..below).filter(|&x| m[x as usize] == S::Free).collect();
                    match a.alloc_below(n, below) {
                        None => prop_assert!(free_below.len() < n as usize, "refused with enough free below"),
                        Some(got) => {
                            let want: Vec<u32> = free_below.iter().copied().take(n as usize).collect();
                            let have: Vec<u32> = got.iter().flat_map(|r| r.start..r.start + r.len).collect();
                            prop_assert_eq!(have, want, "the lowest free slots below {}", below);
                            take(&mut m, &got, S::Free, S::Used(next))?;
                            owned.push((next, got));
                            next += 1;
                        }
                    }
                }
            }
            prop_assert_eq!(a.free_runs(), runs_of(&m, |s| s == S::Free), "free runs (merged, exact)");
            prop_assert_eq!(a.parked_runs(), runs_of(&m, |s| s == S::Parked), "parked runs (merged, exact)");
            prop_assert_eq!(a.free(), m.iter().filter(|s| **s == S::Free).count() as u64);
            prop_assert_eq!(a.parked(), m.iter().filter(|s| **s == S::Parked).count() as u64);
            prop_assert_eq!(a.runs(), a.free_runs().len());
            prop_assert_eq!(a.total(), N);
        }
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 1000, ..ProptestConfig::default() })]
        #[test]
        fn alloc_model(ops in proptest::collection::vec(op(), 1..120)) {
            run(ops)?;
        }
    }
}
