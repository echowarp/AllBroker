//! Per-request lookup index: the request's tokens and an open-addressing table 8-gram → most recent end position.
//! At each new context end `e`, the gram `tokens[e-8..e]` is looked up *before* it is inserted, so a hit `p` is an
//! earlier occurrence (`p < e`). The match is extended backwards (suffix-match length) and the hint is the
//! continuation `tokens[p..min(p + max_chain, e)]`: what followed the earlier occurrence of the current suffix.

pub const GRAM: usize = 8;

#[inline]
fn gram_hash(t: &[u32]) -> u64 {
    // 8 tokens → 64-bit hash (multiply-rotate mix; collisions are verified against the tokens)
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    for &x in t {
        h = (h ^ x as u64).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29);
    }
    h ^ (h >> 31)
}

pub struct RequestIndex {
    pub tokens: Vec<u32>,
    /// slot → end position + 1 (0 = empty)
    table: Vec<u32>,
    used: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Match {
    /// end of the earlier occurrence (the continuation starts here)
    pub pos: usize,
    /// suffix-match length (≥ GRAM)
    pub len: usize,
}

impl RequestIndex {
    pub fn new(tokens: Vec<u32>) -> RequestIndex {
        let cap = (tokens.len() * 2).next_power_of_two().max(1024);
        let mut ix = RequestIndex { tokens, table: vec![0; cap], used: 0 };
        for e in GRAM..=ix.tokens.len() {
            ix.insert(e);
        }
        ix
    }

    fn same_gram(&self, a: usize, b: usize) -> bool {
        self.tokens[a - GRAM..a] == self.tokens[b - GRAM..b]
    }

    /// Slot holding the gram ending at `e`, or the empty slot where it belongs.
    fn find(&self, e: usize) -> usize {
        let mask = self.table.len() - 1;
        let mut i = (gram_hash(&self.tokens[e - GRAM..e]) as usize) & mask;
        loop {
            let v = self.table[i];
            if v == 0 || self.same_gram(v as usize - 1, e) {
                return i;
            }
            i = (i + 1) & mask;
        }
    }

    fn insert(&mut self, e: usize) {
        if (self.used + 1) * 2 > self.table.len() {
            self.grow();
        }
        let i = self.find(e);
        if self.table[i] == 0 {
            self.used += 1;
        }
        self.table[i] = e as u32 + 1;
    }

    fn grow(&mut self) {
        let old = std::mem::take(&mut self.table);
        self.table = vec![0; old.len() * 2];
        self.used = 0;
        for v in old.into_iter().filter(|&v| v != 0) {
            let i = self.find(v as usize - 1);
            self.table[i] = v;
            self.used += 1;
        }
    }

    /// Earlier occurrence of the current suffix (before inserting the current end), with its match length
    /// (capped at `max_len`).
    pub fn lookup(&self, max_len: usize) -> Option<Match> {
        let e = self.tokens.len();
        if e < GRAM + 1 {
            return None;
        }
        let v = self.table[self.find(e)];
        if v == 0 {
            return None;
        }
        let p = v as usize - 1;
        if p >= e {
            return None;
        }
        let mut len = GRAM;
        while len < max_len && len < p && self.tokens[p - len - 1] == self.tokens[e - len - 1] {
            len += 1;
        }
        Some(Match { pos: p, len })
    }

    /// Append tokens; returns the match at the new end (looked up before the last gram is inserted).
    pub fn append(&mut self, new: &[u32], max_len: usize) -> Option<Match> {
        if new.is_empty() {
            return None;
        }
        let start = self.tokens.len();
        self.tokens.extend_from_slice(new);
        let end = self.tokens.len();
        // index every new end except the last one, look the last one up, then index it
        for e in (start + 1).max(GRAM)..end {
            self.insert(e);
        }
        let m = self.lookup(max_len);
        if end >= GRAM {
            self.insert(end);
        }
        m
    }

    /// The continuation after an earlier occurrence ending at `pos`.
    pub fn continuation(&self, pos: usize, max_chain: usize) -> &[u32] {
        let e = self.tokens.len();
        &self.tokens[pos..(pos + max_chain).min(e)]
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn mem_bytes(&self) -> usize {
        self.tokens.capacity() * 4 + self.table.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    #[test]
    fn planted_repeat_gives_aligned_hint() {
        let mut r = Lcg(1);
        let span: Vec<u32> = (0..40).map(|_| r.next() % 50_000).collect();
        let mut prompt: Vec<u32> = (0..1000).map(|_| r.next() % 50_000).collect();
        let at = prompt.len();
        prompt.extend_from_slice(&span); // the span once in the prompt
        prompt.extend((0..300).map(|_| r.next() % 50_000));
        let mut ix = RequestIndex::new(prompt);
        // the output starts repeating the span: after 20 tokens of it, the hint is the span's next 20 tokens
        let mut m = None;
        for t in &span[..20] {
            m = ix.append(&[*t], 64);
        }
        let m = m.expect("hint after 20 repeated tokens");
        assert_eq!(m.pos, at + 20);
        assert_eq!(m.len, 20);
        assert_eq!(ix.continuation(m.pos, 64)[..20], span[20..]);
    }

    #[test]
    fn no_hint_without_repeat_and_short_match_reported() {
        let mut r = Lcg(7);
        let mut ix = RequestIndex::new((0..5000).map(|_| r.next()).collect());
        for _ in 0..200 {
            assert!(ix.append(&[r.next()], 64).is_none());
        }
        // an 8-token repeat: match_len exactly 8
        let g: Vec<u32> = ix.tokens[100..108].to_vec();
        let m = ix.append(&g, 64).unwrap();
        assert_eq!((m.pos, m.len), (108, 8));
    }

    #[test]
    fn most_recent_occurrence_and_multi_token_append() {
        let s: Vec<u32> = (1..=12).collect();
        let mut t = vec![99, 98];
        t.extend(&s);
        t.extend([500, 501]);
        t.extend(&s);
        t.extend([700, 701]);
        let mut ix = RequestIndex::new(t);
        let m = ix.append(&s[..10], 64).unwrap(); // a third copy, appended in one call
        assert_eq!(m.pos, 2 + 12 + 2 + 10); // the second (most recent) copy
        assert_eq!(ix.continuation(m.pos, 4), &[11, 12, 700, 701]);
    }

    #[test]
    fn build_40k_tokens_fast() {
        let mut r = Lcg(3);
        let toks: Vec<u32> = (0..40_000).map(|_| r.next() % 150_000).collect();
        let t = std::time::Instant::now();
        let ix = RequestIndex::new(toks);
        let dt = t.elapsed();
        eprintln!("40k-token index build: {:.2} ms, {} KB", dt.as_secs_f64() * 1e3, ix.mem_bytes() / 1024);
        assert!(cfg!(debug_assertions) || dt.as_millis() < 5, "build took {dt:?}");
    }
}
