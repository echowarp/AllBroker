//! Exact port of gufo's `HashNgramRows` (src/models/qwen38_flash_next/ngram.cpp): the table rows every token
//! gathers, `heads` per token, head-major. An EOS anywhere in the window cuts every older token; a missing
//! predecessor also reads as EOS. The token's own EOS does not cut its context.

pub const MAX_NGRAM: usize = 3;
pub const MAX_HEADS: usize = 16;

#[derive(Debug, Clone)]
pub struct PleParams {
    pub ngram_size: u32,         // 3: bigrams and trigrams
    pub heads_per_ngram: u32,    // 8
    pub heads: u32,              // (ngram_size - 1) * heads_per_ngram
    pub head_dim: u32,           // 160
    pub eos: i64,
    pub multipliers: [u64; MAX_NGRAM],
    pub head_offsets: [u32; MAX_HEADS],
    pub head_vocab: [u32; MAX_HEADS],
    pub rows: u64,
}

impl PleParams {
    /// Reads the `qwen4exp.ple.*` keys the way gufo's config.cpp does.
    pub fn from_gguf(g: &crate::gguf::Gguf) -> Result<PleParams, String> {
        let key = |k: &str| format!("qwen4exp.{k}");
        let u = |k: &str| g.kv.get(&key(k)).and_then(|v| v.as_u64()).ok_or(format!("missing {k}"));
        let a = |k: &str| g.kv.get(&key(k)).and_then(|v| v.as_u64_array()).ok_or(format!("missing {k}"));
        let ngram_size = u("ple.ngram_size")? as u32;
        let heads_per_ngram = u("ple.heads_per_ngram")? as u32;
        let heads = (ngram_size - 1) * heads_per_ngram;
        let mults = a("ple.layer_multipliers")?;
        let offs = a("ple.head_offsets")?;
        let voc = a("ple.head_vocab_sizes")?;
        if ngram_size < 2 || ngram_size as usize > MAX_NGRAM || heads as usize > MAX_HEADS
            || mults.len() != ngram_size as usize || offs.len() != heads as usize || voc.len() != heads as usize
        {
            return Err("unexpected PLE shape".into());
        }
        let mut p = PleParams {
            ngram_size,
            heads_per_ngram,
            heads,
            head_dim: u("embedding_length_per_layer_input")? as u32,
            eos: u("ple.eos_token_id")? as i64,
            multipliers: [0; MAX_NGRAM],
            head_offsets: [0; MAX_HEADS],
            head_vocab: [0; MAX_HEADS],
            rows: 0,
        };
        p.multipliers[..mults.len()].copy_from_slice(&mults);
        for h in 0..heads as usize {
            p.head_offsets[h] = offs[h] as u32;
            p.head_vocab[h] = voc[h] as u32;
            p.rows = p.rows.max(offs[h] + voc[h]);
        }
        Ok(p)
    }
}

/// Newest-first tail of a sequence's previous tokens: the complete hash state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct History {
    pub prev: [i32; MAX_NGRAM - 1],
}

impl Default for History {
    fn default() -> Self {
        History { prev: [-1; MAX_NGRAM - 1] }
    }
}

/// Bits per token in a packed n-gram key (the vocabulary, 248,320 ids, fits in 18 bits).
pub const KEY_BITS: u32 = 18;
const KEY_MASK: u64 = (1 << KEY_BITS) - 1;

/// Packs a hash context of `order` tokens (ctx[0] = the token, ctx[1..] = predecessors after the EOS rules) into a
/// key: ctx[0] in the low bits. Bigram keys use 36 bits, trigram keys 54. The key determines the n-gram's 8 rows.
pub fn pack_key(ctx: &[u64]) -> u64 {
    ctx.iter().rev().fold(0u64, |k, &t| (k << KEY_BITS) | (t & KEY_MASK))
}

pub fn unpack_key(key: u64, order: usize) -> [u64; MAX_NGRAM] {
    let mut c = [0u64; MAX_NGRAM];
    for (s, v) in c.iter_mut().enumerate().take(order) {
        *v = (key >> (KEY_BITS * s as u32)) & KEY_MASK;
    }
    c
}

/// Bigram and trigram keys of every token (what a bundle is keyed by); advances `history` like `hash_rows`.
pub fn ngram_keys(p: &PleParams, history: &mut History, tokens: &[i32], bi: &mut Vec<u64>, tri: &mut Vec<u64>) {
    let n = p.ngram_size as usize;
    let mut ctx = [0u64; MAX_NGRAM];
    for &tok in tokens {
        ctx[0] = tok as i64 as u64;
        let mut cut = false;
        for s in 1..n {
            let t = if cut { -1 } else { history.prev[s - 1] };
            cut = cut || t < 0 || t as i64 == p.eos;
            ctx[s] = if cut { p.eos as u64 } else { t as i64 as u64 };
        }
        bi.push(pack_key(&ctx[..2]));
        if n >= 3 {
            tri.push(pack_key(&ctx[..3]));
        }
        for s in (1..history.prev.len()).rev() {
            history.prev[s] = history.prev[s - 1];
        }
        history.prev[0] = tok;
    }
}

/// The `heads_per_ngram` rows of one n-gram key of `order` (2 = bigram heads 0..8, 3 = trigram heads 8..16).
pub fn rows_of_key(p: &PleParams, key: u64, order: usize, out: &mut [u32]) {
    let ctx = unpack_key(key, order);
    let mut mixed = ctx[0].wrapping_mul(p.multipliers[0]);
    for j in 1..order {
        mixed ^= ctx[j].wrapping_mul(p.multipliers[j]);
    }
    let base = (order - 2) * p.heads_per_ngram as usize;
    for g in 0..p.heads_per_ngram as usize {
        let h = base + g;
        out[g] = (mixed % p.head_vocab[h] as u64) as u32 + p.head_offsets[h];
    }
}

/// Rows for `tokens` (len * heads entries into `rows`); advances `history` past `tokens`.
pub fn hash_rows(p: &PleParams, history: &mut History, tokens: &[i32], rows: &mut [u32]) {
    let n = p.ngram_size as usize;
    let heads = p.heads as usize;
    assert!(rows.len() >= tokens.len() * heads);
    let mut ctx = [0u64; MAX_NGRAM];
    for (i, &tok) in tokens.iter().enumerate() {
        ctx[0] = tok as i64 as u64;
        let mut cut = false;
        for s in 1..n {
            let t = if cut { -1 } else { history.prev[s - 1] };
            cut = cut || t < 0 || t as i64 == p.eos;
            ctx[s] = if cut { p.eos as u64 } else { t as i64 as u64 };
        }
        let out = &mut rows[i * heads..(i + 1) * heads];
        for order in 2..=n {
            let mut mixed = ctx[0].wrapping_mul(p.multipliers[0]);
            for j in 1..order {
                mixed ^= ctx[j].wrapping_mul(p.multipliers[j]);
            }
            let base = (order - 2) * p.heads_per_ngram as usize;
            for g in 0..p.heads_per_ngram as usize {
                let h = base + g;
                out[h] = (mixed % p.head_vocab[h] as u64) as u32 + p.head_offsets[h];
            }
        }
        for s in (1..history.prev.len()).rev() {
            history.prev[s] = history.prev[s - 1];
        }
        history.prev[0] = tok;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> PleParams {
        PleParams {
            ngram_size: 3,
            heads_per_ngram: 8,
            heads: 16,
            head_dim: 160,
            eos: 248044,
            multipliers: [23703573157769, 20109073645365, 8052911324071],
            head_offsets: [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
                           180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275],
            head_vocab: [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
                         20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171],
            rows: 320001446,
        }
    }

    /// Bundles must be exact: the rows recomputed from each token's keys equal hash_rows, across EOS cuts.
    #[test]
    fn keys_reproduce_rows() {
        let p = params();
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut toks = Vec::new();
        for _ in 0..5000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            toks.push(match seed % 10 { 0 => 248044, 1 => ((seed >> 20) % 300) as i32, _ => ((seed >> 20) % 248320) as i32 });
        }
        let mut rows = vec![0u32; toks.len() * 16];
        hash_rows(&p, &mut History::default(), &toks, &mut rows);
        let (mut bi, mut tri) = (Vec::new(), Vec::new());
        ngram_keys(&p, &mut History::default(), &toks, &mut bi, &mut tri);
        let mut r = [0u32; 8];
        for i in 0..toks.len() {
            rows_of_key(&p, bi[i], 2, &mut r);
            assert_eq!(&r[..], &rows[i * 16..i * 16 + 8], "bigram rows at {i}");
            rows_of_key(&p, tri[i], 3, &mut r);
            assert_eq!(&r[..], &rows[i * 16 + 8..i * 16 + 16], "trigram rows at {i}");
        }
    }
}
