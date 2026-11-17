//! NGZ: lossless compressed bf16 n-gram rows.
//!
//! A bf16 row of 160 values (320 B) becomes a fixed 232 B slot (72.5%), so row r stays at a fixed offset (random
//! access, bins, broker cache all unchanged):
//!   byte 0         the row's largest exponent E (bf16 exponent field; 0 = escaped row, see below)
//!   bytes 1..161   per value: sign << 7 | 7-bit mantissa
//!   bytes 161..232 the exponent offsets E - e_i, Huffman coded (canonical, LSB-first bit order), 568 bits
//! Offsets 0..ESC-1 have codes; larger offsets are ESC followed by 6 raw bits. The values are Gaussian-like with one
//! scale per row, so the offsets are small (entropy ~2.56 bits; 2.6 coded): ~415 of the 568 bits on average.
//! Rows that do not fit (none in a 1.6 M-row sample) are stored raw in an escape section at the end (sorted
//! (u32 row, 320 B)), and their slot has E = 0 (no real row has exponent 0: the table has no zeros or subnormals).
//!
//! File: 4096 B header (magic "NGZ1", rows, slot bytes, dim, code lengths, escape count / offset), then the slots at
//! 4096 + r * 232, then the escape section. The engine decodes slots straight to the fp16 embedding x 2^18
//! (engine/ple_rows.h): fp16 = sign << 15 | (e - 94) << 10 | m << 3 for e >= 95, else 0; bit-identical to
//! converting the bf16 row x 2^18 with the host flush.

pub const MAGIC: &[u8; 4] = b"NGZ1";
pub const DIM: usize = 160;
pub const SLOT: usize = 232;
pub const HEADER: u64 = 4096;
pub const STREAM_BYTES: usize = SLOT - 1 - DIM; // 71
pub const NSYM: usize = 16; // offsets 0..14 + ESC
pub const ESC: usize = NSYM - 1;
pub const ESC_BITS: u32 = 6;
pub const MAX_LEN: u32 = 12;

#[derive(Clone)]
pub struct Code {
    pub len: [u8; NSYM],
    /// LSB-first (bit-reversed canonical) codes
    pub code: [u32; NSYM],
    /// decode table over MAX_LEN bits: (symbol, length)
    pub table: Vec<(u8, u8)>,
}

fn reverse(mut c: u32, l: u32) -> u32 {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (c & 1);
        c >>= 1;
    }
    r
}

impl Code {
    /// Huffman lengths from symbol counts, limited to MAX_LEN (counts of rare symbols are raised until it holds).
    pub fn from_counts(counts: &[u64; NSYM]) -> Code {
        let mut c: Vec<u64> = counts.iter().map(|&x| x + 1).collect();
        loop {
            let len = huffman_lengths(&c);
            if len.iter().all(|&l| l as u32 <= MAX_LEN) {
                return Code::from_lengths(&len.try_into().unwrap());
            }
            let total: u64 = c.iter().sum();
            for x in c.iter_mut() {
                *x = (*x).max(total >> (MAX_LEN - 1));
            }
        }
    }

    pub fn from_lengths(len: &[u8; NSYM]) -> Code {
        // canonical: by (length, symbol)
        let mut order: Vec<usize> = (0..NSYM).collect();
        order.sort_by_key(|&s| (len[s], s));
        let mut code = [0u32; NSYM];
        let (mut c, mut prev) = (0u32, 0u8);
        for (i, &s) in order.iter().enumerate() {
            if i > 0 {
                c = (c + 1) << (len[s] - prev);
            } else {
                c = 0;
            }
            prev = len[s];
            code[s] = reverse(c, len[s] as u32);
        }
        let mut table = vec![(0u8, 0u8); 1 << MAX_LEN];
        for s in 0..NSYM {
            let l = len[s] as u32;
            let step = 1u32 << l;
            let mut t = code[s];
            while t < (1 << MAX_LEN) {
                table[t as usize] = (s as u8, l as u8);
                t += step;
            }
        }
        Code { len: *len, code, table }
    }
}

fn huffman_lengths(c: &[u64]) -> Vec<u8> {
    // simple O(n^2) Huffman over a handful of symbols
    let n = c.len();
    let mut nodes: Vec<(u64, Vec<usize>)> = (0..n).map(|i| (c[i], vec![i])).collect();
    let mut len = vec![0u8; n];
    while nodes.len() > 1 {
        nodes.sort_by(|a, b| b.0.cmp(&a.0));
        let a = nodes.pop().unwrap();
        let b = nodes.pop().unwrap();
        for &s in a.1.iter().chain(b.1.iter()) {
            len[s] += 1;
        }
        let mut m = a.1;
        m.extend(b.1);
        nodes.push((a.0 + b.0, m));
    }
    len
}

#[inline]
fn exp(v: u16) -> u32 {
    ((v >> 7) & 0xFF) as u32
}

/// Offset symbol counts of rows (for the code).
pub fn count(rows: &[u16], counts: &mut [u64; NSYM]) {
    for r in rows.chunks_exact(DIM) {
        let e = r.iter().map(|&v| exp(v)).max().unwrap();
        for &v in r {
            let o = (e - exp(v)) as usize;
            counts[o.min(ESC)] += 1;
        }
    }
}

/// Encodes one row; None if it does not fit (escape) or has an exponent-0 value.
pub fn encode(code: &Code, row: &[u16], out: &mut [u8]) -> Option<()> {
    let e = row.iter().map(|&v| exp(v)).max().unwrap();
    if e == 0 || row.iter().any(|&v| exp(v) == 0) {
        return None;
    }
    out[0] = e as u8;
    for (i, &v) in row.iter().enumerate() {
        out[1 + i] = (((v >> 15) as u8) << 7) | (v & 0x7F) as u8;
    }
    let s = &mut out[1 + DIM..SLOT];
    s.fill(0);
    let mut bit = 0usize;
    let mut put = |val: u32, n: u32, s: &mut [u8]| -> bool {
        for k in 0..n {
            if bit >= STREAM_BYTES * 8 {
                return false;
            }
            if (val >> k) & 1 == 1 {
                s[bit >> 3] |= 1 << (bit & 7);
            }
            bit += 1;
        }
        true
    };
    for &v in row {
        let o = (e - exp(v)) as usize;
        let ok = if o < ESC {
            put(code.code[o], code.len[o] as u32, s)
        } else {
            o < 64 && put(code.code[ESC], code.len[ESC] as u32, s) && put(o as u32, ESC_BITS, s)
        };
        if !ok {
            return None;
        }
    }
    Some(())
}

/// Decodes a slot back to bf16 (for verification); false if the slot is an escape marker.
pub fn decode(code: &Code, slot: &[u8], row: &mut [u16]) -> bool {
    let e = slot[0] as u32;
    if e == 0 {
        return false;
    }
    let s = &slot[1 + DIM..SLOT];
    let mut bit = 0usize;
    let get = |bit: usize, n: u32| -> u32 {
        let mut v = 0u32;
        for k in 0..n as usize {
            let b = bit + k;
            if b < STREAM_BYTES * 8 && (s[b >> 3] >> (b & 7)) & 1 == 1 {
                v |= 1 << k;
            }
        }
        v
    };
    for i in 0..DIM {
        let (sym, l) = code.table[get(bit, MAX_LEN) as usize];
        bit += l as usize;
        let o = if sym as usize == ESC {
            let o = get(bit, ESC_BITS);
            bit += ESC_BITS as usize;
            o
        } else {
            sym as u32
        };
        let sm = slot[1 + i] as u16;
        row[i] = ((sm >> 7) << 15) | (((e - o) as u16) << 7) | (sm & 0x7F);
    }
    true
}

pub struct Header {
    pub rows: u64,
    pub len: [u8; NSYM],
    pub n_escape: u64,
    pub escape_offset: u64,
}

impl Header {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = vec![0u8; HEADER as usize];
        h[0..4].copy_from_slice(MAGIC);
        h[4..8].copy_from_slice(&1u32.to_le_bytes());
        h[8..16].copy_from_slice(&self.rows.to_le_bytes());
        h[16..20].copy_from_slice(&(SLOT as u32).to_le_bytes());
        h[20..24].copy_from_slice(&(DIM as u32).to_le_bytes());
        h[24..28].copy_from_slice(&(NSYM as u32).to_le_bytes());
        h[28..32].copy_from_slice(&MAX_LEN.to_le_bytes());
        h[32..40].copy_from_slice(&self.n_escape.to_le_bytes());
        h[40..48].copy_from_slice(&self.escape_offset.to_le_bytes());
        h[48..48 + NSYM].copy_from_slice(&self.len);
        h
    }
    pub fn parse(h: &[u8]) -> Option<Header> {
        if &h[0..4] != MAGIC || u32::from_le_bytes(h[16..20].try_into().ok()?) as usize != SLOT {
            return None;
        }
        let mut len = [0u8; NSYM];
        len.copy_from_slice(&h[48..48 + NSYM]);
        Some(Header {
            rows: u64::from_le_bytes(h[8..16].try_into().ok()?),
            len,
            n_escape: u64::from_le_bytes(h[32..40].try_into().ok()?),
            escape_offset: u64::from_le_bytes(h[40..48].try_into().ok()?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let mut seed = 1u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut rows = vec![0u16; DIM * 1000];
        for r in rows.chunks_exact_mut(DIM) {
            let base = 110 + (rnd() % 10) as u16;
            for v in r.iter_mut() {
                let x = rnd();
                let o = (x % 7) as u16 + if x % 1000 == 0 { 30 } else { 0 };
                *v = (((x >> 20) & 1) as u16) << 15 | (base - o) << 7 | ((x >> 30) & 0x7F) as u16;
            }
        }
        let mut c = [0u64; NSYM];
        count(&rows, &mut c);
        let code = Code::from_counts(&c);
        let mut slot = [0u8; SLOT];
        let mut back = [0u16; DIM];
        for r in rows.chunks_exact(DIM) {
            if encode(&code, r, &mut slot).is_some() {
                assert!(decode(&code, &slot, &mut back));
                assert_eq!(&back[..], r);
            }
        }
    }
}
