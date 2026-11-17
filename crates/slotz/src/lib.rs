//! In-place slot compression: a committed entry's slot rewritten as
//! `LKVZ1` header + chunk table + compressed chunks (4-byte planes + zstd 1 per chunk), the rest of the slot punched.
//! The slot keeps its address; readers (the Arc connector's staging-ring worker, the broker's SSD writer) decompress
//! the chunks they need. Lossless; every chunk carries a checksum of its uncompressed bytes.
//!
//! Layout: magic[8] | chunk u32 | n u32 | raw_len u64 | n x (stored_len u32 (bit 31: raw), checksum u64) | chunks.

pub const MAGIC: &[u8; 8] = b"LKVZ1\0\0\0";
const HEAD: usize = 24;
const ENT: usize = 12;

pub fn checksum(d: &[u8]) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for c in d.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0xff51_afd7_ed55_8ccd).rotate_left(29);
    }
    h
}

fn planes4(d: &[u8]) -> Vec<u8> {
    let n = d.len() / 4;
    let mut out = vec![0u8; d.len()];
    for i in 0..n {
        for b in 0..4 {
            out[b * n + i] = d[i * 4 + b];
        }
    }
    out[n * 4..].copy_from_slice(&d[n * 4..]);
    out
}

fn unplanes4(d: &[u8], out: &mut [u8]) {
    let n = d.len() / 4;
    for i in 0..n {
        for b in 0..4 {
            out[i * 4 + b] = d[b * n + i];
        }
    }
    out[n * 4..].copy_from_slice(&d[n * 4..]);
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16)
}

/// A compressed slot image of a version this build does not know (`LKVZ<digit>\0\0\0`, not LKVZ1): refused, never
/// read as raw KV (docs/protocol.md "Versioning"; a future codec is written only for clients that declared it in
/// FEATURES, and this keeps a reader that slipped through from copying an image as KV).
pub fn unknown_format(slot: &[u8]) -> bool {
    slot.len() >= 8 && &slot[..4] == b"LKVZ" && slot[4].is_ascii_digit() && slot[5..8] == [0, 0, 0] && &slot[..8] != MAGIC
}

pub fn is_compressed(slot: &[u8]) -> bool {
    slot.len() >= HEAD && &slot[..8] == MAGIC
}

/// Compressed image of `raw` (chunks compressed in parallel), or None if it saves less than `min_saving` (0..1).
pub fn compress(raw: &[u8], chunk: usize, min_saving: f64) -> Option<Vec<u8>> {
    let chunk = chunk.max(4096);
    let chunks: Vec<&[u8]> = raw.chunks(chunk).collect();
    let n = chunks.len();
    let mut out: Vec<Vec<u8>> = vec![Vec::new(); n];
    let per = n.div_ceil(threads()).max(1);
    std::thread::scope(|s| {
        for (ti, part) in out.chunks_mut(per).enumerate() {
            let ch = &chunks;
            s.spawn(move || {
                for (k, o) in part.iter_mut().enumerate() {
                    let c = ch[ti * per + k];
                    let z = zstd::bulk::compress(&planes4(c), 1).unwrap_or_default();
                    *o = if !z.is_empty() && z.len() < c.len() { z } else { Vec::new() };
                }
            });
        }
    });
    let mut v = Vec::with_capacity(raw.len());
    v.extend_from_slice(MAGIC);
    v.extend_from_slice(&(chunk as u32).to_le_bytes());
    v.extend_from_slice(&(n as u32).to_le_bytes());
    v.extend_from_slice(&(raw.len() as u64).to_le_bytes());
    for (z, c) in out.iter().zip(&chunks) {
        let l = if z.is_empty() { c.len() as u32 | 1 << 31 } else { z.len() as u32 };
        v.extend_from_slice(&l.to_le_bytes());
        v.extend_from_slice(&checksum(c).to_le_bytes());
    }
    for (z, c) in out.iter().zip(&chunks) {
        v.extend_from_slice(if z.is_empty() { c } else { z });
    }
    ((v.len() as f64) <= raw.len() as f64 * (1.0 - min_saving)).then_some(v)
}

/// Uncompressed length of a compressed slot.
pub fn raw_len(slot: &[u8]) -> Option<usize> {
    is_compressed(slot).then(|| u64::from_le_bytes(slot[16..24].try_into().unwrap()) as usize)
}

/// Bytes of a compressed slot's image (header, chunk table, chunks), from at least its header and chunk table.
pub fn image_len(slot: &[u8]) -> Option<usize> {
    if !is_compressed(slot) {
        return None;
    }
    let n = u32::from_le_bytes(slot[12..16].try_into().unwrap()) as usize;
    let table = slot.get(HEAD..HEAD + n * ENT)?;
    Some(HEAD + n * ENT + table.chunks(ENT).map(|e| (u32::from_le_bytes(e[..4].try_into().unwrap()) & !(1 << 31)) as usize).sum::<usize>())
}

/// Copy uncompressed bytes [off, off + out.len()) of a compressed slot into `out` (decompressing the chunks it
/// covers; checksums verified). False on damage / out of range.
pub fn read_range(slot: &[u8], off: usize, out: &mut [u8]) -> bool {
    if !is_compressed(slot) {
        return false;
    }
    let chunk = u32::from_le_bytes(slot[8..12].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(slot[12..16].try_into().unwrap()) as usize;
    let raw = u64::from_le_bytes(slot[16..24].try_into().unwrap()) as usize;
    if chunk == 0 || off + out.len() > raw || HEAD + n * ENT > slot.len() {
        return false;
    }
    let mut starts = Vec::with_capacity(n);
    let mut at = HEAD + n * ENT;
    for i in 0..n {
        let e = &slot[HEAD + i * ENT..HEAD + (i + 1) * ENT];
        let l = u32::from_le_bytes(e[..4].try_into().unwrap());
        let sum = u64::from_le_bytes(e[4..12].try_into().unwrap());
        starts.push((at, l, sum));
        at += (l & !(1 << 31)) as usize;
    }
    let (first, last) = (off / chunk, (off + out.len()).div_ceil(chunk));
    let mut pos = 0usize;
    for c in first..last.min(n) {
        let (a, l, sum) = starts[c];
        let len = (l & !(1 << 31)) as usize;
        let clen = chunk.min(raw - c * chunk);
        let Some(src) = slot.get(a..a + len) else { return false };
        let mut buf = vec![0u8; clen];
        if l & 1 << 31 != 0 {
            if len != clen {
                return false;
            }
            buf.copy_from_slice(src);
        } else {
            match zstd::bulk::decompress(src, clen) {
                Ok(d) if d.len() == clen => unplanes4(&d, &mut buf),
                _ => return false,
            }
        }
        if checksum(&buf) != sum {
            return false;
        }
        let c0 = c * chunk;
        let from = off.max(c0) - c0;
        let to = (off + out.len()).min(c0 + clen) - c0;
        out[pos..pos + (to - from)].copy_from_slice(&buf[from..to]);
        pos += to - from;
    }
    pos == out.len()
}

/// XXH3-64 of the raw bytes of a compressed slot image (decoded 1 MiB at a time, the chunk checksums verified); None
/// if it does not decode or its raw length is not `bytes` (integrity checksums).
pub fn image_xxh3(img: &[u8], bytes: u64) -> Option<u64> {
    let raw = raw_len(img)?;
    if raw as u64 != bytes {
        return None;
    }
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    let mut buf = vec![0u8; raw.min(1 << 20)];
    let mut off = 0;
    while off < raw {
        let n = (raw - off).min(buf.len());
        if !read_range(img, off, &mut buf[..n]) {
            return None;
        }
        h.update(&buf[..n]);
        off += n;
    }
    Some(h.digest())
}

/// Copy ops (src, dst, size) between raw memory: src inside a pool of `slot` sized slots starting at `base`
/// (`len` bytes); compressed slots are decompressed, raw ones memcpy'd. Ops are cut at chunk boundaries (compressed)
/// or into 1 MiB pieces (raw) and the pieces run in parallel. Returns false if any piece failed (damaged slot).
///
/// # Safety
/// The caller guarantees src / dst ranges are mapped, dst ranges don't overlap each other or the pool slots read.
pub unsafe fn copy_ops(base: u64, len: u64, slot: u64, ops: &[(u64, u64, u64)]) -> bool {
    // pieces: (slot start or 0 for raw, src, dst, size)
    let mut pieces: Vec<(u64, u64, u64, u64)> = Vec::new();
    for &(src, dst, size) in ops {
        let in_pool = slot > 0 && src >= base && src + size <= base + len;
        let s0 = if in_pool { base + (src - base) / slot * slot } else { 0 };
        let sl = if in_pool { std::slice::from_raw_parts(s0 as *const u8, slot.min(base + len - s0) as usize) } else { &[][..] };
        if in_pool && unknown_format(sl) {
            return false;                                   // an image this build cannot decode: never copied as KV
        }
        let step = if in_pool && is_compressed(sl) {
            u32::from_le_bytes(sl[8..12].try_into().unwrap()) as u64
        } else {
            1 << 20
        };
        let comp = in_pool && is_compressed(sl);
        let mut o = 0u64;
        while o < size {
            let at = src + o;
            let n = if comp { (step - (at - s0) % step).min(size - o) } else { step.min(size - o) };
            pieces.push((if comp { s0 } else { 0 }, at, dst + o, n));
            o += n;
        }
    }
    let ok = std::sync::atomic::AtomicBool::new(true);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let nt = threads().min(pieces.len()).max(1);
    std::thread::scope(|s| {
        for _ in 0..nt {
            let (ok, next, pieces) = (&ok, &next, &pieces);
            s.spawn(move || loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(s0, src, dst, n)) = pieces.get(i) else { break };
                if s0 == 0 {
                    std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n as usize);
                } else {
                    let sl = std::slice::from_raw_parts(s0 as *const u8, slot.min(base + len - s0) as usize);
                    let out = std::slice::from_raw_parts_mut(dst as *mut u8, n as usize);
                    if !read_range(sl, (src - s0) as usize, out) {
                        ok.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    });
    ok.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ranges() {
        let raw: Vec<u8> = (0..(3 * 1000 * 4 + 777)).map(|i| ((i / 7) % 13) as u8).collect();
        let z = compress(&raw, 4000 * 4 / 4 * 1, 0.05).expect("compressible");
        assert!(is_compressed(&z) && raw_len(&z) == Some(raw.len()));
        for (off, n) in [(0, raw.len()), (5, 100), (3999, 2), (4000, 4000), (8000, raw.len() - 8000), (11000, 1777)] {
            let mut out = vec![0u8; n];
            assert!(read_range(&z, off, &mut out), "range {off}+{n}");
            assert_eq!(out, raw[off..off + n]);
        }
        let mut bad = z.clone();
        let l = bad.len();
        bad[l - 1] ^= 0xff;
        let mut out = vec![0u8; raw.len()];
        assert!(!read_range(&bad, 0, &mut out));
        let mut x: u64 = 0x1234_5678_9abc_def1;
        let noise: Vec<u8> = (0..20000).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 32) as u8 }).collect();
        assert!(compress(&noise, 4096, 0.05).is_none());
    }

    #[test]
    fn copy_ops_mixed_pool() {
        // pool of 3 slots of 64 KiB: slot 0 compressed, slot 1 raw, slot 2 compressed; ops across chunk boundaries
        let slot = 65536usize;
        let raws: Vec<Vec<u8>> = (0..3).map(|k| (0..slot).map(|i| ((i / 4 + k * 3) % 29) as u8 * (i % 4 != 3) as u8).collect()).collect();
        let mut pool = vec![0u8; 3 * slot];
        for k in 0..3 {
            if k == 1 {
                pool[k * slot..(k + 1) * slot].copy_from_slice(&raws[k]);
            } else {
                let z = compress(&raws[k], 8192, 0.05).unwrap();
                pool[k * slot..k * slot + z.len()].copy_from_slice(&z);
            }
        }
        let base = pool.as_ptr() as u64;
        let mut out = vec![0u8; 3 * 20000];
        let ob = out.as_mut_ptr() as u64;
        let ops = [(base + 100, ob, 20000), (base + slot as u64 + 7, ob + 20000, 20000), (base + 2 * slot as u64 + 40000, ob + 40000, 20000)];
        assert!(unsafe { copy_ops(base, pool.len() as u64, slot as u64, &ops) });
        assert_eq!(&out[..20000], &raws[0][100..20100]);
        assert_eq!(&out[20000..40000], &raws[1][7..20007]);
        assert_eq!(&out[40000..], &raws[2][40000..60000]);
    }

    /// Noise (incompressible) and runs (compressible), 4 KiB each, alternating.
    fn mixed(n_chunks: usize) -> Vec<u8> {
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        (0..n_chunks * 4096).map(|i| {
            if (i / 4096) % 2 == 0 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            } else {
                (i / 512) as u8
            }
        }).collect()
    }

    #[test]
    fn raw_chunks_roundtrip_and_flagged() {
        // (found with cargo mutants): incompressible chunks are stored raw (bit 31 of the table entry)
        let raw = mixed(6);
        let z = compress(&raw, 4096, 0.0).expect("half of it compresses");
        for i in 0..6 {
            let l = u32::from_le_bytes(z[HEAD + i * ENT..HEAD + i * ENT + 4].try_into().unwrap());
            assert_eq!(l & 1 << 31 != 0, i % 2 == 0, "chunk {i}: raw flag");
            if i % 2 == 0 {
                assert_eq!(l & !(1 << 31), 4096);
            }
        }
        for (off, n) in [(0, raw.len()), (0, 4096), (4090, 12), (8192, 4096), (12_000, 9000), (raw.len() - 1, 1)] {
            let mut out = vec![0u8; n];
            assert!(read_range(&z, off, &mut out), "range {off}+{n}");
            assert_eq!(out, raw[off..off + n]);
        }
        // a raw chunk whose stored length is wrong is damage, not data
        let mut bad = z.clone();
        bad[HEAD..HEAD + 4].copy_from_slice(&(4095u32 | 1 << 31).to_le_bytes());
        let mut out = vec![0u8; 4096];
        assert!(!read_range(&bad, 0, &mut out));
    }

    #[test]
    fn bounds_and_threshold() {
        let raw: Vec<u8> = (0..20_000).map(|i| (i / 100) as u8).collect();
        let z = compress(&raw, 4096, 0.0).unwrap();
        let mut out = vec![0u8; 10];
        assert!(!read_range(&z, raw.len() - 9, &mut out), "past the end");
        assert!(read_range(&z, raw.len() - 10, &mut out));
        let mut none = [];
        assert!(read_range(&z, raw.len(), &mut none));
        // a slot cut inside its chunk table: refused, no panic
        let n = 5;
        for cut in [HEAD + n * ENT - 1, HEAD + 2 * ENT, HEAD] {
            assert!(!read_range(&z[..cut], 0, &mut out), "cut at {cut}");
        }
        assert!(!read_range(&raw, 0, &mut out), "not compressed");
        // min_saving: ~40% saving data passes 0.3, fails 0.5
        let half = mixed(10);
        let zl = compress(&half, 4096, 0.0).unwrap().len() as f64 / half.len() as f64;
        assert!(zl > 0.5 && zl < 0.7, "ratio {zl}");
        assert!(compress(&half, 4096, 0.3).is_some());
        assert!(compress(&half, 4096, 0.5).is_none());
    }

    #[test]
    fn short_zstd_frame_is_damage() {
        // a chunk whose zstd stream decodes to fewer bytes than the chunk: refused (not a panic, not short data)
        let short = vec![7u8; 4000];
        let z = zstd::bulk::compress(&planes4(&short), 1).unwrap();
        let mut slot = MAGIC.to_vec();
        slot.extend_from_slice(&4096u32.to_le_bytes());
        slot.extend_from_slice(&1u32.to_le_bytes());
        slot.extend_from_slice(&4096u64.to_le_bytes());
        slot.extend_from_slice(&(z.len() as u32).to_le_bytes());
        slot.extend_from_slice(&checksum(&[7u8; 4096]).to_le_bytes());
        slot.extend_from_slice(&z);
        let mut out = vec![0u8; 4096];
        assert!(!read_range(&slot, 0, &mut out));
    }

    #[test]
    fn unknown_codec_version_is_refused_not_copied_raw() {
        // a slot whose image is "LKVZ2" (a future codec): copy_ops fails instead of copying the image as KV
        let slot = 64usize << 10;
        let mut pool = vec![7u8; slot * 2];
        pool[slot..slot + 8].copy_from_slice(b"LKVZ2\0\0\0");
        let mut out = vec![0u8; 4096];
        let base = pool.as_ptr() as u64;
        let ok = unsafe { copy_ops(base, pool.len() as u64, slot as u64, &[(base + slot as u64, out.as_mut_ptr() as u64, 4096)]) };
        assert!(!ok, "an unknown image version must not be copied");
        assert!(out.iter().all(|x| *x == 0), "nothing copied");
        // raw slots (also ones that merely start with "LKVZ") and LKVZ1 images still copy
        pool[slot..slot + 8].copy_from_slice(b"LKVZ2abc");
        let ok = unsafe { copy_ops(base, pool.len() as u64, slot as u64, &[(base + slot as u64, out.as_mut_ptr() as u64, 4096)]) };
        assert!(ok && out[..8] == *b"LKVZ2abc");
        let raw: Vec<u8> = (0..slot).map(|i| (i % 7) as u8).collect();
        let z = compress(&raw, 16 << 10, 0.0).unwrap();
        pool[..z.len()].copy_from_slice(&z);
        let ok = unsafe { copy_ops(base, pool.len() as u64, slot as u64, &[(base, out.as_mut_ptr() as u64, 4096)]) };
        assert!(ok && out[..] == raw[..4096]);
    }
}
