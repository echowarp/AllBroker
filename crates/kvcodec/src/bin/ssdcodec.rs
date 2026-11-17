//! ssdcodec: lossless compression of SSD-tier entry files (`*.kvd`, crates/broker/src/disk.rs) for the broker's SSD
//! tier: ratio and single / multi-thread GB/s per codec, per data set. Numbers only (run it on public-prompt entries).
//!
//!   ssdcodec [--gufo] [--chunk-mib 4] [--threads 16] FILE...
//!
//! Each file: the 4 KiB LKVD1 header is skipped; --gufo also skips gufo's continuation image header (identity +
//! tokens), so only the snapshot payload is measured. The data is cut into chunks (the SSD tier would compress per
//! chunk so restores decompress in parallel); variants per chunk: zstd on the raw bytes, on 2-byte planes (16-bit
//! K/V), on 4-byte planes (fp32 state), and "best": the smallest of those per chunk (one tag byte), plus lz4.
use std::time::Instant;

fn planes(d: &[u8], size: usize) -> Vec<u8> {
    let n = d.len() / size;
    let mut out = vec![0u8; d.len()];
    for i in 0..n {
        for b in 0..size {
            out[b * n + i] = d[i * size + b];
        }
    }
    out[n * size..].copy_from_slice(&d[n * size..]);
    out
}

fn unplanes(d: &[u8], size: usize) -> Vec<u8> {
    let n = d.len() / size;
    let mut out = vec![0u8; d.len()];
    for i in 0..n {
        for b in 0..size {
            out[i * size + b] = d[b * n + i];
        }
    }
    out[n * size..].copy_from_slice(&d[n * size..]);
    out
}

#[derive(Clone, Copy, PartialEq)]
enum V {
    Raw(i32),
    P2(i32),
    P4(i32),
    Best(i32),
    Lz4,
}

fn name(v: V) -> String {
    match v {
        V::Raw(l) => format!("zstd {l}"),
        V::P2(l) => format!("2-byte planes + zstd {l}"),
        V::P4(l) => format!("4-byte planes + zstd {l}"),
        V::Best(l) => format!("best per chunk (raw / 2 / 4 planes), zstd {l}"),
        V::Lz4 => "lz4".into(),
    }
}

fn enc(v: V, c: &[u8]) -> Vec<u8> {
    match v {
        V::Raw(l) => zstd::bulk::compress(c, l).unwrap(),
        V::P2(l) => zstd::bulk::compress(&planes(c, 2), l).unwrap(),
        V::P4(l) => zstd::bulk::compress(&planes(c, 4), l).unwrap(),
        V::Best(l) => {
            let cands = [(0u8, enc(V::Raw(l), c)), (2, enc(V::P2(l), c)), (4, enc(V::P4(l), c))];
            let (t, b) = cands.into_iter().min_by_key(|x| x.1.len()).unwrap();
            let mut o = vec![t];
            o.extend(b);
            o
        }
        V::Lz4 => lz4_flex::compress_prepend_size(c),
    }
}

fn dec(v: V, z: &[u8], n: usize) -> Vec<u8> {
    match v {
        V::Raw(_) => zstd::bulk::decompress(z, n).unwrap(),
        V::P2(_) => unplanes(&zstd::bulk::decompress(z, n).unwrap(), 2),
        V::P4(_) => unplanes(&zstd::bulk::decompress(z, n).unwrap(), 4),
        V::Best(l) => match z[0] {
            0 => dec(V::Raw(l), &z[1..], n),
            2 => dec(V::P2(l), &z[1..], n),
            _ => dec(V::P4(l), &z[1..], n),
        },
        V::Lz4 => lz4_flex::decompress_size_prepended(z).unwrap(),
    }
}

fn main() {
    let mut gufo = false;
    let mut chunk = 4usize << 20;
    let mut threads = 16usize;
    let mut files = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--gufo" => gufo = true,
            "--chunk-mib" => chunk = it.next().unwrap().parse::<usize>().unwrap() << 20,
            "--threads" => threads = it.next().unwrap().parse().unwrap(),
            _ => files.push(a),
        }
    }
    let mut data = Vec::new();
    for f in &files {
        let b = std::fs::read(f).unwrap();
        let bytes = u64::from_le_bytes(b[57..65].try_into().unwrap()) as usize;
        let mut d = &b[4096..4096 + bytes];
        if gufo {
            let id = u32::from_le_bytes(d[16..20].try_into().unwrap()) as usize;
            let tok = u32::from_le_bytes(d[20..24].try_into().unwrap()) as usize;
            d = &d[96 + id + 4 * tok..];
        }
        data.extend_from_slice(d);
    }
    let chunks: Vec<&[u8]> = data.chunks(chunk).collect();
    let total = data.len() as f64;
    let zeros = data.iter().filter(|&&x| x == 0).count() as f64 / total;
    println!("{} files, {:.1} MB payload, {} chunks of {} MiB, zero bytes {:.2}%", files.len(), total / 1e6, chunks.len(), chunk >> 20, zeros * 100.0);
    println!("| codec | ratio | compress 1 thread | decompress 1 thread | compress {threads} threads | decompress {threads} threads |");
    println!("|---|---|---|---|---|---|");
    for v in [V::Lz4, V::Raw(-3), V::Raw(1), V::Raw(3), V::P2(1), V::P4(1), V::Best(1), V::Best(3)] {
        // single thread over a sample (up to 64 chunks), multi-thread over all
        let sample: Vec<&[u8]> = chunks.iter().step_by((chunks.len() / 64).max(1)).copied().collect();
        let sb: usize = sample.iter().map(|c| c.len()).sum();
        let t = Instant::now();
        let zs: Vec<Vec<u8>> = sample.iter().map(|c| enc(v, c)).collect();
        let c1 = sb as f64 / t.elapsed().as_secs_f64() / 1e9;
        let t = Instant::now();
        for (z, c) in zs.iter().zip(&sample) {
            let d = dec(v, z, c.len());
            assert!(d == *c, "roundtrip");
        }
        let d1 = sb as f64 / t.elapsed().as_secs_f64() / 1e9;
        // all chunks, threads
        let t = Instant::now();
        let mut out: Vec<Vec<u8>> = vec![Vec::new(); chunks.len()];
        std::thread::scope(|s| {
            for (ti, part) in out.chunks_mut(chunks.len().div_ceil(threads)).enumerate() {
                let base = ti * chunks.len().div_ceil(threads);
                let ch = &chunks;
                s.spawn(move || {
                    for (k, o) in part.iter_mut().enumerate() {
                        *o = enc(v, ch[base + k]);
                    }
                });
            }
        });
        let cm = total / t.elapsed().as_secs_f64() / 1e9;
        let zb: usize = out.iter().map(|z| z.len()).sum();
        let t = Instant::now();
        std::thread::scope(|s| {
            for (ti, part) in out.chunks(chunks.len().div_ceil(threads)).enumerate() {
                let base = ti * chunks.len().div_ceil(threads);
                let ch = &chunks;
                s.spawn(move || {
                    for (k, z) in part.iter().enumerate() {
                        let d = dec(v, z, ch[base + k].len());
                        std::hint::black_box(&d);
                    }
                });
            }
        });
        let dm = total / t.elapsed().as_secs_f64() / 1e9;
        println!("| {} | {:.3} | {:.2} GB/s | {:.2} GB/s | {:.1} GB/s | {:.1} GB/s |", name(v), total / zb as f64, c1, d1, cm, dm);
    }
}
