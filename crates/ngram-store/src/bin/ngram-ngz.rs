//! Compressed bf16 n-gram rows (src/ngz.rs).
//!   ngram-ngz build <rows.bin> <out.ngz> [--threads N]   code from a 1/64 sample, then every row encoded (and decoded
//!                                                        back and compared) in parallel; escapes at the end
//!   ngram-ngz verify <rows.bin> <file.ngz> [--threads N] every slot read back from disk, decoded, compared with rows.bin
//! Plain positional reads (no mmap).

use ngram_store::ngz::{self, Code, Header, DIM, HEADER, NSYM, SLOT};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::Instant;

const RB: usize = DIM * 2;
const CHUNK_ROWS: u64 = 1 << 18; // 80 MiB of bf16 rows

fn read_rows(f: &File, r0: u64, n: u64) -> Vec<u16> {
    let mut b = vec![0u8; (n as usize) * RB];
    f.read_exact_at(&mut b, r0 * RB as u64).expect("read rows");
    b.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: ngram-ngz build|verify <rows.bin> <file.ngz> [--threads N]");
        std::process::exit(2);
    }
    let threads: usize = a.iter().position(|x| x == "--threads").and_then(|i| a.get(i + 1)).map(|v| v.parse().unwrap()).unwrap_or(8);
    let src = File::open(&a[2]).expect("rows.bin");
    let rows = src.metadata().unwrap().len() / RB as u64;
    let chunks = rows.div_ceil(CHUNK_ROWS);
    let t0 = Instant::now();
    match a[1].as_str() {
        "build" => {
            // 1. code from every 64th chunk
            let counts = Mutex::new([0u64; NSYM]);
            std::thread::scope(|s| {
                for t in 0..threads {
                    let (src, counts) = (&src, &counts);
                    s.spawn(move || {
                        let mut c = [0u64; NSYM];
                        for k in (0..chunks).step_by(64).skip(t).step_by(threads) {
                            let n = CHUNK_ROWS.min(rows - k * CHUNK_ROWS);
                            ngz::count(&read_rows(src, k * CHUNK_ROWS, n), &mut c);
                        }
                        let mut g = counts.lock().unwrap();
                        for i in 0..NSYM {
                            g[i] += c[i];
                        }
                    });
                }
            });
            let counts = counts.into_inner().unwrap();
            let code = Code::from_counts(&counts);
            let total: u64 = counts.iter().sum();
            let bits: f64 = (0..NSYM).map(|i| counts[i] as f64 * (code.len[i] as f64 + if i == ngz::ESC { ngz::ESC_BITS as f64 } else { 0.0 })).sum::<f64>() / total as f64;
            println!("code lengths {:?}, {:.3} offset bits per value (sample)", code.len, bits);
            // 2. encode + check every row
            let out = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&a[3]).expect("out");
            out.set_len(HEADER + rows * SLOT as u64).unwrap();
            let escapes = Mutex::new(Vec::<(u32, Vec<u8>)>::new());
            let done = AtomicU64::new(0);
            std::thread::scope(|s| {
                for t in 0..threads {
                    let (src, out, code, escapes, done) = (&src, &out, &code, &escapes, &done);
                    s.spawn(move || {
                        let mut back = [0u16; DIM];
                        for k in (t as u64..chunks).step_by(threads) {
                            let r0 = k * CHUNK_ROWS;
                            let n = CHUNK_ROWS.min(rows - r0);
                            let v = read_rows(src, r0, n);
                            let mut slots = vec![0u8; n as usize * SLOT];
                            let mut esc = Vec::new();
                            for (i, (row, slot)) in v.chunks_exact(DIM).zip(slots.chunks_exact_mut(SLOT)).enumerate() {
                                if ngz::encode(code, row, slot).is_some() {
                                    assert!(ngz::decode(code, slot, &mut back) && back[..] == *row, "row {} does not round-trip", r0 + i as u64);
                                } else {
                                    slot.fill(0);
                                    let raw: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
                                    esc.push(((r0 + i as u64) as u32, raw));
                                }
                            }
                            out.write_all_at(&slots, HEADER + r0 * SLOT as u64).expect("write");
                            escapes.lock().unwrap().extend(esc);
                            let d = done.fetch_add(n, Relaxed) + n;
                            if (k / threads as u64) % 64 == 0 && t == 0 {
                                eprintln!("{:.0}% ({:.0} s)", d as f64 * 100.0 / rows as f64, t0.elapsed().as_secs_f64());
                            }
                        }
                    });
                }
            });
            let mut esc = escapes.into_inner().unwrap();
            esc.sort_by_key(|e| e.0);
            let eoff = HEADER + rows * SLOT as u64;
            let mut buf = Vec::with_capacity(esc.len() * (4 + RB));
            for (r, raw) in &esc {
                buf.extend_from_slice(&r.to_le_bytes());
                buf.extend_from_slice(raw);
            }
            out.write_all_at(&buf, eoff).unwrap();
            let h = Header { rows, len: code.len, n_escape: esc.len() as u64, escape_offset: eoff };
            out.write_all_at(&h.to_bytes(), 0).unwrap();
            out.sync_all().unwrap();
            let size = eoff + buf.len() as u64;
            println!("{}: {} rows, {} escaped, {:.2} GB ({:.1}% of bf16), {:.0} s; every encoded row decoded and compared: OK",
                     a[3], rows, esc.len(), size as f64 / 1e9, size as f64 * 100.0 / (rows * RB as u64) as f64, t0.elapsed().as_secs_f64());
        }
        "verify" => {
            let f = File::open(&a[3]).expect("ngz");
            let mut hb = vec![0u8; HEADER as usize];
            f.read_exact_at(&mut hb, 0).unwrap();
            let h = Header::parse(&hb).expect("not an NGZ1 file");
            assert_eq!(h.rows, rows, "row count");
            let code = Code::from_lengths(&h.len);
            let mut eb = vec![0u8; h.n_escape as usize * (4 + RB)];
            f.read_exact_at(&mut eb, h.escape_offset).unwrap();
            let esc: std::collections::HashMap<u32, &[u8]> =
                eb.chunks_exact(4 + RB).map(|c| (u32::from_le_bytes(c[..4].try_into().unwrap()), &c[4..])).collect();
            let bad = AtomicU64::new(0);
            std::thread::scope(|s| {
                for t in 0..threads {
                    let (src, f, code, esc, bad) = (&src, &f, &code, &esc, &bad);
                    s.spawn(move || {
                        let mut back = [0u16; DIM];
                        for k in (t as u64..chunks).step_by(threads) {
                            let r0 = k * CHUNK_ROWS;
                            let n = CHUNK_ROWS.min(rows - r0);
                            let v = read_rows(src, r0, n);
                            let mut slots = vec![0u8; n as usize * SLOT];
                            f.read_exact_at(&mut slots, HEADER + r0 * SLOT as u64).unwrap();
                            for (i, (row, slot)) in v.chunks_exact(DIM).zip(slots.chunks_exact(SLOT)).enumerate() {
                                let ok = if ngz::decode(code, slot, &mut back) {
                                    back[..] == *row
                                } else {
                                    let raw: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
                                    esc.get(&((r0 + i as u64) as u32)).map(|e| **e == raw[..]).unwrap_or(false)
                                };
                                if !ok {
                                    bad.fetch_add(1, Relaxed);
                                }
                            }
                        }
                    });
                }
            });
            let b = bad.load(Relaxed);
            println!("verify {}: {} rows ({} escaped) against {}: {} mismatches ({:.0} s)", a[3], rows, h.n_escape, a[2], b, t0.elapsed().as_secs_f64());
            if b > 0 {
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("build | verify");
            std::process::exit(2);
        }
    }
}
