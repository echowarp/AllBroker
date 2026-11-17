//! ngram-bench <model.gguf> info
//! ngram-bench <model.gguf> hash <tokens...>              rows for a token sequence (parity with gufo's hash)
//! ngram-bench <model.gguf> io <uring|pread> <rows/gather> <gathers> [depth|threads]
//!     Uniform random rows = worst case (no locality, no cache). Prints per-gather latency percentiles and
//!     throughput. `pread` mimics gufo's current reader: a pool of threads doing O_DIRECT pread per row.

use ngram_store::{codec::Codec, gguf::Gguf, hash, reader::RowFile, TABLE_TENSOR};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;
use std::time::Instant;

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: see the header of src/bin/ngram-bench.rs");
        std::process::exit(2);
    }
    let g = Gguf::open(Path::new(&a[1])).expect("gguf");
    let p = hash::PleParams::from_gguf(&g).expect("ple params");
    let t = g.tensors.get(TABLE_TENSOR).expect("table tensor");
    let codec = Codec::from_ggml(t.ggml_type).expect("unsupported table type");
    let rb = codec.row_bytes(p.head_dim as usize);
    match a[2].as_str() {
        "info" => {
            println!("table {} in {} at offset {}", t.name, t.path.display(), t.file_offset);
            println!("dims {:?}, type {} ({:?}), {} bytes/row, rows {} (params say {})", t.dims, t.ggml_type, codec, rb, t.dims[1], p.rows);
            println!("ngram_size {}, heads {}, eos {}, multipliers {:?}", p.ngram_size, p.heads, p.eos, &p.multipliers[..p.ngram_size as usize]);
            println!("head_offsets {:?}", &p.head_offsets[..p.heads as usize]);
            println!("head_vocab {:?}", &p.head_vocab[..p.heads as usize]);
        }
        "hash" => {
            let toks: Vec<i32> = a[3..].iter().map(|s| s.parse().unwrap()).collect();
            let mut h = hash::History::default();
            let mut rows = vec![0u32; toks.len() * p.heads as usize];
            hash::hash_rows(&p, &mut h, &toks, &mut rows);
            for (i, c) in rows.chunks(p.heads as usize).enumerate() {
                println!("{} {}", toks[i], c.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(" "));
            }
        }
        "io" => {
            let mode = a[3].as_str();
            let per: usize = a[4].parse().unwrap();
            let gathers: usize = a[5].parse().unwrap();
            let knob: u32 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(if mode == "uring" { 256 } else { 32 });
            let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ Instant::now().elapsed().as_nanos() as u64 ^ std::process::id() as u64;
            let mut lat = Vec::with_capacity(gathers);
            let mut out = vec![0u8; per * rb];
            let mut rows = vec![0u32; per];
            let mut dec = vec![0f32; p.head_dim as usize];
            let t0 = Instant::now();
            match mode {
                "uring" => {
                    let mut f = RowFile::open(&t.path, t.file_offset, rb, t.dims[1], knob).expect("open");
                    for _ in 0..gathers {
                        for r in rows.iter_mut() {
                            *r = (rng(&mut seed) % t.dims[1]) as u32;
                        }
                        let s = Instant::now();
                        f.read_rows(&rows, &mut out).expect("read");
                        lat.push(s.elapsed().as_secs_f64() * 1e6);
                    }
                    println!("reads submitted {}, MB read {:.1}", f.reads_submitted, f.bytes_read as f64 / 1e6);
                }
                "pread" => {
                    let file = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&t.path).unwrap();
                    let threads = knob as usize;
                    for _ in 0..gathers {
                        for r in rows.iter_mut() {
                            *r = (rng(&mut seed) % t.dims[1]) as u32;
                        }
                        let s = Instant::now();
                        std::thread::scope(|sc| {
                            for chunk in rows.chunks(per.div_ceil(threads)) {
                                let file = &file;
                                sc.spawn(move || {
                                    let layout = std::alloc::Layout::from_size_align(8192, 4096).unwrap();
                                    let buf = unsafe { std::alloc::alloc(layout) };
                                    let b = unsafe { std::slice::from_raw_parts_mut(buf, 8192) };
                                    for &r in chunk {
                                        let off = t.file_offset + r as u64 * rb as u64;
                                        let begin = off & !4095;
                                        let end = (off + rb as u64).div_ceil(4096) * 4096;
                                        file.read_exact_at(&mut b[..(end - begin) as usize], begin).unwrap();
                                    }
                                    unsafe { std::alloc::dealloc(buf, layout) };
                                });
                            }
                        });
                        lat.push(s.elapsed().as_secs_f64() * 1e6);
                    }
                }
                _ => panic!("mode uring|pread"),
            }
            let secs = t0.elapsed().as_secs_f64();
            codec.decode(&out[..rb], &mut dec); // touch the decoder once
            let (p50, p90, p99) = (pct(&mut lat, 0.5), pct(&mut lat, 0.9), pct(&mut lat, 0.99));
            println!(
                "{mode} knob {knob}: {per} rows x {gathers} gathers: p50 {p50:.0} us, p90 {p90:.0} us, p99 {p99:.0} us; {:.0} rows/s",
                (per * gathers) as f64 / secs
            );
        }
        "sizes" => {
            // Random aligned reads of each size inside the table: QD1 latency, 16-at-once (one token's gather),
            // and QD64 throughput. Answers "what does reading more than 4 KiB cost on this drive?".
            use io_uring::{opcode, types, IoUring};
            use std::os::fd::AsRawFd;
            let file = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&t.path).unwrap();
            let fd = types::Fd(file.as_raw_fd());
            let span = t.dims[1] * rb as u64;
            let mut ring = IoUring::new(128).unwrap();
            let layout = std::alloc::Layout::from_size_align(64 * 65536, 65536).unwrap();
            let buf = unsafe { std::alloc::alloc(layout) };
            let mut seed = 0x1234_5678_9ABC_DEF1u64 ^ std::process::id() as u64;
            for kib in [4u64, 8, 16, 32, 64] {
                let sz = kib * 1024;
                let mut run = |batch: usize, rounds: usize| -> (Vec<f64>, f64) {
                    let mut lat = Vec::with_capacity(rounds);
                    let t0 = Instant::now();
                    for _ in 0..rounds {
                        let s = Instant::now();
                        for k in 0..batch {
                            let off = t.file_offset / sz * sz + rng(&mut seed) % (span / sz) * sz;
                            let e = opcode::Read::new(fd, unsafe { buf.add(k * 65536) }, sz as u32).offset(off).build();
                            unsafe { ring.submission().push(&e).unwrap() };
                        }
                        ring.submit_and_wait(batch).unwrap();
                        let n = ring.completion().count();
                        assert_eq!(n, batch);
                        lat.push(s.elapsed().as_secs_f64() * 1e6);
                    }
                    (lat, t0.elapsed().as_secs_f64())
                };
                let (mut l1, _) = run(1, 2000);
                let (mut l16, _) = run(16, 1000);
                let (_, secs) = run(64, 300);
                let iops = 64.0 * 300.0 / secs;
                println!(
                    "{kib:2} KiB: QD1 p50 {:5.0} us p99 {:5.0} | 16 at once p50 {:5.0} us p99 {:5.0} | QD64 {:7.0} IOPS {:6.0} MB/s",
                    pct(&mut l1, 0.5), pct(&mut l1, 0.99), pct(&mut l16, 0.5), pct(&mut l16, 0.99), iops, iops * sz as f64 / 1e6
                );
            }
        }
        _ => panic!("unknown command"),
    }
}
