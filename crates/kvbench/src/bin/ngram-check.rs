//! End-to-end check of the broker's n-gram service against direct table reads.
//!
//!   ngram-check <socket> <table_name> <model.gguf> [runs=300] [tokens_per_run=16] [--holdout FILE.u32]
//!
//! `--holdout`: runs are consecutive windows of real token streams (ngram-profile holdout format) with their true
//! history, so bundled n-grams (broker `--ngram-bins`) are exercised; the default is random tokens.
//!
//! Gathers random token runs (random histories incl. none and EOS) through the broker and compares every staged
//! row byte with a direct O_DIRECT read of the same rows (ngram-store hash + reader). Then times cold gathers (new
//! tokens: disk), warm gathers (same tokens: cache), and prefetch-then-gather.

use kvclient::{Client, NgramSeq};
use ngram_store::{codec::Codec, gguf::Gguf, hash, layout::read_holdout, reader::RowFile, TABLE_TENSOR};
use std::path::Path;
use std::time::{Duration, Instant};

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: ngram-check <socket> <table_name> <model.gguf> [runs] [tokens_per_run]");
        std::process::exit(2);
    }
    let holdout = a.iter().position(|x| x == "--holdout").and_then(|i| a.get(i + 1)).cloned();
    let pos: Vec<&String> = a.iter().enumerate().filter(|(i, x)| !x.starts_with("--") && (*i == 0 || a[i - 1] != "--holdout"))
        .map(|(_, x)| x).collect();
    let runs: usize = pos.get(4).map(|s| s.parse().unwrap()).unwrap_or(300);
    let per: usize = pos.get(5).map(|s| s.parse().unwrap()).unwrap_or(16);
    let g = Gguf::open(Path::new(&a[3])).expect("gguf");
    let p = hash::PleParams::from_gguf(&g).expect("params");
    let t = g.tensors.get(TABLE_TENSOR).expect("table");
    let rb = Codec::from_ggml(t.ggml_type).expect("codec").row_bytes(p.head_dim as usize);
    let mut direct = RowFile::open(&t.path, t.file_offset, rb, t.dims[1], 64).expect("direct reader");
    let mut c = Client::connect(Path::new(&a[1]), 99, "ngram-check", Duration::from_secs(5)).expect("connect");
    let st = c.ngram_open(&a[2], t.dims[1], rb as u32).expect("ngram_open");
    println!("staging window {} MiB, {} B rows, {} heads", st.bytes >> 20, st.row_bytes, st.heads);

    let mut seed = 0xA076_1D64_78BD_642Fu64 ^ std::process::id() as u64;
    let mk = |seed: &mut u64| -> NgramSeq {
        let tok = |s: &mut u64| -> i32 { if rng(s) % 12 == 0 { p.eos as i32 } else { (rng(s) % 248_320) as i32 } };
        let prev = [if rng(seed) % 5 == 0 { -1 } else { tok(seed) }, if rng(seed) % 5 == 0 { -1 } else { tok(seed) }];
        NgramSeq { prev, tokens: (0..per).map(|_| tok(seed)).collect() }
    };
    // 1. correctness + cold latency
    let seqs: Vec<NgramSeq> = match &holdout {
        None => (0..runs).map(|_| mk(&mut seed)).collect(),
        Some(f) => {
            let docs = read_holdout(Path::new(f), runs * per * 2).expect("holdout");
            let mut v = Vec::new();
            'outer: for d in &docs {
                for (w, c) in d.chunks(per).enumerate() {
                    if v.len() == runs {
                        break 'outer;
                    }
                    let at = w * per;
                    let prev = [if at >= 1 { d[at - 1] } else { -1 }, if at >= 2 { d[at - 2] } else { -1 }];
                    v.push(NgramSeq { prev, tokens: c.to_vec() });
                }
            }
            println!("holdout {f}: {} runs of up to {per} tokens", v.len());
            v
        }
    };
    let (mut bad, mut checked) = (0usize, 0usize);
    let mut cold = Vec::new();
    let mut disk_total = 0u64;
    for s in &seqs {
        let t0 = Instant::now();
        let (n, disk) = c.ngram_gather(vec![s.clone()]).expect("gather");
        cold.push(t0.elapsed().as_secs_f64() * 1e6);
        disk_total += disk as u64;
        let mut h = hash::History { prev: s.prev };
        let mut rows = vec![0u32; s.tokens.len() * p.heads as usize];
        hash::hash_rows(&p, &mut h, &s.tokens, &mut rows);
        assert_eq!(n as usize, rows.len());
        let mut want = vec![0u8; rows.len() * rb];
        direct.read_rows(&rows, &mut want).expect("direct read");
        let got = st.rows(rows.len());
        for i in 0..rows.len() {
            checked += 1;
            if got[i * rb..(i + 1) * rb] != want[i * rb..(i + 1) * rb] {
                bad += 1;
            }
        }
    }
    println!("correctness: {checked} rows compared with direct reads, {bad} mismatches");
    // 2. warm: same runs again (cached)
    let mut warm = Vec::new();
    let mut warm_disk = 0u64;
    for s in &seqs {
        let t0 = Instant::now();
        let (_, d) = c.ngram_gather(vec![s.clone()]).expect("gather");
        warm.push(t0.elapsed().as_secs_f64() * 1e6);
        warm_disk += d as u64;
    }
    // 3. prefetch, give it 1 ms, then gather
    let mut pre = Vec::new();
    let mut pre_disk = 0u64;
    for _ in 0..runs.min(200) {
        let s = mk(&mut seed);
        c.ngram_prefetch(vec![s.clone()]).expect("prefetch");
        std::thread::sleep(Duration::from_millis(1));
        let t0 = Instant::now();
        let (_, d) = c.ngram_gather(vec![s]).expect("gather");
        pre.push(t0.elapsed().as_secs_f64() * 1e6);
        pre_disk += d as u64;
    }
    let rows_per = per * p.heads as usize;
    println!("cold  {runs} gathers of {rows_per} rows: p50 {:.0} us, p99 {:.0} us, {:.1} disk rows/gather",
             pct(&mut cold, 0.5), pct(&mut cold, 0.99), disk_total as f64 / runs as f64);
    println!("warm  {runs} gathers: p50 {:.0} us, p99 {:.0} us, {:.2} disk rows/gather",
             pct(&mut warm, 0.5), pct(&mut warm, 0.99), warm_disk as f64 / runs as f64);
    println!("prefetched (1 ms ahead) {} gathers: p50 {:.0} us, p99 {:.0} us, {:.2} disk rows/gather",
             pre.len(), pct(&mut pre, 0.5), pct(&mut pre, 0.99), pre_disk as f64 / pre.len() as f64);
    if bad > 0 {
        std::process::exit(1);
    }
}
