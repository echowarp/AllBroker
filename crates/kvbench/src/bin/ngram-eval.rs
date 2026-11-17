//! Replay held-out text through a broker's n-gram service the way an engine reads it, and report disk work.
//!
//!   ngram-eval <socket> <table_name> <model.gguf> <holdout.u32> <metrics host:port> [--per N] [--tokens N]
//!
//! Documents are streamed in order, `--per` tokens per gather (1 = decode, 512 = prefill chunks), each with its
//! true history. Reports gathers, gather latency p50/p99 (client side), and from the broker's counters (deltas):
//! disk I/Os per token (row reads + bin reads), requested rows that missed, and cache hit rate.

use kvclient::{Client, NgramSeq};
use ngram_store::{codec::Codec, gguf::Gguf, hash, layout::read_holdout, TABLE_TENSOR};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::time::{Duration, Instant};

fn metrics(addr: &str) -> HashMap<String, f64> {
    // The listener may come up after the socket, and a reset can follow a complete reply: retry until the
    // n-gram counters are in the body.
    let mut body = String::new();
    for _ in 0..50 {
        body.clear();
        if let Ok(mut s) = TcpStream::connect(addr) {
            if write!(s, "GET /metrics HTTP/1.0\r\nHost: {addr}\r\n\r\n").is_ok() {
                let mut buf = Vec::new();
                let _ = s.read_to_end(&mut buf);
                body = String::from_utf8_lossy(&buf).into_owned();
                if body.contains("llmt_kv_ngram_rows_total") {
                    break;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    body.lines().filter(|l| l.starts_with("llmt_kv_ngram_")).filter_map(|l| {
        let (k, v) = l.rsplit_once(' ')?;
        Some((k.to_string(), v.parse().ok()?))
    }).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: ngram-eval <socket> <table> <model.gguf> <holdout.u32> <metrics host:port> [--per N] [--tokens N]");
        std::process::exit(2);
    }
    let flag = |n: &str, d: f64| a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).map(|v| v.parse().unwrap()).unwrap_or(d);
    let per = flag("--per", 1.0) as usize;
    let max_tokens = flag("--tokens", 200e3) as usize;
    let g = Gguf::open(Path::new(&a[3])).expect("gguf");
    let p = hash::PleParams::from_gguf(&g).expect("params");
    let t = g.tensors.get(TABLE_TENSOR).expect("table");
    let rb = Codec::from_ggml(t.ggml_type).expect("codec").row_bytes(p.head_dim as usize);
    let docs = read_holdout(Path::new(&a[4]), max_tokens).expect("holdout");
    let mut c = Client::connect(Path::new(&a[1]), 99, "ngram-eval", Duration::from_secs(10)).expect("connect");
    c.ngram_open(&a[2], t.dims[1], rb as u32).expect("ngram_open");
    let m0 = metrics(&a[5]);
    let mut lat = Vec::new();
    let mut tokens = 0usize;
    let t0 = Instant::now();
    for d in &docs {
        for (w, ch) in d.chunks(per).enumerate() {
            let at = w * per;
            let prev = [if at >= 1 { d[at - 1] } else { -1 }, if at >= 2 { d[at - 2] } else { -1 }];
            let s = Instant::now();
            c.ngram_gather(vec![NgramSeq { prev, tokens: ch.to_vec() }]).expect("gather");
            lat.push(s.elapsed().as_secs_f64() * 1e6);
            tokens += ch.len();
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let m1 = metrics(&a[5]);
    let dm = |k: &str| m1.get(k).copied().unwrap_or(0.0) - m0.get(k).copied().unwrap_or(0.0);
    let pre = "llmt_kv_ngram_";
    let rows = dm(&format!("{pre}rows_total"));
    let row_reads = dm(&format!("{pre}row_reads_total"));
    let b16 = dm(&format!("{pre}bin_reads_total{{size=\"16k\"}}"));
    let b4 = dm(&format!("{pre}bin_reads_total{{size=\"4k\"}}"));
    let missed = dm(&format!("{pre}disk_rows_total"));
    lat.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let q = |f: f64| lat[((lat.len() - 1) as f64 * f) as usize];
    let tk = tokens as f64;
    println!("{:<12} per {:>4}  tokens {:>7}  hit {:5.1}%  I/O per token {:6.3} (rows {:6.3}, bin16 {:6.3}, bin4 {:6.3})  \
              missed rows/tok {:6.2}  gather p50 {:6.0} us p99 {:6.0} us  {:.1} s",
             Path::new(&a[4]).file_stem().unwrap().to_string_lossy(), per, tokens, 100.0 * (1.0 - missed / rows.max(1.0)),
             (row_reads + b16 + b4) / tk, row_reads / tk, b16 / tk, b4 / tk, missed / tk, q(0.5), q(0.99), wall);
}
