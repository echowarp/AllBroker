//! Replay held-out token streams against n-gram table layouts and RAM budgets.
//!
//!   ngram-replay <model.gguf> <holdout_dir> <ram_gb,ram_gb,...> <raw|layout_path>... [--tokens N] [--pin F]
//!
//! For every holdout category (<holdout_dir>/<category>.u32) and every layout and RAM budget, prints hit rate,
//! disk reads per token (4 KiB row reads, 16 KiB bin reads), share of tokens that touch the disk, the exposed
//! latency per token if nothing is prefetched (measured 990 PRO latencies), and the disk time of a 2,048-token
//! prefill chunk. `--pin F`: fraction of the RAM budget given to preloaded hot bins (default 0.5). `--tokens N`:
//! per-category token limit (default 4e6); the first 10% of each category warms the cache and isn't counted.

use ngram_store::{codec::Codec, gguf::Gguf, hash::PleParams, layout::*, TABLE_TENSOR};
use std::path::{Path, PathBuf};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: ngram-replay <model.gguf> <holdout_dir> <ram_gb,...> <raw|layout>... [--tokens N] [--pin F]");
        std::process::exit(2);
    }
    let flag = |name: &str, default: f64| -> f64 {
        a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(|v| v.parse().unwrap()).unwrap_or(default)
    };
    let g = Gguf::open(Path::new(&a[1])).expect("gguf");
    let p = PleParams::from_gguf(&g).expect("ple params");
    let t = g.tensors.get(TABLE_TENSOR).expect("table");
    let row_bytes = Codec::from_ggml(t.ggml_type).expect("codec").row_bytes(p.head_dim as usize);
    let budgets: Vec<f64> = a[3].split(',').map(|s| s.parse().unwrap()).collect();
    let max_tokens = flag("--tokens", 4e6) as usize;
    let pin = flag("--pin", 0.5);
    let modes: Vec<String> = a[4..].iter().take_while(|x| !x.starts_with("--")).cloned().collect();
    let layouts: Vec<(String, Option<Layout>)> = modes.iter().map(|m| {
        if m == "raw" { (m.clone(), None) } else { (m.clone(), Some(Layout::load(Path::new(m)).expect("layout"))) }
    }).collect();

    let mut cats: Vec<PathBuf> = std::fs::read_dir(&a[2]).unwrap().filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "u32").unwrap_or(false)).collect();
    cats.sort();
    println!("{:<9} {:<28} {:>6} {:>7} {:>8} {:>8} {:>8} {:>10} {:>11}",
             "category", "layout", "RAM GB", "hit %", "4K/tok", "16K/tok", "io tok%", "us/token", "ms/2k pref");
    for c in &cats {
        let docs = read_holdout(c, max_tokens).expect("holdout");
        let n: usize = docs.iter().map(|d| d.len()).sum();
        let cat = c.file_stem().unwrap().to_string_lossy().to_string();
        for (name, l) in &layouts {
            for &gb in &budgets {
                let ram = (gb * 1e9) as u64;
                let mode = match l {
                    None => Mode::Raw,
                    Some(l) => Mode::Bins { layout: l, pinned_bytes: (ram as f64 * pin) as u64 },
                };
                let s = replay(&p, &docs, &mode, ram, row_bytes, n as u64 / 10);
                let tk = s.tokens.max(1) as f64;
                let short = Path::new(name).file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or(name.clone());
                println!("{:<9} {:<28} {:>6.1} {:>7.1} {:>8.2} {:>8.2} {:>8.1} {:>10.1} {:>11.1}",
                         cat, short, gb, 100.0 * s.hits as f64 / s.lookups.max(1) as f64, s.reads_4k as f64 / tk,
                         s.reads_16k as f64 / tk, 100.0 * s.tokens_with_io as f64 / tk, s.exposed_us / tk, s.prefill_ms_per_2k());
            }
        }
    }
}
