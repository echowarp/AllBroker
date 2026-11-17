//! Build a bundle layout from an ngram-profile output.
//!
//!   ngram-bins <profile_dir> <out.layout> <model.gguf> [--bigrams N] [--trigrams N] [--tri-per-bigram K]
//!              [--dup FRACTION] [--no-chains] [--chain-bigrams N] [--row-bytes N]
//!
//! Hot tier: chains from the hottest N bigrams in 16 KiB bins. Rare tier: all other selected bundles in 4 KiB bins.
//!
//! Writes the layout (which keys go into which 16 KiB bin, hottest bin first). The bundle data file (the rows
//! themselves, copied from the table) is written later, for the layout that wins the replay comparison.

use ngram_store::{codec::Codec, gguf::Gguf, hash::PleParams, layout::*, TABLE_TENSOR};
use std::path::Path;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: ngram-bins <profile_dir> <out.layout> <model.gguf> [--bigrams N] [--trigrams N] [--tri-per-bigram K] [--dup F] [--no-chains]");
        std::process::exit(2);
    }
    let flag = |name: &str, default: f64| -> f64 {
        a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(|v| v.parse().unwrap()).unwrap_or(default)
    };
    let g = Gguf::open(Path::new(&a[3])).expect("gguf");
    let p = PleParams::from_gguf(&g).expect("ple params");
    let t = g.tensors.get(TABLE_TENSOR).expect("table");
    // --row-bytes N: bundles of another codec's rows (bf16: 320), same keys / hashing
    let row_bytes = match a.iter().position(|x| x == "--row-bytes").and_then(|i| a.get(i + 1)) {
        Some(v) => v.parse().expect("--row-bytes"),
        None => Codec::from_ggml(t.ggml_type).expect("codec").row_bytes(p.head_dim as usize),
    };
    let o = BuildOptions {
        bigrams: flag("--bigrams", 10e6) as usize,
        trigrams: flag("--trigrams", 10e6) as usize,
        bundle_bytes: row_bytes * p.heads_per_ngram as usize,
        tri_per_bigram: flag("--tri-per-bigram", 2.0) as usize,
        dup_fraction: flag("--dup", 0.0),
        no_chains: a.iter().any(|x| x == "--no-chains"),
        chain_bigrams: flag("--chain-bigrams", 2e6) as usize,
    };
    let t0 = Instant::now();
    let dir = Path::new(&a[1]);
    let bi = read_profile(&dir.join("bigram.bin"), o.bigrams).expect("bigram.bin");
    let tri = read_profile(&dir.join("trigram.bin"), o.trigrams).expect("trigram.bin");
    eprintln!("profile: {} bigram, {} trigram keys ({:.1} s)", bi.len(), tri.len(), t0.elapsed().as_secs_f64());
    let l = Layout::build(&bi, &tri, &o);
    l.save(Path::new(&a[2])).expect("save");
    let tier = |small: bool| {
        let idx: Vec<usize> = (0..l.bins.len()).filter(|&i| l.small[i] == small).collect();
        let bundles: usize = idx.iter().map(|&i| l.bins[i].len()).sum();
        let cap = if small { l.per_small } else { l.per_bin };
        let full = idx.iter().filter(|&&i| l.bins[i].len() == cap).count();
        let bytes: usize = idx.iter().map(|&i| l.bin_bytes(i)).sum();
        format!("{} bins, {} bundles, {:.2} GB, {:.0}% full", idx.len(), bundles, bytes as f64 / 1e9,
                100.0 * full as f64 / idx.len().max(1) as f64)
    };
    println!("{}: 16 KiB tier: {} | 4 KiB tier: {} | {} duplicates | {} B bundles | {:.1} s",
             a[2], tier(false), tier(true), l.duplicates, o.bundle_bytes, t0.elapsed().as_secs_f64());
}
