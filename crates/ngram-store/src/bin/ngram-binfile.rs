//! Write the bins file (bundle data) for a layout: every bundle's rows copied from the table, bins in layout order,
//! directory at the end (format: src/bins.rs). Then re-reads a sample of keys through `BinFile` and compares them
//! with direct table reads.
//!
//!   ngram-binfile <layout> <model.gguf> <out.ngbins> [--check N] [--rows FILE@OFFSET/ROW_BYTES]

use ngram_store::{bins::{self, BinFile}, codec::Codec, gguf::Gguf, hash::{self, PleParams}, layout::Layout,
                  reader::RowFile, TABLE_TENSOR};
use std::path::Path;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: ngram-binfile <layout> <model.gguf> <out.ngbins> [--check N]");
        std::process::exit(2);
    }
    let check: usize = a.iter().position(|x| x == "--check").and_then(|i| a.get(i + 1)).map(|v| v.parse().unwrap()).unwrap_or(2000);
    let t0 = Instant::now();
    let layout = Layout::load(Path::new(&a[1])).expect("layout");
    let g = Gguf::open(Path::new(&a[2])).expect("gguf");
    let p = PleParams::from_gguf(&g).expect("ple params");
    let t = g.tensors.get(TABLE_TENSOR).expect("table");
    // --rows FILE@OFFSET/ROW_BYTES: bundle rows from another file (bf16 rows.bin), as the broker's --ngram-rows
    let (rb, mut table) = match a.iter().position(|x| x == "--rows").and_then(|i| a.get(i + 1)) {
        Some(v) => {
            let (path, rest) = v.rsplit_once('@').expect("--rows FILE@OFFSET/ROW_BYTES");
            let (off, rb) = rest.split_once('/').expect("--rows FILE@OFFSET/ROW_BYTES");
            let rb: usize = rb.parse().unwrap();
            (rb, RowFile::open(Path::new(path), off.parse().unwrap(), rb, t.dims[1], 256).expect("rows reader"))
        }
        None => {
            let rb = Codec::from_ggml(t.ggml_type).expect("codec").row_bytes(p.head_dim as usize);
            (rb, RowFile::open(&t.path, t.file_offset, rb, t.dims[1], 256).expect("table reader"))
        }
    };
    let h = bins::write(&layout, &p, &mut table, Path::new(&a[3])).expect("write");
    let size = std::fs::metadata(&a[3]).map(|m| m.len()).unwrap_or(0);
    println!("{}: {} x 16 KiB + {} x 4 KiB bins, {} keys, {:.2} GB, {:.1} s",
             a[3], h.n16, h.n4, h.dir_n, size as f64 / 1e9, t0.elapsed().as_secs_f64());

    // Verify: random keys -> their bin through the directory -> bundle bytes == direct table rows.
    let mut bf = BinFile::open(Path::new(&a[3]), 64).expect("open bins");
    bf.matches(&p, rb, t.dims[1]).expect("identity");
    let keys: Vec<u64> = layout.dir.keys().copied().collect();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let hpn = p.heads_per_ngram as usize;
    let (mut ok, mut bad) = (0usize, 0usize);
    let mut rows = [0u32; 8];
    for _ in 0..check.min(keys.len()) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let k = keys[(seed % keys.len() as u64) as usize];
        let b = bf.lookup(k).expect("key in directory");
        assert_eq!(b, layout.dir[&k], "directory bin");
        let (key, order) = bins::split_key(k);
        hash::rows_of_key(&p, key, order, &mut rows[..hpn]);
        let mut want = vec![0u8; hpn * rb];
        table.read_rows(&rows[..hpn], &mut want).expect("direct read");
        bf.submit(&[b]).expect("submit");
        let mut found = false;
        bf.wait(|_, ks, data| {
            if let Some(i) = ks.iter().position(|&x| x == k) {
                found = true;
                if data[i * hpn * rb..(i + 1) * hpn * rb] == want[..] { ok += 1 } else { bad += 1 }
            }
        }).expect("bin read");
        if !found {
            bad += 1;
        }
    }
    println!("check: {ok} bundles equal to direct table reads, {bad} mismatches");
    if bad > 0 {
        std::process::exit(1);
    }
}
