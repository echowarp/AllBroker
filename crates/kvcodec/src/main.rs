//! kvcodec: compression study on a snapshot of the RAM KV tier (dense per-tensor dumps of the pool).
//! Prints aggregate numbers only (a snapshot holds real KV data and is never copied anywhere).
//!
//!   kvcodec FILE... [--page 3461120] [--every 16] [--row 520] [--heads 4]
//!
//! FILEs are dense tensor dumps (tNN.bin without gaps). Pages are sampled every N-th page per file and classified:
//! a GDN-state page ends in zero padding (the tier's page size is the attention page size); an attention page is int8
//! KV in NHD rows of `row` bytes (token-major, `heads` rows per token).

use std::io::Read;
use std::time::Instant;

struct Args {
    files: Vec<String>,
    page: usize,
    every: usize,
    row: usize,
    heads: usize,
}

fn args() -> Args {
    let mut a = Args { files: vec![], page: 3_461_120, every: 16, row: 520, heads: 4 };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--page" => a.page = it.next().unwrap().parse().unwrap(),
            "--every" => a.every = it.next().unwrap().parse().unwrap(),
            "--row" => a.row = it.next().unwrap().parse().unwrap(),
            "--heads" => a.heads = it.next().unwrap().parse().unwrap(),
            _ => a.files.push(x),
        }
    }
    a
}

/// Order-0 entropy in bits per symbol.
fn entropy(h: &[u64]) -> f64 {
    let n: u64 = h.iter().sum();
    if n == 0 {
        return 0.0;
    }
    h.iter().filter(|&&c| c > 0).map(|&c| {
        let p = c as f64 / n as f64;
        -p * p.log2()
    }).sum()
}

fn hist(b: impl Iterator<Item = u8>) -> [u64; 256] {
    let mut h = [0u64; 256];
    for x in b {
        h[x as usize] += 1;
    }
    h
}

#[derive(Default)]
struct Codec {
    name: String,
    raw: u64,
    out: u64,
    ct: f64,
    dt: f64,
}

impl Codec {
    fn add(&mut self, raw: usize, out: usize, ct: f64, dt: f64) {
        self.raw += raw as u64;
        self.out += out as u64;
        self.ct += ct;
        self.dt += dt;
    }
    fn line(&self) -> String {
        format!(
            "  {:34} ratio {:5.3}  ({:5.1}% of raw)  compress {:7.0} MB/s  decompress {:7.0} MB/s",
            self.name,
            self.raw as f64 / self.out.max(1) as f64,
            100.0 * self.out as f64 / self.raw.max(1) as f64,
            self.raw as f64 / 1e6 / self.ct.max(1e-9),
            self.raw as f64 / 1e6 / self.dt.max(1e-9)
        )
    }
}

/// Run zstd levels and lz4 on `data`, recording into `cs` (one Codec per method, same order every call).
fn run_codecs(prefix: &str, data: &[u8], cs: &mut Vec<Codec>, slot: usize) {
    let methods: [(&str, i32); 4] = [("lz4", 0), ("zstd-1", 1), ("zstd-3", 3), ("zstd-9", 9)];
    for (i, (m, lvl)) in methods.iter().enumerate() {
        let idx = slot + i;
        if cs.len() <= idx {
            cs.push(Codec { name: format!("{prefix} {m}"), ..Default::default() });
        }
        let t = Instant::now();
        let (out, n) = if *lvl == 0 {
            let c = lz4_flex::compress(data);
            let n = c.len();
            (c, n)
        } else {
            let c = zstd::bulk::compress(data, *lvl).unwrap();
            let n = c.len();
            (c, n)
        };
        let ct = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let back = if *lvl == 0 {
            lz4_flex::decompress(&out, data.len()).unwrap()
        } else {
            zstd::bulk::decompress(&out, data.len()).unwrap()
        };
        let dt = t.elapsed().as_secs_f64();
        assert_eq!(back.len(), data.len());
        cs[idx].add(data.len(), n, ct, dt);
    }
}

/// Byte-plane split: element of `size` bytes -> `size` planes.
fn planes(d: &[u8], size: usize) -> Vec<u8> {
    let n = d.len() / size;
    let mut out = vec![0u8; n * size];
    for i in 0..n {
        for b in 0..size {
            out[b * n + i] = d[i * size + b];
        }
    }
    out
}

/// Column-major by row: all rows' byte 0, then byte 1, ... (groups each channel's values together).
fn by_column(d: &[u8], row: usize) -> Vec<u8> {
    let n = d.len() / row;
    let mut out = vec![0u8; n * row];
    for r in 0..n {
        for c in 0..row {
            out[c * n + r] = d[r * row + c];
        }
    }
    out
}

fn main() {
    let a = args();
    let mut attn_pages: Vec<Vec<u8>> = vec![];
    let mut state_pages: Vec<(Vec<u8>, usize)> = vec![]; // (page, content length)
    let mut n_pages = 0;
    for f in &a.files {
        let mut file = std::fs::File::open(f).unwrap();
        let mut d = Vec::new();
        file.read_to_end(&mut d).unwrap();
        let pages = d.len() / a.page;
        for p in (0..pages).step_by(a.every) {
            n_pages += 1;
            let pg = d[p * a.page..(p + 1) * a.page].to_vec();
            let content = pg.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
            if a.page - content >= 64 << 10 {
                state_pages.push((pg, content));
            } else {
                attn_pages.push(pg);
            }
        }
    }
    println!(
        "{} files, {} sampled pages of {} B: attention {}, GDN state {}",
        a.files.len(),
        n_pages,
        a.page,
        attn_pages.len(),
        state_pages.len()
    );

    // ---------------- attention: layout discovery
    if !attn_pages.is_empty() {
        let row = a.row;
        let mut col_h = vec![[0u64; 256]; row];
        for pg in &attn_pages {
            for r in pg.chunks_exact(row) {
                for (c, x) in r.iter().enumerate() {
                    col_h[c][*x as usize] += 1;
                }
            }
        }
        let ce: Vec<f64> = col_h.iter().map(|h| entropy(h)).collect();
        println!("\nATTENTION ({} pages, {:.0} MB)", attn_pages.len(), attn_pages.len() as f64 * a.page as f64 / 1e6);
        println!("  per-column order-0 entropy over all rows (bits/byte), column ranges of {row}-byte rows:");
        let mut start = 0;
        for c in 1..=row {
            if c == row || (ce[c] - ce[start]).abs() > 0.6 || c - start >= 128 {
                let m = ce[start..c].iter().sum::<f64>() / (c - start) as f64;
                println!("    cols {start:3}..{:3}: {m:.2}", c - 1);
                start = c;
            }
        }
        // entropy bounds
        let total: usize = attn_pages.len() * a.page;
        let page_h = hist(attn_pages.iter().flat_map(|p| p.iter().copied()));
        let h_page = entropy(&page_h);
        let h_col: f64 = ce.iter().sum::<f64>() / row as f64;
        // per (head, column) and token-delta per (head, column)
        let heads = a.heads;
        let mut hc = vec![[0u64; 256]; heads * row];
        let mut dc = vec![[0u64; 256]; heads * row];
        for pg in &attn_pages {
            let rows: Vec<&[u8]> = pg.chunks_exact(row).collect();
            for (i, r) in rows.iter().enumerate() {
                let h = i % heads;
                for c in 0..row {
                    hc[h * row + c][r[c] as usize] += 1;
                    if i >= heads {
                        let prev = rows[i - heads][c];
                        dc[h * row + c][r[c].wrapping_sub(prev) as usize] += 1;
                    }
                }
            }
        }
        let h_hc = hc.iter().map(|h| entropy(h)).sum::<f64>() / (heads * row) as f64;
        let h_dc = dc.iter().map(|h| entropy(h)).sum::<f64>() / (heads * row) as f64;
        // best of value vs delta per (head, column)
        let h_best = hc.iter().zip(&dc).map(|(x, y)| entropy(x).min(entropy(y))).sum::<f64>() / (heads * row) as f64;
        println!("  entropy bounds (bits/byte; 8 = incompressible) -> ideal ratio for an order-0 coder such as rANS:");
        for (n, h) in [
            ("whole page, one histogram", h_page),
            ("per column (all heads)", h_col),
            ("per head x column", h_hc),
            ("per head x column, token delta", h_dc),
            ("per head x column, best of value/delta", h_best),
        ] {
            println!("    {n:40} {h:.3}  -> {:.3}x", 8.0 / h);
        }
        let mut cs = vec![];
        for pg in &attn_pages {
            run_codecs("raw", pg, &mut cs, 0);
            run_codecs("by column", &by_column(pg, row), &mut cs, 4);
        }
        println!("  codecs (single thread, {:.0} MB):", total as f64 / 1e6);
        for c in &cs {
            println!("{}", c.line());
        }
    }

    // ---------------- GDN state
    if !state_pages.is_empty() {
        let content: usize = state_pages.iter().map(|x| x.1).sum();
        let total = state_pages.len() * a.page;
        let lens: Vec<usize> = state_pages.iter().map(|x| x.1).collect();
        println!(
            "\nGDN STATE ({} pages): content {:.1}% of the page (min {} max {} B), rest zero padding",
            state_pages.len(),
            100.0 * content as f64 / total as f64,
            lens.iter().min().unwrap(),
            lens.iter().max().unwrap()
        );
        // element size: entropy of byte planes for 2 and 4
        for size in [2usize, 4] {
            let mut hs = vec![[0u64; 256]; size];
            for (pg, n) in &state_pages {
                for e in pg[..n / size * size].chunks_exact(size) {
                    for b in 0..size {
                        hs[b][e[b] as usize] += 1;
                    }
                }
            }
            let es: Vec<String> = hs.iter().map(|h| format!("{:.2}", entropy(h))).collect();
            let mean = hs.iter().map(|h| entropy(h)).sum::<f64>() / size as f64;
            println!("  as {size}-byte elements, byte-plane entropies (low..high byte): [{}] -> ideal {:.3}x", es.join(", "), 8.0 / mean);
        }
        // fp16 / bf16 / fp32 plausibility: fraction of huge or non-finite values
        let mut f16_bad = 0u64;
        let mut f32_bad = 0u64;
        let mut n16 = 0u64;
        let mut n32 = 0u64;
        for (pg, n) in &state_pages {
            for e in pg[..n / 4 * 4].chunks_exact(4) {
                let v = f32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                n32 += 1;
                if !v.is_finite() || v.abs() > 1e6 || (v != 0.0 && v.abs() < 1e-30) {
                    f32_bad += 1;
                }
                for h in e.chunks_exact(2) {
                    let bits = u16::from_le_bytes([h[0], h[1]]);
                    n16 += 1;
                    if bits & 0x7c00 == 0x7c00 {
                        f16_bad += 1;
                    }
                }
            }
        }
        println!(
            "  dtype check: as fp32 {:.2}% implausible; as fp16 {:.2}% inf/NaN patterns",
            100.0 * f32_bad as f64 / n32 as f64,
            100.0 * f16_bad as f64 / n16 as f64
        );
        let mut cs = vec![];
        for (pg, n) in &state_pages {
            run_codecs("full page (with padding)", pg, &mut cs, 0);
            run_codecs("content only", &pg[..*n], &mut cs, 4);
            run_codecs("content, 2-byte planes", &planes(&pg[..*n], 2), &mut cs, 8);
            run_codecs("content, 4-byte planes", &planes(&pg[..*n], 4), &mut cs, 12);
        }
        println!("  codecs (single thread, {:.0} MB of pages):", total as f64 / 1e6);
        for c in &cs {
            println!("{}", c.line());
        }
    }
}
