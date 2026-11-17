//! Mean of decoded table rows per head (sampled), to compare with the EXL3 checkpoint's head_bias.
//!   ngram-rowstats <model.gguf> <out.f32> [samples_per_head=2000]
use ngram_store::{codec::Codec, gguf::Gguf, hash::PleParams, reader::RowFile, TABLE_TENSOR};
use std::io::Write;
use std::path::Path;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(Path::new(&a[1])).unwrap();
    let p = PleParams::from_gguf(&g).unwrap();
    let t = g.tensors.get(TABLE_TENSOR).unwrap();
    let codec = Codec::from_ggml(t.ggml_type).unwrap();
    let rb = codec.row_bytes(p.head_dim as usize);
    let mut rf = RowFile::open(&t.path, t.file_offset, rb, t.dims[1], 64).unwrap();
    let n: u32 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[2]).unwrap());
    let mut seed = 0x1234_5678u64;
    for h in 0..p.heads as usize {
        let rows: Vec<u32> = (0..n).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            p.head_offsets[h] + (seed % p.head_vocab[h] as u64) as u32 }).collect();
        let mut buf = vec![0u8; rows.len() * rb];
        rf.read_rows(&rows, &mut buf).unwrap();
        let mut mean = vec![0f64; p.head_dim as usize];
        let mut row = vec![0f32; p.head_dim as usize];
        for i in 0..rows.len() {
            codec.decode(&buf[i * rb..(i + 1) * rb], &mut row);
            for (m, v) in mean.iter_mut().zip(&row) { *m += *v as f64 / n as f64; }
        }
        for m in &mean { out.write_all(&(*m as f32).to_le_bytes()).unwrap(); }
    }
}
