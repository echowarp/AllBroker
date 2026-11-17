//! Bins file (`.ngbins`): the bundle data for one layout. Exact copies of table rows, so serving a row from a
//! bundle gives the same bytes as reading the table.
//!
//! ```text
//! [ header, 16 KiB ][ 16 KiB bins ... ][ 4 KiB bins ... ][ directory: sorted (u64 key, u32 bin) ]
//! ```
//! A bin is one aligned read: u32 count | u32 reserved | count x u64 key | count x bundle. A bundle is the
//! `heads_per_ngram` rows of its key in head order (`hash::rows_of_key`). Keys: bigram plain, trigram | `layout::TRI`.
//! The directory points each key at its primary bin; it is read into RAM when the file is opened.

use crate::hash::{self, PleParams};
use crate::layout::{Layout, BIN_BYTES, PAGE_BYTES, TRI};
use crate::reader::RowFile;
use io_uring::{opcode, types, IoUring};
use std::alloc::{alloc_zeroed, dealloc, Layout as AllocLayout};
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

const MAGIC: &[u8; 4] = b"NGBF";
const VERSION: u32 = 1;
const HEADER_BYTES: u64 = BIN_BYTES as u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub row_bytes: u32,
    pub heads_per_ngram: u32,
    pub table_rows: u64,
    pub multipliers: [u64; hash::MAX_NGRAM],
    pub n16: u32,
    pub n4: u32,
    pub dir_off: u64,
    pub dir_n: u64,
}

impl Header {
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(HEADER_BYTES as usize);
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&self.row_bytes.to_le_bytes());
        b.extend_from_slice(&self.heads_per_ngram.to_le_bytes());
        b.extend_from_slice(&self.table_rows.to_le_bytes());
        for m in self.multipliers {
            b.extend_from_slice(&m.to_le_bytes());
        }
        b.extend_from_slice(&self.n16.to_le_bytes());
        b.extend_from_slice(&self.n4.to_le_bytes());
        b.extend_from_slice(&self.dir_off.to_le_bytes());
        b.extend_from_slice(&self.dir_n.to_le_bytes());
        b.resize(HEADER_BYTES as usize, 0);
        b
    }
    fn decode(b: &[u8]) -> io::Result<Header> {
        if b.len() < 80 || &b[0..4] != MAGIC || u32::from_le_bytes(b[4..8].try_into().unwrap()) != VERSION {
            return Err(io::Error::other("not an ngbins v1 file"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        Ok(Header {
            row_bytes: u32_at(8),
            heads_per_ngram: u32_at(12),
            table_rows: u64_at(16),
            multipliers: [u64_at(24), u64_at(32), u64_at(40)],
            n16: u32_at(48),
            n4: u32_at(52),
            dir_off: u64_at(56),
            dir_n: u64_at(64),
        })
    }
    pub fn bundle_bytes(&self) -> usize {
        (self.row_bytes * self.heads_per_ngram) as usize
    }
    pub fn bins(&self) -> u32 {
        self.n16 + self.n4
    }
    /// File offset and length of bin `b`.
    pub fn bin_range(&self, b: u32) -> (u64, usize) {
        if b < self.n16 {
            (HEADER_BYTES + b as u64 * BIN_BYTES as u64, BIN_BYTES)
        } else {
            (HEADER_BYTES + self.n16 as u64 * BIN_BYTES as u64 + (b - self.n16) as u64 * PAGE_BYTES as u64, PAGE_BYTES)
        }
    }
}

/// Order of a key's n-gram (2 = bigram, 3 = trigram) and the key without its tag.
pub fn split_key(k: u64) -> (u64, usize) {
    if k & TRI != 0 { (k & !TRI, 3) } else { (k, 2) }
}

/// Writes the bins file for `layout`, copying every bundle's rows from `table`. Bins keep the layout's order
/// (16 KiB bins first, hottest first, then 4 KiB bins), so a file prefix is the hottest data.
pub fn write(layout: &Layout, p: &PleParams, table: &mut RowFile, out: &Path) -> io::Result<Header> {
    let hpn = p.heads_per_ngram as usize;
    let rb = table.row_bytes;
    if layout.bundle_bytes != rb * hpn {
        return Err(io::Error::other(format!("layout bundles are {} B, table bundles {} B", layout.bundle_bytes, rb * hpn)));
    }
    let n16 = layout.small.iter().take_while(|s| !**s).count() as u32;
    if layout.small[n16 as usize..].iter().any(|s| !*s) {
        return Err(io::Error::other("layout: 16 KiB bins must come before 4 KiB bins"));
    }
    let n4 = layout.bins.len() as u32 - n16;
    let mut h = Header { row_bytes: rb as u32, heads_per_ngram: hpn as u32, table_rows: table.rows,
                         multipliers: p.multipliers, n16, n4, dir_off: 0, dir_n: 0 };
    let mut w = BufWriter::with_capacity(8 << 20, File::create(out)?);
    w.write_all(&h.encode())?;
    // Bins in waves: gather the rows of many bins in one direct read, then lay the bins out.
    let mut rows = [0u32; 8];
    let mut want: Vec<u32> = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    let mut bin = Vec::with_capacity(BIN_BYTES);
    const WAVE: usize = 256;
    for wave in layout.bins.chunks(WAVE).enumerate().map(|(i, c)| (i * WAVE, c)) {
        let (first, bins) = wave;
        want.clear();
        for keys in bins {
            for &k in keys {
                let (key, order) = split_key(k);
                hash::rows_of_key(p, key, order, &mut rows[..hpn]);
                want.extend_from_slice(&rows[..hpn]);
            }
        }
        data.resize(want.len() * rb, 0);
        table.read_rows(&want, &mut data)?;
        let mut at = 0usize;
        for (i, keys) in bins.iter().enumerate() {
            let size = h.bin_range((first + i) as u32).1;
            let need = 8 + keys.len() * (8 + h.bundle_bytes());
            if need > size {
                return Err(io::Error::other(format!("bin {} holds {} B in {} B", first + i, need, size)));
            }
            bin.clear();
            bin.extend_from_slice(&(keys.len() as u32).to_le_bytes());
            bin.extend_from_slice(&0u32.to_le_bytes());
            for &k in keys {
                bin.extend_from_slice(&k.to_le_bytes());
            }
            let n = keys.len() * h.bundle_bytes();
            bin.extend_from_slice(&data[at..at + n]);
            at += n;
            bin.resize(size, 0);
            w.write_all(&bin)?;
        }
    }
    let mut dir: Vec<(u64, u32)> = layout.dir.iter().map(|(&k, &b)| (k, b)).collect();
    dir.sort_unstable();
    h.dir_off = HEADER_BYTES + n16 as u64 * BIN_BYTES as u64 + n4 as u64 * PAGE_BYTES as u64;
    h.dir_n = dir.len() as u64;
    for (k, b) in &dir {
        w.write_all(&k.to_le_bytes())?;
        w.write_all(&b.to_le_bytes())?;
    }
    let mut f = w.into_inner().map_err(|e| e.into_error())?;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&h.encode())?;
    f.sync_all()?;
    Ok(h)
}

/// Page-aligned buffer for O_DIRECT bin reads.
struct Aligned {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for Aligned {}
impl Aligned {
    fn ensure(&mut self, len: usize) {
        if len <= self.len {
            return;
        }
        let len = len.next_power_of_two().max(1 << 20);
        self.free();
        self.ptr = unsafe { alloc_zeroed(AllocLayout::from_size_align(len, PAGE_BYTES).unwrap()) };
        assert!(!self.ptr.is_null(), "out of memory");
        self.len = len;
    }
    fn free(&mut self) {
        if !self.ptr.is_null() {
            unsafe { dealloc(self.ptr, AllocLayout::from_size_align(self.len, PAGE_BYTES).unwrap()) };
            self.ptr = std::ptr::null_mut();
            self.len = 0;
        }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        self.free();
    }
}

/// An open bins file: header, directory in RAM, and an io_uring for O_DIRECT bin reads (submit, then wait, so
/// row reads from the table can run in between).
pub struct BinFile {
    pub h: Header,
    file: File,
    keys: Vec<u64>,
    dir_bins: Vec<u32>,
    ring: IoUring,
    depth: usize,
    buf: Aligned,
    /// bins of the reads in flight and their offsets in `buf`
    pending: Vec<(u32, usize)>,
    submitted: usize,
    pub reads_16k: u64,
    pub reads_4k: u64,
}

impl BinFile {
    pub fn open(path: &Path, depth: u32) -> io::Result<BinFile> {
        let mut f = File::open(path)?;
        let mut hb = vec![0u8; HEADER_BYTES as usize];
        f.read_exact(&mut hb)?;
        let h = Header::decode(&hb)?;
        // The directory is read once (buffered), then dropped from the page cache.
        f.seek(SeekFrom::Start(h.dir_off))?;
        let mut raw = vec![0u8; h.dir_n as usize * 12];
        f.read_exact(&mut raw)?;
        unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        let mut keys = Vec::with_capacity(h.dir_n as usize);
        let mut dir_bins = Vec::with_capacity(h.dir_n as usize);
        for e in raw.chunks_exact(12) {
            keys.push(u64::from_le_bytes(e[0..8].try_into().unwrap()));
            dir_bins.push(u32::from_le_bytes(e[8..12].try_into().unwrap()));
        }
        if keys.windows(2).any(|w| w[0] >= w[1]) || dir_bins.iter().any(|&b| b >= h.bins()) {
            return Err(io::Error::other("ngbins directory is not sorted or points past the bins"));
        }
        let file = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path)?;
        Ok(BinFile { h, file, keys, dir_bins, ring: IoUring::new(depth.next_power_of_two())?, depth: depth as usize,
                     buf: Aligned { ptr: std::ptr::null_mut(), len: 0 }, pending: Vec::new(), submitted: 0,
                     reads_16k: 0, reads_4k: 0 })
    }

    /// Checks that the file was built for this table and hash.
    pub fn matches(&self, p: &PleParams, row_bytes: usize, rows: u64) -> Result<(), String> {
        if self.h.row_bytes as usize != row_bytes || self.h.table_rows != rows || self.h.multipliers != p.multipliers
            || self.h.heads_per_ngram != p.heads_per_ngram
        {
            return Err(format!("bins file is for {} rows x {} B (hash {:x?}), table has {} x {} B (hash {:x?})",
                               self.h.table_rows, self.h.row_bytes, self.h.multipliers, rows, row_bytes, p.multipliers));
        }
        Ok(())
    }

    pub fn keys(&self) -> usize {
        self.keys.len()
    }
    /// Bytes of RAM the directory takes.
    pub fn dir_bytes(&self) -> u64 {
        self.keys.capacity() as u64 * 8 + self.dir_bins.capacity() as u64 * 4
    }

    pub fn lookup(&self, key: u64) -> Option<u32> {
        self.keys.binary_search(&key).ok().map(|i| self.dir_bins[i])
    }

    /// Submits reads of `bins` (distinct). Pair with `wait`.
    pub fn submit(&mut self, bins: &[u32]) -> io::Result<()> {
        assert!(self.pending.is_empty(), "previous bin reads not waited for");
        let total: usize = bins.iter().map(|&b| self.h.bin_range(b).1).sum();
        self.buf.ensure(total);
        let mut at = 0usize;
        for &b in bins {
            self.pending.push((b, at));
            at += self.h.bin_range(b).1;
        }
        self.submitted = 0;
        self.push_wave()
    }

    fn push_wave(&mut self) -> io::Result<()> {
        let fd = types::Fd(self.file.as_raw_fd());
        let end = (self.submitted + self.depth).min(self.pending.len());
        for i in self.submitted..end {
            let (b, at) = self.pending[i];
            let (off, len) = self.h.bin_range(b);
            let e = opcode::Read::new(fd, unsafe { self.buf.ptr.add(at) }, len as u32).offset(off).build().user_data(i as u64);
            unsafe { self.ring.submission().push(&e).map_err(|_| io::Error::other("sq full"))? };
            if len == BIN_BYTES { self.reads_16k += 1 } else { self.reads_4k += 1 }
        }
        let n = end - self.submitted;
        self.submitted = end;
        if n > 0 {
            self.ring.submit()?;
        }
        Ok(())
    }

    /// Waits for the submitted reads; calls `each(bin, keys, bundles)` for every bin read.
    pub fn wait(&mut self, mut each: impl FnMut(u32, &[u64], &[u8])) -> io::Result<()> {
        let mut done = 0usize;
        let mut err: Option<io::Error> = None;
        while done < self.pending.len() {
            let in_flight = self.submitted - done;
            if in_flight == 0 {
                self.push_wave()?;
                continue;
            }
            self.ring.submit_and_wait(1)?;
            while let Some(cqe) = self.ring.completion().next() {
                let (b, _) = self.pending[cqe.user_data() as usize];
                if cqe.result() != self.h.bin_range(b).1 as i32 && err.is_none() {
                    err = Some(if cqe.result() < 0 { io::Error::from_raw_os_error(-cqe.result()) }
                               else { io::Error::other("short bin read") });
                }
                done += 1;
            }
            if done == self.submitted && self.submitted < self.pending.len() {
                self.push_wave()?;
            }
        }
        let pending = std::mem::take(&mut self.pending);
        if let Some(e) = err {
            return Err(e);
        }
        let bb = self.h.bundle_bytes();
        for (b, at) in pending {
            let len = self.h.bin_range(b).1;
            let bin = unsafe { std::slice::from_raw_parts(self.buf.ptr.add(at), len) };
            let n = u32::from_le_bytes(bin[0..4].try_into().unwrap()) as usize;
            if 8 + n * (8 + bb) > len {
                return Err(io::Error::other(format!("bin {b}: corrupt count {n}")));
            }
            let keys: Vec<u64> = bin[8..8 + 8 * n].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
            each(b, &keys, &bin[8 + 8 * n..8 + 8 * n + n * bb]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip_and_ranges() {
        let h = Header { row_bytes: 90, heads_per_ngram: 8, table_rows: 320_001_536, multipliers: [3, 5, 7], n16: 10,
                         n4: 4, dir_off: 12345, dir_n: 99 };
        assert_eq!(Header::decode(&h.encode()).unwrap(), h);
        assert_eq!(h.bin_range(0), (16384, 16384));
        assert_eq!(h.bin_range(9), (16384 * 10, 16384));
        assert_eq!(h.bin_range(10), (16384 * 11, 4096));
        assert_eq!(h.bin_range(13), (16384 * 11 + 3 * 4096, 4096));
    }
}
