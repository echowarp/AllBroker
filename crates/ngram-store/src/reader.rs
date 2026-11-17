//! Direct (O_DIRECT, no page cache, no mmap) reads of table rows through io_uring. A row may straddle two 4 KiB
//! pages; spans on the same pages are merged, all spans of a gather are submitted at once (up to the ring depth),
//! and the row bytes are copied out of page-aligned buffers.

use io_uring::{opcode, types, IoUring};
use std::alloc::{alloc, dealloc, Layout};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

pub const PAGE: u64 = 4096;

/// Page-aligned scratch memory, grown on demand.
struct Aligned {
    ptr: *mut u8,
    len: usize,
}

// The buffer is owned exclusively by its RowFile (no aliases escape), so moving it to another thread is sound.
unsafe impl Send for Aligned {}

impl Aligned {
    fn new() -> Self {
        Aligned { ptr: std::ptr::null_mut(), len: 0 }
    }
    fn ensure(&mut self, len: usize) {
        if len <= self.len {
            return;
        }
        let len = len.next_power_of_two().max(1 << 20);
        self.free();
        self.ptr = unsafe { alloc(Layout::from_size_align(len, PAGE as usize).unwrap()) };
        assert!(!self.ptr.is_null(), "out of memory");
        self.len = len;
    }
    fn free(&mut self) {
        if !self.ptr.is_null() {
            unsafe { dealloc(self.ptr, Layout::from_size_align(self.len, PAGE as usize).unwrap()) };
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

/// One merged read: [begin, end) in the file, landing at `buf_off` in the scratch buffer.
#[derive(Clone, Copy)]
struct Span {
    begin: u64,
    end: u64,
    buf_off: usize,
}

pub struct RowFile {
    file: File,
    base: u64,
    pub row_bytes: usize,
    pub rows: u64,
    ring: IoUring,
    depth: usize,
    buf: Aligned,
    spans: Vec<Span>,
    order: Vec<(u64, u32)>, // (row begin offset, index into the request)
    pub reads_submitted: u64,
    pub bytes_read: u64,
}

impl RowFile {
    pub fn open(path: &Path, base: u64, row_bytes: usize, rows: u64, depth: u32) -> io::Result<RowFile> {
        let file = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path)?;
        Ok(RowFile {
            file,
            base,
            row_bytes,
            rows,
            ring: IoUring::new(depth.next_power_of_two())?,
            depth: depth as usize,
            buf: Aligned::new(),
            spans: Vec::new(),
            order: Vec::new(),
            reads_submitted: 0,
            bytes_read: 0,
        })
    }

    /// Reads the compressed bytes of `rows` into `out` (rows.len() * row_bytes, in request order).
    pub fn read_rows(&mut self, rows: &[u32], out: &mut [u8]) -> io::Result<()> {
        let rb = self.row_bytes as u64;
        assert!(out.len() >= rows.len() * self.row_bytes);
        // Sort by file position, merge page spans that touch or overlap.
        self.order.clear();
        for (i, &r) in rows.iter().enumerate() {
            if r as u64 >= self.rows {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "row out of range"));
            }
            self.order.push((self.base + r as u64 * rb, i as u32));
        }
        self.order.sort_unstable();
        self.spans.clear();
        let mut total = 0usize;
        for &(off, _) in &self.order {
            let b = off & !(PAGE - 1);
            let e = (off + rb).div_ceil(PAGE) * PAGE;
            match self.spans.last_mut() {
                Some(s) if b <= s.end => s.end = s.end.max(e),
                _ => {
                    if let Some(s) = self.spans.last() {
                        total = s.buf_off + (s.end - s.begin) as usize;
                    }
                    self.spans.push(Span { begin: b, end: e, buf_off: total });
                }
            }
        }
        if let Some(s) = self.spans.last() {
            total = s.buf_off + (s.end - s.begin) as usize;
        }
        self.buf.ensure(total);
        // Submit in waves of at most `depth` reads.
        let fd = types::Fd(self.file.as_raw_fd());
        let mut next = 0;
        while next < self.spans.len() {
            let wave = (self.spans.len() - next).min(self.depth);
            for (k, s) in self.spans[next..next + wave].iter().enumerate() {
                let len = (s.end - s.begin) as u32;
                let e = opcode::Read::new(fd, unsafe { self.buf.ptr.add(s.buf_off) }, len)
                    .offset(s.begin)
                    .build()
                    .user_data((next + k) as u64);
                unsafe { self.ring.submission().push(&e).map_err(|_| io::Error::other("sq full"))? };
            }
            self.ring.submit_and_wait(wave)?;
            let mut done = 0;
            while done < wave {
                let next_cqe = self.ring.completion().next();
                let cqe = match next_cqe {
                    Some(c) => c,
                    None => {
                        self.ring.submit_and_wait(1)?;
                        continue;
                    }
                };
                let s = self.spans[cqe.user_data() as usize];
                let want = (s.end - s.begin) as i32;
                if cqe.result() < 0 {
                    return Err(io::Error::from_raw_os_error(-cqe.result()));
                }
                // A short read is only legal at the end of the file, past the last row.
                if cqe.result() < want && s.end <= self.base + self.rows * rb {
                    return Err(io::Error::other("short read"));
                }
                done += 1;
            }
            self.reads_submitted += wave as u64;
            self.bytes_read += self.spans[next..next + wave].iter().map(|s| s.end - s.begin).sum::<u64>();
            next += wave;
        }
        // Copy every requested row out of its span.
        let mut si = 0;
        for &(off, idx) in &self.order {
            while self.spans[si].end < off + rb {
                si += 1;
            }
            let s = self.spans[si];
            let src = unsafe { std::slice::from_raw_parts(self.buf.ptr.add(s.buf_off + (off - s.begin) as usize), self.row_bytes) };
            out[idx as usize * self.row_bytes..(idx as usize + 1) * self.row_bytes].copy_from_slice(src);
        }
        Ok(())
    }
}

/// Reads [base, base + dst.len()) of `path` into `dst` with O_DIRECT (no page cache, no mmap), 64 MiB at a time: a
/// whole table loaded into RAM (the broker's resident tables). `keep_going` is asked between chunks (false: Err).
pub fn read_range(path: &Path, base: u64, dst: &mut [u8], keep_going: &dyn Fn() -> bool) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    const CHUNK: u64 = 64 << 20;
    let file = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path)?;
    let mut buf = Aligned::new();
    buf.ensure((CHUNK + 2 * PAGE) as usize);
    let end = base + dst.len() as u64;
    let mut at = base;
    while at < end {
        if !keep_going() {
            return Err(io::Error::other("stopped"));
        }
        let stop = (at + CHUNK).min(end);
        let b = at & !(PAGE - 1);
        let e = stop.div_ceil(PAGE) * PAGE;
        let want = (e - b) as usize;
        let mut got = 0usize;
        while got < want {
            let s = unsafe { std::slice::from_raw_parts_mut(buf.ptr.add(got), want - got) };
            let n = file.read_at(s, b + got as u64)?;
            if n == 0 {
                break; // end of file (the last page may be short)
            }
            got += n;
        }
        let take = (stop - at) as usize;
        if got < (at - b) as usize + take {
            return Err(io::Error::other("short read"));
        }
        let src = unsafe { std::slice::from_raw_parts(buf.ptr.add((at - b) as usize), take) };
        dst[(at - base) as usize..(at - base) as usize + take].copy_from_slice(src);
        at = stop;
    }
    Ok(())
}
