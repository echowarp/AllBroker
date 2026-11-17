//! Owned shared mappings of the pool files (memmap2; the client's PoolMap / SlotRange is the same design).
//! A mapping is unmapped exactly once when its `Shm` goes; its bytes are reached through `Span`s that borrow the
//! `Shm`, so a span cannot outlive the mapping (compile error). `MmapRaw`: engines write this memory from other
//! processes, so no Rust reference (`&[u8]`) over it is ever formed: spans only copy in and out, bounds-checked.
use std::fs::File;
use std::io;
use std::marker::PhantomData;

pub struct Shm {
    map: memmap2::MmapRaw,
    len: usize,
}

impl Shm {
    /// Map `len` bytes of `f` shared, read-write (`write`) or read-only.
    pub fn map(f: &File, len: usize, write: bool) -> io::Result<Shm> {
        let mut o = memmap2::MmapOptions::new();
        o.len(len);
        let map = if write { o.map_raw(f)? } else { o.map_raw_read_only(f)? };
        Ok(Shm { map, len })
    }

    /// `n` bytes at byte offset `off` (bounds-checked).
    ///
    /// # Safety
    /// The caller must own the bytes for what it does with them: the slots of an entry it holds the core lock for,
    /// leases, or reserved for it (the same rules as the engines' reservations and leases).
    pub unsafe fn span(&self, off: usize, n: usize) -> Span<'_> {
        assert!(off.checked_add(n).is_some_and(|e| e <= self.len), "pool span {off}+{n} outside {}", self.len);
        Span { ptr: self.map.as_mut_ptr().add(off), len: n, _map: PhantomData }
    }
}

/// Bytes of a mapping borrowed from a [`Shm`].
pub struct Span<'a> {
    ptr: *mut u8,
    len: usize,
    _map: PhantomData<&'a Shm>,
}

impl Span<'_> {
    /// Copy `out.len()` bytes from `off` into `out`.
    pub fn read(&self, off: usize, out: &mut [u8]) {
        assert!(off.checked_add(out.len()).is_some_and(|e| e <= self.len), "read outside the span");
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.add(off), out.as_mut_ptr(), out.len()) }
    }
    /// The first `n` bytes (at most the span).
    pub fn to_vec(&self, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n.min(self.len)];
        self.read(0, &mut v);
        v
    }
    /// Copy `src` into the span at `off`.
    pub fn write(&self, off: usize, src: &[u8]) {
        assert!(off.checked_add(src.len()).is_some_and(|e| e <= self.len), "write outside the span");
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(off), src.len()) }
    }
    /// Copy all of `src` (another span of the same or another mapping, not overlapping this one) to `off`.
    pub fn copy_from(&self, off: usize, src: &Span<'_>) {
        assert!(off.checked_add(src.len).is_some_and(|e| e <= self.len), "copy outside the span");
        unsafe { std::ptr::copy_nonoverlapping(src.ptr, self.ptr.add(off), src.len) }
    }
}
