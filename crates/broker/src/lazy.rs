//! On-demand pool, gradual eviction and compaction: no extra RAM while it is not used, entries leave gradually,
//! and the broker compacts (defragments) the pool.
//!
//! The pool file keeps its size (engines map it once), but its memory is only allocated for slots that hold entries:
//! a RESERVE (and a disk-tier / hot-set load) allocates the reserved slots with fallocate before the client gets them
//! (so a client never writes into a hole: on a full tmpfs that would be SIGBUS), and every sweep punches the holes of
//! free slots back (FALLOC_FL_PUNCH_HOLE), so the RAM returns to the kernel. `--pool-gib` is the ceiling.
//!
//! The sweep (every `--sweep-s`):
//! - memory pressure (MemAvailable < `--mem-reserve-gib`): evict in policy order until the deficit (+1 GiB) is freed;
//! - age: evict entries unused for `--evict-age-s`, least recently used first, at most `--evict-step-pct` of the live
//!   slots per sweep (gradual);
//! - no clients for `--idle-shrink-s`: evict everything (shrink to zero), then idle exit as before;
//! - before evicting, the SSD tier's write pass runs (admitted entries reach the SSD; never a dump), so evicted entries
//!   that earned it come back from the SSD;
//! - punch the free slots, in batches of `TRIM_BATCH_BYTES` with the core lock released between batches;
//! - compaction when no request came for 3 s: move unleased entries from the top of the pool into free slots below
//!   (lowest first), up to `--compact-mib` per sweep, holding the core lock (no lease can start meanwhile; the SSD
//!   writer is paused), then punch the old slots: the high-water mark (and with huge pages / coarse granularity the
//!   resident set) shrinks. With 4 KiB tmpfs pages the freed slots already return their RAM, so this mostly keeps
//!   the pool's address range dense.
//! Pinning clients (cudaHostRegister / Level Zero import of the whole pool) pin the pages a slot has when they
//! register it; punching that slot later drops those pages from the file while the GPU keeps using them (its stores go
//! where no other engine reads, its loads see stale data). So punching stops for good once a client asks for the pool
//! state (POOL_STATE / POOL_ACK: every pinning client does so before it registers anything), and after a restart on
//! a reused pool nothing is punched for `--pool-reuse-hold-s` (engines that pinned it reconnect within that time)
//! and the backed slots are read back from the file instead of punching it. `--pool-pin-safe off` drops the guard
//! (only for hosts whose clients never pin: then free slots return their memory again). The memory then comes back
//! only through the dynamic-pool controller, whose parking waits for the pinning clients' acks.
use super::Shared;
use kvproto::Range;
use std::fs;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use tracing_mutex::stdsync::Mutex;
use std::time::{Duration, Instant};

pub struct Backing {
    file: fs::File,
    map: super::shm::Shm,
    slot: u64,
    #[allow(dead_code)]
    pool_bytes: u64,
    backed: Vec<bool>,
    pub resident: u64,
    /// Kept allocated ahead (KV must never be written to disk or to
    /// zram): the lowest `keep_warm` free slots are allocated by the broker itself (warm thread), so their memory is
    /// charged to the broker's cgroup (MemorySwapMax=0: never swapped, never in zswap; under pressure the broker
    /// evicts instead). New entries take the lowest free slots (Alloc::low_first), so engines write into memory the
    /// broker allocated instead of faulting in pages charged to their own (swappable) cgroup. 0 = nothing kept.
    pub keep_warm: u64,
    warming: Vec<bool>,
    /// slots the sweep took out of the allocator and is punching (outside the locks): the warm thread must not
    /// fallocate them meanwhile (it would race the punch: `backed` and the file would disagree until the next sweep)
    freeing: Vec<bool>,
    /// last RESERVE on this pool (ms, broker clock): the warm margin is only kept while a client stores
    pub last_reserve_ms: u64,
    /// set once a client that may pin the pool appeared (shared by every class): no hole punching from then on
    pins: Arc<AtomicBool>,
    /// reused pool: no hole punching before this (engines that pinned the old incarnation reconnect meanwhile)
    hold_until: Option<Instant>,
    /// punches skipped by the two rules above (counter)
    pub punches_held: AtomicU64,
}


#[derive(Default)]
pub struct Stats {
    pub sweeps: AtomicU64,
    pub freed_bytes: AtomicU64,
    pub last_freed_bytes: AtomicU64,
    pub evicted_aged: AtomicU64,
    pub evicted_pressure: AtomicU64,
    pub evicted_idle: AtomicU64,
    pub evicted_ceiling: AtomicU64,
    /// pressure protection: bytes protected in the last pressure sweep (gauge), sweeps where the
    /// protection gave way (MemAvailable below the floor), sweeps that stopped at the protected set
    pub protected_bytes: AtomicU64,
    pub protect_shed: AtomicU64,
    pub protect_held: AtomicU64,
    pub compact_moves: AtomicU64,
    pub rcompress_entries: AtomicU64,
    pub rcompress_skipped: AtomicU64,
    pub rcompress_yielded: AtomicU64,
    /// the entry changed between the copy and the rewrite (evicted and stored again): copy dropped
    pub rcompress_raced: AtomicU64,
    /// compression passes skipped because a connected client of the class did not declare slot codec 1 (FEATURES)
    pub rcompress_no_reader: AtomicU64,
    pub rcompress_raw: AtomicU64,
    /// integrity checksums (checksum_loop): bytes hashed, passes that stopped for a request, copies that raced
    pub cksum_bytes: AtomicU64,
    pub cksum_yielded: AtomicU64,
    pub cksum_raced: AtomicU64,
    pub rcompress_stored: AtomicU64,
    pub compact_bytes: AtomicU64,
    pub backing_failures: AtomicU64,
    /// bytes the warm thread allocated ahead (counter)
    pub warm_bytes: AtomicU64,
}

impl Backing {
    /// Opens the pool. A new pool is punched whole (nothing maps it yet). A reused pool (`reuse_hold`: a restart) is
    /// not touched: engines may still map and pin its pages, so the backed slots are read back from the file
    /// (SEEK_DATA) and punching waits `reuse_hold`. Its contents are stale either way: the index starts empty.
    pub fn new(path: &Path, pool_bytes: u64, slot: u64, pins: Arc<AtomicBool>, reuse_hold: Option<Duration>) -> std::io::Result<Backing> {
        let file = fs::OpenOptions::new().read(true).write(true).open(path)?;
        let map = super::shm::Shm::map(&file, pool_bytes as usize, true)?;
        let n = (pool_bytes / slot) as usize;
        let mut b = Backing { file, map, slot, pool_bytes, backed: vec![false; n], resident: 0, keep_warm: 0,
                              warming: vec![false; n], freeing: vec![false; n], last_reserve_ms: 0, pins, hold_until: reuse_hold.map(|d| Instant::now() + d),
                              punches_held: AtomicU64::new(0) };
        if reuse_hold.is_some() {
            for (s, e) in data_extents(&b.file, pool_bytes)? {
                for i in (s / slot) as usize..(e.div_ceil(slot) as usize).min(n) {
                    b.backed[i] = true;
                }
            }
            b.resident = b.backed.iter().filter(|x| **x).count() as u64 * slot;
        } else {
            b.punch(0, pool_bytes)?;
        }
        Ok(b)
    }

    /// Hole punching is allowed now (see the module notes on pinning clients).
    pub fn may_punch(&self) -> bool {
        !self.pins.load(Relaxed) && self.hold_until.is_none_or(|t| Instant::now() >= t)
    }

    /// Ok(false): not punched (a pinning client may hold these pages, or a reused pool's hold is on).
    fn punch(&self, off: u64, len: u64) -> std::io::Result<bool> {
        if !self.may_punch() {
            self.punches_held.fetch_add(1, Relaxed);
            return Ok(false);
        }
        let r = unsafe {
            libc::fallocate(self.file.as_raw_fd(), libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, off as libc::off_t, len as libc::off_t)
        };
        if r != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(true)
    }

    /// The pool's file system can take these slots (plus 1 GiB): statvfs, no allocation (fast enough for a RESERVE).
    pub fn room_for(&mut self, ranges: &[Range]) -> bool {
        if super::fp::hit("lazy-room-for") {
            return false;
        }
        let need: u64 = ranges.iter().filter(|r| !self.backed[r.start as usize]).map(|r| r.len as u64 * self.slot).sum();
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstatvfs(self.file.as_raw_fd(), &mut st) } != 0 {
            return true;
        }
        st.f_bavail as u64 * st.f_frsize as u64 >= need + (1u64 << 30)
    }

    /// Allocate the memory of these slots (before a client or loader writes them). Err: the pool's file system is full.
    pub fn ensure(&mut self, ranges: &[Range]) -> std::io::Result<()> {
        if super::fp::hit("lazy-ensure") {
            return Err(std::io::Error::from_raw_os_error(libc::ENOSPC));
        }
        for r in ranges {
            let mut s = r.start as usize;
            let end = (r.start + r.len) as usize;
            while s < end {
                if self.backed[s] {
                    s += 1;
                    continue;
                }
                let mut e = s;
                while e < end && !self.backed[e] {
                    e += 1;
                }
                let (off, len) = (s as u64 * self.slot, (e - s) as u64 * self.slot);
                let rc = unsafe { libc::posix_fallocate(self.file.as_raw_fd(), off as libc::off_t, len as libc::off_t) };
                if rc != 0 {
                    let _ = self.punch(off, len);
                    return Err(std::io::Error::from_raw_os_error(rc));
                }
                for b in &mut self.backed[s..e] {
                    *b = true;
                }
                self.resident += len;
                s = e;
            }
        }
        Ok(())
    }

    /// Punch these free runs (all of them: a client may have allocated pages behind the broker's back, e.g. a
    /// connector that fallocates the pool at connect); returns the bytes the pool file gave back.
    #[allow(dead_code)]
    pub fn trim(&mut self, free: &[Range]) -> u64 {
        let free = self.past_warm(free);
        self.punch_runs(&free)
    }

    /// trim() for one batch: the free slots at or after slot `from`, at most `max_bytes` of them (at least one slot).
    /// The warm margin is worked out on the whole free list, as in trim(). Returns the bytes freed and the first slot
    /// of the next batch (None: the end of the free list was reached).
    #[cfg(test)]
    pub fn trim_from(&mut self, free: &[Range], from: u32, max_bytes: u64) -> (u64, Option<u32>) {
        let (part, next) = self.plan_from(free, from, max_bytes);
        (self.punch_runs(&part), next)
    }

    /// The runs trim_from() would punch, and where the next batch starts.
    pub fn plan_from(&self, free: &[Range], from: u32, max_bytes: u64) -> (Vec<Range>, Option<u32>) {
        let free = self.past_warm(free);
        let mut left = (max_bytes / self.slot).max(1) as u32;
        let (mut part, mut next) = (Vec::new(), None);
        for r in &free {
            let (s, e) = (r.start.max(from), r.start + r.len);
            if s >= e {
                continue;
            }
            if left == 0 {
                next = Some(s);
                break;
            }
            let n = (e - s).min(left);
            part.push(Range { start: s, len: n });
            left -= n;
            if s + n < e {
                next = Some(s + n);
                break;
            }
        }
        (part, next)
    }

    /// The sweep took `part` out of the allocator and will punch it: the warm thread skips these slots until
    /// `unmark_freeing`. Called under the pool lock in the same hold that planned the batch (no warm plan can slip in).
    fn mark_freeing(&mut self, part: &[Range]) {
        for r in part {
            for i in r.start as usize..(r.start + r.len) as usize {
                self.freeing[i] = true;
            }
        }
    }

    fn unmark_freeing(&mut self, part: &[Range]) {
        for r in part {
            for i in r.start as usize..(r.start + r.len) as usize {
                self.freeing[i] = false;
            }
        }
    }

    /// Bookkeeping after `part` was punched outside the pool lock (see `punch_unlocked`): the slots are not backed.
    /// A slot the warm thread started allocating meanwhile keeps its flag (its own pages are its business).
    fn punched(&mut self, part: &[Range]) {
        for r in part {
            for i in r.start as usize..(r.start + r.len) as usize {
                if self.backed[i] && !self.warming[i] {
                    self.backed[i] = false;
                    self.resident -= self.slot;
                }
            }
        }
    }

    #[allow(dead_code)]
    fn punch_runs(&mut self, free: &[Range]) -> u64 {
        let before = self.file_bytes();
        for r in free {
            if !matches!(self.punch(r.start as u64 * self.slot, r.len as u64 * self.slot), Ok(true)) {
                continue;   // pages kept: the slots stay backed
            }
            for b in &mut self.backed[r.start as usize..(r.start + r.len) as usize] {
                if *b {
                    *b = false;
                    self.resident -= self.slot;
                }
            }
        }
        before.saturating_sub(self.file_bytes())
    }

    /// `free` (address order) minus the warm margin: the lowest `keep_warm` free slots that the broker allocated
    /// (or is allocating) stay; a slot in the margin that a client faulted in itself is punched, so the warm thread
    /// allocates it again as the broker's.
    fn past_warm(&self, free: &[Range]) -> Vec<Range> {
        let mut out = Vec::new();
        let mut left = self.keep_warm;
        for r in free {
            let mut run: Option<(u32, u32)> = None;
            for s in r.start..r.start + r.len {
                // a slot the warm thread is allocating right now is never punched (its fallocate may be in flight)
                let keep = self.warming[s as usize] || (left > 0 && self.backed[s as usize]);
                if left > 0 {
                    left -= 1;
                }
                if keep {
                    if let Some((a, n)) = run.take() {
                        out.push(Range { start: a, len: n });
                    }
                } else {
                    run = match run {
                        Some((a, n)) => Some((a, n + 1)),
                        None => Some((s, 1)),
                    };
                }
            }
            if let Some((a, n)) = run {
                out.push(Range { start: a, len: n });
            }
        }
        out
    }

    /// Warm margin, step 1 (under the lock): the unallocated slots among the lowest `keep_warm` free ones, marked
    /// as being allocated; the caller fallocates them with `fd()` outside the lock, then calls `warm_done`.
    pub fn warm_plan(&mut self, free: &[Range]) -> Vec<Range> {
        let mut out: Vec<Range> = Vec::new();
        let mut left = self.keep_warm;
        'runs: for r in free {
            for s in r.start..r.start + r.len {
                if left == 0 {
                    break 'runs;
                }
                left -= 1;
                let i = s as usize;
                if self.backed[i] || self.warming[i] || self.freeing[i] {
                    continue;   // allocated, being allocated, or being punched by the sweep
                }
                self.warming[i] = true;
                match out.last_mut() {
                    Some(l) if l.start + l.len == s => l.len += 1,
                    _ => out.push(Range { start: s, len: 1 }),
                }
            }
        }
        out
    }

    /// Warm margin, step 2: `runs` from `warm_plan` were fallocated (`ok`) or not.
    pub fn warm_done(&mut self, runs: &[Range], ok: &[bool]) {
        for (r, &ok) in runs.iter().zip(ok) {
            for s in r.start..r.start + r.len {
                let i = s as usize;
                self.warming[i] = false;
                if ok && !self.backed[i] {
                    self.backed[i] = true;
                    self.resident += self.slot;
                }
            }
        }
    }

    pub fn fd(&self) -> i32 {
        self.file.as_raw_fd()
    }

    pub fn slot_bytes(&self) -> u64 {
        self.slot
    }

    #[allow(dead_code)]
    fn trim_backed_only(&mut self, free: &[Range]) -> u64 {
        let mut freed = 0;
        for r in free {
            let mut s = r.start as usize;
            let end = (r.start + r.len) as usize;
            while s < end {
                if !self.backed[s] {
                    s += 1;
                    continue;
                }
                let mut e = s;
                while e < end && self.backed[e] {
                    e += 1;
                }
                let (off, len) = (s as u64 * self.slot, (e - s) as u64 * self.slot);
                if matches!(self.punch(off, len), Ok(true)) {
                    for b in &mut self.backed[s..e] {
                        *b = false;
                    }
                    self.resident -= len;
                    freed += len;
                }
                s = e;
            }
        }
        freed
    }

    /// Bytes the pool file really holds (tmpfs blocks).
    /// (Resident = what the pool file really holds: clients allocate pages by writing them.)
    pub fn file_bytes(&self) -> u64 {
        use std::os::unix::fs::MetadataExt;
        self.file.metadata().map(|m| m.blocks() * 512).unwrap_or(0)
    }


    /// A copy of `len` bytes at slot `slot` (the in-place compressor works on it outside the core lock).
    pub fn read(&self, slot: u32, len: usize) -> Vec<u8> {
        let off = slot as u64 * self.slot;
        // the caller holds the core lock's go-ahead for this entry (compress_candidates / begin_rewrite)
        unsafe { self.map.span(off as usize, len) }.to_vec(len)
    }

    /// `len` bytes at byte offset `off` of the pool (the integrity checksummer: copy_guard decides afterwards whether
    /// the copy described the entry).
    pub fn read_at(&self, off: u64, len: usize) -> Vec<u8> {
        unsafe { self.map.span(off as usize, len) }.to_vec(len)
    }

    /// What to punch after a `len`-byte image was written at the start of slot `slot` (the rest of the slot):
    /// (fd, offset, length). None: nothing to punch (the image fills the slot, or punching is held: counted).
    pub fn tail_range(&self, slot: u32, len: u64) -> Option<(i32, u64, u64)> {
        let keep = len.div_ceil(4096) * 4096;
        if keep >= self.slot {
            return None;
        }
        if !self.may_punch() {
            self.punches_held.fetch_add(1, Relaxed);
            return None;
        }
        Some((self.fd(), slot as u64 * self.slot + keep, self.slot - keep))
    }

    /// Write a compressed image at the start of slot `slot` (the caller punches the rest with `punch_tail`).
    pub fn write_slot_image(&self, slot: u32, z: &[u8]) {
        unsafe { self.map.span((slot as u64 * self.slot) as usize, z.len()) }.write(0, z);
    }

    /// Copy slot runs `from` -> `to` (same total length; disjoint).
    pub fn copy(&self, from: &[Range], to: &[Range]) {
        let flat = |rs: &[Range]| -> Vec<(u64, u64)> { rs.iter().map(|r| (r.start as u64 * self.slot, r.len as u64 * self.slot)).collect() };
        let (src, dst) = (flat(from), flat(to));
        let (mut i, mut io, mut j, mut jo) = (0usize, 0u64, 0usize, 0u64);
        while i < src.len() && j < dst.len() {
            let n = (src[i].1 - io).min(dst[j].1 - jo);
            // under the core lock: the entry is unleased, its new slots allocated for it (relocation)
            let (a, b) = unsafe { (self.map.span((src[i].0 + io) as usize, n as usize), self.map.span((dst[j].0 + jo) as usize, n as usize)) };
            b.copy_from(0, &a);
            io += n;
            jo += n;
            if io == src[i].1 {
                i += 1;
                io = 0;
            }
            if jo == dst[j].1 {
                j += 1;
                jo = 0;
            }
        }
    }
}

pub struct SweepCfg {
    pub every: Duration,
    pub age_ms: u64,
    pub step_pct: f64,
    pub idle_shrink: Duration,
    pub reserve: u64,
    pub compact_bytes: u64,
    /// unified broker: a namespace (domain) without connected clients ages this many times faster
    pub inactive_div: u64,
    /// pressure protection (0 = off): the sessions used in the last `protect_window_ms` and the shared
    /// class, newest first, up to `protect_bytes`, survive memory-pressure culls; MemAvailable below `protect_floor`
    /// overrides it
    pub protect_bytes: u64,
    pub protect_window_ms: u64,
    pub protect_floor: u64,
}

/// Warm margin upkeep (every 200 ms, warm thread): per pool, while a client stored in the last `active_ms` and
/// MemAvailable is above `reserve`, keep the lowest `warm_bytes` of free slots allocated by the broker; otherwise
/// none (the next sweep punches them).
pub fn warm(sh: &Shared, warm_bytes: u64, active_ms: u64, reserve: u64) {
    let now = sh.now();
    let calm = reserve == 0 || mem_available() >= reserve;
    for i in 0..sh.n_classes() {
        let c = sh.cls(i);
        let Some(bk) = c.backing else { continue };
        // lock order core -> pool (as RESERVE and the sweep): the free runs first, the pool lock after (the
        // reverse order here deadlocked against a RESERVE and stalled the engine)
        let free = c.core.lock().unwrap().free_runs();
        let (plan, fd, slot) = {
            let mut b = bk.lock().unwrap();
            let on = calm && b.last_reserve_ms > 0 && now.saturating_sub(b.last_reserve_ms) <= active_ms;
            b.keep_warm = if on { warm_bytes.div_ceil(b.slot_bytes()).max(1) } else { 0 };
            if b.keep_warm == 0 {
                continue;
            }
            (b.warm_plan(&free), b.fd(), b.slot_bytes())
        };
        if plan.is_empty() {
            continue;
        }
        // allocate outside the pool lock (a 56 MiB slot takes ~10-20 ms; RESERVE must not wait for it)
        let ok: Vec<bool> = plan.iter().map(|r| !super::fp::hit("warm-fallocate") && unsafe {
            libc::posix_fallocate(fd, (r.start as u64 * slot) as libc::off_t, (r.len as u64 * slot) as libc::off_t) == 0
        }).collect();
        sh.lazy.warm_bytes.fetch_add(plan.iter().zip(&ok).filter(|x| *x.1).map(|x| x.0.len as u64 * slot).sum(), Relaxed);
        bk.lock().unwrap().warm_done(&plan, &ok);
    }
}

fn mem_available() -> u64 {
    // test hook: LLMT_KV_MEMAVAIL_FILE holds MemAvailable in bytes
    if let Ok(p) = std::env::var("LLMT_KV_MEMAVAIL_FILE") {
        if let Some(v) = fs::read_to_string(p).ok().and_then(|t| t.trim().parse::<u64>().ok()) {
            return v;
        }
    }
    super::mem_available()
}

/// Bytes of free slots punched per core-lock hold (at least one slot). The punch runs under the core lock (a RESERVE
/// hands out free slots without allocating them, and the client's writes allocate the pages: a slot punched after
/// it was reserved would lose them), and lookups, reserves and commits wait for that lock while a client call times
/// out after a few ms. Punching 1-2 GiB at once (a memory-pressure eviction) held it for longer than that.
/// Punch the tail of a rewritten slot (`Backing::tail_range`) without any lock held: the slot belongs to an entry being
/// rewritten (not free, not leased), so neither the allocator nor the warm thread touches it; `backed` stays set. The
/// pool lock is taken for the checks and the byte count only. Returns the bytes the file gave back.
pub fn punch_tail(bk: &Mutex<Backing>, slot: u32, len: u64) -> u64 {
    let (args, before) = {
        let b = bk.lock().unwrap();
        (b.tail_range(slot, len), b.file_bytes())
    };
    let Some((fd, off, n)) = args else { return 0 };
    unsafe { libc::fallocate(fd, libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, off as libc::off_t, n as libc::off_t) };
    before.saturating_sub(bk.lock().unwrap().file_bytes())
}

pub const TRIM_BATCH_BYTES: u64 = 64 << 20;

/// Punch `part` (slots the caller took out of the allocator, so nothing can hand them out) without holding the pool
/// lock: the lock is taken for the checks and the bookkeeping, not for the fallocate (~10 ms per 56 MiB slot of
/// tmpfs, long enough to time out a 2 ms client lookup on a lock held meanwhile). Returns the bytes the file gave back.
pub fn punch_unlocked(bk: &Mutex<Backing>, part: &[Range]) -> u64 {
    let (fd, slot, before) = {
        let b = bk.lock().unwrap();
        if !b.may_punch() {
            b.punches_held.fetch_add(1, Relaxed);
            drop(b);
            bk.lock().unwrap().unmark_freeing(part);
            return 0;
        }
        (b.fd(), b.slot_bytes(), b.file_bytes())
    };
    let mut done = Vec::new();
    for r in part {
        let rc = unsafe {
            libc::fallocate(fd, libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, (r.start as u64 * slot) as libc::off_t, (r.len as u64 * slot) as libc::off_t)
        };
        if rc == 0 {
            done.push(*r);   // else: pages kept, the slots stay backed
        }
    }
    let mut b = bk.lock().unwrap();
    b.punched(&done);
    b.unmark_freeing(part);
    before.saturating_sub(b.file_bytes())
}

/// Punch a pool's free slots (Backing::trim) in batches without holding the core lock for any punch: `lock_free`
/// takes the lock that guards the free list and returns its guard with the current free runs; `take` (under that
/// guard) takes the batch's slots out of the allocator and returns the ones it got (slots handed out meanwhile are
/// not among them); the guard is dropped, the batch is punched, and `give` returns the slots to the allocator. A slot
/// is therefore never punched after a RESERVE got it, and never handed out before its punch is done.
pub fn trim_batched<G>(mut lock_free: impl FnMut() -> (G, Vec<Range>), mut take: impl FnMut(&mut G, &[Range]) -> Vec<Range>,
                       mut give: impl FnMut(&[Range]), bk: &Mutex<Backing>, batch: u64) -> u64 {
    let (mut freed, mut from) = (0u64, 0u32);
    loop {
        let (mut guard, free) = lock_free();
        // plan, take and mark in one pool-lock hold (core -> pool order): the warm thread, which plans under the pool
        // lock, sees the slots as being freed from the moment they leave the allocator
        let (got, next) = {
            let mut b = bk.lock().unwrap();
            let (part, next) = b.plan_from(&free, from, batch);
            let got = take(&mut guard, &part);
            b.mark_freeing(&got);
            (got, next)
        };
        drop(guard);
        if !got.is_empty() {
            freed += punch_unlocked(bk, &got);
            give(&got);
        }
        match next {
            Some(n) => {
                from = n;
                std::thread::yield_now();
            }
            None => return freed,
        }
    }
}

fn trim_class(core: &Mutex<kvcore::Core>, bk: &Mutex<Backing>) -> u64 {
    trim_batched(|| {
        let g = core.lock().unwrap();
        let free = g.free_runs();
        (g, free)
    }, |g, part| g.begin_freeing(part), |got| core.lock().unwrap().end_freeing(got), bk, TRIM_BATCH_BYTES)
}

/// One sweep over every pool (slab class). `idle`: no client connected for the idle-shrink time.
pub fn sweep(sh: &Shared, cfg: &SweepCfg, idle: bool) {
    let n = sh.n_classes();
    if !(0..n).any(|i| sh.cls(i).backing.is_some()) {
        return;
    }
    let mem = mem_available();
    let pressure = cfg.reserve > 0 && mem < cfg.reserve;
    let now = sh.now();
    // the SSD write pass first: admitted entries reach the SSD before their RAM copy may go (never a dump)
    for i in 0..n {
        if let Some(d) = sh.cls(i).disk {
            d.flush(sh);
        }
    }
    if pressure || idle {
        // no warm margin under memory pressure or without clients (the warm thread restores it when calm)
        for i in 0..n {
            if let Some(bk) = sh.cls(i).backing {
                bk.lock().unwrap().keep_warm = 0;
            }
        }
    }
    let active = sh.active_ns();
    let mut freed = 0u64;
    // per pool: gradual age eviction (namespaces without clients age `inactive_div` times faster), idle shrink
    for i in 0..n {
        let c = sh.cls(i);
        let Some(bk) = c.backing else { continue };
        // strict SSD mode: what this age pass is about to evict is written first, if its session proved
        // repeated use (never under pressure: that loop below writes nothing)
        if !idle && cfg.age_ms > 0 {
            if let Some((min_resumes, min_span)) = c.disk.and_then(|d| d.strict_rule()) {
                let (age, fast) = (cfg.age_ms, cfg.age_ms / cfg.inactive_div.max(1));
                let before = |ns: u32| now.saturating_sub(if active.contains(&(i, ns)) { age } else { fast });
                let cands = {
                    let core = c.core.lock().unwrap();
                    let (live, _) = core.live_and_high_water();
                    let step = ((live as f64 * cfg.step_pct / 100.0).ceil() as u64).max(1);
                    core.strict_age_candidates(&before, step, min_resumes, min_span)
                };
                if let Some(d) = c.disk {
                    d.write_aged(sh, &cands);
                }
            }
        }
        let mut core = c.core.lock().unwrap();
        if idle {
            let k = core.evict_aged(u64::MAX, 0);
            sh.lazy.evicted_idle.fetch_add(k, Relaxed);
        } else if cfg.age_ms > 0 {
            let (live, _) = core.live_and_high_water();
            let step = ((live as f64 * cfg.step_pct / 100.0).ceil() as u64).max(1);
            let (age, fast) = (cfg.age_ms, cfg.age_ms / cfg.inactive_div.max(1));
            let before = |ns: u32| now.saturating_sub(if active.contains(&(i, ns)) { age } else { fast });
            let k = core.evict_aged_by(&before, step);
            sh.lazy.evicted_aged.fetch_add(k, Relaxed);
        }
        drop(core);
        freed += trim_class(c.core, bk);
    }
    // global: one RAM ceiling and memory pressure, in one heat order across pools (least recently used first)
    let lives: Vec<u64> = (0..n).map(|i| {
        let c = sh.cls(i);
        c.core.lock().unwrap().live_and_high_water().0 * c.hello.slot_bytes
    }).collect();
    let mut total: u64 = lives.iter().sum();
    let mut target = sh.ceiling;
    if pressure {
        target = target.min(total.saturating_sub(cfg.reserve - mem + (1u64 << 30)));
    }
    if total > target {
        let mut guards: Vec<_> = (0..n).map(|i| sh.cls(i).core.lock().unwrap()).collect();
        // pressure protection: the active sessions' chains (+ newest states) and the shared class,
        // newest first, within the budget; cold entries go first and the pool shrinks down to that set
        let mut protect: Vec<std::collections::HashSet<u64>> = vec![Default::default(); n];
        if pressure && cfg.protect_bytes > 0 {
            if mem < cfg.protect_floor {
                sh.lazy.protect_shed.fetch_add(1, Relaxed);
            } else {
                let since = now.saturating_sub(cfg.protect_window_ms);
                let mut all = Vec::new();
                for (i, g) in guards.iter().enumerate() {
                    let sb = sh.cls(i).hello.slot_bytes;
                    for (shared, lu, id, slots) in g.protect_candidates(since) {
                        all.push((shared, lu, i, id, slots * sb));
                    }
                }
                all.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
                let mut used = 0u64;
                for (_, _, i, id, bytes) in all {
                    if used + bytes > cfg.protect_bytes {
                        break;
                    }
                    used += bytes;
                    protect[i].insert(id);
                }
                sh.lazy.protected_bytes.store(used, Relaxed);
            }
        }
        while total > target {
            let Some(i) = (0..n).filter_map(|i| guards[i].oldest_evictable_except(&protect[i]).map(|t| (t, i))).min().map(|x| x.1) else {
                if protect.iter().any(|p| !p.is_empty()) {
                    sh.lazy.protect_held.fetch_add(1, Relaxed);
                }
                break;
            };
            let k = guards[i].evict_oldest_except(&protect[i]);
            total = total.saturating_sub(k * sh.cls(i).hello.slot_bytes);
            if pressure {
                sh.lazy.evicted_pressure.fetch_add(k, Relaxed);
            } else {
                sh.lazy.evicted_ceiling.fetch_add(k, Relaxed);
            }
        }
        drop(guards);                           // punch outside the eviction's all-class hold, a batch at a time
        for i in 0..n {
            if let Some(bk) = sh.cls(i).backing {
                freed += trim_class(sh.cls(i).core, bk);
            }
        }
    }
    // in-place RAM compression (classes with rcodec): work outside the core lock, which is taken only to start / end
    // the rewrite (lookups miss the entry during it: no lease can start; nothing waits on the lock for the copy).
    // Only when quiet (compression is idle work): no request for --ram-compress-quiet-ms, and it
    // stops at the first request that arrives, so an entry is never mid-rewrite while an engine looks it up.
    let rq0 = sh.last_req_ms.load(Relaxed);
    let rquiet = sh.rcompress_quiet_ms == 0 || now.saturating_sub(rq0) >= sh.rcompress_quiet_ms;
    for i in 0..n {
        if !rquiet {
            break;
        }
        let chunk = sh.rchunk.get(i).copied().unwrap_or(0) as usize;
        let c = sh.cls(i);
        let Some(bk) = c.backing else { continue };
        if chunk == 0 || now < sh.rcompress_after_ms {
            continue;
        }
        if sh.bg_worker {
            break;                      // the background worker compresses (BG_JOBS)
        }
        // the images are LKVZ1: only if every connected reader of the class decodes it (FEATURES; legacy clients do)
        if !sh.class_reads(i, kvproto::FMT_SLOT_CODEC, 1) {
            sh.lazy.rcompress_no_reader.fetch_add(1, Relaxed);
            continue;
        }
        let slot = c.hello.slot_bytes;
        let t = Instant::now();
        // the SSD writer copies slots without a lease: no in-place rewrite while it runs
        let _pause = c.disk.map(|d| d.pause());
        let cands = c.core.lock().unwrap().compress_candidates(now - sh.rcompress_after_ms, 64);
        for (ns, key, ranges, bytes) in cands {
            if t.elapsed() >= Duration::from_millis(1000) {
                break;
            }
            if sh.rcompress_quiet_ms > 0 && sh.last_req_ms.load(Relaxed) != rq0 {
                sh.lazy.rcompress_yielded.fetch_add(1, Relaxed);
                break; // a request arrived: back off until quiet again
            }
            if ranges.len() != 1 || ranges[0].len != 1 || bytes > slot {
                continue; // one-slot entries only (the Arc connector's)
            }
            // identity of the bytes we copy (found with the rcompress-read failpoint): between the copy
            // and begin_rewrite the entry can be evicted and the same key stored again, with other bytes, in the
            // same slot; begin_rewrite (same key, same ranges) would then accept it and the old image would replace
            // the new entry's bytes. Only the entry we copied may be rewritten.
            let Some(g0) = c.core.lock().unwrap().copy_guard(ns, &key) else { continue };
            if g0.1 != ranges || g0.2 || g0.3 {
                continue;
            }
            let raw = bk.lock().unwrap().read(ranges[0].start, bytes as usize);
            super::fp::hit("rcompress-read");   // failpoints build: the entry may change while we work on the copy
            let z = if super::fp::hit("rcompress-fail") { None } else { kvslotz::compress(&raw, chunk, 0.05) };
            let same = |core: &kvcore::Core| core.copy_guard(ns, &key).as_ref() == Some(&g0);
            let Some(z) = z else {
                let mut core = c.core.lock().unwrap();
                if same(&core) && core.begin_rewrite(ns, &key, &ranges) {
                    core.end_rewrite(ns, &key, false); // not worth it: never tried again
                }
                sh.lazy.rcompress_skipped.fetch_add(1, Relaxed);
                continue;
            };
            {
                let mut core = c.core.lock().unwrap();
                if !same(&core) {
                    sh.lazy.rcompress_raced.fetch_add(1, Relaxed);
                    continue;
                }
                if !core.begin_rewrite(ns, &key, &ranges) {
                    continue;
                }
            }
            bk.lock().unwrap().write_slot_image(ranges[0].start, &z);
            let freed_now = punch_tail(bk, ranges[0].start, z.len() as u64);
            c.core.lock().unwrap().end_rewrite(ns, &key, true);
            sh.lazy.rcompress_entries.fetch_add(1, Relaxed);
            sh.lazy.rcompress_raw.fetch_add(bytes, Relaxed);
            sh.lazy.rcompress_stored.fetch_add(z.len() as u64, Relaxed);
            freed += freed_now;
        }
    }
    // compaction: only when quiet, never under pressure
    let quiet = now.saturating_sub(sh.last_req_ms.load(Relaxed)) >= 3000;
    let mut moved_bytes = 0u64;
    if sh.bg_worker {
        // the background worker moves entries (BG_COMPACT_JOBS) within this budget until the next sweep
        sh.compact_budget.store(if cfg.compact_bytes > 0 && quiet && !pressure && !idle { cfg.compact_bytes } else { 0 }, Relaxed);
    } else if cfg.compact_bytes > 0 && quiet && !pressure && !idle {
        let t = Instant::now();
        'pools: for i in 0..n {
            let c = sh.cls(i);
            let Some(bk) = c.backing else { continue };
            let slot = c.hello.slot_bytes;
            let _pause = c.disk.map(|d| d.pause());
            while moved_bytes < cfg.compact_bytes {
                if t.elapsed() >= Duration::from_millis(500) {
                    break 'pools;
                }
                let mut core = c.core.lock().unwrap();
                let (live, hw) = core.live_and_high_water();
                if hw as u64 <= live {
                    break; // dense
                }
                let Some((ns, key, old, new)) = core.relocation_candidate() else { break };
                let mut b = bk.lock().unwrap();
                if b.ensure(&new).is_err() {
                    core.finish_relocation(ns, &key, &[], new); // frees the new slots
                    break;
                }
                b.copy(&old, &new);
                if core.finish_relocation(ns, &key, &old, new) {
                    moved_bytes += old.iter().map(|r| r.len as u64).sum::<u64>() * slot;
                    sh.lazy.compact_moves.fetch_add(1, Relaxed);
                }
                // punch outside both locks (quarantined slots, as the sweep's own trim)
                drop(b);
                drop(core);
                freed += trim_class(c.core, bk);
            }
        }
        sh.lazy.compact_bytes.fetch_add(moved_bytes, Relaxed);
    }
    sh.lazy.sweeps.fetch_add(1, Relaxed);
    sh.lazy.freed_bytes.fetch_add(freed, Relaxed);
    sh.lazy.last_freed_bytes.store(freed, Relaxed);
    if freed >= (1u64 << 30) || moved_bytes > 0 {
        let resident: u64 = (0..n).filter_map(|i| sh.cls(i).backing.as_ref().map(|b| b.lock().unwrap().file_bytes())).sum();
        eprintln!("sweep: freed {:.2} GiB{}{}, compacted {:.2} GiB; resident {:.2} GiB",
                  freed as f64 / (1u64 << 30) as f64, if pressure { " (memory pressure)" } else { "" }, if idle { " (no clients)" } else { "" },
                  moved_bytes as f64 / (1u64 << 30) as f64, resident as f64 / (1u64 << 30) as f64);
    }
}

pub fn metrics(sh: &Shared, s: &mut String) {
    use std::fmt::Write as _;
    let n = sh.n_classes();
    if !(0..n).any(|i| sh.cls(i).backing.is_some()) {
        return;
    }
    let (mut file, mut live, mut hw_sum) = (0u64, 0u64, 0u64);
    for i in 0..n {
        let c = sh.cls(i);
        let Some(bk) = c.backing else { continue };
        let f = bk.lock().unwrap().file_bytes();
        let (l, hw) = c.core.lock().unwrap().live_and_high_water();
        let slot = c.hello.slot_bytes;
        let (l, hw) = (l * slot, hw as u64 * slot);
        file += f;
        live += l;
        hw_sum += hw;
        for (m, v) in [("pool_class_file_bytes", f), ("pool_class_live_bytes", l), ("pool_class_high_water_bytes", hw),
                       ("pool_class_ceiling_bytes", c.hello.pool_bytes), ("pool_class_slot_bytes", slot)] {
            let _ = writeln!(s, "llmt_kv_{m}{{class=\"{}\"}} {v}", c.name);
        }
    }
    for (m, v) in [("pool_resident_bytes", file), ("pool_file_bytes", file), ("pool_ceiling_bytes", sh.ceiling),
                   ("pool_live_bytes", live), ("pool_high_water_bytes", hw_sum),
                   ("sweep_last_freed_bytes", sh.lazy.last_freed_bytes.load(Relaxed)),
                   ("pressure_protected_bytes", sh.lazy.protected_bytes.load(Relaxed)),
                   ("pool_punch_blocked", sh.pins.load(Relaxed) as u64)] {
        let _ = writeln!(s, "# TYPE llmt_kv_{m} gauge\nllmt_kv_{m} {v}");
    }
    let held: u64 = (0..n).filter_map(|i| sh.cls(i).backing.as_ref().map(|b| b.lock().unwrap().punches_held.load(Relaxed))).sum();
    let _ = writeln!(s, "# TYPE llmt_kv_pool_punches_held_total counter\nllmt_kv_pool_punches_held_total {held}");
    for (m, v) in [("sweeps", &sh.lazy.sweeps), ("sweep_freed_bytes", &sh.lazy.freed_bytes),
                   ("sweep_evicted_aged_slots", &sh.lazy.evicted_aged), ("sweep_evicted_pressure_slots", &sh.lazy.evicted_pressure),
                   ("sweep_evicted_ceiling_slots", &sh.lazy.evicted_ceiling),
                   ("pressure_protect_shed", &sh.lazy.protect_shed), ("pressure_protect_held", &sh.lazy.protect_held),
                   ("sweep_evicted_idle_slots", &sh.lazy.evicted_idle), ("compact_moves", &sh.lazy.compact_moves),
                   ("compact_bytes", &sh.lazy.compact_bytes), ("backing_failures", &sh.lazy.backing_failures), ("warm_bytes", &sh.lazy.warm_bytes),
                   ("ram_compressed_entries", &sh.lazy.rcompress_entries), ("ram_compress_skipped", &sh.lazy.rcompress_skipped), ("ram_compress_yielded", &sh.lazy.rcompress_yielded), ("ram_compress_raced", &sh.lazy.rcompress_raced),
                   ("ram_compress_no_reader", &sh.lazy.rcompress_no_reader),
                   ("ram_compressed_raw_bytes", &sh.lazy.rcompress_raw), ("ram_compressed_stored_bytes", &sh.lazy.rcompress_stored),
                   ("cksum_bytes", &sh.lazy.cksum_bytes), ("cksum_yielded", &sh.lazy.cksum_yielded), ("cksum_raced", &sh.lazy.cksum_raced)] {
        let _ = writeln!(s, "# TYPE llmt_kv_{m}_total counter\nllmt_kv_{m}_total {}", v.load(Relaxed));
    }
}

/// Integrity checksums (for hosts without ECC; idle work only, never on the load path; suspect entries
/// are thrown out, nothing is repaired). Every 100 ms, after --ram-compress-quiet-ms without a request: hash entries
/// never hashed (oldest first), then those last verified more than --cksum-reverify-s ago, within a tenth of the
/// per-second budget (unified memory: the weights' bandwidth is not ours to take), stopping at the first request.
/// The thread runs at nice 19. XXH3-64 over the entry's raw bytes (a compressed slot is decoded chunk by chunk, its
/// chunk checksums verified too); the copy is guarded like the compressor's (copy_guard): an entry that changed while
/// it was read is left for the next pass. A mismatch evicts it with the rest of its chain (Core::checksum_result).
pub fn checksum_loop(sh: &Shared) {
    unsafe { libc::setpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t, 19) };
    let tick = Duration::from_millis(100);
    loop {
        std::thread::sleep(tick);
        let now = sh.now();
        let rq0 = sh.last_req_ms.load(Relaxed);
        if now.saturating_sub(rq0) < sh.rcompress_quiet_ms.max(1000) {
            continue;
        }
        let mut budget = sh.cksum_bytes_s / 10;
        'pools: for i in 0..sh.n_classes() {
            let c = sh.cls(i);
            let Some(bk) = c.backing else { continue };
            let slot = c.hello.slot_bytes;
            let cands = c.core.lock().unwrap().checksum_candidates(now.saturating_sub(sh.cksum_reverify_ms), 16);
            for (ns, key, ranges, bytes, _, _) in cands {
                if budget == 0 {
                    break 'pools;
                }
                if sh.last_req_ms.load(Relaxed) != rq0 {
                    sh.lazy.cksum_yielded.fetch_add(1, Relaxed);
                    break 'pools;
                }
                let Some(g0) = c.core.lock().unwrap().copy_guard(ns, &key) else { continue };
                if g0.1 != ranges || g0.3 {
                    continue;
                }
                let sum = entry_sum(&bk.lock().unwrap(), &ranges, bytes, slot, g0.2);
                super::fp::hit("cksum-read");   // failpoints build: the entry may change after the read
                let sum = if super::fp::hit("cksum-flip") { sum.map(|s| s ^ 1) } else { sum };
                budget = budget.saturating_sub(bytes.max(1));
                sh.lazy.cksum_bytes.fetch_add(bytes, Relaxed);
                match c.core.lock().unwrap().checksum_result(ns, &key, &g0, sum, sh.now()) {
                    kvcore::ChecksumResult::Raced => {
                        sh.lazy.cksum_raced.fetch_add(1, Relaxed);
                    }
                    kvcore::ChecksumResult::Corrupt => {
                        eprintln!("integrity: entry of {} bytes failed its checksum (class {i}, ns {ns}): evicted with \
                                   the rest of its chain", bytes);
                    }
                    _ => {}
                }
            }
        }
    }
}

/// XXH3-64 of the raw bytes of a compressed slot image (kvslotz::image_xxh3).
pub fn image_sum(img: &[u8], bytes: u64) -> Option<u64> {
    kvslotz::image_xxh3(img, bytes)
}

/// XXH3-64 of an entry's raw bytes, read 1 MiB at a time; None if a compressed slot does not decode.
fn entry_sum(b: &Backing, ranges: &[Range], bytes: u64, slot: u64, compressed: bool) -> Option<u64> {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    if compressed {
        // the in-place compressor's images: one-slot entries
        let r = ranges.first()?;
        let head = b.read(r.start, (slot as usize).min(64 << 10));
        kvslotz::raw_len(&head)?;                  // a compressed image at all
        let len = match kvslotz::image_len(&head) {
            Some(l) => l,
            None => kvslotz::image_len(&b.read(r.start, slot as usize))?,   // a chunk table past the first 64 KiB
        };
        return image_sum(&b.read(r.start, len.min(slot as usize)), bytes);
    }
    let mut left = bytes;
    for r in ranges {
        let mut at = r.start as u64 * slot;
        let end = at + (r.len as u64 * slot).min(left);
        while at < end {
            let n = (end - at).min(1 << 20);
            h.update(&b.read_at(at, n as usize));
            left -= n;
            at += n;
        }
    }
    Some(h.digest())
}

/// Byte ranges of `file` that hold data (SEEK_DATA / SEEK_HOLE), within `len`.
fn data_extents(file: &fs::File, len: u64) -> std::io::Result<Vec<(u64, u64)>> {
    let fd = file.as_raw_fd();
    let (mut out, mut pos) = (Vec::new(), 0u64);
    while pos < len {
        let d = unsafe { libc::lseek(fd, pos as libc::off_t, libc::SEEK_DATA) };
        if d < 0 {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::ENXIO) {
                break; // no data past pos
            }
            return Err(e);
        }
        let h = unsafe { libc::lseek(fd, d, libc::SEEK_HOLE) };
        if h < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let (d, h) = (d as u64, (h as u64).min(len));
        if d >= len {
            break;
        }
        out.push((d, h));
        pos = h;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    const SLOT: u64 = 1 << 20;

    fn pool(name: &str, slots: u64) -> std::path::PathBuf {
        let dir = if Path::new("/dev/shm").is_dir() { Path::new("/dev/shm") } else { Path::new("/tmp") };
        let p = dir.join(format!("llmt-lazy-test-{}-{name}", std::process::id()));
        let f = fs::File::create(&p).unwrap();
        f.set_len(slots * SLOT).unwrap();
        p
    }

    fn blocks(p: &Path) -> u64 {
        fs::metadata(p).unwrap().blocks() * 512
    }

    fn write_slot(p: &Path, slot: u64) {
        use std::os::unix::fs::FileExt;
        fs::OpenOptions::new().write(true).open(p).unwrap().write_all_at(&vec![0xab; SLOT as usize], slot * SLOT).unwrap();
    }

    #[test]
    fn reused_pool_is_read_back_not_punched() {
        let p = pool("reuse", 8);
        write_slot(&p, 2);
        write_slot(&p, 5);
        let before = blocks(&p);
        let b = Backing::new(&p, 8 * SLOT, SLOT, Arc::new(AtomicBool::new(false)), Some(Duration::from_secs(60))).unwrap();
        assert_eq!(blocks(&p), before, "a reused pool keeps its pages");
        assert_eq!(b.backed.iter().enumerate().filter(|(_, x)| **x).map(|(i, _)| i).collect::<Vec<_>>(), vec![2, 5]);
        assert_eq!(b.resident, 2 * SLOT);
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn hold_and_pins_stop_trim() {
        let p = pool("hold", 4);
        for s in 0..4 {
            write_slot(&p, s);
        }
        let pins = Arc::new(AtomicBool::new(false));
        let mut b = Backing::new(&p, 4 * SLOT, SLOT, pins.clone(), Some(Duration::from_secs(60))).unwrap();
        assert_eq!(b.trim(&[Range { start: 0, len: 4 }]), 0, "hold: nothing punched");
        assert!(b.backed.iter().all(|x| *x));
        b.hold_until = None;
        pins.store(true, Relaxed);
        assert_eq!(b.trim(&[Range { start: 0, len: 4 }]), 0, "pinning client: nothing punched");
        assert!(b.punches_held.load(Relaxed) >= 2);
        pins.store(false, Relaxed);
        assert_eq!(b.trim(&[Range { start: 0, len: 4 }]), 4 * SLOT, "no pins, no hold: free slots punched");
        assert!(b.backed.iter().all(|x| !*x));
        fs::remove_file(&p).unwrap();
    }

    fn slot_has_data(p: &Path, slot: u64) -> bool {
        use std::os::unix::fs::FileExt;
        let mut buf = vec![0u8; 16];
        fs::File::open(p).unwrap().read_exact_at(&mut buf, slot * SLOT).unwrap();
        buf.iter().all(|x| *x == 0xab)
    }

    /// A model of the core's free list for trim_batched: take removes what it can from it, give adds back.
    fn take_from(free: &mut Vec<Range>, part: &[Range]) -> Vec<Range> {
        let mut got = Vec::new();
        for r in part {
            let mut ok = true;
            for s in r.start..r.start + r.len {
                ok &= free.iter().any(|f| f.start <= s && s < f.start + f.len);
            }
            if !ok {
                continue;
            }
            let mut out = Vec::new();
            for f in free.drain(..) {
                let (fe, re) = (f.start + f.len, r.start + r.len);
                if re <= f.start || r.start >= fe {
                    out.push(f);
                    continue;
                }
                if f.start < r.start {
                    out.push(Range { start: f.start, len: r.start - f.start });
                }
                if re < fe {
                    out.push(Range { start: re, len: fe - re });
                }
            }
            *free = out;
            got.push(*r);
        }
        got
    }

    #[test]
    fn trim_batched_releases_the_lock_between_batches() {
        let p = pool("batch", 8);
        let bk = Mutex::new(Backing::new(&p, 8 * SLOT, SLOT, Arc::new(AtomicBool::new(false)), None).unwrap());
        for s in 0..8 {
            write_slot(&p, s);        // Backing::new punched the new pool: written again, as clients do
        }
        // the "core lock": a real mutex; trim_batched must drop each guard before it asks again (else this deadlocks)
        let core = std::sync::Mutex::new(vec![Range { start: 0, len: 8 }]);
        let mut calls = 0;
        let freed = trim_batched(|| {
            calls += 1;
            let g = core.lock().unwrap();
            let f = g.clone();
            (g, f)
        }, |g, part| take_from(g, part), |got| core.lock().unwrap().push(got[0]), &bk, 2 * SLOT);
        assert_eq!(calls, 4, "8 free slots in batches of 2: one lock hold per batch");
        assert_eq!(freed, 8 * SLOT);
        assert_eq!(blocks(&p), 0);
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn trim_batched_never_punches_a_slot_handed_out_between_batches() {
        let p = pool("race", 8);
        let bk = Mutex::new(Backing::new(&p, 8 * SLOT, SLOT, Arc::new(AtomicBool::new(false)), None).unwrap());
        for s in 0..8 {
            write_slot(&p, s);
        }
        let core = std::sync::Mutex::new(vec![Range { start: 0, len: 8 }]);
        let mut calls = 0;
        let freed = trim_batched(|| {
            calls += 1;
            let mut g = core.lock().unwrap();
            if calls == 2 {
                // between batch 1 and 2 a RESERVE hands out slot 3 and the client writes it
                *g = vec![Range { start: 2, len: 1 }, Range { start: 4, len: 4 }];
                write_slot(&p, 3);
            }
            let f = g.clone();
            (g, f)
        }, |g, part| take_from(g, part), |got| core.lock().unwrap().push(got[0]), &bk, 2 * SLOT);
        assert_eq!(freed, 7 * SLOT);
        assert!(slot_has_data(&p, 3), "the reserved slot keeps its bytes");
        assert_eq!(blocks(&p), SLOT);
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn trim_batched_holds_slots_out_of_the_free_list_until_they_are_punched() {
        let p = pool("freeing", 6);
        let bk = Mutex::new(Backing::new(&p, 6 * SLOT, SLOT, Arc::new(AtomicBool::new(false)), None).unwrap());
        for s in 0..6 {
            write_slot(&p, s);
        }
        let core = std::sync::Mutex::new(vec![Range { start: 0, len: 6 }]);
        let mut gives = 0;
        let freed = trim_batched(|| {
            let g = core.lock().unwrap();
            let f = g.clone();
            (g, f)
        }, |g, part| take_from(g, part), |got| {
            gives += 1;
            let mut g = core.lock().unwrap();
            // while a batch is punched the core lock is free (a lookup gets it) and its slots are in nobody's free
            // list: a RESERVE cannot take them; they come back only after the punch is done
            assert!(g.iter().all(|f| got.iter().all(|r| f.start + f.len <= r.start || r.start + r.len <= f.start)), "not yet free");
            for r in got {
                assert!(!slot_has_data(&p, r.start as u64), "punched before it is free again");
            }
            g.push(got[0]);
        }, &bk, SLOT);
        assert_eq!((gives, freed), (6, 6 * SLOT));
        assert_eq!(core.lock().unwrap().len(), 6, "every slot is back");
        assert_eq!(blocks(&p), 0);
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn trim_from_keeps_the_warm_margin_across_batches() {
        let p = pool("warmb", 6);
        let mut b = Backing::new(&p, 6 * SLOT, SLOT, Arc::new(AtomicBool::new(false)), None).unwrap();
        for s in 0..6 {
            write_slot(&p, s);
        }
        b.backed[0] = true;
        b.backed[1] = true;
        b.keep_warm = 2;
        let free = [Range { start: 0, len: 6 }];
        let (mut from, mut freed, mut batches) = (0, 0, 0);
        loop {
            let (f, next) = b.trim_from(&free, from, SLOT);
            freed += f;
            batches += 1;
            match next {
                Some(n) => from = n,
                None => break,
            }
        }
        assert_eq!(freed, 4 * SLOT, "slots 0 and 1 are the warm margin");
        assert!(slot_has_data(&p, 0) && slot_has_data(&p, 1));
        assert_eq!(batches, 4);
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn warm_thread_skips_slots_being_punched_and_the_sweep_skips_slots_being_warmed() {
        let p = pool("warmpunch", 6);
        let bk = Mutex::new(Backing::new(&p, 6 * SLOT, SLOT, Arc::new(AtomicBool::new(false)), None).unwrap());
        for s in 0..6 {
            write_slot(&p, s);
        }
        bk.lock().unwrap().keep_warm = 6;
        let core = std::sync::Mutex::new(vec![Range { start: 0, len: 6 }]);
        // slot 5 is being warmed (fallocate in flight) when the sweep plans: it must not be punched
        let warm_first = bk.lock().unwrap().warm_plan(&[Range { start: 5, len: 1 }]);
        assert_eq!(warm_first, vec![Range { start: 5, len: 1 }]);
        bk.lock().unwrap().keep_warm = 0;       // pressure: no margin, everything free would be punched
        {
            // the window of a punch in flight: slots 0-1 are marked freeing; the warm thread, working from its stale
            // free list, plans nothing for them (and nothing for 5, already warming)
            let mut b = bk.lock().unwrap();
            b.mark_freeing(&[Range { start: 0, len: 2 }]);
            b.keep_warm = 6;
            let plan = b.warm_plan(&[Range { start: 0, len: 6 }]);
            assert_eq!(plan, vec![Range { start: 2, len: 3 }], "0-1 are being punched, 5 is being warmed");
            b.warm_done(&plan, &[false, false, false]);        // not allocated: clear the marks again
            b.unmark_freeing(&[Range { start: 0, len: 2 }]);
            b.keep_warm = 0;
        }
        let freed = trim_batched(|| {
            let g = core.lock().unwrap();
            let f = g.clone();
            (g, f)
        }, |g, part| take_from(g, part), |got| {
            core.lock().unwrap().push(got[0]);
        }, &bk, SLOT);
        assert_eq!(freed, 5 * SLOT, "slot 5 (being warmed) is not punched");
        assert!(slot_has_data(&p, 5), "the warmed slot keeps its pages");
        for s in 0..5 {
            assert!(!slot_has_data(&p, s));
        }
        let mut b = bk.lock().unwrap();
        assert!(b.freeing.iter().all(|x| !x), "marks cleared after the punch");
        b.warm_done(&warm_first, &[true]);
        assert!(b.backed[5] && !b.warming[5]);
        assert_eq!(b.resident, SLOT);
        drop(b);
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn freeing_marks_are_cleared_when_punching_is_held() {
        let p = pool("heldmark", 2);
        let bk = Mutex::new(Backing::new(&p, 2 * SLOT, SLOT, Arc::new(AtomicBool::new(true)), None).unwrap());
        bk.lock().unwrap().mark_freeing(&[Range { start: 0, len: 2 }]);
        assert_eq!(punch_unlocked(&bk, &[Range { start: 0, len: 2 }]), 0);
        assert!(bk.lock().unwrap().freeing.iter().all(|x| !x));
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn punch_tail_outside_the_lock_keeps_the_head() {
        let p = pool("tail", 2);
        let bk = Mutex::new(Backing::new(&p, 2 * SLOT, SLOT, Arc::new(AtomicBool::new(false)), None).unwrap());
        write_slot(&p, 1);
        {
            use std::os::unix::fs::FileExt;
            let f = fs::OpenOptions::new().write(true).open(&p).unwrap();
            f.write_all_at(&vec![0xcd; SLOT as usize], SLOT).unwrap();
        }
        let freed = punch_tail(&bk, 1, 4096);
        assert_eq!(freed, SLOT - 4096);
        assert_eq!(blocks(&p), 4096);
        assert_eq!(punch_tail(&bk, 1, SLOT), 0, "a full slot has no tail");
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn new_pool_is_punched() {
        let p = pool("new", 4);
        write_slot(&p, 1);
        let b = Backing::new(&p, 4 * SLOT, SLOT, Arc::new(AtomicBool::new(false)), None).unwrap();
        assert_eq!(blocks(&p), 0);
        assert_eq!(b.resident, 0);
        fs::remove_file(&p).unwrap();
    }
}
