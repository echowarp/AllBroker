//! `kvbench race`: data-integrity stress against a real broker. Every byte a
//! reader sees under a lease must belong to the key it looked up; any foreign or torn byte fails the run.
//!
//! - Writers store entries whose slot bytes encode their key id, a random generation, the length and a
//!   compressible pattern derived from both (so in-place RAM compression kicks in). Many threads share a small key
//!   space on a small pool: constant eviction, slot reuse, keys stored again after eviction.
//! - Late writers: now and then a writer drops its connection half way through its copy and keeps writing that
//!   slot for a while, as an engine whose copy outlives its reservation (the broker must not hand
//!   that slot to anyone else meanwhile).
//! - Readers lease (acquiring LOOKUP), read (decompressing compressed slots), verify every byte; some hold the lease
//!   while sleeping and verify again before releasing (compaction / compression / eviction must leave a leased entry
//!   alone).
//! - Quiet windows: every QUIET_EVERY the writers pause for QUIET_FOR (> the broker's 3 s compaction quiet time)
//!   while readers keep holding leases, so compaction and compression run concurrently with leased reads.
//! - Broker restarts are external (a driver kills / restarts it); clients reconnect and re-map the pool.
//! Exit status 1 if any foreign / torn read was seen.
use kvclient::{Client, Key, PoolMap, ReserveItem, Status};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

const NS: u32 = 11;
const QUIET_EVERY: Duration = Duration::from_secs(8);
const QUIET_FOR: Duration = Duration::from_secs(4);

#[derive(Default)]
struct C {
    writes: AtomicU64,
    commits: AtomicU64,
    late: AtomicU64,
    slow: AtomicU64,
    reads: AtomicU64,
    reads_compressed: AtomicU64,
    held: AtomicU64,
    foreign: AtomicU64,
    torn: AtomicU64,
    errors: AtomicU64,
    reconnects: AtomicU64,
}

fn key(id: u32) -> Key {
    let mut k = [0u8; 32];
    k[..4].copy_from_slice(&id.to_le_bytes());
    k[4..12].copy_from_slice(b"kvrace01");
    k
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Byte `i` of the payload body for (id, gen): one value per 4 KiB block with a small ripple (compressible, and
/// every block differs between generations).
fn body(id: u32, gen: u64, i: usize) -> u8 {
    let s = mix((id as u64) << 40 ^ gen ^ (i / 4096) as u64);
    (s as u8) ^ ((i / 61) % 4) as u8
}

fn fill(buf: &mut [u8], id: u32, gen: u64) {
    buf[..4].copy_from_slice(&id.to_le_bytes());
    buf[4..12].copy_from_slice(&gen.to_le_bytes());
    let n = buf.len() as u64;
    buf[12..20].copy_from_slice(&n.to_le_bytes());
    for (i, b) in buf.iter_mut().enumerate().skip(20) {
        *b = body(id, gen, i);
    }
}

/// None = good; Some(true) = another key's / generation's bytes (foreign); Some(false) = torn (header ok, body not).
fn verify(buf: &[u8], id: u32) -> Option<bool> {
    let hid = u32::from_le_bytes(buf[..4].try_into().unwrap());
    let len = u64::from_le_bytes(buf[12..20].try_into().unwrap()) as usize;
    if hid != id || len != buf.len() {
        return Some(true);
    }
    let gen = u64::from_le_bytes(buf[4..12].try_into().unwrap());
    if buf.iter().enumerate().skip(20).any(|(i, b)| *b != body(id, gen, i)) {
        return Some(false);
    }
    None
}

struct Conn {
    c: Client,
    pool: PoolMap,
    tag: u16,
}

fn open(sock: &Path, tag: u16) -> Option<Conn> {
    let c = Client::connect(sock, tag, &format!("kvrace-{tag}"), Duration::from_millis(200)).ok()?;
    let pool = PoolMap::open(&c.hello).ok()?;
    Some(Conn { c, pool, tag })
}

fn reopen(sock: &Path, cn: &mut Option<Conn>, tag: u16, st: &C) {
    *cn = None;
    for _ in 0..200 {
        if let Some(x) = open(sock, tag) {
            *cn = Some(x);
            st.reconnects.fetch_add(1, Relaxed);
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Read an entry's bytes (decompressing an in-place compressed slot) and verify them.
fn read_check(pool: &PoolMap, e: &kvproto::EntryDesc, id: u32, st: &C) {
    let mut out = vec![0u8; e.bytes as usize];
    let slot = unsafe { pool.range(e.ranges[0]) };
    let img = if e.ranges.len() == 1 { slot.to_vec(slot.len()) } else { Vec::new() };
    if e.ranges.len() == 1 && kvslotz::is_compressed(&img) {
        st.reads_compressed.fetch_add(1, Relaxed);
        if !kvslotz::read_range(&img, 0, &mut out) {
            st.torn.fetch_add(1, Relaxed);
            return;
        }
    } else {
        let mut off = 0;
        for r in &e.ranges {
            let s = unsafe { pool.range(*r) };
            let n = s.len().min(out.len() - off);
            s.read(0, &mut out[off..off + n]);
            off += n;
            if off == out.len() {
                break;
            }
        }
    }
    st.reads.fetch_add(1, Relaxed);
    match verify(&out, id) {
        None => {}
        Some(true) => {
            st.foreign.fetch_add(1, Relaxed);
        }
        Some(false) => {
            st.torn.fetch_add(1, Relaxed);
        }
    }
}

pub fn run(sock: PathBuf, threads: usize, seconds: u64, keys: usize) {
    let st = Arc::new(C::default());
    let quiet = Arc::new(AtomicBool::new(false));
    let end = Instant::now() + Duration::from_secs(seconds);
    let mut hs = Vec::new();
    {
        let quiet = quiet.clone();
        hs.push(std::thread::spawn(move || {
            while Instant::now() < end {
                std::thread::sleep(QUIET_EVERY);
                quiet.store(true, Relaxed);
                std::thread::sleep(QUIET_FOR);
                quiet.store(false, Relaxed);
            }
        }));
    }
    for t in 0..threads {
        let (sock, st, quiet) = (sock.clone(), st.clone(), quiet.clone());
        let writer = t % 2 == 0;
        hs.push(std::thread::spawn(move || {
            let tag = 300 + t as u16;
            let mut cn = open(&sock, tag);
            let mut rng = mix(0xa5a5 ^ t as u64 ^ std::process::id() as u64);
            let mut next = move || {
                rng = mix(rng);
                rng
            };
            while Instant::now() < end {
                if cn.is_none() {
                    reopen(&sock, &mut cn, tag, &st);
                    continue;
                }
                let id = (next() % keys as u64) as u32;
                if writer {
                    if quiet.load(Relaxed) {
                        std::thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                    let x = cn.as_mut().unwrap();
                    let slot = x.pool.slot_bytes as u64;
                    let bytes = 64 + next() % (slot - 64);
                    let it = ReserveItem { key: key(id), parent: None, bytes, kind: 0, format: 1, cost_hint: 0, flags: 0,
                                           ttl_ms: 0, meta: vec![] };
                    let r = match x.c.reserve(NS, vec![it]) {
                        Ok(r) => r,
                        Err(_) => {
                            st.errors.fetch_add(1, Relaxed);
                            reopen(&sock, &mut cn, tag, &st);
                            continue;
                        }
                    };
                    let (s, ranges) = &r[0];
                    if *s != Status::Ok || ranges.len() != 1 {
                        continue;
                    }
                    let r0 = ranges[0];
                    let gen = next();
                    let mut img = vec![0u8; bytes as usize];
                    fill(&mut img, id, gen);
                    st.writes.fetch_add(1, Relaxed);
                    if next() % 40 == 1 {
                        // slow writer (still connected): half the copy, a stall past the reservation timeout (the
                        // broker drops the reservation: its slots must stay quarantined), the rest, then the commit,
                        // which must be refused
                        let half = img.len() / 2;
                        let dst = unsafe { x.pool.range(r0) };
                        dst.write(0, &img[..half]);
                        std::thread::sleep(Duration::from_millis(450 + next() % 500));
                        dst.write(half, &img[half..]);
                        st.slow.fetch_add(1, Relaxed);
                        if let Ok(v) = x.c.commit(NS, &[key(id)]) {
                            if v[0] == Status::Ok {
                                st.commits.fetch_add(1, Relaxed);
                            }
                        }
                        continue;
                    }
                    if next() % 40 == 0 {
                        // late writer: the connection goes half way through the copy (the broker aborts the
                        // reservation), the copy goes on for up to 300 ms
                        // the old connection goes (socket closed: the broker aborts the reservation); the late copy
                        // writes through a SlotRange borrowed from the old mapping, so the mapping cannot be dropped
                        // before the copy ends (that was a segfault with raw pointers; now it does not compile)
                        let Conn { c: old_c, pool: old_pool, .. } = cn.take().unwrap();
                        let dst = unsafe { old_pool.range(r0) };
                        let half = img.len() / 2;
                        dst.write(0, &img[..half]);
                        drop(old_c);
                        reopen(&sock, &mut cn, tag, &st);
                        st.late.fetch_add(1, Relaxed);
                        let t_end = Instant::now() + Duration::from_millis(next() % 300);
                        while Instant::now() < t_end {
                            // garbage of this writer, for a key nobody may read from this slot now
                            dst.fill(half, img.len() - half, (next() & 0xff) as u8);
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        continue;
                    }
                    unsafe { x.pool.range(r0) }.write(0, &img);
                    match x.c.commit(NS, &[key(id)]) {
                        Ok(v) if v[0] == Status::Ok => {
                            st.commits.fetch_add(1, Relaxed);
                        }
                        Ok(_) => {}
                        Err(_) => {
                            st.errors.fetch_add(1, Relaxed);
                            reopen(&sock, &mut cn, tag, &st);
                        }
                    }
                } else {
                    if quiet.load(Relaxed) {
                        // no requests in a quiet window (the broker compacts only after 3 s without any); leases
                        // taken before it are still held and read below
                        std::thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                    let x = cn.as_mut().unwrap();
                    let l = match x.c.lookup(NS, &[key(id)], true) {
                        Ok(l) => l,
                        Err(_) => {
                            st.errors.fetch_add(1, Relaxed);
                            reopen(&sock, &mut cn, tag, &st);
                            continue;
                        }
                    };
                    if l.n_hit == 0 {
                        continue;
                    }
                    read_check(&x.pool, &l.entries[0], id, &st);
                    // hold some leases (always in quiet windows): eviction, compaction, compression must wait
                    if next() % 8 == 0 {
                        // hold the lease while eviction / compression / compaction run (some long enough to span a
                        // quiet window), verifying the bytes again and again
                        st.held.fetch_add(1, Relaxed);
                        let t_end = Instant::now() + Duration::from_millis(50 + next() % 6000);
                        while Instant::now() < t_end {
                            std::thread::sleep(Duration::from_millis(250));
                            read_check(&x.pool, &l.entries[0], id, &st);
                        }
                    }
                    if x.c.release(&[l.lease_id]).is_err() {
                        st.errors.fetch_add(1, Relaxed);
                        reopen(&sock, &mut cn, tag, &st);
                    }
                }
                let _ = cn.as_ref().map(|c| c.tag);
            }
        }));
    }
    for h in hs {
        let _ = h.join();
    }
    let g = |a: &AtomicU64| a.load(Relaxed);
    println!("race: writes {} commits {} late-writers {} slow-writers {} | reads {} (compressed {}) leases held {} | FOREIGN {} TORN {} | \
              errors {} reconnects {}", g(&st.writes), g(&st.commits), g(&st.late), g(&st.slow), g(&st.reads), g(&st.reads_compressed),
             g(&st.held), g(&st.foreign), g(&st.torn), g(&st.errors), g(&st.reconnects));
    if g(&st.foreign) + g(&st.torn) > 0 {
        println!("FAIL: a reader saw bytes that are not its key's");
        std::process::exit(1);
    }
    println!("PASS: every leased read verified");
}

/// `kvbench compact`: compaction moves entries while readers hold leases on others. Deterministic:
/// fill the pool (every other entry ephemeral with a 2 s TTL, so expiry leaves holes below the rest), lease the
/// highest few, go quiet (no request at all) for 7 s while the lease holders keep verifying their bytes, then check
/// that compaction moved entries, that every surviving entry reads back intact at its (new) slots and that the
/// leased ones neither moved nor changed. Exit status 1 on any wrong byte or a leased entry that moved.
pub fn compact(sock: PathBuf) {
    let st = Arc::new(C::default());
    let mut a = open(&sock, 501).expect("connect");
    let n_slots = (a.pool.len / a.pool.slot_bytes) as u32;
    let slot = a.pool.slot_bytes as u64;
    let n = n_slots * 9 / 10;
    let mut gen = 1u64;
    let mut placed = Vec::new();
    for id in 0..n {
        let eph = id % 2 == 0;
        let bytes = slot - 64 - (id as u64 % 512);
        let it = ReserveItem { key: key(id), parent: None, bytes, kind: 0, format: 1, cost_hint: 0,
                               flags: if eph { kvproto::ENTRY_EPHEMERAL } else { 0 }, ttl_ms: if eph { 2000 } else { 0 },
                               meta: vec![] };
        let r = a.c.reserve(NS, vec![it]).expect("reserve");
        if r[0].0 != Status::Ok {
            continue;
        }
        gen = mix(gen);
        let mut img = vec![0u8; bytes as usize];
        fill(&mut img, id, gen);
        unsafe { a.pool.range(r[0].1[0]) }.write(0, &img);
        a.c.commit(NS, &[key(id)]).expect("commit");
        if !eph {
            placed.push((id, r[0].1[0].start));
        }
    }
    let moves0 = a.c.stats().unwrap().into_iter().find(|(k, _)| k == "relocations").map(|x| x.1).unwrap_or(0);
    // lease the 4 highest surviving entries: they must not move
    placed.sort_by_key(|x| std::cmp::Reverse(x.1));
    let mut held = Vec::new();
    for (id, s) in placed.iter().take(4) {
        let l = a.c.lookup(NS, &[key(*id)], true).expect("lookup");
        assert_eq!(l.n_hit, 1, "entry {id} present");
        held.push((*id, *s, l));
    }
    std::thread::sleep(Duration::from_millis(2500));            // the ephemeral half expires (next sweep frees it)
    // quiet: no requests at all for 7 s (compaction needs 3 s); lease holders verify their bytes meanwhile
    let t_end = Instant::now() + Duration::from_secs(7);
    while Instant::now() < t_end {
        for (id, _, l) in &held {
            read_check(&a.pool, &l.entries[0], *id, &st);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let moves = a.c.stats().unwrap().into_iter().find(|(k, _)| k == "relocations").map(|x| x.1).unwrap_or(0) - moves0;
    let mut moved_leased = 0;
    for (id, s, l) in &held {
        read_check(&a.pool, &l.entries[0], *id, &st);
        // (a lookup without acquire describes nothing: lease again to see the entry's slot)
        let again = a.c.lookup(NS, &[key(*id)], true).expect("lookup");
        if again.n_hit != 1 || again.entries.first().map(|e| e.ranges[0].start) != Some(*s) {
            moved_leased += 1;
        }
        a.c.release(&[again.lease_id, l.lease_id]).expect("release");
    }
    let mut checked = 0;
    for (id, _) in &placed {
        let l = a.c.lookup(NS, &[key(*id)], true).expect("lookup");
        if l.n_hit == 1 {
            read_check(&a.pool, &l.entries[0], *id, &st);
            checked += 1;
            a.c.release(&[l.lease_id]).expect("release");
        }
    }
    let g = |x: &AtomicU64| x.load(Relaxed);
    println!("compact: {n} stored ({} kept), relocations {moves}, leased entries moved {moved_leased}, entries verified \
              {checked}, reads {} | FOREIGN {} TORN {}", placed.len(), g(&st.reads), g(&st.foreign), g(&st.torn));
    let ok = moves > 0 && moved_leased == 0 && g(&st.foreign) + g(&st.torn) == 0;
    println!("{}", if ok { "PASS" } else if moves == 0 { "FAIL: no compaction happened" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}

/// Integrity checksums, against a broker started with a short --cksum-reverify-s: a chain of 6 entries
/// (explicit parents) and 6 unrelated ones are stored, the broker hashes them while nothing asks; then one byte of the
/// chain's third entry is flipped in the pool (as a RAM fault would), and after the re-verification the third entry
/// and everything after it in the chain must be gone, the rest intact. Requests stay away while the broker works
/// (any request pauses it).
pub fn cksum(sock: PathBuf) {
    let st = C::default();
    let mut a = open(&sock, 502).expect("connect");
    let slot = a.pool.slot_bytes as u64;
    let store = |a: &mut Conn, id: u32, parent: Option<u32>| {
        let bytes = slot - 128;
        let it = ReserveItem { key: key(id), parent: parent.map(key), bytes, kind: 0, format: 1, cost_hint: 0, flags: 0,
                               ttl_ms: 0, meta: vec![] };
        let r = a.c.reserve(NS, vec![it]).expect("reserve");
        assert_eq!(r[0].0, Status::Ok, "reserve {id}");
        let mut img = vec![0u8; bytes as usize];
        fill(&mut img, id, 1);
        unsafe { a.pool.range(r[0].1[0]) }.write(0, &img);
        a.c.commit(NS, &[key(id)]).expect("commit");
    };
    for id in 0..6 {
        store(&mut a, 100 + id, (id > 0).then(|| 100 + id - 1));
        store(&mut a, 200 + id, None);
    }
    let stat = |a: &mut Conn, k: &str| a.c.stats().unwrap().into_iter().find(|(n, _)| n == k).map(|x| x.1).unwrap_or(0);
    std::thread::sleep(Duration::from_secs(5));
    let stored = stat(&mut a, "cksum_stored");
    // the fault: flip one byte of entry 102 (the lease only finds its slot, and is given back before the flip)
    let l = a.c.lookup(NS, &[key(102)], true).expect("lookup");
    let r = l.entries[0].ranges[0];
    a.c.release(&[l.lease_id]).expect("release");
    let mut b = [0u8; 1];
    unsafe { a.pool.range(r) }.read(4096, &mut b);
    b[0] ^= 0x10;
    unsafe { a.pool.range(r) }.write(4096, &b);
    std::thread::sleep(Duration::from_secs(6));
    let (corrupt, cascade) = (stat(&mut a, "cksum_corrupt"), stat(&mut a, "cascade_evicted"));
    let verified = stat(&mut a, "cksum_verified");
    let mut present = Vec::new();
    for id in (100..106).chain(200..206) {
        let l = a.c.lookup(NS, &[key(id)], true).expect("lookup");
        if l.n_hit == 1 {
            read_check(&a.pool, &l.entries[0], id, &st);
            present.push(id);
            a.c.release(&[l.lease_id]).expect("release");
        }
    }
    let want: Vec<u32> = (100..102).chain(200..206).collect();
    println!("cksum: stored {stored}, verified {verified}, corrupt {corrupt}, cascade {cascade} | present {present:?} | \
              reads {} FOREIGN {} TORN {}", st.reads.load(Relaxed), st.foreign.load(Relaxed), st.torn.load(Relaxed));
    let ok = stored >= 12 && corrupt == 1 && cascade == 3 && present == want
        && st.foreign.load(Relaxed) == 0 && st.torn.load(Relaxed) == 0;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
