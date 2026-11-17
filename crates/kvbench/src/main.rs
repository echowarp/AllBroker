//! kvbench: end-to-end scenarios and a synthetic load test against a running llmt-kvbroker.
//! Synthetic keys and bytes only.
//!
//!   kvbench scenarios --socket PATH          race/crash/WAIT/data-path checks (needs a pool of >= 64 slots)
//!   kvbench shared --socket PATH             shared-prefix class (--shared-gib): evicted last (see fn shared)
//!   kvbench tail --socket PATH               partial-tail entries (kind 4): evicted first, age faster (see fn tail)
//!   kvbench phases --socket PATH             agent phase, low-priority one-off phase, agents return (PHASES_TAGGED=0:
//!                                            the same traffic without request classes, for a before/after number)
//!   kvbench hints --socket PATH              prefix-affinity hints HINT / WHERE (docs/prefix-hints.md; see fn hints)
//!   kvbench load --socket PATH [--threads 4] [--seconds 10] [--keys 32] [--batch 8] [--steps-per-s 0 = flat out]

mod race;
mod replay;

use kvclient::{Client, Key, PoolMap, ReserveItem, Status};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const NS: u32 = 7;

fn key(a: u64, b: u64) -> Key {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&a.to_le_bytes());
    k[8..16].copy_from_slice(&b.to_le_bytes());
    k
}

fn item(k: Key, parent: Option<Key>, bytes: u64) -> ReserveItem {
    ReserveItem { key: k, parent, bytes, kind: 2, format: 1, cost_hint: 10, flags: 0, ttl_ms: 0, meta: b"seg:kv+state".to_vec() }
}

fn connect(sock: &PathBuf, engine: u16) -> Client {
    Client::connect(sock, engine, &format!("kvbench-{engine}"), Duration::from_millis(500)).expect("connect")
}

fn stat(c: &mut Client, name: &str) -> u64 {
    c.stats().unwrap().into_iter().find(|(k, _)| k == name).map(|x| x.1).unwrap_or_else(|| panic!("no stat {name}"))
}

fn check(name: &str, ok: bool) {
    println!("{} {name}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}

/// Cache domains: different descriptors never see each other's entries (same namespace and key); the same descriptor
/// (any line order) hits; clients without a domain (the older wire format) keep their plain namespaces.
fn domains(sock: PathBuf) {
    let run = std::process::id() as u64;
    let d27 = "model=Qwen/Qwen3.8-27B\nweights=hf:Qwen3.8-27B@rev-x/exl3-4.0bpw\nengine=example-arc\nlayout=kv:fp16-blk1664;gdn:k12-ssm:float32-conv:auto\nkv_format=fp16";
    let dfn = "model=Qwen/Qwen3.8-Flash-Next\nweights=hf:qwen3.8-flash-next-exl3-4.05\nengine=split-dual\nlayout=kv-page:2MiB-v1;gdn-piece:16\nkv_format=fp16\nple=bf16z";
    let mut a = connect(&sock, 1);
    let mut b = connect(&sock, 2);
    let mut c = connect(&sock, 3);
    let mut legacy = connect(&sock, 4);
    let ida = a.domain(d27).unwrap();
    let idb = b.domain(dfn).unwrap();
    // same descriptor, other line order -> same domain
    let d27_reordered: String = d27.lines().rev().collect::<Vec<_>>().join("\n");
    let idc = c.domain(&d27_reordered).unwrap();
    check("different descriptors -> different domain ids", ida != idb);
    check("same descriptor (any line order) -> same domain id", ida == idc);
    check("descriptor without model / weights / engine / layout refused", connect(&sock, 5).domain("model=x").is_err());
    check("a connection can't switch to another domain", a.domain(dfn).is_err());
    let k = key(run, 1);
    let r = a.reserve(NS, vec![item(k, None, 100)]).unwrap();
    check("domain A reserves", r[0].0 == Status::Ok);
    check("domain A commits", a.commit(NS, &[k]).unwrap()[0] == Status::Ok);
    check("domain B (other model) misses the same namespace + key", b.lookup(NS, &[k], false).unwrap().n_hit == 0);
    check("domain B can store its own entry under the same key", b.reserve(NS, vec![item(k, None, 100)]).unwrap()[0].0 == Status::Ok);
    check("no-domain client misses it too", legacy.lookup(NS, &[k], false).unwrap().n_hit == 0);
    check("same domain, other connection hits", c.lookup(NS, &[k], false).unwrap().n_hit == 1);
    let k2 = key(run, 2);
    legacy.reserve(NS, vec![item(k2, None, 100)]).unwrap();
    legacy.commit(NS, &[k2]).unwrap();
    check("domain client misses a no-domain entry", a.lookup(NS, &[k2], false).unwrap().n_hit == 0);
    check("no-domain clients keep their plain namespace (older wire format)", connect(&sock, 6).lookup(NS, &[k2], false).unwrap().n_hit == 1);
    println!("domains: OK (A {ida:016x}, B {idb:016x})");
}

/// SSD tier (broker --disk-dir): `disk write` stores N one-slot entries with known bytes and waits for the
/// write-behind; after a broker restart `disk check` finds them all again with the same bytes; `disk promote` fills the
/// pool past its size so the first entries are evicted from RAM, then looks one up (a miss that queues its promotion)
/// and again (a hit, same bytes).
fn disk(sock: PathBuf, phase: &str) {
    let mut a = connect(&sock, 1);
    let slot = a.hello.slot_bytes as usize;
    let pool = PoolMap::open(&a.hello).expect("map pool");
    let n_slots = a.hello.n_slots as u64;
    let pat = |i: u64, j: usize| ((i.wrapping_mul(0x9E37_79B9) as usize ^ j.wrapping_mul(31)) & 0xff) as u8;
    let store_p = |a: &mut Client, i: u64, parent: Option<u64>| {
        let k = key(0xd15c, i);
        let r = a.reserve(NS, vec![item(k, parent.map(|p| key(0xd15c, p)), slot as u64)]).unwrap();
        if r[0].0 != Status::Ok {
            return false;
        }
        for rg in &r[0].1 {
            let b = unsafe { pool.range(*rg) };
            b.write(0, &(0..b.len()).map(|j| pat(i, j)).collect::<Vec<u8>>());
        }
        a.commit(NS, &[k]).unwrap()[0] == Status::Ok
    };
    let store = |a: &mut Client, i: u64| store_p(a, i, None);
    let verify = |a: &mut Client, i: u64| -> Option<bool> {
        let l = a.lookup(NS, &[key(0xd15c, i)], true).unwrap();
        if l.n_hit == 0 {
            return None;
        }
        let ok = l.entries[0].ranges.iter().all(|rg| unsafe { pool.range(*rg) }.to_vec(usize::MAX).iter().enumerate().all(|(j, x)| *x == pat(i, j)));
        a.release(&[l.lease_id]).unwrap();
        Some(ok)
    };
    match phase {
        "write" => {
            // admission (--disk-min-hits 2): 0..4 one-off (never hit), 4..8 hit twice, chain 20 <- 21 with only the
            // child hit twice (prefix-closed: the parent goes too)
            for i in 0..8 {
                check(&format!("store entry {i}"), store(&mut a, i));
            }
            check("store chain root 20", store(&mut a, 20));
            check("store chain child 21", store_p(&mut a, 21, Some(20)));
            for i in (4..8).chain([21]) {
                verify(&mut a, i);
                verify(&mut a, i);
            }
            std::thread::sleep(Duration::from_secs(7));   // write-behind (--disk-flush-s 5)
            println!("disk write: 10 entries stored, 5 reused");
        }
        "check" => {
            std::thread::sleep(Duration::from_secs(2));   // start-up load
            for i in 0..4 {
                check(&format!("one-off entry {i} not written to the SSD (gone after the restart)"), verify(&mut a, i).is_none());
            }
            for i in (4..8).chain([20, 21]) {
                // an on-demand pool loads only the core at start: the first lookup queues the promotion
                let mut got = verify(&mut a, i);
                for _ in 0..30 {
                    if got.is_some() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                    got = verify(&mut a, i);
                }
                check(&format!("reused entry {i} back after the broker restart, same bytes"), got == Some(true));
            }
        }
        "promote" => {
            for i in 100..100 + 4 * n_slots {
                store(&mut a, i);
                verify(&mut a, i);   // twice: frequent (ARC T2) like the restored ones, so the fill turns them over
                verify(&mut a, i);
            }
            let gone: Vec<u64> = (4..8).filter(|&i| a.lookup(NS, &[key(0xd15c, i)], false).unwrap().n_hit == 0).collect();
            check("some of the first entries evicted from RAM by the fill", !gone.is_empty());
            let i = gone[0];
            check("lookup of an evicted entry misses (and queues its promotion)", verify(&mut a, i).is_none());
            std::thread::sleep(Duration::from_secs(2));
            check("evicted entry promoted from the SSD tier, same bytes", verify(&mut a, i) == Some(true));
            // PROBE (no lease): fill again, then the evicted ones read 2 (SSD only, promotion queued), then 1 (RAM)
            for i in 10_000..10_000 + 4 * n_slots {
                store(&mut a, i);
                verify(&mut a, i);
                verify(&mut a, i);
            }
            let ks: Vec<_> = (4..8).map(|i| key(0xd15c, i)).collect();
            let pr = a.probe(NS, &ks).unwrap();
            check("PROBE: entries evicted by the second fill are on the SSD tier only (2)", pr.contains(&2) && !pr.contains(&0));
            std::thread::sleep(Duration::from_secs(2));
            let pr2 = a.probe(NS, &[ks[pr.iter().position(|&x| x == 2).unwrap()], key(0xd15c, 999_999)]).unwrap();
            check("PROBE queued the promotion: back in RAM (1); unknown key absent (0)", pr2 == vec![1, 0]);
        }
        _ => panic!("disk write|check|promote"),
    }
}

/// On-demand pool (lazy.rs): run against a broker with --sweep-s 1 --evict-age-s 8 --idle-shrink-s 3 --compact-mib 64
/// --mem-reserve-gib 1 and LLMT_KV_MEMAVAIL_FILE=<file> (written here for the pressure phase). LAZY_PHASE=main|pressure.
fn lazy(sock: PathBuf, phase: &str) {
    use std::os::unix::fs::MetadataExt;
    let mut a = connect(&sock, 1);
    let slot = a.hello.slot_bytes;
    let pool = PoolMap::open(&a.hello).expect("map pool");
    let path = a.hello.pool_path.clone();
    let resident = || std::fs::metadata(&path).map(|m| m.blocks() * 512).unwrap_or(0);
    let pat = |i: u64, j: usize| ((i.wrapping_mul(0x9E37_79B9) as usize ^ j.wrapping_mul(131)) & 0xff) as u8;
    let store = |a: &mut Client, i: u64| {
        let k = key(0x1a2, i);
        let r = a.reserve(NS, vec![item(k, None, slot)]).unwrap();
        if r[0].0 != Status::Ok {
            return false;
        }
        for rg in &r[0].1 {
            let b = unsafe { pool.range(*rg) };
            b.write(0, &(0..b.len()).map(|j| pat(i, j)).collect::<Vec<u8>>());
        }
        a.commit(NS, &[k]).unwrap()[0] == Status::Ok
    };
    let verify = |a: &mut Client, i: u64| -> Option<bool> {
        let l = a.lookup(NS, &[key(0x1a2, i)], true).unwrap();
        if l.n_hit == 0 {
            return None;
        }
        let ok = l.entries[0].ranges.iter().all(|rg| unsafe { pool.range(*rg) }.to_vec(usize::MAX).iter().enumerate().all(|(j, x)| *x == pat(i, j)));
        a.release(&[l.lease_id]).unwrap();
        Some(ok)
    };
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    match phase {
        "main" => {
            let r0 = resident();
            println!("pool {} slots of {} KiB; resident at start {:.1} MiB", a.hello.n_slots, slot >> 10, mib(r0));
            check("pool starts (near) empty", r0 < slot);
            for i in 0..64 {
                check(&format!("store {i}"), store(&mut a, i));
            }
            let r1 = resident();
            println!("after 64 entries: {:.1} MiB", mib(r1));
            check("resident grows with commits (64 slots)", r1 >= 64 * slot && r1 < 66 * slot);
            // keep the even entries in use for 11 s: the odd ones age out (--evict-age-s 8), gradually
            let t = Instant::now();
            while t.elapsed() < Duration::from_secs(11) {
                for i in (0..64).step_by(2) {
                    verify(&mut a, i);
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            let gone = (0..64).filter(|i| i % 2 == 1).filter(|&i| a.lookup(NS, &[key(0x1a2, i)], false).unwrap().n_hit == 0).count();
            let r2 = resident();
            println!("after the age sweep: {gone} of 32 unused entries evicted, resident {:.1} MiB", mib(r2));
            check("aged entries evicted, used ones kept", gone == 32);
            check("resident drops with them (punched)", r2 <= 33 * slot);
            // quiet (> 3 s without requests): compaction moves the even entries (spread over 64 slots) down
            let rel0 = stat(&mut a, "relocations");
            std::thread::sleep(Duration::from_secs(5));
            let rel = stat(&mut a, "relocations") - rel0;
            println!("compaction: {rel} entries moved");
            check("compaction moved entries into the low holes", rel > 0);
            for i in (0..64).step_by(2) {
                check(&format!("entry {i} after compaction: same bytes"), verify(&mut a, i) == Some(true));
            }
            check("resident after compaction still 32 slots", resident() <= 33 * slot);
            println!("lazy main: done (disconnecting: the idle shrink follows)");
        }
        "pressure" => {
            let f = std::env::var("LLMT_KV_MEMAVAIL_FILE").expect("LLMT_KV_MEMAVAIL_FILE");
            std::fs::write(&f, format!("{}", 64u64 << 30)).unwrap();
            for i in 1000..1032 {
                check(&format!("store {i}"), store(&mut a, i));
                verify(&mut a, i);
            }
            let r1 = resident();
            std::fs::write(&f, format!("{}", (1u64 << 30) - 1)).unwrap(); // below --mem-reserve-gib 1
            std::thread::sleep(Duration::from_secs(3));
            let r2 = resident();
            std::fs::write(&f, format!("{}", 64u64 << 30)).unwrap();
            println!("pressure: resident {:.1} -> {:.1} MiB", mib(r1), mib(r2));
            check("memory pressure evicts recently used entries too and punches them", r2 < r1 / 2);
        }
        _ => panic!("lazy main|pressure"),
    }
}

/// Unified broker (slab classes): run against a broker with --class big=1024 --class small=512 --pool-gib 0.046875
/// (48 MiB ceiling) --sweep-s 1 --evict-age-s 8 --inactive-age-div 4 --compact-mib 0.
fn classes(sock: PathBuf) {
    use std::os::unix::fs::MetadataExt;
    let mut a = Client::connect(&sock, 1, "big", Duration::from_millis(500)).expect("connect big");
    let mut b = Client::connect(&sock, 2, "small", Duration::from_millis(500)).expect("connect small");
    check("HELLO name picks the slab class: 1 MiB and 512 KiB slots", a.hello.slot_bytes == 1 << 20 && b.hello.slot_bytes == 512 << 10);
    check("separate pool files per class", a.hello.pool_path != b.hello.pool_path);
    let pa = PoolMap::open(&a.hello).expect("map a");
    let pb = PoolMap::open(&b.hello).expect("map b");
    let file = |c: &Client| std::fs::metadata(&c.hello.pool_path).map(|m| m.blocks() * 512).unwrap_or(0);
    let store = |c: &mut Client, pool: &PoolMap, tag: u64, i: u64| {
        let slot = c.hello.slot_bytes;
        let k = key(tag, i);
        let r = c.reserve(NS, vec![item(k, None, slot)]).unwrap();
        if r[0].0 != Status::Ok {
            return false;
        }
        for rg in &r[0].1 {
            { let b = unsafe { pool.range(*rg) }; b.write(0, &(0..b.len()).map(|j| (i as usize ^ j) as u8).collect::<Vec<u8>>()); }
        }
        c.commit(NS, &[k]).unwrap()[0] == Status::Ok
    };
    let has = |c: &mut Client, tag: u64, i: u64| c.lookup(NS, &[key(tag, i)], false).unwrap().n_hit == 1;
    // 32 MiB in the big class (older), then 32 MiB in the small one: 64 MiB > the 48 MiB ceiling
    for i in 0..32 {
        check(&format!("big {i}"), store(&mut a, &pa, 0xb16, i));
    }
    std::thread::sleep(Duration::from_millis(1200));
    for i in 0..64 {
        check(&format!("small {i}"), store(&mut b, &pb, 0x5a1, i));
    }
    std::thread::sleep(Duration::from_millis(2500));
    let big_left = (0..32).filter(|&i| has(&mut a, 0xb16, i)).count();
    let small_left = (0..64).filter(|&i| has(&mut b, 0x5a1, i)).count();
    let total = big_left as u64 * (1 << 20) + small_left as u64 * (512 << 10);
    println!("after the ceiling sweep: big {big_left}/32, small {small_left}/64, live {} MiB; files {} + {} MiB",
             total >> 20, file(&a) >> 20, file(&b) >> 20);
    check("one ceiling over both classes (<= 48 MiB live)", total <= 48 << 20);
    check("global heat order: the older class's entries went first", big_left <= 16 && small_left == 64);
    check("freed RAM returned (big class file shrank)", file(&a) <= (big_left as u64 + 1) << 20);
    // inactive domain: the big client leaves; its entries age 4x faster (8 s / 4 = 2 s) while small (connected,
    // touched) stays
    drop(a);
    for _ in 0..28 {
        for i in 0..64 {
            has(&mut b, 0x5a1, i);
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let mut a2 = Client::connect(&sock, 3, "kvbench-other", Duration::from_millis(500)).expect("connect");   // unknown name: class 0 (big)
    let big_now = (0..32).filter(|&i| has(&mut a2, 0xb16, i)).count();
    let small_left = (0..64).filter(|&i| has(&mut b, 0x5a1, i)).count();
    println!("14 s after the big client left: big {big_now}/{big_left}, small {small_left}/64");
    check("a namespace without clients ages 4x faster, 25 % of the pool per sweep: big emptied", big_now == 0);
    check("the active namespace stays", small_left == 64);
}

/// Compression is idle work: run against a broker like rcomp's but with the default
/// --ram-compress-quiet-ms 2000. While a client keeps looking entries up (every 150 ms) nothing is rewritten; once it
/// stops, the next sweeps compress everything.
fn rquiet(sock: PathBuf) {
    let mut a = Client::connect(&sock, 1, "z", Duration::from_millis(500)).expect("connect");
    let slot = a.hello.slot_bytes as usize;
    let pool = PoolMap::open(&a.hello).expect("map");
    let pat = |i: u64, j: usize| -> u8 { if j % 4 == 3 { 0 } else { ((i as usize * 5 + j / 4) % 53) as u8 } };
    for i in 0..8u64 {
        let k = key(0x2d0, i);
        let r = a.reserve(NS, vec![item(k, None, slot as u64)]).unwrap();
        check(&format!("store {i}"), r[0].0 == Status::Ok);
        for rg in &r[0].1 {
            { let b = unsafe { pool.range(*rg) }; b.write(0, &(0..b.len()).map(|j| pat(i, j)).collect::<Vec<u8>>()); }
        }
        a.commit(NS, &[k]).unwrap();
    }
    let compressed = |c: &mut Client| -> usize {
        let mut n = 0;
        for i in 0..8u64 {
            let l = c.lookup(NS, &[key(0x2d0, i)], true).unwrap();
            if l.n_hit == 1 {
                n += kvslotz::is_compressed(&unsafe { pool.range(l.entries[0].ranges[0]) }.to_vec(usize::MAX)) as usize;
                c.release(&[l.lease_id]).unwrap();
            }
        }
        n
    };
    // busy: a lookup of another key every 150 ms for 5 s (entries age past --ram-compress-after-s meanwhile)
    let t = std::time::Instant::now();
    while t.elapsed() < Duration::from_secs(5) {
        let _ = a.lookup(NS, &[key(0x2d1, 0)], false).unwrap();
        std::thread::sleep(Duration::from_millis(150));
    }
    let busy = compressed(&mut a);
    check("busy client: nothing compressed", busy == 0);
    std::thread::sleep(Duration::from_secs(5));
    let quiet = compressed(&mut a);
    println!("compressed while busy {busy}/8, after 5 s quiet {quiet}/8");
    check("quiet: all compressed", quiet == 8);
}

/// In-place RAM compression (class rcodec): run against a broker with --class z=1024,rcodec=1,chunk=262144
/// --pool-gib 0.0625 --sweep-s 1 --ram-compress-after-s 1 --compact-mib 0 --ram-compress-quiet-ms 0. A leased entry is
/// never rewritten; readers racing the sweep (lease just before / after a rewrite) always see the right bytes (raw or
/// compressed slot).
fn rcomp(sock: PathBuf) {
    use std::os::unix::fs::MetadataExt;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    let mut a = Client::connect(&sock, 1, "z", Duration::from_millis(500)).expect("connect");
    let slot = a.hello.slot_bytes as usize;
    let pool = PoolMap::open(&a.hello).expect("map");
    let path = a.hello.pool_path.clone();
    let file = || std::fs::metadata(&path).map(|m| m.blocks() * 512).unwrap_or(0);
    // compressible like KV: small values in 4-byte words
    let pat = |i: u64, j: usize| -> u8 { if j % 4 == 3 { 0 } else { ((i as usize * 7 + j / 4) % 61) as u8 } };
    for i in 0..16u64 {
        let k = key(0x2c0, i);
        let r = a.reserve(NS, vec![item(k, None, slot as u64)]).unwrap();
        check(&format!("store {i}"), r[0].0 == Status::Ok);
        for rg in &r[0].1 {
            { let b = unsafe { pool.range(*rg) }; b.write(0, &(0..b.len()).map(|j| pat(i, j)).collect::<Vec<u8>>()); }
        }
        a.commit(NS, &[k]).unwrap();
    }
    let f0 = file();
    // read an entry through the reader path (compressed slots decompressed), under a lease
    let read = |c: &mut Client, pool: &PoolMap, i: u64| -> Option<(bool, bool)> {
        let l = c.lookup(NS, &[key(0x2c0, i)], true).unwrap();
        if l.n_hit == 0 {
            return None;
        }
        let rg = l.entries[0].ranges[0];
        let s0 = unsafe { pool.range(rg) }.to_vec(usize::MAX);
        let compressed = kvslotz::is_compressed(&s0);
        let mut out = vec![0u8; slot];
        let ok = if compressed { kvslotz::read_range(&s0, 0, &mut out) } else { out.copy_from_slice(&s0[..slot]); true };
        let same = ok && out.iter().enumerate().all(|(j, x)| *x == pat(i, j));
        c.release(&[l.lease_id]).unwrap();
        Some((compressed, same))
    };
    // hold a lease on entry 0 across the sweeps
    let held = a.lookup(NS, &[key(0x2c0, 0)], true).unwrap();
    check("lease on entry 0", held.n_hit == 1);
    // a racing reader: lookups + reads of every entry while the compressor runs
    let stop = AtomicBool::new(false);
    let (bad, misses, reads) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    std::thread::scope(|s| {
        s.spawn(|| {
            let mut b = Client::connect(&sock, 2, "z", Duration::from_millis(500)).expect("connect b");
            let pb = PoolMap::open(&b.hello).expect("map b");
            while !stop.load(Relaxed) {
                for i in 1..16 {
                    match read(&mut b, &pb, i) {
                        Some((_, true)) => { reads.fetch_add(1, Relaxed); }
                        Some((_, false)) => { bad.fetch_add(1, Relaxed); }
                        None => { misses.fetch_add(1, Relaxed); }
                    }
                    // each entry read every ~2.2 s: older than --ram-compress-after-s between reads, so the
                    // compressor rewrites slots between (and racing) these lookups
                    std::thread::sleep(Duration::from_millis(150));
                }
            }
        });
        std::thread::sleep(Duration::from_secs(9));
        stop.store(true, Relaxed);
    });
    let comp_seen = (1..16).filter(|&i| read(&mut a, &pool, i).is_some_and(|x| x.0)).count();
    println!("racing reader: {} good reads, {} wrong, {} misses (lookups during a rewrite miss); {comp_seen}/15 slots compressed meanwhile",
             reads.load(Relaxed), bad.load(Relaxed), misses.load(Relaxed));
    check("the compressor ran while the reader was reading", comp_seen > 0);
    check("racing reads never see wrong bytes", bad.load(Relaxed) == 0);
    let s0 = unsafe { pool.range(held.entries[0].ranges[0]) }.to_vec(usize::MAX);
    check("the leased entry was not rewritten", !kvslotz::is_compressed(&s0) && s0.iter().enumerate().all(|(j, x)| *x == pat(0, j)));
    // the reader's lookups kept the others recently used: wait until they are older than 1 s
    std::thread::sleep(Duration::from_secs(3));
    let mut comp = 0;
    for i in 1..16 {
        let r = read(&mut a, &pool, i);
        check(&format!("entry {i} reads back identical"), r.is_some_and(|x| x.1));
        comp += r.is_some_and(|x| x.0) as u32;
    }
    let f1 = file();
    println!("compressed {comp}/15 unleased entries; pool file {} -> {} KiB", f0 >> 10, f1 >> 10);
    check("unleased entries compressed in place", comp == 15);
    check("their slot tails punched (pool file smaller)", f1 < f0 * 3 / 4);
    a.release(&[held.lease_id]).unwrap();
    std::thread::sleep(Duration::from_secs(3));
    let r = read(&mut a, &pool, 0);
    check("entry 0 compressed after its lease ended, identical", r == Some((true, true)));
}

fn scenarios(sock: PathBuf) {
    let run = std::process::id() as u64; // keys unique per run, so a long-lived broker can be reused
    let mut a = connect(&sock, 1);
    let mut b = connect(&sock, 2);
    let slot = a.hello.slot_bytes;
    println!("broker {} epoch {:016x}, {} slots of {} KiB", a.hello.broker_version, a.hello.epoch, a.hello.n_slots, slot >> 10);

    // 1. data path: A writes a pattern through the pool, B leases and reads it back
    let pool_a = PoolMap::open(&a.hello).expect("map pool");
    let pool_b = PoolMap::open(&b.hello).expect("map pool");
    let k1 = key(run, 1);
    let r = a.reserve(NS, vec![item(k1, None, slot * 2 + 1)]).unwrap();
    check("reserve 2 slots + 1 byte -> 3 slots", r[0].0 == Status::Ok && r[0].1.iter().map(|x| x.len).sum::<u32>() == 3);
    for rg in &r[0].1 {
        let s = unsafe { pool_a.range(*rg) };
        s.write(0, &(0..s.len()).map(|i| (i as u64 ^ run) as u8).collect::<Vec<u8>>());
    }
    check("reader can't see a reservation", b.lookup(NS, &[k1], true).unwrap().n_hit == 0);
    check("second writer gets BUSY", b.reserve(NS, vec![item(k1, None, 10)]).unwrap()[0].0 == Status::Busy);
    check("second writer can't commit", b.commit(NS, &[k1]).unwrap()[0] == Status::NotOwner);
    check("commit", a.commit(NS, &[k1]).unwrap()[0] == Status::Ok);
    let l = b.lookup(NS, &[k1], true).unwrap();
    let same = l.entries[0].ranges.iter().all(|rg| {
        let s = unsafe { pool_b.range(*rg) }.to_vec(usize::MAX);
        s.iter().enumerate().all(|(i, x)| *x == (i as u64 ^ run) as u8)
    });
    check("reader sees the writer's bytes through its own mapping", l.n_hit == 1 && same);
    check("meta and producer come back", l.entries[0].meta == b"seg:kv+state" && l.entries[0].producer == 1);
    check("EXISTS after commit", a.reserve(NS, vec![item(k1, None, 10)]).unwrap()[0].0 == Status::Exists);

    // 2. crash with a live lease and live reservations: B's socket closes
    let free_before = stat(&mut a, "slots_free");
    b.reserve(NS, vec![item(key(run, 2), Some(k1), slot * 4), item(key(run, 3), None, slot)]).unwrap();
    check("reservations take slots", stat(&mut a, "slots_free") == free_before - 5);
    let leases_before = stat(&mut a, "leases_active");
    drop(b);
    std::thread::sleep(Duration::from_millis(50));
    check("disconnect frees reservations", stat(&mut a, "slots_free") == free_before);
    check("disconnect ends leases", stat(&mut a, "leases_active") == leases_before - 1);
    check("aborted reservation gone", a.lookup(NS, &[k1, key(run, 2)], false).unwrap().n_hit == 1);

    // 3. WAIT across connections (prefill offload): C waits, A commits
    let k4 = key(run, 4);
    let sock2 = sock.clone();
    let waiter = std::thread::spawn(move || {
        let mut c = connect(&sock2, 3);
        let t = Instant::now();
        let st = c.wait(NS, k4, 5000).unwrap();
        (st, t.elapsed())
    });
    std::thread::sleep(Duration::from_millis(100));
    a.reserve(NS, vec![item(k4, Some(k1), 100)]).unwrap();
    a.commit(NS, &[k4]).unwrap();
    let (st, el) = waiter.join().unwrap();
    check("WAIT returns on commit", st == Status::Ok && el < Duration::from_millis(1000));
    let mut c = connect(&sock, 3);
    check("WAIT times out", c.wait(NS, key(run, 99), 50).unwrap() == Status::Timeout);

    // 4. eviction never takes a leased entry; FULL when everything is leased
    let n = a.hello.n_slots as u64;
    let l1 = c.lookup(NS, &[k1], true).unwrap();
    let mut filled = 0;
    for i in 0..n {
        let k = key(run, 1000 + i);
        if a.reserve(NS, vec![item(k, None, slot)]).unwrap()[0].0 == Status::Ok {
            a.commit(NS, &[k]).unwrap();
            filled += 1;
        }
    }
    check("pool cycled through eviction", filled == n);
    check("leased entry survived eviction", c.lookup(NS, &[k1], false).unwrap().n_hit == 1);
    let big = a.reserve(NS, vec![item(key(run, 5), None, slot * n)]).unwrap();
    check("whole-pool reserve is FULL while a lease pins slots", big[0].0 == Status::Full);
    c.release(&[l1.lease_id]).unwrap();
    let big = a.reserve(NS, vec![item(key(run, 5), None, slot * n)]).unwrap();
    check("after release the whole pool can be reserved", big[0].0 == Status::Ok);
    a.abort(NS, &[key(run, 5)]).unwrap();

    // 5. garbage on the socket kills only that connection
    let mut raw = std::os::unix::net::UnixStream::connect(&sock).unwrap();
    raw.write_all(&[0xff, 0xff, 0xff, 0x7f, 1, 2, 3]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    check("broker survives a bad frame", a.stats().is_ok());
    let mut d = Client::connect(&sock, 4, "v0", Duration::from_millis(500));
    check("requests before HELLO are refused", {
        let mut raw = std::os::unix::net::UnixStream::connect(&sock).unwrap();
        raw.write_all(&kvproto::request_frame(1, &kvproto::Request::Stats)).unwrap();
        let mut buf = Vec::new();
        let _ = kvproto::read_frame(&mut raw, &mut buf).unwrap();
        kvproto::Reply::decode(kvproto::Op::Stats, &buf).unwrap().0 == Status::NoHello
    });
    check("client connects", d.as_mut().is_ok());
    println!("all scenarios passed");
}

/// Partial-tail entries (kind 4, Arc tail-hold spill): evicted before any other entry, age `--tail-age-div`
/// times faster, no TTL. Run against a fresh broker with a 64-slot pool:
///   --pool-gib 0.015625 --slot-kib 256 --low-water-pct 0 --sweep-s 1 --evict-age-s 8 --tail-age-div 4 --compact-mib 0
/// Request classes under room pressure (synthetic keys; shapes scaled from a production traffic test: an agent
/// phase, then single-stream low-priority one-off prompts that overflow the pool, then the agents come back).
/// Agents: AGENTS sessions over one shared system prefix (SYS blocks), each turn re-reads its chain and stores TURN_NEW
/// blocks; low phase: LOW_REQS requests over their own shared prefix (LOW_SYS blocks, reused by every such request)
/// plus LOW_OWN one-off blocks each. tagged: agents TOUCH_TAGGED class agent with a session, low requests class low +
/// ENTRY_LOW stores; untagged: the same requests as today (TOUCH_REQUEST, no flags). Prints the agents' chain blocks
/// still in RAM when they return and the low prefix's hit rate during its phase.
#[allow(non_snake_case)]
fn phases(sock: PathBuf, tagged: bool) {
    let env = |n: &str, d: u64| std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (AGENTS, SYS, TURNS, TURN_NEW) = (env("PHASES_AGENTS", 12), env("PHASES_SYS", 8), env("PHASES_TURNS", 6), 3u64);
    let (LOW_REQS, LOW_SYS, LOW_OWN) = (env("PHASES_LOW_REQS", 150), env("PHASES_LOW_SYS", 10), env("PHASES_LOW_OWN", 8));
    let run = std::process::id() as u64;
    let mut c = connect(&sock, 1);
    let slot = c.hello.slot_bytes;
    let tags = tagged && c.features & kvclient::FEAT_REQUEST_TAGS != 0;
    println!("broker {}, {} slots of {} KiB, request tags {}", c.hello.broker_version, c.hello.n_slots, slot >> 10,
             if tags { "on" } else if tagged { "requested, broker without them" } else { "off (today's behaviour)" });
    let k_sys = |i: u64| key(run, 0x5000 + i);
    let k_own = |s: u64, i: u64| key(run, 0x10_0000 + s * 0x1000 + i);
    let k_low_sys = |i: u64| key(run, 0x6000 + i);
    let k_low = |r: u64, i: u64| key(run, 0x20_0000 + r * 0x100 + i);
    // one request: lookup its chain (hit length), touch it with its class, store what is missing
    let mut request = |c: &mut Client, chain: &[Key], class: u8, sess: u64| -> u32 {
        let hit = c.lookup(NS, chain, false).unwrap().n_hit;
        if tags {
            c.touch_tagged(NS, &[(0, chain.to_vec())], sess, 0, class, 0, 0).unwrap();
        } else {
            c.touch_request(NS, &[(0, chain.to_vec())]).unwrap();
        }
        let flags = if tags && class == kvclient::CLASS_LOW { kvclient::ENTRY_LOW } else { 0 };
        for k in &chain[hit as usize..] {
            let it = ReserveItem { key: *k, parent: None, bytes: slot, kind: 0, format: 1, cost_hint: 0, flags, ttl_ms: 0,
                                   meta: vec![] };
            if c.reserve(NS, vec![it]).unwrap()[0].0 == Status::Ok {
                c.commit(NS, &[*k]).unwrap();
            }
        }
        hit
    };
    let agent_chain = |s: u64, turns: u64| -> Vec<Key> {
        (0..SYS).map(k_sys).chain((0..turns * TURN_NEW).map(|i| k_own(s, i))).collect()
    };
    // 1. agent phase: sessions take turns, each turn continues its chain
    let mut agent_hit = (0u64, 0u64);
    for t in 1..=TURNS {
        for s in 0..AGENTS {
            let ch = agent_chain(s, t);
            let h = request(&mut c, &ch, kvclient::CLASS_AGENT, 0x100 + s) as u64;
            if t > 1 {
                agent_hit.0 += h;
                agent_hit.1 += (agent_chain(s, t - 1)).len() as u64;
            }
        }
    }
    // 2. low phase: one-off prompts over a shared low prefix, one at a time
    let mut low_pfx = (0u64, 0u64);
    for r in 0..LOW_REQS {
        let ch: Vec<Key> = (0..LOW_SYS).map(k_low_sys).chain((0..LOW_OWN).map(|i| k_low(r, i))).collect();
        let h = request(&mut c, &ch, kvclient::CLASS_LOW, 0x9000 + r) as u64;
        if r > 0 {
            low_pfx.0 += h.min(LOW_SYS);
            low_pfx.1 += LOW_SYS;
        }
    }
    // 3. the agents return: how much of each chain is still in RAM
    let (mut back, mut total) = (0u64, 0u64);
    for s in 0..AGENTS {
        let ch = agent_chain(s, TURNS);
        back += c.lookup(NS, &ch, false).unwrap().n_hit as u64;
        total += ch.len() as u64;
    }
    let st: std::collections::HashMap<String, u64> = c.stats().unwrap().into_iter().collect();
    let g = |n: &str| st.get(n).copied().unwrap_or(0);
    println!("agent phase: follow-up turns found {:.0}% of their previous chain in RAM", 100.0 * agent_hit.0 as f64 / agent_hit.1.max(1) as f64);
    println!("low phase: the low requests' shared prefix hit {:.0}% ({} of {} blocks)", 100.0 * low_pfx.0 as f64 / low_pfx.1.max(1) as f64,
             low_pfx.0, low_pfx.1);
    println!("agents return: {back} of {total} chain blocks in RAM ({:.0}%)", 100.0 * back as f64 / total.max(1) as f64);
    println!("broker: committed {}, low committed {}, evicted low {} / normal {}, low picked over the policy {}, raised {}",
             g("committed"), g("low_committed"), g("low_evicted"), g("normal_evicted"), g("low_picked"), g("low_promoted"));
    println!("phases: OK");
}

fn tail(sock: PathBuf) {
    const TAIL: u8 = 4;
    let run = std::process::id() as u64;
    let mut a = connect(&sock, 1);
    let slot = a.hello.slot_bytes;
    let n = a.hello.n_slots as u64;
    let pool = PoolMap::open(&a.hello).expect("map pool");
    println!("broker {}, {} slots of {} KiB", a.hello.broker_version, n, slot >> 10);
    let it = |k: Key, kind: u8, slots: u64| ReserveItem { key: k, parent: None, bytes: slots * slot, kind, format: 1,
                                                           cost_hint: 0, flags: 0, ttl_ms: 0, meta: b"L1".to_vec() };
    let pat = |i: u64, j: usize| ((i.wrapping_mul(0x9E37_79B9) as usize ^ j.wrapping_mul(131)) & 0xff) as u8;
    let put = |a: &mut Client, k: Key, kind: u8, slots: u64, p: u64| -> bool {
        let r = a.reserve(NS, vec![it(k, kind, slots)]).unwrap();
        if r[0].0 != Status::Ok {
            return false;
        }
        for rg in &r[0].1 {
            { let b = unsafe { pool.range(*rg) }; b.write(0, &(0..b.len()).map(|j| pat(p, j)).collect::<Vec<u8>>()); }
        }
        a.commit(NS, &[k]).unwrap()[0] == Status::Ok
    };
    let has = |a: &mut Client, k: Key| a.lookup(NS, &[k], false).unwrap().n_hit == 1;

    // 1. a tail entry (4 slots: 1 attention page + 3 GDN state pages) round-trips its bytes under a lease
    let t1 = key(run, 0x7a11);
    check("store a 4-slot tail entry", put(&mut a, t1, TAIL, 4, 1));
    check("entries_kind4 counts it", stat(&mut a, "entries_kind4") == 1);
    let l = a.lookup(NS, &[t1], true).unwrap();
    let same = l.n_hit == 1 && l.entries[0].kind == TAIL
        && l.entries[0].ranges.iter().all(|rg| unsafe { pool.range(*rg) }.to_vec(usize::MAX).iter().enumerate().all(|(j, x)| *x == pat(1, j)));
    check("tail entry reads back (kind 4, same bytes)", same);
    a.release(&[l.lease_id]).unwrap();

    // 2. RESERVE evicts tails first: fill the pool with prefix entries (kind 0, 1 slot each, stored BEFORE the second
    //    tail so they are older); the tails go before any of them
    let t2 = key(run, 0x7a12);
    let pre: Vec<Key> = (0..(n - 8)).map(|i| key(run, 0x100 + i)).collect();
    for (i, k) in pre.iter().enumerate() {
        check(&format!("prefix {i}"), put(&mut a, *k, 0, 1, 0x100 + i as u64));
    }
    check("store a second tail", put(&mut a, t2, TAIL, 4, 2));
    check("pool full", stat(&mut a, "slots_free") == 0);
    check("one more prefix entry fits", put(&mut a, key(run, 0x900), 0, 4, 0x900));
    check("... by evicting a tail (the least recently used one)", !has(&mut a, t1) && has(&mut a, t2));
    check("... and no prefix entry", pre.iter().all(|k| has(&mut a, *k)));
    check("next store takes the other tail", put(&mut a, key(run, 0x901), 0, 4, 0x901) && !has(&mut a, t2));
    check("prefix entries still all there", pre.iter().all(|k| has(&mut a, *k)));

    // 3. age: --evict-age-s 8 / --tail-age-div 4: an unused tail goes after ~2 s, prefix entries stay (keep them used)
    let k0 = key(run, 0x100);
    let t3 = key(run, 0x7a13);
    let ok = put(&mut a, t3, TAIL, 4, 3); // evicts the coldest prefix entries if needed
    check("store a third tail", ok);
    let t = Instant::now();
    let mut gone_at = None;
    while t.elapsed() < Duration::from_secs(6) {
        let _ = a.touch(NS, &pre[..pre.len() / 2]);
        if gone_at.is_none() && stat(&mut a, "entries_kind4") == 0 {   // (a lookup would refresh the tail)
            gone_at = Some(t.elapsed());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    println!("unused tail evicted after {:?} (evict age 8 s, tail div 4 -> ~2 s + sweep 1 s)", gone_at);
    check("unused tail aged out within 6 s (prefix age 8 s)", gone_at.is_some_and(|d| d < Duration::from_secs(6)));
    check("recently used prefix entries kept", pre[..pre.len() / 2].iter().all(|k| has(&mut a, *k)) && has(&mut a, k0));
    println!("tail: done");
}

/// Prefix-affinity hints (docs/prefix-hints.md): two engines hint the same router anchor, a router (observer engine id)
/// asks WHERE; the answer follows what is committed in RAM, and WHERE moves no counter (read-only).
fn hints(sock: PathBuf) {
    use kvclient::{FEAT_INVALIDATE, FEAT_PREFIX_HINTS, FEAT_TOUCH_REQUEST, HINT_VRAM, OBSERVER_ENGINE_ID};
    let run = std::process::id() as u64;
    let feat = |c: &mut Client| c.negotiate(FEAT_TOUCH_REQUEST, vec![]).unwrap();
    let mut a = Client::connect(&sock, 11, "arc", Duration::from_millis(500)).expect("connect");
    let mut b = Client::connect(&sock, 12, "rtx5090", Duration::from_millis(500)).expect("connect");
    let mut r = Client::connect(&sock, OBSERVER_ENGINE_ID, "router", Duration::from_millis(500)).expect("connect");
    feat(&mut a);
    feat(&mut b);
    r.negotiate(0, vec![]).unwrap();
    check("broker offers FEAT_PREFIX_HINTS", r.features & FEAT_PREFIX_HINTS != 0);
    let slot = a.hello.slot_bytes;
    let it = |k: Key, kind: u8| ReserveItem { key: k, parent: None, bytes: slot, kind, format: 1, cost_hint: 0, flags: 0,
                                             ttl_ms: 0, meta: vec![] };
    // engine A: 5 attention chunks, GDN state at chunks 2 and 4 (0-based); stores chunks 0..4 (4 of them)
    let attn: Vec<Key> = (0..5).map(|i| key(run, 0x4100 + i)).collect();
    let st: Vec<Key> = (0..5).map(|i| key(run, 0x4200 + i)).collect();
    let mut items: Vec<ReserveItem> = attn[..4].iter().map(|k| it(*k, 0)).collect();
    items.push(it(st[3], 1));
    items.push(it(st[1], 1));
    check("engine A stores 4 attention chunks + 2 states", a.reserve(NS, items.clone()).unwrap().iter().all(|x| x.0 == Status::Ok));
    let keys: Vec<Key> = items.iter().map(|x| x.key).collect();
    check("... and commits them", a.commit(NS, &keys).unwrap().iter().all(|s| *s == Status::Ok));
    let anchor = { let mut x = [0u8; 16]; x[..8].copy_from_slice(&run.to_le_bytes()); x[8] = 1; x };
    let older = { let mut x = anchor; x[8] = 2; x };
    let groups = vec![(0u8, attn.clone()), (1u8, st.clone())];
    check("engine A: HINT", a.hint(NS, &anchor, 1664, HINT_VRAM, &groups).is_ok());
    check("engine A: HINT of an older, shorter anchor", a.hint(NS, &older, 1664, 0, &[(0, attn[..2].to_vec()), (1, st[..2].to_vec())]).is_ok());
    // engine B hinted the same anchor but holds nothing (its keys differ: other compat tag)
    let battn: Vec<Key> = (0..5).map(|i| key(run, 0x4300 + i)).collect();
    check("engine B: HINT (nothing stored)", b.hint(NS, &anchor, 1664, 0, &[(0, battn)]).is_ok());
    check("a HINT without keys is refused", r.hint(NS, &anchor, 1664, 0, &[(0, vec![])]).is_err());
    let before: Vec<(String, u64)> = a.stats().unwrap().into_iter()
        .filter(|(k, _)| ["lookups", "hit_entries", "misses", "partial_hits", "full_hits"].contains(&k.as_str())).collect();
    let unknown = [0xeeu8; 16];
    let w = r.where_(&[unknown, anchor, older]).unwrap();
    println!("WHERE -> {w:?}");
    let pa = w.iter().find(|p| p.engine_id == 11);
    let pb = w.iter().find(|p| p.engine_id == 12);
    check("WHERE: one place per engine", w.len() == 2);
    check("WHERE: engine A, anchor 1 (the longest it hinted), 4 of 5 chunks resident", pa.is_some_and(|p| p.anchor_idx == 1
        && p.n_chunks == 5 && p.n_resident == 4 && p.chunk_tokens == 1664 && p.flags == HINT_VRAM && p.name == "arc"));
    check("WHERE: engine B, 0 resident", pb.is_some_and(|p| p.n_resident == 0 && p.name == "rtx5090"));
    for _ in 0..20 {
        r.where_(&[anchor]).unwrap();
    }
    let after: Vec<(String, u64)> = a.stats().unwrap().into_iter()
        .filter(|(k, _)| ["lookups", "hit_entries", "misses", "partial_hits", "full_hits"].contains(&k.as_str())).collect();
    check("WHERE is read-only (lookup / hit counters unchanged)", before == after && !before.is_empty());
    check("broker counts WHEREs", stat(&mut r, "where_requests") >= 21);
    // the state at chunk 3 goes: the resumable prefix falls back to the state at chunk 1 (2 chunks)
    a.negotiate(FEAT_TOUCH_REQUEST | FEAT_INVALIDATE, vec![]).unwrap();
    check("invalidate the chunk-3 state", a.invalidate(NS, &[st[3]]).map(|v| v[0] == Status::Ok).unwrap_or(false));
    let w = r.where_(&[anchor]).unwrap();
    check("WHERE follows the index: 2 chunks resident now", w.iter().any(|p| p.engine_id == 11 && p.n_resident == 2));
    // a re-HINT from the same engine replaces its hint
    check("engine A re-hints with 2 chunks", a.hint(NS, &anchor, 1664, 0, &[(0, attn[..2].to_vec())]).is_ok());
    let w = r.where_(&[anchor]).unwrap();
    check("replaced (n_chunks 2, flags 0)", w.iter().any(|p| p.engine_id == 11 && p.n_chunks == 2 && p.n_resident == 2 && p.flags == 0));
    check("WHERE of nothing known: empty", r.where_(&[unknown]).unwrap().is_empty());
    println!("hints: OK");
}

/// Shared-prefix class: run against a fresh broker with a 64-slot pool and --shared-gib large enough for
/// a few slots, e.g. --pool-gib 0.015625 --slot-kib 256 --low-water-pct 0 --shared-gib 0.002 (8 slots).
/// Two chains continuing from the same 2-chunk prefix make it shared (metadata parent links, "L1" + 16 B);
/// filling the pool then evicts everything else first.
fn shared(sock: PathBuf) {
    let run = std::process::id() as u64;
    let mut a = connect(&sock, 1);
    let slot = a.hello.slot_bytes;
    let n = a.hello.n_slots as u64;
    let meta = |parent: Option<Key>| {
        let mut m = b"L1".to_vec();
        m.extend_from_slice(&parent.map(|p| p[..16].to_vec()).unwrap_or(vec![0u8; 16]));
        m.extend_from_slice(&[0u8; 14]);
        m
    };
    let put = |a: &mut Client, k: Key, parent: Option<Key>| -> bool {
        let it = ReserveItem { key: k, parent: None, bytes: slot, kind: 0, format: 1, cost_hint: 0, flags: 0, ttl_ms: 0,
                               meta: meta(parent) };
        let r = a.reserve(NS, vec![it]).unwrap();
        r[0].0 == Status::Ok && a.commit(NS, &[k]).unwrap()[0] == Status::Ok
    };
    let has = |a: &mut Client, k: Key| a.lookup(NS, &[k], false).unwrap().n_hit == 1;
    let (s1, s2) = (key(run, 0x5101), key(run, 0x5102));
    check("system prompt chunk 1", put(&mut a, s1, None));
    check("system prompt chunk 2", put(&mut a, s2, Some(s1)));
    check("session A continues", put(&mut a, key(run, 0xa1), Some(s2)));
    check("not shared after one continuation", stat(&mut a, "shared_entries") == 0);
    check("session B continues", put(&mut a, key(run, 0xb1), Some(s2)));
    check("two continuations (a regenerate) are not shared", stat(&mut a, "shared_entries") == 0);
    check("session C continues", put(&mut a, key(run, 0xc1), Some(s2)));
    check("shared after three: the prefix and its root", stat(&mut a, "shared_entries") == 2);
    for i in 0..(n + 8) {
        check(&format!("filler {i}"), put(&mut a, key(run, 0x9000 + i), None));
    }
    check("the shared prefix survived a full pool turnover", has(&mut a, s1) && has(&mut a, s2));
    check("unshared older entries did not", !has(&mut a, key(run, 0xa1)) && !has(&mut a, key(run, 0xb1))
          && !has(&mut a, key(run, 0xc1)));
    println!("shared: done");
}

fn pct(v: &mut [u32], p: f64) -> u32 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() - 1) as f64 * p) as usize]
}

/// Each thread is one "engine" running scheduler steps: lookup a 32-key chain (acquire), release, reserve a batch
/// of new chain entries, commit them. Prints per-op latency and total requests/s.
fn load(sock: PathBuf, threads: usize, seconds: u64, keys: usize, batch: usize, rate: u64) {
    let bar = Arc::new(Barrier::new(threads));
    let hs: Vec<_> = (0..threads)
        .map(|t| {
            let sock = sock.clone();
            let bar = bar.clone();
            std::thread::spawn(move || {
                let mut c = connect(&sock, 100 + t as u16);
                c.set_timeout(Duration::from_millis(100));
                let slot = c.hello.slot_bytes;
                let mut lat: [Vec<u32>; 4] = Default::default(); // lookup, release, reserve, commit
                let mut errors = 0u64;
                let run = std::process::id() as u64;
                let mut chain = 0u64;
                bar.wait();
                let start = Instant::now();
                let end = start + Duration::from_secs(seconds);
                let mut step = 0u64;
                while Instant::now() < end {
                    if rate > 0 {
                        // paced: one scheduler step every 1/rate s, threads offset so they don't fire together
                        let due = start + Duration::from_secs_f64((step as f64 + t as f64 / threads as f64) / rate as f64);
                        step += 1;
                        if let Some(d) = due.checked_duration_since(Instant::now()) {
                            std::thread::sleep(d);
                        }
                    }
                    // half the lookups hit a chain this thread stored before (prefix re-use), half are new
                    let base = if chain > 0 && chain % 2 == 0 { chain - 1 } else { chain };
                    let ks: Vec<Key> = (0..keys as u64).map(|i| key(run ^ (t as u64) << 40 ^ base, i)).collect();
                    let t0 = Instant::now();
                    let l = c.lookup(NS, &ks, true);
                    lat[0].push(t0.elapsed().as_micros() as u32);
                    let hit = match l {
                        Ok(l) => {
                            if l.lease_id > 0 {
                                let t0 = Instant::now();
                                let _ = c.release(&[l.lease_id]);
                                lat[1].push(t0.elapsed().as_micros() as u32);
                            }
                            l.n_hit as usize
                        }
                        Err(_) => {
                            errors += 1;
                            0
                        }
                    };
                    if hit < ks.len() {
                        let todo: Vec<Key> = ks[hit..(hit + batch).min(ks.len())].to_vec();
                        let items: Vec<ReserveItem> = todo
                            .iter()
                            .enumerate()
                            .map(|(i, k)| item(*k, if hit + i > 0 { Some(ks[hit + i - 1]) } else { None }, slot))
                            .collect();
                        let t0 = Instant::now();
                        let r = c.reserve(NS, items);
                        lat[2].push(t0.elapsed().as_micros() as u32);
                        if let Ok(r) = r {
                            let ok: Vec<Key> = todo.iter().zip(&r).take_while(|(_, x)| x.0 == Status::Ok).map(|(k, _)| *k).collect();
                            let t0 = Instant::now();
                            let _ = c.commit(NS, &ok);
                            lat[3].push(t0.elapsed().as_micros() as u32);
                            // drop reservations behind a refused one (no parent -> later ones were refused too)
                            let rest: Vec<Key> = todo.iter().zip(&r).skip(ok.len()).filter(|(_, x)| x.0 == Status::Ok).map(|(k, _)| *k).collect();
                            if !rest.is_empty() {
                                let _ = c.abort(NS, &rest);
                            }
                        } else {
                            errors += 1;
                        }
                    }
                    if hit + batch >= ks.len() {
                        chain += 1;
                    }
                }
                (lat, errors)
            })
        })
        .collect();
    let mut all: [Vec<u32>; 4] = Default::default();
    let mut errors = 0;
    for h in hs {
        let (lat, e) = h.join().unwrap();
        errors += e;
        for i in 0..4 {
            all[i].extend(lat[i].iter());
        }
    }
    let total: usize = all.iter().map(|v| v.len()).sum();
    println!(
        "threads {threads}, {seconds} s, {keys} keys/lookup, batch {batch}, steps/s/thread {rate}: {total} requests, {:.0} req/s, errors {errors}",
        total as f64 / seconds as f64
    );
    for (i, name) in ["lookup", "release", "reserve", "commit"].iter().enumerate() {
        let v = &mut all[i];
        let n = v.len();
        println!(
            "  {name:8} n={n:8}  p50 {:5} us  p99 {:5} us  p99.9 {:5} us  max {:6} us",
            pct(v, 0.5),
            pct(v, 0.99),
            pct(v, 0.999),
            pct(v, 1.0)
        );
    }
    let mut c = connect(&sock, 99);
    for (k, v) in c.stats().unwrap() {
        if ["slots_free", "entries", "hit_entries", "committed", "evict_room", "busy", "full", "no_parent"].contains(&k.as_str()) {
            print!("{k}={v} ");
        }
    }
    println!();
    if errors > 0 {
        eprintln!("note: errors are client timeouts (100 ms) or refused requests");
    }
}

/// Memory-pressure protection. Broker: 64-slot pool of 256 KiB, --sweep-s 1, --mem-reserve-gib 8,
/// --protect-gib 0.002 (8 slots), --protect-window-s 3, --protect-floor-gib 2, LLMT_KV_MEMAVAIL_FILE=$MEMAVAIL_FILE
/// (this scenario writes MemAvailable there). PROTECT=off: the same broker without --protect-gib (today's behaviour).
fn protect(sock: PathBuf) {
    let f = std::env::var("MEMAVAIL_FILE").expect("MEMAVAIL_FILE");
    let on = std::env::var("PROTECT").unwrap_or_default() != "off";
    let set_mem = |gib: f64| std::fs::write(&f, format!("{}", (gib * (1u64 << 30) as f64) as u64)).unwrap();
    set_mem(64.0);
    let run = std::process::id() as u64;
    let mut a = connect(&sock, 1);
    let slot = a.hello.slot_bytes;
    let it = |k: Key, parent: Option<Key>| ReserveItem { key: k, parent, bytes: slot, kind: 0, format: 1, cost_hint: 0, flags: 0,
                                                          ttl_ms: 0, meta: b"L1".to_vec() };
    let put = |a: &mut Client, k: Key, parent: Option<Key>| {
        let r = a.reserve(NS, vec![it(k, parent)]).unwrap();
        r[0].0 == Status::Ok && a.commit(NS, &[k]).unwrap()[0] == Status::Ok
    };
    let has = |a: &mut Client, k: Key| a.lookup(NS, &[k], false).unwrap().n_hit == 1;
    // 20 cold entries (an old session), then let them age past the 3 s protection window
    let cold: Vec<Key> = (0..20).map(|i| key(run, 0xc0 + i)).collect();
    for (i, k) in cold.iter().enumerate() {
        check(&format!("cold {i}"), put(&mut a, *k, if i > 0 { Some(cold[i - 1]) } else { None }));
    }
    std::thread::sleep(Duration::from_secs(4));
    // the active session: a 6-entry chain, touched now
    let act: Vec<Key> = (0..6).map(|i| key(run, 0xa0 + i)).collect();
    for (i, k) in act.iter().enumerate() {
        check(&format!("active {i}"), put(&mut a, *k, if i > 0 { Some(act[i - 1]) } else { None }));
    }
    let _ = a.touch(NS, &act);
    // memory pressure: MemAvailable 4 GiB < reserve 8 GiB -> the sweep culls everything it may
    set_mem(4.0);
    std::thread::sleep(Duration::from_millis(2500));
    let _ = a.touch(NS, &act);
    let cold_left = cold.iter().filter(|k| has(&mut a, **k)).count();
    let act_left = act.iter().filter(|k| has(&mut a, **k)).count();
    println!("pressure: cold entries left {cold_left} of 20, active session entries left {act_left} of 6");
    check("pressure culls every cold entry", cold_left == 0);
    if on {
        check("the active session's chain survives (protected)", act_left == 6);
        // extreme pressure: MemAvailable 1 GiB < floor 2 GiB -> the protection gives way
        set_mem(1.0);
        std::thread::sleep(Duration::from_millis(2500));
        let act_left = act.iter().filter(|k| has(&mut a, **k)).count();
        println!("extreme pressure: active session entries left {act_left} of 6");
        check("below the floor the protection gives way", act_left == 0);
    } else {
        check("without protection the active session goes too (today's behaviour)", act_left == 0);
    }
    set_mem(64.0);
    println!("protect: done");
}

/// Strict SSD writes. Broker: --disk-dir $DISK_DIR --disk-write-mode strict --disk-session-resumes 3
/// --disk-session-gap-s 1 --disk-session-span-s 4 --evict-age-s 3 --sweep-s 1 --disk-flush-s 1 (seconds instead of
/// minutes / hours). A short-lived session ages out: nothing written. A long-term session (3 resumes after >= 1 s
/// idle over >= 4 s) ages out: its chain and its newest state written, the superseded state not. Pressure never writes.
fn strict(sock: PathBuf) {
    let dir = std::env::var("DISK_DIR").expect("DISK_DIR");
    let files = || std::fs::read_dir(&dir).map(|d| d.filter(|e| e.as_ref().is_ok_and(|e| e.path().extension().is_some_and(|x| x == "kvd"))).count()).unwrap_or(0);
    let run = std::process::id() as u64;
    let mut a = connect(&sock, 1);
    let slot = a.hello.slot_bytes;
    let it = |k: Key, parent: Option<Key>, kind: u8| ReserveItem { key: k, parent, bytes: slot, kind, format: 1, cost_hint: 0,
                                                                    flags: 0, ttl_ms: 0, meta: b"L1".to_vec() };
    let put = |a: &mut Client, k: Key, parent: Option<Key>, kind: u8| {
        let r = a.reserve(NS, vec![it(k, parent, kind)]).unwrap();
        r[0].0 == Status::Ok && a.commit(NS, &[k]).unwrap()[0] == Status::Ok
    };
    let hit2 = |a: &mut Client, k: Key| {
        for _ in 0..3 {
            let l = a.lookup(NS, &[k], true).unwrap();
            if l.n_hit == 1 {
                a.release(&[l.lease_id]).unwrap();
            }
        }
    };
    // the short-lived session: a chain of 3, looked up repeatedly (write-through mode would write it: >= 2 hits)
    let q: Vec<Key> = (0..3).map(|i| key(run, 0x50 + i)).collect();
    let _ = a.touch(NS, &q);
    for (i, k) in q.iter().enumerate() {
        check(&format!("short session {i}"), put(&mut a, *k, if i > 0 { Some(q[i - 1]) } else { None }, 0));
        hit2(&mut a, *k);
    }
    // the long-term session: attention chain s0 <- s1 <- s2, a state at s1 (old), later a state at s2 (newest)
    let s: Vec<Key> = (0..3).map(|i| key(run, 0x70 + i)).collect();
    let (st_old, st_new) = (key(run, 0x7e1), key(run, 0x7e2));
    let _ = a.touch(NS, &s[..2]);
    let _ = a.touch(NS, &[st_old]);
    check("long session s0", put(&mut a, s[0], None, 0));
    check("long session s1", put(&mut a, s[1], Some(s[0]), 0));
    check("long session old state", put(&mut a, st_old, None, 1));
    // 3 resumes, each after >= 1 s idle; the chain grows on the last one
    for r in 0..3 {
        std::thread::sleep(Duration::from_millis(1600));
        if r < 2 {
            let _ = a.touch(NS, &s[..2]);
            let _ = a.touch(NS, &[st_old]);
        } else {
            let _ = a.touch(NS, &s);
            let _ = a.touch(NS, &[st_old, st_new]);
            check("long session s2", put(&mut a, s[2], Some(s[1]), 0));
            check("long session newest state", put(&mut a, st_new, None, 1));
        }
    }
    check("nothing written while the sessions are in RAM (no write-through)", files() == 0);
    // both sessions go idle: everything ages out after 3 s (sweep every 1 s)
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(9) && (stat(&mut a, "entries_kind0") > 0 || stat(&mut a, "entries_kind1") > 0) {
        std::thread::sleep(Duration::from_millis(250));
    }
    check("both sessions aged out of RAM", stat(&mut a, "entries_kind0") == 0 && stat(&mut a, "entries_kind1") == 0);
    let n = files();
    println!("SSD files after the age-out: {n} (expected 4: s0, s1, s2 and the newest state)");
    check("the long-term session's chain + newest state written once (4 files); the short one and the superseded \
           state not", n == 4);
    println!("strict: done");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_default();
    let mut sock = PathBuf::from("/run/llmt-kv/broker.sock");
    let (mut threads, mut seconds, mut keys, mut batch, mut rate) = (4usize, 10u64, 32usize, 8usize, 0u64);
    let (mut file, mut scale, mut speed, mut reason) = (PathBuf::new(), 1u64, 1.0f64, String::from("manual"));
    if mode == "replay" {
        file = args.next().expect("kvbench replay FILE --socket S").into();
    }
    while let Some(f) = args.next() {
        let v = args.next().expect("flag value");
        match f.as_str() {
            "--scale" => scale = v.parse().unwrap(),
            "--speed" => speed = v.parse().unwrap(),
            "--reason" => reason = v,
            "--socket" => sock = v.into(),
            "--threads" => threads = v.parse().unwrap(),
            "--seconds" => seconds = v.parse().unwrap(),
            "--keys" => keys = v.parse().unwrap(),
            "--batch" => batch = v.parse().unwrap(),
            "--steps-per-s" => rate = v.parse().unwrap(),
            _ => panic!("unknown flag {f}"),
        }
    }
    match mode.as_str() {
        "scenarios" => scenarios(sock),
        "domains" => domains(sock),
        "disk" => disk(sock, &std::env::var("DISK_PHASE").unwrap_or_default()),
        "lazy" => lazy(sock, &std::env::var("LAZY_PHASE").unwrap_or_default()),
        "classes" => classes(sock),
        "rcomp" => rcomp(sock),
        "rquiet" => rquiet(sock),
        "tail" => tail(sock),
        "hints" => hints(sock),
        "shared" => shared(sock),
        "phases" => phases(sock, std::env::var("PHASES_TAGGED").map(|v| v != "0").unwrap_or(true)),
        "protect" => protect(sock),
        "strict" => strict(sock),
        "load" => load(sock, threads, seconds, keys, batch, rate),
        "race" => race::run(sock, threads, seconds, keys),
        "compact" => race::compact(sock),
        "cksum" => race::cksum(sock),
        "replay" => replay::run(&file, sock, scale, speed),
        "trace-dump" => replay::dump(sock, &reason),
        "ping" => replay::ping(sock, seconds),
        _ => {
            eprintln!("usage: kvbench scenarios|tail|shared|load|race --socket PATH [--threads N --seconds S --keys K --batch B --steps-per-s R]");
            std::process::exit(2);
        }
    }
}
