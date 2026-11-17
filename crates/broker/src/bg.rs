//! Background worker (`llmt-kvbroker --bg-worker SOCKET`): the broker's CPU-heavy background work,
//! integrity checksums (XXH3-64 over each entry's raw bytes), in-place slot compression (planes + zstd) and SSD tier
//! writes (copy, check against the RAM checksum, compress, write; the broker plans and records them) and compaction
//! copies (an entry's slots to lower ones; it stays readable at the old ones meanwhile), in a separate
//! process at the lowest priority (nice 19, idle I/O; its systemd unit adds CPUWeight=1), so the broker itself only
//! answers requests and frees memory (CPUWeight=1000). Same binary as the broker: always the same version.
//!
//! The worker maps each pool class like an engine and asks the broker for work (BG_JOBS: empty unless the broker is
//! quiet). Every decision stays in the broker's core: a job carries the entry's identity (guard), and a result or a
//! rewrite only counts while the entry is still exactly that one. A rewrite is begun (the entry stops being readable),
//! the image written into the slot by the worker, then ended (the broker punches the rest of the slot); a worker that
//! disappears in between leaves a torn slot, which the broker evicts. Without a worker the work pauses.
use kvclient::{BgJob, Client, PoolMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const TICK: Duration = Duration::from_millis(100);
const ENGINE_ID: u16 = kvclient::BG_ENGINE_ID;

struct Pool {
    c: Client,
    map: PoolMap,
    slot: u64,
}

pub fn run(sock: PathBuf) -> ! {
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, 19);
        libc::syscall(libc::SYS_ioprio_set, 1, 0, 3 << 13); // IOPRIO_WHO_PROCESS, IOPRIO_CLASS_IDLE
    }
    eprintln!("background worker: broker {}", sock.display());
    loop {
        match session(&sock) {
            Ok(()) => {}
            Err(e) => eprintln!("background worker: {e}; reconnecting"),
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

fn connect(sock: &Path, name: &str) -> Result<Pool, String> {
    let mut c = Client::connect(sock, ENGINE_ID, name, Duration::from_secs(5)).map_err(|e| format!("connect: {e:?}"))?;
    if c.features & kvclient::FEAT_BG_WORKER == 0 {
        return Err("the broker runs its background work itself (--bg internal) or is too old".into());
    }
    c.negotiate(kvclient::FEAT_BG_WORKER | kvclient::FEAT_TOUCH_REQUEST, kvclient::client_formats())
        .map_err(|e| format!("negotiate: {e:?}"))?;
    let map = PoolMap::open(&c.hello).map_err(|e| format!("map the pool: {e}"))?;
    let slot = c.hello.slot_bytes;
    Ok(Pool { c, map, slot })
}

fn session(sock: &Path) -> Result<(), String> {
    let mut first = connect(sock, "")?;
    let (_, _, classes, _) = first.c.bg_jobs(0, 0).map_err(|e| format!("bg_jobs: {e:?}"))?;
    let mut pools = vec![first];
    for name in classes.iter().skip(1) {
        pools.push(connect(sock, name)?);
    }
    eprintln!("background worker: connected, classes {classes:?}");
    loop {
        std::thread::sleep(TICK);
        for p in pools.iter_mut() {
            checksums(p).map_err(|e| format!("checksums: {e:?}"))?;
            compress(p).map_err(|e| format!("compression: {e:?}"))?;
            ssd_writes(p).map_err(|e| format!("SSD writes: {e:?}"))?;
            compact(p).map_err(|e| format!("compaction: {e:?}"))?;
        }
    }
}

/// Raw bytes of a job's entry (decompressing is the caller's business): the slots' contents, `bytes` long.
fn read_entry(p: &Pool, j: &BgJob, compressed: bool) -> Vec<u8> {
    if compressed {
        let s = unsafe { p.map.range(j.guard.ranges[0]) };
        return s.to_vec(s.len());
    }
    let mut out = vec![0u8; j.bytes as usize];
    let mut off = 0;
    for r in &j.guard.ranges {
        let s = unsafe { p.map.range(*r) };
        let n = s.len().min(out.len() - off);
        s.read(0, &mut out[off..off + n]);
        off += n;
    }
    out
}

/// One tick of checksums: at most a tenth of the broker's bytes-per-second budget (unified memory: the weights'
/// bandwidth is not ours to take).
fn checksums(p: &mut Pool) -> kvclient::Result<()> {
    let (bytes_s, _, _, jobs) = p.c.bg_jobs(kvclient::BG_CHECKSUM, 16)?;
    let mut budget = bytes_s / 10;
    for j in jobs {
        if budget == 0 {
            break;
        }
        let data = read_entry(p, &j, j.guard.compressed);
        let sum = if j.guard.compressed {
            kvslotz::image_xxh3(&data, j.bytes)
        } else {
            Some(xxhash_rust::xxh3::xxh3_64(&data))
        };
        super::fp::hit("bg-checksum-read");
        let sum = if super::fp::hit("bg-checksum-flip") { sum.map(|s| s ^ 1) } else { sum };
        budget = budget.saturating_sub(j.bytes.max(1));
        p.c.bg_checksum(j.namespace, j.key, j.guard, sum)?;
    }
    Ok(())
}

/// One tick of in-place compression: copy a slot, compress it, and only if the broker still agrees (quiet, the
/// entry unchanged) write the image back. At most 1 s per tick.
fn compress(p: &mut Pool) -> kvclient::Result<()> {
    let (_, chunk, _, jobs) = p.c.bg_jobs(kvclient::BG_COMPRESS, 16)?;
    let t = Instant::now();
    for j in jobs {
        if chunk == 0 || t.elapsed() >= Duration::from_secs(1) {
            break;
        }
        if j.bytes > p.slot || j.guard.ranges.len() != 1 || j.guard.ranges[0].len != 1 {
            continue;
        }
        let raw = read_entry(p, &j, false);
        let z = if super::fp::hit("rcompress-fail") { None } else { kvslotz::compress(&raw, chunk as usize, 0.05) };
        let Some(z) = z else {
            p.c.bg_rewrite(kvclient::BG_SKIP, j.namespace, j.key, j.guard, 0)?;
            continue;
        };
        if p.c.bg_rewrite(kvclient::BG_BEGIN, j.namespace, j.key, j.guard.clone(), 0)? != 0 {
            break; // a request arrived, or the entry changed: back off until the broker is quiet again
        }
        unsafe { p.map.range(j.guard.ranges[0]) }.write(0, &z);
        super::fp::hit("bg-rewrite-written"); // failpoints build: the worker may die here (torn rewrite)
        p.c.bg_rewrite(kvclient::BG_END, j.namespace, j.key, j.guard, z.len() as u64)?;
    }
    Ok(())
}

/// One tick of SSD tier writes (super::disk::execute: the same code as the broker's --bg internal path); at most 2 s.
fn ssd_writes(p: &mut Pool) -> kvclient::Result<()> {
    let jobs = p.c.bg_ssd_jobs(8)?;
    let t = Instant::now();
    for j in jobs {
        if t.elapsed() >= Duration::from_secs(2) {
            // not started: the broker plans it again once the worker's in-flight jobs are reported or it reconnects
            p.c.bg_ssd_done(j.namespace, j.key, 3, 0)?;
            continue;
        }
        let job = super::disk::WriteJob {
            ns: j.namespace, key: j.key, guard: j.guard.clone(), kind: j.kind, format: j.format, producer: j.producer,
            hits: j.hits, bytes: j.bytes, last_used_ms: j.last_used_ms, meta: j.meta.clone(), path: PathBuf::from(&j.path),
            codec: j.codec,
        };
        let mut err = None;
        let map = &p.map;
        let c = &mut p.c;
        let out = super::disk::execute(&job, p.slot,
            |r, out| unsafe { map.range(r) }.read(0, out),
            |raw_sum| match c.bg_checksum(j.namespace, j.key, j.guard.clone(), raw_sum) {
                Ok(0 | 1) => 0,
                Ok(3) => 3,
                Ok(_) => 2,
                Err(e) => {
                    err = Some(e);
                    3
                }
            });
        if let Some(e) = err {
            return Err(e);
        }
        let (code, bytes) = out.code();
        p.c.bg_ssd_done(j.namespace, j.key, code, bytes)?;
    }
    Ok(())
}

/// One tick of compaction: moves one at a time (the broker hands out none unless quiet and within its budget), at most
/// 500 ms. The copy runs outside the broker's lock; the broker moves the entry only if it is still unchanged.
fn compact(p: &mut Pool) -> kvclient::Result<()> {
    let t = Instant::now();
    let mut buf = Vec::new();
    while t.elapsed() < Duration::from_millis(500) {
        let Some(j) = p.c.bg_compact_jobs()?.pop() else { break };
        let flat = |rs: &[kvclient::Range]| -> Vec<kvclient::Range> {
            rs.iter().flat_map(|r| (r.start..r.start + r.len).map(|s| kvclient::Range { start: s, len: 1 })).collect()
        };
        let (from, to) = (flat(&j.guard.ranges), flat(&j.new));
        let ok = from.len() == to.len();
        if ok {
            for (a, b) in from.iter().zip(&to) {
                let src = unsafe { p.map.range(*a) };
                buf.resize(src.len(), 0);
                src.read(0, &mut buf);
                unsafe { p.map.range(*b) }.write(0, &buf);
            }
        }
        super::fp::hit("bg-compact-copied"); // failpoints build: the worker may die here (a move left open)
        p.c.bg_compact_done(j.namespace, j.key, ok)?;
    }
    Ok(())
}
