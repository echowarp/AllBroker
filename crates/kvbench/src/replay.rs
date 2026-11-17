//! `kvbench replay FILE --socket S [--scale N] [--speed X]`: replay a broker operation trace (crates/trace,
//! written by TRACE_DUMP or on an anomaly) against a test broker and compare what it answers with what production
//! answered. Keys are the trace's 16-byte prefixes padded with zeros (segment links and metadata parents keep matching);
//! every reserved slot gets the key prefix written at its start and every hit is verified against it (wrong bytes =
//! FOREIGN). `--scale N`: entry sizes divided by N, for a test broker whose classes have slot sizes divided by N
//! (printed from the trace header). `--speed X`: X times real time (default 1; 0 = as fast as possible: the broker's
//! own clock then runs slower than production's, so ages, grace periods and lease timeouts differ).
use kvclient::{Client, Key, PoolMap, ReserveItem, Status};
use kvtrace::{Body, Record, P16};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

struct Conn {
    c: Client,
    pool: PoolMap,
    /// the trace's slot bytes for this connection's class, and the test broker's
    slot_t: u64,
}

fn key(p: &P16) -> Key {
    let mut k = [0u8; 32];
    k[..16].copy_from_slice(p);
    k
}

#[derive(Default)]
struct Tally {
    ops: HashMap<u16, u64>,
    skipped_no_conn: u64,
    lookups: u64,
    blocks_asked: u64,
    hits_trace: u64,
    hits_replay: u64,
    lookups_same: u64,
    lookups_fewer: u64,
    lookups_more: u64,
    reserve_status_diff: u64,
    foreign: u64,
    errors: u64,
}

pub fn run(file: &Path, sock: PathBuf, scale: u64, speed: f64) {
    let data = std::fs::read(file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    let (h, recs) = kvtrace::decode_file(&data).unwrap_or_else(|e| {
        eprintln!("{}: not a trace this build reads: {e:?}", file.display());
        std::process::exit(2);
    });
    println!("trace: broker {}, epoch {:016x}, reason {}, {} records over {:.1} s", h.broker_version, h.epoch, h.reason,
             recs.len(), recs.last().map_or(0.0, |r| r.t_us as f64 / 1e6) - recs.iter().find(|r| r.t_us > 0).map_or(0.0, |r| r.t_us as f64 / 1e6));
    let classes: Vec<String> = h.classes.iter().map(|(n, s)| format!("{n}={}", (s / scale.max(1)).div_ceil(1024))).collect();
    println!("test broker classes (slot KiB at scale {scale}): --class {}", classes.join(" --class "));
    let slot_of = |name: &str| h.classes.iter().find(|c| c.0 == name).map_or(h.classes.first().map_or(1, |c| c.1), |c| c.1);
    let mut conns: HashMap<u32, Conn> = HashMap::new();
    let mut leases: HashMap<u64, u64> = HashMap::new(); // trace lease id -> replay lease id
    let mut t = Tally::default();
    let stats0 = stats_any(&sock);
    let start = Instant::now();
    let t_first = recs.iter().find(|r| r.t_us > 0).map_or(0, |r| r.t_us);
    for r in &recs {
        if speed > 0.0 && r.t_us > t_first {
            let due = Duration::from_secs_f64((r.t_us - t_first) as f64 / 1e6 / speed);
            if let Some(w) = due.checked_sub(start.elapsed()) {
                std::thread::sleep(w);
            }
        }
        *t.ops.entry(r.op).or_default() += 1;
        if r.op == kvtrace::OP_CLOSE {
            conns.remove(&r.conn);
            continue;
        }
        if let Body::Hello { engine_id, name } = &r.body {
            if !conns.contains_key(&r.conn) {
                match Client::connect(&sock, *engine_id, name, Duration::from_secs(2)) {
                    Ok(c) => {
                        let pool = PoolMap::open(&c.hello).expect("map the test broker's pool");
                        conns.insert(r.conn, Conn { c, pool, slot_t: slot_of(name) });
                    }
                    Err(e) => {
                        eprintln!("conn {}: cannot connect as {name}: {e:?}", r.conn);
                        t.errors += 1;
                    }
                }
            }
            continue;
        }
        let Some(cn) = conns.get_mut(&r.conn) else {
            t.skipped_no_conn += 1;
            continue;
        };
        if replay_one(cn, r, &mut leases, &mut t, scale).is_err() {
            t.errors += 1;
        }
    }
    let stats1 = stats_any(&sock);
    report(&t, &stats0, &stats1, start.elapsed());
}

fn stats_any(sock: &Path) -> Vec<(String, u64)> {
    Client::connect(sock, 0, "kvbench-replay-stats", Duration::from_secs(2)).and_then(|mut c| c.stats()).unwrap_or_default()
}

/// Bytes of a traced entry at the test broker's slot size: the same number of slots, the last one filled alike.
fn scaled(bytes: u64, slot_t: u64, slot_r: u64) -> u64 {
    if slot_t == slot_r || slot_t == 0 {
        return bytes;
    }
    let n = bytes.div_ceil(slot_t).max(1);
    let rem = bytes - (n - 1) * slot_t;
    (n - 1) * slot_r + (rem * slot_r).div_ceil(slot_t).clamp(1, slot_r)
}

fn replay_one(cn: &mut Conn, r: &Record, leases: &mut HashMap<u64, u64>, t: &mut Tally, _scale: u64) -> kvclient::Result<()> {
    let slot_r = cn.c.hello.slot_bytes;
    match &r.body {
        Body::Lookup { ns, acquire, keys, n_hit, lease_id } => {
            let ks: Vec<Key> = keys.iter().map(key).collect();
            let l = cn.c.lookup(*ns, &ks, *acquire)?;
            t.lookups += 1;
            t.blocks_asked += ks.len() as u64;
            t.hits_trace += *n_hit as u64;
            t.hits_replay += l.n_hit as u64;
            match l.n_hit.cmp(n_hit) {
                std::cmp::Ordering::Equal => t.lookups_same += 1,
                std::cmp::Ordering::Less => t.lookups_fewer += 1,
                std::cmp::Ordering::Greater => t.lookups_more += 1,
            }
            for (i, e) in l.entries.iter().enumerate() {
                let mut head = [0u8; 16];
                let s = unsafe { cn.pool.range(e.ranges[0]) };
                s.read(0, &mut head);
                let img = s.to_vec(s.len().min(64 << 10));
                if !kvslotz::is_compressed(&img) && head != keys[i] {
                    t.foreign += 1;
                }
            }
            if *acquire && l.lease_id != 0 {
                if *lease_id != 0 {
                    leases.insert(*lease_id, l.lease_id);
                } else {
                    cn.c.release(&[l.lease_id])?;   // production missed: hold nothing the trace does not release
                }
            }
        }
        Body::Reserve { ns, items, statuses } => {
            let its: Vec<ReserveItem> = items.iter().map(|it| ReserveItem {
                key: key(&it.key), parent: it.parent.as_ref().map(key), bytes: scaled(it.bytes, cn.slot_t, slot_r),
                kind: it.kind, format: it.format, cost_hint: 0, flags: it.flags, ttl_ms: it.ttl_ms, meta: it.meta.clone(),
            }).collect();
            let rep = cn.c.reserve(*ns, its)?;
            for (i, (st, ranges)) in rep.iter().enumerate() {
                if statuses.get(i).is_some_and(|s| *s != *st as u16) {
                    t.reserve_status_diff += 1;
                }
                if *st == Status::Ok {
                    for rg in ranges {
                        unsafe { cn.pool.range(*rg) }.write(0, &items[i].key);
                    }
                }
            }
        }
        Body::Keys { ns, keys } => {
            let ks: Vec<Key> = keys.iter().map(key).collect();
            match r.op {
                4 => drop(cn.c.commit(*ns, &ks)?),
                5 => drop(cn.c.abort(*ns, &ks)?),
                7 => drop(cn.c.touch(*ns, &ks)?),
                16 => drop(cn.c.probe(*ns, &ks)?),
                _ => {}
            }
        }
        Body::Heat { ns, hot, keys } => drop(cn.c.heat(*ns, &keys.iter().map(key).collect::<Vec<_>>(), *hot)?),
        Body::Release { lease_ids } => {
            let ids: Vec<u64> = lease_ids.iter().filter_map(|l| leases.remove(l)).collect();
            if !ids.is_empty() {
                cn.c.release(&ids)?;
            }
        }
        Body::TouchRequest { ns, groups } => {
            let gs: Vec<(u8, Vec<Key>)> = groups.iter().map(|(k, ks)| (*k, ks.iter().map(key).collect())).collect();
            cn.c.touch_request(*ns, &gs)?;
        }
        Body::Domain { descriptor } => drop(cn.c.domain(descriptor)?),
        Body::Wait { .. } | Body::Hello { .. } | Body::None => {}
    }
    Ok(())
}

fn report(t: &Tally, s0: &[(String, u64)], s1: &[(String, u64)], took: Duration) {
    let mut ops: Vec<_> = t.ops.iter().collect();
    ops.sort();
    let names: Vec<String> = ops.iter().map(|(op, n)| {
        let name = kvproto_name(**op);
        format!("{name} {n}")
    }).collect();
    println!("replayed in {:.1} s: {}", took.as_secs_f64(), names.join(", "));
    println!("records of connections never seen (no HELLO): {}; errors {}", t.skipped_no_conn, t.errors);
    let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
    println!("lookups {}: blocks asked {}, hit in production {} ({:.1} %), in the replay {} ({:.1} %); per lookup same {} \
              fewer {} more {}", t.lookups, t.blocks_asked, t.hits_trace, pct(t.hits_trace, t.blocks_asked), t.hits_replay,
             pct(t.hits_replay, t.blocks_asked), t.lookups_same, t.lookups_fewer, t.lookups_more);
    println!("reserve statuses different from production: {}; FOREIGN reads {}", t.reserve_status_diff, t.foreign);
    let d = |k: &str| {
        let g = |s: &[(String, u64)]| s.iter().find(|x| x.0 == k).map_or(0, |x| x.1);
        g(s1).saturating_sub(g(s0))
    };
    let keys = ["seg_runs_linked", "seg_runs_skipped", "seg_skip_shape", "seg_skip_len", "seg_skip_kind", "seg_skip_layout",
                "seg_tagged_linked", "seg_tagged_unlinked", "evict_room", "evict_low_water", "cascade_evicted"];
    let v: Vec<String> = keys.iter().map(|k| format!("{k} {}", d(k))).collect();
    println!("test broker: {}", v.join(", "));
    println!("{}", if t.foreign == 0 && t.errors == 0 { "replay: no wrong bytes" } else { "replay: PROBLEMS (see above)" });
}

fn kvproto_name(op: u16) -> String {
    if op == kvtrace::OP_CLOSE {
        return "close".into();
    }
    kvproto::Op::from_u16(op).map_or_else(|| format!("op{op}"), |o| o.name().into())
}

/// `kvbench trace-dump --socket S`: ask the broker to write its trace now.
pub fn dump(sock: PathBuf, reason: &str) {
    let mut c = Client::connect(&sock, 0, "kvbench-trace", Duration::from_secs(5)).expect("connect");
    match c.trace_dump(reason) {
        Ok(p) => println!("{p}"),
        Err(e) => {
            eprintln!("trace dump refused: {e:?}");
            std::process::exit(1);
        }
    }
}

/// `kvbench ping --socket S --seconds N`: PROBE round trips at ~1 kHz with the connector's 2 ms timeout (for
/// broker readiness under CPU load): RTT percentiles of the answered ones, timeouts, the longest run of timeouts
/// (the old connector rule dropped the connection at 16).
pub fn ping(sock: PathBuf, seconds: u64) {
    let mut c = Client::connect(&sock, 0, "kvbench-ping", Duration::from_secs(2)).expect("connect");
    c.set_timeout(Duration::from_millis(2));
    let (mut rtt, mut to, mut run, mut longest) = (Vec::new(), 0u64, 0u64, 0u64);
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(seconds) {
        let s = Instant::now();
        match c.probe(0, &[[7u8; 32]]) {
            Ok(_) => {
                rtt.push(s.elapsed().as_micros() as u64);
                run = 0;
            }
            Err(kvclient::Error::Timeout) => {
                to += 1;
                run += 1;
                longest = longest.max(run);
            }
            Err(e) => panic!("ping: {e:?}"),
        }
        std::thread::sleep(Duration::from_micros(1000));
    }
    rtt.sort_unstable();
    let p = |q: f64| rtt.get(((rtt.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(0);
    println!("ping: {} answered, rtt us p50 {} p99 {} p99.9 {} max {} | timeouts {} ({:.2} %), longest run {}",
             rtt.len(), p(0.5), p(0.99), p(0.999), rtt.last().copied().unwrap_or(0), to,
             100.0 * to as f64 / (to + rtt.len() as u64).max(1) as f64, longest);
}
