//! Operation trace (crates/trace): every request and the broker's answer in a bounded in-memory ring,
//! written to a file only on an anomaly (a burst of skipped segment runs, a failed integrity checksum, a handler
//! panic) or on request (TRACE_DUMP), for `kvbench replay`. Metadata only (opaque key prefixes, counts, times).
//! Memory: the ring holds at most --trace-mib and is dropped after a minute without connections (dual-use memory).
//! SSD: at most --trace-dumps-day files a day, KEEP files kept (oldest removed).
use kvtrace::{Body, Header, Item, Record};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const KEEP: usize = 10;
const CONNS: usize = 4096;

pub struct Tracer {
    ring: Mutex<(VecDeque<Vec<u8>>, usize)>,
    /// every connection's HELLO / DOMAIN records (time 0 in a dump): the ring's window rarely reaches back to them, and
    /// replay needs them to attribute the traffic (pool class, namespace scope). Bounded: CONNS newest connections.
    conns: Mutex<std::collections::BTreeMap<u32, Vec<Record>>>,
    cap: usize,
    dir: PathBuf,
    per_day: u32,
    t0: Instant,
    /// (UTC day, dumps that day, last dump)
    budget: Mutex<(u64, u32, Option<Instant>)>,
    pub dumps: AtomicU64,
    pub refused: AtomicU64,
    pub records: AtomicU64,
}

pub fn p16s(keys: &[kvproto::Key]) -> Vec<kvtrace::P16> {
    keys.iter().map(kvtrace::p16).collect()
}

/// The traced body of a request (before it is handled: the request is consumed by `handle`).
pub fn body_of(req: &kvproto::Request) -> Body {
    use kvproto::Request as Q;
    match req {
        Q::Hello { engine_id, name, .. } => Body::Hello { engine_id: *engine_id, name: name.clone() },
        Q::Lookup { namespace, acquire, keys } => {
            Body::Lookup { ns: *namespace, acquire: *acquire, keys: p16s(keys), n_hit: 0, lease_id: 0 }
        }
        Q::Reserve { namespace, items } => Body::Reserve {
            ns: *namespace,
            items: items.iter().map(|it| Item { key: kvtrace::p16(&it.key), parent: it.parent.as_ref().map(kvtrace::p16),
                                                bytes: it.bytes, kind: it.kind, format: it.format, flags: it.flags,
                                                ttl_ms: it.ttl_ms, meta: it.meta.clone() }).collect(),
            statuses: Vec::new(),
        },
        Q::Commit { namespace, keys } | Q::Abort { namespace, keys } | Q::Touch { namespace, keys }
        | Q::Probe { namespace, keys } => Body::Keys { ns: *namespace, keys: p16s(keys) },
        Q::Heat { namespace, hot, keys } => Body::Heat { ns: *namespace, hot: *hot, keys: p16s(keys) },
        Q::Release { lease_ids } => Body::Release { lease_ids: lease_ids.clone() },
        // TOUCH_TAGGED traces as TOUCH_REQUEST (the trace format keeps no session tags or classes)
        Q::TouchRequest { namespace, groups } | Q::TouchTagged { namespace, groups, .. } => {
            Body::TouchRequest { ns: *namespace, groups: groups.iter().map(|(k, ks)| (*k, p16s(ks))).collect() }
        }
        Q::Wait { namespace, key, timeout_ms } => Body::Wait { ns: *namespace, key: kvtrace::p16(key), timeout_ms: *timeout_ms },
        Q::Domain { descriptor } => Body::Domain { descriptor: descriptor.clone() },
        _ => Body::None,
    }
}

/// Fill in what the broker answered.
pub fn with_reply(body: &mut Body, reply: &kvproto::Reply) {
    match (body, reply) {
        (Body::Lookup { n_hit, lease_id, .. }, kvproto::Reply::Lookup { n_hit: h, lease_id: l, .. }) => {
            *n_hit = *h;
            *lease_id = *l;
        }
        (Body::Reserve { statuses, .. }, kvproto::Reply::Reserve(v)) => {
            *statuses = v.iter().map(|(s, _)| *s as u16).collect();
        }
        _ => {}
    }
}

impl Tracer {
    pub fn new(mib: u64, dir: PathBuf, per_day: u32) -> Tracer {
        Tracer { ring: Mutex::new((VecDeque::new(), 0)), conns: Mutex::new(Default::default()), cap: (mib << 20) as usize, dir, per_day, t0: Instant::now(),
                 budget: Mutex::new((0, 0, None)), dumps: AtomicU64::new(0), refused: AtomicU64::new(0),
                 records: AtomicU64::new(0) }
    }

    pub fn now_us(&self) -> u64 {
        self.t0.elapsed().as_micros() as u64
    }

    pub fn record(&self, rec: &Record) {
        if matches!(rec.body, Body::Hello { .. } | Body::Domain { .. }) && rec.status == 0 {
            let mut c = self.conns.lock().unwrap();
            let mut r = rec.clone();
            r.t_us = 0;
            r.lat_us = 0;
            c.entry(rec.conn).or_default().push(r);
            while c.len() > CONNS {
                c.pop_first();
            }
        }
        let b = kvtrace::encode_record(rec);
        let mut g = self.ring.lock().unwrap();
        g.1 += b.len();
        g.0.push_back(b);
        while g.1 > self.cap {
            let Some(old) = g.0.pop_front() else { break };
            g.1 -= old.len();
        }
        self.records.fetch_add(1, Relaxed);
    }

    /// Drop the ring (no connections for a while): its memory goes back.
    pub fn clear(&self) {
        let mut g = self.ring.lock().unwrap();
        if g.1 > 0 {
            *g = (VecDeque::new(), 0);
        }
        *self.conns.lock().unwrap() = Default::default();
    }

    pub fn ring_bytes(&self) -> u64 {
        self.ring.lock().unwrap().1 as u64
    }

    /// Write the HELLO / DOMAIN records of the connections in the ring (time 0) and the ring to
    /// <dir>/trace-<unix ms>-<reason>.lkvt; None when the day's budget is used up (anomaly dumps also wait 10 min after
    /// the previous one) or the write fails.
    pub fn dump(&self, header: Header, anomaly: bool) -> Option<PathBuf> {
        {
            let mut b = self.budget.lock().unwrap();
            let day = header.unix_ms / 86_400_000;
            if b.0 != day {
                *b = (day, 0, b.2);
            }
            if b.1 >= self.per_day || (anomaly && b.2.is_some_and(|t| t.elapsed() < Duration::from_secs(600))) {
                self.refused.fetch_add(1, Relaxed);
                return None;
            }
            b.1 += 1;
            b.2 = Some(Instant::now());
        }
        let ring: Vec<Vec<u8>> = self.ring.lock().unwrap().0.iter().cloned().collect();
        let seen: std::collections::BTreeSet<u32> =
            ring.iter().filter_map(|b| b.get(8..12).map(|x| u32::from_le_bytes(x.try_into().unwrap()))).collect();
        let mut recs: Vec<Vec<u8>> = {
            let c = self.conns.lock().unwrap();
            seen.iter().filter_map(|k| c.get(k)).flatten().map(kvtrace::encode_record).collect()
        };
        recs.extend(ring);
        let n = recs.len();
        let data = kvtrace::encode_file(&header, recs);
        let reason: String = header.reason.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).take(40).collect();
        let path = self.dir.join(format!("trace-{}-{reason}.lkvt", header.unix_ms));
        let tmp = path.with_extension("tmp");
        let ok = std::fs::create_dir_all(&self.dir).and_then(|_| std::fs::write(&tmp, &data)).and_then(|_| std::fs::rename(&tmp, &path));
        if let Err(e) = ok {
            eprintln!("trace: cannot write {}: {e}", path.display());
            let _ = std::fs::remove_file(&tmp);
            return None;
        }
        self.dumps.fetch_add(1, Relaxed);
        // keep the newest KEEP dumps
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            let mut v: Vec<PathBuf> = rd.flatten().map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "lkvt")).collect();
            v.sort();
            for old in v.iter().take(v.len().saturating_sub(KEEP)) {
                let _ = std::fs::remove_file(old);
            }
        }
        eprintln!("trace: {} ({} records, {:.1} MiB, reason {})", path.display(),
                  n, data.len() as f64 / (1u64 << 20) as f64,
                  header.reason);
        Some(path)
    }
}
