//! Chains listener: one thread per engine connection, no state shared with the KV index. Per connection: the open
//! requests' indexes. Hints are written back on the same socket, flushed once per burst of engine messages (an engine
//! sends one burst per step).
use crate::index::RequestIndex;
use crate::proto::{self, Msg};
use crate::store::{self, Store};
use std::sync::Mutex;
use std::time::Duration;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Config {
    pub socket: PathBuf,
    /// cap on open requests per connection and on tokens per request (memory guard)
    pub max_open: usize,
    pub max_tokens: usize,
    /// suffix matches are measured up to this length
    pub max_match: usize,
    /// global store: memory cap (bytes), snapshot file and interval, idle time before GC runs
    pub global: bool,
    pub store_mem: usize,
    pub snapshot: Option<PathBuf>,
    pub snapshot_s: u64,
    pub idle_ms: u64,
    /// heat half-life, prune grace period (s) and prune threshold (store.rs)
    pub heat_half_life_s: u32,
    pub prune_grace_s: u32,
    pub prune_heat: f32,
}

impl Config {
    pub fn new(socket: impl Into<PathBuf>) -> Config {
        Config { socket: socket.into(), max_open: 4096, max_tokens: 2_000_000, max_match: 1024, global: true,
                 store_mem: 1 << 30, snapshot: None, snapshot_s: 1800, idle_ms: 200,
                 heat_half_life_s: store::HALF_LIFE_S, prune_grace_s: store::PRUNE_GRACE_S, prune_heat: store::PRUNE_HEAT }
    }

    fn configure(&self, s: &mut Store) {
        (s.mem_cap, s.half_life_s, s.prune_grace_s, s.prune_heat) =
            (self.store_mem, self.heat_half_life_s, self.prune_grace_s, self.prune_heat);
    }
}

#[derive(Default)]
pub struct Stats {
    pub connections: AtomicU64,
    pub opens: AtomicU64,
    pub closes: AtomicU64,
    pub appends: AtomicU64,
    pub appended_tokens: AtomicU64,
    pub hints: AtomicU64,
    pub hint_tokens: AtomicU64,
    pub no_hints: AtomicU64,
    pub think_appends: AtomicU64,
    pub desyncs: AtomicU64,
    pub unknown_req: AtomicU64,
    pub offered: AtomicU64,
    pub accepted: AtomicU64,
    pub open_now: AtomicU64,
    pub lookup_ns: AtomicU64,
    pub open_ns: AtomicU64,
    pub errors: AtomicU64,
    pub global_hints: AtomicU64,
    pub global_hint_tokens: AtomicU64,
    pub global_busy: AtomicU64,
    pub docs_queued: AtomicU64,
    pub docs_dropped: AtomicU64,
    pub seeds: AtomicU64,
    pub gc_runs: AtomicU64,
    pub snapshots: AtomicU64,
    pub snapshot_ms: AtomicU64,
    /// last snapshot: longest store-lock hold while serializing (µs; one index shard), all holds, bytes written
    pub snapshot_lock_us: AtomicU64,
    pub snapshot_lock_total_us: AtomicU64,
    /// last sweep: longest store-lock hold (µs; one index shard)
    pub sweep_lock_us: AtomicU64,
    pub snapshot_bytes: AtomicU64,
    pub snapshot_errors: AtomicU64,
    /// a restored store discarded because the first engine's model_tag differed
    pub restore_discarded: AtomicU64,
}

impl Stats {
    pub fn json(&self) -> String {
        let f = |a: &AtomicU64| a.load(Relaxed);
        let ap = f(&self.appends).max(1);
        format!(
            "{{\"connections\":{},\"open_requests\":{},\"opens\":{},\"closes\":{},\"appends\":{},\"appended_tokens\":{},\
             \"think_appends\":{},\"hints\":{},\"hint_tokens\":{},\"no_hints\":{},\"desyncs\":{},\"unknown_req\":{},\
             \"offered\":{},\"accepted\":{},\"lookup_us_mean\":{:.2},\"open_ms_total\":{:.1},\"errors\":{},\
             \"global_hints\":{},\"global_hint_tokens\":{},\"global_busy\":{},\"docs_queued\":{},\"docs_dropped\":{},\
             \"seeds\":{},\"gc_runs\":{},\"snapshots\":{},\"snapshot_ms\":{},\"snapshot_lock_us\":{},\"snapshot_bytes\":{},\
             \"snapshot_lock_total_us\":{},\"sweep_lock_us\":{},\"snapshot_errors\":{},\"restore_discarded\":{}}}",
            f(&self.connections), f(&self.open_now), f(&self.opens), f(&self.closes), f(&self.appends),
            f(&self.appended_tokens), f(&self.think_appends), f(&self.hints), f(&self.hint_tokens), f(&self.no_hints),
            f(&self.desyncs), f(&self.unknown_req), f(&self.offered), f(&self.accepted),
            f(&self.lookup_ns) as f64 / ap as f64 / 1000.0, f(&self.open_ns) as f64 / 1e6, f(&self.errors),
            f(&self.global_hints), f(&self.global_hint_tokens), f(&self.global_busy), f(&self.docs_queued),
            f(&self.docs_dropped), f(&self.seeds), f(&self.gc_runs), f(&self.snapshots), f(&self.snapshot_ms),
            f(&self.snapshot_lock_us), f(&self.snapshot_bytes), f(&self.snapshot_lock_total_us), f(&self.sweep_lock_us),
            f(&self.snapshot_errors), f(&self.restore_discarded)
        )
    }
}

struct Req {
    ix: RequestIndex,
    hint_active: bool,
    prompt_len: usize,
}

/// Shared by all connections: the global store and the queue the idle GC thread drains (inserts never run on the
/// lookup path).
pub struct Global {
    pub store: Mutex<Store>,
    pending: Mutex<Pending>,
    last_activity_ms: AtomicU64,
    start: Instant,
    /// one snapshot writer at a time (GC thread / shutdown)
    snap_io: Mutex<()>,
    /// store gauges as of the last GC run (stats and metrics never wait for the store lock)
    view: Mutex<View>,
}

#[derive(Default)]
struct Pending {
    docs: Vec<(Vec<u32>, u8, u16, bool)>,
    tokens: usize,
    feedback: Vec<(u64, u16, u16)>,
}

const PENDING_MAX_TOKENS: usize = 16 << 20;

/// Store gauges copied after each GC run.
#[derive(Default, Clone, Debug)]
pub struct View {
    pub stats: store::StoreStats,
    pub heat: store::HeatReport,
    pub docs: usize,
    pub tokens: usize,
    pub keys: usize,
    pub occ: usize,
    pub mem_bytes: usize,
    pub model_tag: String,
}

impl View {
    fn of(s: &Store) -> View {
        View { stats: s.stats.clone(), heat: s.heat.clone(), docs: s.docs.len(), tokens: s.n_tokens(), keys: s.index.len(),
               occ: s.n_occ(), mem_bytes: s.mem_bytes(), model_tag: s.model_tag.clone() }
    }
}

impl Global {
    fn now_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Serialize under the store lock, one index shard per lock hold, and write outside the lock (temp file, fsync,
    /// 0600, rename).
    fn snapshot(&self, path: &std::path::Path, st: &Stats) -> std::io::Result<()> {
        let _io = self.snap_io.lock().unwrap();
        let t = Instant::now();
        let mut bytes = Vec::new();
        let (mut max_hold, mut total_hold) = (0u64, 0u64);
        for i in 0..=store::SHARDS {
            let s = self.store.lock().unwrap();
            let t = Instant::now();
            if i == 0 {
                bytes.reserve(s.mem_bytes() * 2 / 3 + 4096);
                s.serialize_head(store::unix_now(), &mut bytes);
            } else {
                s.serialize_shard(i - 1, &mut bytes);
            }
            let us = t.elapsed().as_micros() as u64;
            (max_hold, total_hold) = (max_hold.max(us), total_hold + us);
        }
        st.snapshot_lock_us.store(max_hold, Relaxed);
        st.snapshot_lock_total_us.store(total_hold, Relaxed);
        let r = Store::write_snapshot(path, &bytes);
        match &r {
            Ok(()) => {
                st.snapshots.fetch_add(1, Relaxed);
                st.snapshot_ms.store(t.elapsed().as_millis() as u64, Relaxed);
                st.snapshot_bytes.store(bytes.len() as u64, Relaxed);
            }
            Err(e) => {
                st.snapshot_errors.fetch_add(1, Relaxed);
                eprintln!("chains: snapshot {}: {e}", path.display());
            }
        }
        r
    }

    /// Insert queued documents and feedback, evict over the cap, and sweep when due (one index shard per store-lock
    /// hold). Returns true if the store changed.
    fn run_gc(&self, st: &Stats) -> bool {
        let p = std::mem::take(&mut *self.pending.lock().unwrap());
        let ns = store::unix_now();
        let (mut changed, due) = {
            let mut s = self.store.lock().unwrap();
            for (d, src, eng, pinned) in &p.docs {
                s.insert_doc(d, *src, *eng, *pinned, ns);
            }
            for (r, o, a) in &p.feedback {
                s.feedback(*r, *o, *a, ns);
            }
            let changed = s.evict_to_cap(ns) || !p.docs.is_empty() || !p.feedback.is_empty();
            let due = s.sweep_due(ns);
            if due {
                s.sweep_begin(ns);
            } else {
                *self.view.lock().unwrap() = View::of(&s);
            }
            (changed, due)
        };
        if due {
            let mut max_hold = 0u64;
            for i in 0..=store::SHARDS {
                let mut s = self.store.lock().unwrap();
                let t = Instant::now();
                if i < store::SHARDS {
                    s.sweep_shard(i, ns);
                } else {
                    changed |= s.sweep_end(ns);
                    *self.view.lock().unwrap() = View::of(&s);
                }
                max_hold = max_hold.max(t.elapsed().as_micros() as u64);
            }
            st.sweep_lock_us.store(max_hold, Relaxed);
        }
        changed
    }
}

fn store_json(g: &Global) -> String {
    let v = g.view.lock().unwrap().clone();
    let (s, h) = (&v.stats, &v.heat);
    format!(
        ",\"store\":{{\"docs_in\":{},\"docs\":{},\"tokens\":{},\"keys\":{},\"occ\":{},\"pinned\":{},\"seeds\":{},\
         \"mem_mib\":{:.1},\"lookups\":{},\"hits\":{},\"docs_evicted\":{},\"docs_freed\":{},\"promoted\":{},\"demoted\":{},\
         \"pruned\":{},\"rescued\":{},\"prune_candidates\":{},\"copies_evicted\":{},\"occ_dropped\":{},\"decays\":{},\
         \"sweeps\":{},\"sweep_ms\":{:.1},\"heat_mean\":{:.3},\"heat_hist\":{:?},\"doc_heat_hist\":{:?},\"model_tag\":{:?}}}}}",
        s.docs_in, v.docs, v.tokens, v.keys, v.occ, h.pinned, h.seeds, v.mem_bytes as f64 / (1 << 20) as f64, s.lookups,
        s.hits, s.docs_evicted, s.docs_freed, s.promoted, s.demoted, s.pruned, s.rescued, h.candidates, s.copies_evicted,
        s.occ_dropped, s.decays, s.sweeps, s.sweep_us as f64 / 1e3, h.heat_sum / v.occ.max(1) as f64, h.occ, h.docs,
        v.model_tag)
}

/// Idle-only maintenance at idle CPU priority: insert queued documents, apply feedback, evict/prune/decay, snapshot.
fn gc_thread(cfg: Config, g: Arc<Global>, st: Arc<Stats>) {
    unsafe {
        let p = libc::sched_param { sched_priority: 0 };
        libc::sched_setscheduler(0, libc::SCHED_IDLE, &p);
        libc::setpriority(libc::PRIO_PROCESS, libc::gettid() as u32, 19);
    }
    let (mut last_tick, mut last_snap, mut dirty) = (g.now_ms(), g.now_ms(), false);
    loop {
        std::thread::sleep(Duration::from_millis(200));
        let now = g.now_ms();
        let idle = now.saturating_sub(g.last_activity_ms.load(Relaxed)) >= cfg.idle_ms;
        let (queued, pressure) = {
            let p = g.pending.lock().unwrap();
            (!p.docs.is_empty() || !p.feedback.is_empty(), p.tokens > PENDING_MAX_TOKENS / 2)
        };
        if !idle && !pressure {
            continue;
        }
        // periodic sweeps (prune candidates, pruning, counter halving) run even without new input
        let tick = now - last_tick >= store::SWEEP_S as u64 * 1000;
        let snap_due = dirty && cfg.snapshot.is_some() && now - last_snap >= cfg.snapshot_s * 1000;
        if !queued && !tick && !snap_due {
            continue;
        }
        if tick {
            last_tick = now;
        }
        dirty |= g.run_gc(&st);
        st.gc_runs.fetch_add(1, Relaxed);
        if let Some(path) = &cfg.snapshot {
            if dirty && now - last_snap >= cfg.snapshot_s * 1000 {
                if g.snapshot(path, &st).is_ok() {
                    dirty = false;
                }
                last_snap = now;
            }
        }
    }
}

/// Handle to a running chains listener: stats, store gauges, and the shutdown snapshot.
#[derive(Clone)]
pub struct Chains {
    pub stats: Arc<Stats>,
    pub global: Arc<Global>,
    pub cfg: Config,
}

impl Chains {
    pub fn snapshot_path(&self) -> Option<&std::path::Path> {
        self.cfg.snapshot.as_deref().filter(|_| self.cfg.global)
    }

    /// Shutdown: insert what is queued, evict / sweep, and write the snapshot (if configured). Returns the time it
    /// took.
    pub fn save_now(&self) -> Option<std::io::Result<Duration>> {
        let path = self.snapshot_path()?;
        let t = Instant::now();
        self.global.run_gc(&self.stats);
        Some(self.global.snapshot(path, &self.stats).map(|_| t.elapsed()))
    }

    pub fn view(&self) -> View {
        self.global.view.lock().unwrap().clone()
    }

    /// One journal line (the broker's --stats-log-s summary).
    pub fn summary(&self) -> String {
        let v = self.view();
        let f = |a: &AtomicU64| a.load(Relaxed);
        let st = &self.stats;
        format!(
            "stats: chains: {} docs, {} tokens, {} keys, {} occurrences ({} permanent, {} seeds), {:.1} MiB; \
             hints own {} / global {}, accepted {}/{} tokens; heat mean {:.2}, histogram {:?} (le {:?}, +Inf), \
             prune candidates {}, pruned {}, rescued {}, demoted {}, promoted {}, docs evicted {} / freed {}; \
             sweep {:.1} ms (lock ≤ {:.1} ms); snapshots {} (last {} ms, lock ≤ {:.1} ms / {:.1} ms total, {:.1} MiB)",
            v.docs, v.tokens, v.keys, v.occ, v.heat.pinned, v.heat.seeds, v.mem_bytes as f64 / 1048576.0,
            f(&st.hints), f(&st.global_hints), f(&st.accepted), f(&st.offered), v.heat.heat_sum / v.occ.max(1) as f64,
            v.heat.occ, store::HEAT_BUCKETS, v.heat.candidates, v.stats.pruned, v.stats.rescued, v.stats.demoted,
            v.stats.promoted, v.stats.docs_evicted, v.stats.docs_freed, v.stats.sweep_us as f64 / 1e3,
            f(&st.sweep_lock_us) as f64 / 1e3, f(&st.snapshots), f(&st.snapshot_ms), f(&st.snapshot_lock_us) as f64 / 1e3,
            f(&st.snapshot_lock_total_us) as f64 / 1e3, f(&st.snapshot_bytes) as f64 / 1048576.0)
    }

    /// Prometheus text (llmt_kv_chains_*).
    pub fn metrics(&self) -> String {
        use std::fmt::Write as _;
        let v = self.view();
        let st = &self.stats;
        let mut s = String::new();
        let f = |a: &AtomicU64| a.load(Relaxed);
        for (k, x) in [("docs", v.docs as u64), ("tokens", v.tokens as u64), ("keys", v.keys as u64),
                       ("occurrences", v.occ as u64), ("permanent", v.heat.pinned), ("seeds", v.heat.seeds),
                       ("mem_bytes", v.mem_bytes as u64), ("prune_candidates", v.heat.candidates),
                       ("open_requests", f(&st.open_now)), ("connections", f(&st.connections)),
                       ("snapshot_bytes", f(&st.snapshot_bytes))] {
            let _ = writeln!(s, "# TYPE llmt_kv_chains_{k} gauge\nllmt_kv_chains_{k} {x}");
        }
        let _ = writeln!(s, "# TYPE llmt_kv_chains_snapshot_seconds gauge\nllmt_kv_chains_snapshot_seconds {}",
                         f(&st.snapshot_ms) as f64 / 1e3);
        let _ = writeln!(s, "# TYPE llmt_kv_chains_snapshot_lock_seconds gauge\nllmt_kv_chains_snapshot_lock_seconds {}",
                         f(&st.snapshot_lock_us) as f64 / 1e6);
        let _ = writeln!(s, "# TYPE llmt_kv_chains_sweep_seconds gauge\nllmt_kv_chains_sweep_seconds {}",
                         v.stats.sweep_us as f64 / 1e6);
        let _ = writeln!(s, "# TYPE llmt_kv_chains_sweep_lock_seconds gauge\nllmt_kv_chains_sweep_lock_seconds {}",
                         f(&st.sweep_lock_us) as f64 / 1e6);
        for (k, x) in [("hints", f(&st.hints)), ("global_hints", f(&st.global_hints)), ("offered_tokens", f(&st.offered)),
                       ("accepted_tokens", f(&st.accepted)), ("docs_in", v.stats.docs_in),
                       ("docs_evicted", v.stats.docs_evicted), ("docs_freed", v.stats.docs_freed),
                       ("pruned", v.stats.pruned), ("rescued", v.stats.rescued), ("demoted", v.stats.demoted),
                       ("promoted", v.stats.promoted), ("copies_evicted", v.stats.copies_evicted),
                       ("snapshots", f(&st.snapshots)), ("snapshot_errors", f(&st.snapshot_errors)),
                       ("global_busy", f(&st.global_busy))] {
            let _ = writeln!(s, "# TYPE llmt_kv_chains_{k}_total counter\nllmt_kv_chains_{k}_total {x}");
        }
        // heat distribution of the occurrences as a (cumulative) histogram
        let _ = writeln!(s, "# TYPE llmt_kv_chains_heat histogram");
        let mut cum = 0;
        for (i, b) in store::HEAT_BUCKETS.iter().enumerate() {
            cum += v.heat.occ[i];
            let _ = writeln!(s, "llmt_kv_chains_heat_bucket{{le=\"{b}\"}} {cum}");
        }
        cum += v.heat.occ[store::HEAT_BUCKETS.len()];
        let _ = writeln!(s, "llmt_kv_chains_heat_bucket{{le=\"+Inf\"}} {cum}");
        let _ = writeln!(s, "llmt_kv_chains_heat_sum {}\nllmt_kv_chains_heat_count {cum}", v.heat.heat_sum);
        s
    }
}

/// Bind the socket and serve in background threads. Errors here never affect the KV broker.
pub fn serve(cfg: Config) -> std::io::Result<Chains> {
    if let Some(d) = cfg.socket.parent() {
        std::fs::create_dir_all(d)?;
    }
    let _ = std::fs::remove_file(&cfg.socket);
    let l = UnixListener::bind(&cfg.socket)?;
    std::fs::set_permissions(&cfg.socket, std::fs::Permissions::from_mode(0o660))?;
    let stats = Arc::new(Stats::default());
    let mut store = match cfg.snapshot.as_ref().filter(|p| cfg.global && p.exists()) {
        Some(p) => {
            let t = Instant::now();
            match Store::load(p, cfg.store_mem, store::unix_now()) {
                Ok(s) => {
                    eprintln!("chains: restored {} in {:.2} s ({} docs, {} tokens, {} keys, {} occurrences, {} permanent, \
                               model_tag {:?})", p.display(), t.elapsed().as_secs_f64(), s.docs.len(), s.n_tokens(),
                              s.index.len(), s.n_occ(), s.n_pinned(), s.model_tag);
                    s
                }
                Err(e) => {
                    eprintln!("chains: cannot restore {}: {e}; starting empty", p.display());
                    Store::new(cfg.store_mem)
                }
            }
        }
        None => Store::new(cfg.store_mem),
    };
    cfg.configure(&mut store);
    let view = Mutex::new(View::of(&store));
    let global = Arc::new(Global { store: Mutex::new(store), pending: Mutex::new(Pending::default()),
                                   last_activity_ms: AtomicU64::new(0), start: Instant::now(),
                                   snap_io: Mutex::new(()), view });
    if cfg.global {
        let (c2, g2, s2) = (cfg.clone(), global.clone(), stats.clone());
        std::thread::Builder::new().name("chains-gc".into()).spawn(move || gc_thread(c2, g2, s2))?;
    }
    let handle = Chains { stats: stats.clone(), global: global.clone(), cfg: cfg.clone() };
    let st = stats.clone();
    let gl = global.clone();
    std::thread::Builder::new().name("chains-accept".into()).spawn(move || {
        for conn in l.incoming() {
            let Ok(s) = conn else { continue };
            let (st, cfg, gl) = (st.clone(), cfg.clone(), gl.clone());
            st.connections.fetch_add(1, Relaxed);
            let _ = std::thread::Builder::new().name("chains-conn".into()).spawn(move || {
                if let Err(e) = serve_conn(&cfg, &st, &gl, s) {
                    if e.kind() != std::io::ErrorKind::UnexpectedEof && e.kind() != std::io::ErrorKind::BrokenPipe {
                        st.errors.fetch_add(1, Relaxed);
                        eprintln!("chains: connection closed: {e}");
                    }
                }
                st.connections.fetch_sub(1, Relaxed);
            });
        }
    })?;
    Ok(handle)
}

fn serve_conn(cfg: &Config, st: &Stats, g: &Global, s: UnixStream) -> std::io::Result<()> {
    let mut r = BufReader::with_capacity(1 << 20, s.try_clone()?);
    let mut w = BufWriter::with_capacity(1 << 16, s);
    let mut buf = Vec::new();
    let mut reqs: HashMap<u64, Req> = HashMap::new();
    let (mut gate, mut max_chain, mut hello, mut engine) = (16usize, 64usize, false, 0u16);
    let mut out = Vec::new();
    let res = (|| -> std::io::Result<()> {
        loop {
            let Some(ty) = proto::read_frame(&mut r, &mut buf)? else { return Ok(()) };
            let m = proto::decode(ty, &buf)?;
            out.clear();
            match m {
                Msg::Hello { proto: pv, gate: gt, max_chain: mc, engine_id, model_tag } => {
                    let status = if pv != proto::PROTOCOL {
                        1
                    } else {
                        // one model per store: the first non-empty tag wins, a different one is refused. A tag
                        // restored from a snapshot is provisional: the first engine with a different tag discards
                        // the restored store (another model's chains are useless) instead of being refused.
                        let mut s = g.store.lock().unwrap();
                        if s.tag_provisional && !model_tag.is_empty() {
                            if model_tag != s.model_tag {
                                eprintln!("chains: snapshot model_tag {:?} != engine {:?}: discarding the restored store",
                                          s.model_tag, model_tag);
                                s.clear();
                                st.restore_discarded.fetch_add(1, Relaxed);
                            }
                            s.tag_provisional = false;
                        }
                        if s.model_tag.is_empty() {
                            s.model_tag = model_tag.clone();
                        }
                        if !model_tag.is_empty() && model_tag != s.model_tag { 2 } else { 0 }
                    };
                    engine = engine_id;
                    gate = (gt as usize).max(proto_min_gate());
                    max_chain = (mc as usize).clamp(1, 64);
                    hello = status == 0;
                    proto::encode(&Msg::HelloAck { proto: proto::PROTOCOL, status }, &mut out);
                }
                _ if !hello => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "message before hello")),
                Msg::Open { req, tokens, .. } => {
                    if reqs.len() >= cfg.max_open && !reqs.contains_key(&req) {
                        st.errors.fetch_add(1, Relaxed);
                        continue;
                    }
                    let t = Instant::now();
                    let mut toks = tokens;
                    toks.truncate(cfg.max_tokens);
                    let prompt_len = toks.len();
                    let old = reqs.insert(req, Req { ix: RequestIndex::new(toks), hint_active: false, prompt_len });
                    st.open_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                    st.opens.fetch_add(1, Relaxed);
                    if old.is_none() {
                        st.open_now.fetch_add(1, Relaxed);
                    }
                }
                Msg::Append { req, known_len, phase, tokens } => {
                    g.last_activity_ms.store(g.now_ms(), Relaxed);
                    st.appends.fetch_add(1, Relaxed);
                    st.appended_tokens.fetch_add(tokens.len() as u64, Relaxed);
                    let Some(q) = reqs.get_mut(&req) else {
                        st.unknown_req.fetch_add(1, Relaxed);
                        continue;
                    };
                    if q.ix.len() + tokens.len() != known_len as usize || q.ix.len() + tokens.len() > cfg.max_tokens {
                        // lost or reordered appends: the index no longer matches the engine; drop it (the client
                        // re-opens the request with its full token list)
                        st.desyncs.fetch_add(1, Relaxed);
                        if q.hint_active {
                            proto::encode(&Msg::NoHint { req, known_len }, &mut out);
                        }
                        reqs.remove(&req);
                        st.open_now.fetch_sub(1, Relaxed);
                    } else {
                        let t = Instant::now();
                        let m = q.ix.append(&tokens, cfg.max_match);
                        let hint = if phase == proto::PHASE_THINK {
                            st.think_appends.fetch_add(1, Relaxed);
                            None
                        } else {
                            m.filter(|m| m.len >= gate)
                        };
                        // own context first; the global store only on a miss, and never waiting for it
                        let global = if hint.is_none() && phase != proto::PHASE_THINK && cfg.global {
                            match g.store.try_lock() {
                                Ok(mut s) => {
                                    let t = &q.ix.tokens;
                                    s.lookup(&t[t.len().saturating_sub(store::LEFT)..], gate, max_chain, store::unix_now())
                                }
                                Err(_) => {
                                    st.global_busy.fetch_add(1, Relaxed);
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        match (hint, global) {
                            (None, Some(gh)) => {
                                st.global_hints.fetch_add(1, Relaxed);
                                st.global_hint_tokens.fetch_add(gh.tokens.len() as u64, Relaxed);
                                proto::encode(&Msg::Hint { req, known_len, match_len: gh.match_len.min(u16::MAX as usize) as u16,
                                    source: proto::SRC_GLOBAL, chain_ref: gh.chain_ref, tokens: gh.tokens }, &mut out);
                                q.hint_active = true;
                            }
                            (Some(m), _) => {
                                let toks = q.ix.continuation(m.pos, max_chain).to_vec();
                                st.hints.fetch_add(1, Relaxed);
                                st.hint_tokens.fetch_add(toks.len() as u64, Relaxed);
                                let chain_ref = (m.pos as u64) | ((m.len.min(u16::MAX as usize) as u64) << 32);
                                proto::encode(&Msg::Hint { req, known_len, match_len: m.len.min(u16::MAX as usize) as u16,
                                    source: proto::SRC_OWN, chain_ref, tokens: toks }, &mut out);
                                q.hint_active = true;
                            }
                            (None, None) if q.hint_active => {
                                st.no_hints.fetch_add(1, Relaxed);
                                proto::encode(&Msg::NoHint { req, known_len }, &mut out);
                                q.hint_active = false;
                            }
                            (None, None) => {}
                        }
                        st.lookup_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                    }
                }
                Msg::Close { req, status } => {
                    if let Some(q) = reqs.get(&req) {
                        // a finished request's output feeds the global store (inserted on the idle GC thread)
                        let out_toks = &q.ix.tokens[q.prompt_len.min(q.ix.len())..];
                        if cfg.global && status == 0 && out_toks.len() > crate::index::GRAM {
                            let mut p = g.pending.lock().unwrap();
                            if p.tokens + out_toks.len() <= PENDING_MAX_TOKENS {
                                p.tokens += out_toks.len();
                                p.docs.push((out_toks.to_vec(), store::SRC_OUTPUT, engine, false));
                                st.docs_queued.fetch_add(1, Relaxed);
                            } else {
                                st.docs_dropped.fetch_add(1, Relaxed);
                            }
                        }
                    }
                    if reqs.remove(&req).is_some() {
                        st.closes.fetch_add(1, Relaxed);
                        st.open_now.fetch_sub(1, Relaxed);
                    }
                }
                Msg::Feedback { items } => {
                    let mut gfb = Vec::new();
                    for (_, o, a, c) in items {
                        st.offered.fetch_add(o as u64, Relaxed);
                        st.accepted.fetch_add(a as u64, Relaxed);
                        if c & store::GLOBAL_REF != 0 {
                            gfb.push((c, o, a));
                        }
                    }
                    if !gfb.is_empty() {
                        g.pending.lock().unwrap().feedback.extend(gfb);
                    }
                }
                Msg::Seed { flags, tokens } => {
                    st.seeds.fetch_add(1, Relaxed);
                    let mut p = g.pending.lock().unwrap();
                    if p.tokens + tokens.len() <= PENDING_MAX_TOKENS {
                        p.tokens += tokens.len();
                        p.docs.push((tokens, store::SRC_CORPUS, engine, flags & 1 != 0));
                    } else {
                        st.docs_dropped.fetch_add(1, Relaxed);
                    }
                }
                Msg::StatsReq => {
                    let mut j = st.json();
                    j.pop(); // the closing brace: the store section goes inside
                    j.push_str(&store_json(g));
                    proto::encode(&Msg::Stats(j), &mut out)
                }
                other => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("unexpected {other:?}"))),
            }
            if !out.is_empty() {
                w.write_all(&out)?;
            }
            // one flush per burst: only when no further input is already buffered
            if r.buffer().is_empty() {
                w.flush()?;
            }
        }
    })();
    st.open_now.fetch_sub(reqs.len() as u64, Relaxed);
    res
}

/// Hints shorter than the 8-gram are impossible; the gate is at least that.
fn proto_min_gate() -> usize {
    crate::index::GRAM
}
