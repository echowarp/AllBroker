//! llmt-kvbroker: control plane for the shared host-RAM KV tier (protocol: docs/protocol.md).
//!
//! Threads: accept loop; one reader per connection (decode, run against the index under one lock, write the reply
//! outside the lock); a 10 ms timer (timeouts, ephemeral expiry, low-water eviction, WAIT timeouts); a tiny HTTP
//! server for Prometheus text. KV bytes never pass through here: engines mmap the pool file.

mod disk;
mod fp;
mod lazy;
mod ngram;
mod shm;
mod trace;
mod bg;

use kvcore::policy::PolicyKind;
use kvcore::{ConnId, Core, Notify};
use kvproto::{read_frame, reply_frame, HelloReply, Op, Reply, Request, Status, PROTOCOL_VERSION};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, Read, Write};
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use tracing_mutex::stdsync::Mutex;
use std::time::{Duration, Instant};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const LAT_BUCKETS_US: [u64; 11] = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 100000];

struct Opts {
    pool_dir: PathBuf,
    pool_bytes: u64,
    slot_bytes: u64,
    socket: PathBuf,
    metrics: String,
    policy: PolicyKind,
    prefault: bool,
    reserve_timeout_ms: u64,
    lease_timeout_ms: u64,
    low_water_pct: f64,
    new_pool: bool,
    reuse_grace_ms: u64,
    mlock: bool,
    stats_log_s: u64,
    events_dir: String,
    /// lookup chains listener (crates/chains): socket path or "off" (default)
    chains: String,
    chains_global: bool,
    chains_mem_mib: usize,
    chains_snapshot: String,
    chains_snapshot_s: u64,
    chains_heat_half_life_s: u32,
    chains_prune_grace_s: u32,
    chains_prune_heat: f32,
    stale_unhit_s: u64,
    ns_floor_pct: f64,
    ns_cap_pct: f64,
    scrub: String,
    scrub_mbps: u64,
    hotset_dir: String,
    hotset_gib: f64,
    hotset_min_hits: u64,
    hotset_save_s: u64,
    hotset_stable_saves: u32,
    /// rank the hot set by use (distinct request chains touching an entry, decayed; hot.rs) or by RAM-tier hits
    hotset_rank_use: bool,
    hotset_min_uses: f64,
    hotset_use_half_life_s: u64,
    /// SSD tier (disk.rs): directory or "off", budget, per-namespace cap, write-behind period
    disk_dir: String,
    disk_gib: f64,
    disk_ns_gib: f64,
    disk_flush_s: u64,
    disk_min_hits: u8,
    disk_state_min_hits: u8,
    disk_write_gib_day: f64,
    /// Dynamic pool (the broker changes its own memory use on demand): the pool file keeps its
    /// size (engines map it once and check it on reconnect), but slots can be PARKED: evicted, taken out of the
    /// allocator, their 2 MiB-aligned interior hole-punched and (on hugetlbfs) the huge-page reservation lowered, so the
    /// memory returns to the kernel; later re-allocated and unparked. Explicit: POST /pool?target_gib=N on the metrics
    /// port. Autonomous: when MemAvailable drops below --mem-reserve-gib the broker culls its worst entries to make up
    /// the deficit; after --pool-regrow-s of calm it grows back 2 GiB at a time. Never below --pool-min-gib.
    pool_min_gib: f64,
    mem_reserve_gib: f64,
    /// pressure protection: under memory pressure the sweep keeps the entries of sessions used in the last
    /// --protect-window-s and the shared class, newest first, up to --protect-gib (0 = off); below
    /// --protect-floor-gib MemAvailable the protection gives way (least recently used first)
    protect_gib: f64,
    protect_window_s: u64,
    protect_floor_gib: f64,
    /// SSD write mode: "through" (admitted entries written while in RAM, as before) or "strict" (written
    /// only when they leave RAM by age, and only for sessions with proven repeated use; shared class as before)
    disk_write_mode: String,
    disk_session_resumes: u32,
    disk_session_gap_s: u64,
    disk_session_span_s: u64,
    pool_regrow_s: u64,
    pool_ack_timeout_s: u64,
    hugepages_ctl: bool,
    /// N-gram table service: NAME=GGUF path (any shard), repeatable; off when none.
    ngram_tables: Vec<(String, PathBuf)>,
    ngram_bins: Vec<(String, PathBuf)>,
    /// rows from another file (other codec): NAME=FILE@OFFSET/ROW_BYTES
    ngram_rows: Vec<(String, ngram::RowsSrc)>,
    /// tables kept wholly in RAM once loaded
    ngram_resident: Vec<String>,
    ngram_cache_gib: f64,
    ngram_windows: usize,
    ngram_window_mib: u64,
    ngram_idle_s: u64,
    /// exit after this many seconds without connections (0 = never; for socket-activated dev instances)
    idle_exit_s: u64,
    /// On-demand pool (lazy.rs): memory only for slots holding entries; gradual eviction; compaction
    pool_lazy: bool,
    /// on-demand pool: no hole punching once a client that may pin the pool appeared (lazy.rs notes)
    pool_pin_safe: bool,
    /// on-demand pool reused at start (a restart): no hole punching for this long
    pool_reuse_hold_s: u64,
    sweep_s: u64,
    /// on-demand pools: free memory kept allocated ahead by the broker per pool while a client stores (MiB; 0 = off)
    warm_mib: u64,
    /// exit (systemd restarts the broker) when an internal lock could not be taken for this long (0 = off)
    watchdog_s: u64,
    evict_age_s: u64,
    evict_step_pct: f64,
    idle_shrink_s: u64,
    compact_mib: u64,
    idle_exit_max_s: u64,
    /// Unified broker: slab classes NAME=SLOT_KIB[,ram=GiB][,ssd=GiB][,write=GiB]; a client picks one by
    /// its HELLO name (unknown names: the first). --pool-gib is then the one RAM ceiling over all of them.
    classes: Vec<ClassSpec>,
    inactive_div: u64,
    tail_age_div: u64,
    heat_hit_ms: u64,
    low_margin_s: u64,
    shared_gib: f64,
    shared_age_mult: u64,
    shared_min_cont: u8,
    shared_max_depth: u32,
    /// segment-wise eviction (kvcore seg.rs): on by default; --seg-evict off = per-entry order (rollback)
    seg_evict: bool,
    seg_grace_s: u64,
    /// weights entries (kind 5, e.g. a vision tower) are valued per slot against KV: one KV slot of --slot-kib costs
    /// this many ms to recompute (kvcore Config::weights_kv_slot_ms; other classes scaled by their slot size)
    weights_kv_slot_ms: u64,
    /// SSD tier codec: 1 = 4-byte planes + zstd 1 (default), 0 = raw
    disk_codec: u8,
    ram_compress_after_s: u64,
    ram_compress_quiet_ms: u64,
    /// integrity checksums: bandwidth cap of the idle checksum thread (0 = off), re-verification age
    cksum_mib_s: u64,
    cksum_reverify_s: u64,
    /// operation trace (trace.rs): ring size (0 = off), dump directory, dumps per day
    trace_mib: u64,
    trace_dir: String,
    trace_dumps_day: u32,
    /// request threads' scheduler slice (EEVDF, sched_setattr runtime; 0 = kernel default)
    latency_slice_us: u64,
    /// prefix-affinity hints (HINT / WHERE, docs/prefix-hints.md): side-table size; 0 = off (the feature is not offered)
    hint_mib: u64,
    /// where the CPU-heavy idle work (checksums, in-place compression) runs: a background worker process (default;
    /// `llmt-kvbroker --bg-worker SOCKET`, its own low-weight unit) or inside the broker (internal: rollback)
    bg_worker: bool,
}

#[derive(Clone, Debug)]
struct ClassSpec {
    name: String,
    slot: u64,
    ram: Option<f64>,
    ssd: Option<f64>,
    write: Option<f64>,
    /// in-place RAM compression chunk (bytes; 0 = off): rcodec=1,chunk=N
    rchunk: u64,
    /// SSD write mode of this class: strict=1 / strict=0; None = --disk-write-mode
    strict: Option<bool>,
}

fn parse_class(v: &str) -> Option<ClassSpec> {
    let (name, rest) = v.split_once('=')?;
    let mut it = rest.split(',');
    let slot = it.next()?.parse::<u64>().ok()? << 10;
    let mut c = ClassSpec { name: name.to_string(), slot, ram: None, ssd: None, write: None, rchunk: 0, strict: None };
    let mut rcodec = false;
    for kv in it {
        let (k, x) = kv.split_once('=')?;
        let x: f64 = x.parse().ok()?;
        match k {
            "ram" => c.ram = Some(x),
            "ssd" => c.ssd = Some(x),
            "write" => c.write = Some(x),
            "rcodec" => rcodec = x != 0.0,
            "chunk" => c.rchunk = x as u64,
            "strict" => c.strict = Some(x != 0.0),
            _ => return None,
        }
    }
    if rcodec && c.rchunk == 0 {
        c.rchunk = c.slot;
    }
    if !rcodec {
        c.rchunk = 0;
    }
    (!c.name.is_empty() && !c.name.contains('/') && c.slot > 0).then_some(c)
}

fn usage() -> ! {
    eprintln!(
        "usage: llmt-kvbroker [--pool-gib 24] [--slot-kib 2048] [--pool-dir /dev/shm/llmt-kv]
                     [--socket /run/llmt-kv/broker.sock] [--metrics 127.0.0.1:9103 | --metrics off]
                     [--policy arc|lru] [--prefault] [--reserve-timeout-ms 300000]
                     [--lease-timeout-ms 60000] [--low-water-pct 1] [--new-pool] [--reuse-grace-ms 3000]
                     [--no-mlock] [--stats-log-s 600 (0 = off)]
                     [--events-dir /var/lib/llmt-kv | off]
                     [--stale-unhit-s 0] [--ns-floor-pct 0] [--ns-cap-pct 0]   (shared-pool fairness; 0 = off)
                     [--scrub report|evict|off] [--scrub-mbps 500]
                     [--hotset-dir /var/lib/llmt-kv/hotset | off] [--hotset-gib 2] [--hotset-min-hits 3]
                     [--pool-min-gib 6] [--mem-reserve-gib 4 (0 = no autonomous culling)] [--pool-regrow-s 60]
                     [--pool-ack-timeout-s 0 (= wait for every pinning client)] [--no-hugepages-ctl]
                     (dynamic pool; POST /pool?target_gib=N | ?reserve_gib=N at runtime)
                     [--disk-dir DIR | off] [--disk-gib 40] [--disk-ns-gib = disk-gib] [--disk-flush-s 5] [--disk-min-hits 2]
                     [--disk-state-min-hits 1] [--disk-write-gib-day 4] (SSD tier; replaces the hot set's files)
                     [--disk-write-mode through|strict] [--disk-session-resumes 3] [--disk-session-gap-s 600]
                     [--disk-session-span-s 3600] (strict: write only what leaves RAM by age, only for sessions resumed
                      >= N times after >= gap idle over >= span; the newest state only; shared class as before)
                     [--protect-gib 0 (off)] [--protect-window-s 900] [--protect-floor-gib 2] (memory pressure keeps
                      the sessions used in the window + the shared class, newest first, up to this; below the floor
                      MemAvailable the protection gives way)
                     [--hotset-save-s 1800] [--hotset-stable-saves 3] [--hotset-rank use|hits]
                     [--hotset-min-uses 8] [--hotset-use-half-life-s 86400] [--chains /run/llmt-kv/chains.sock | off]
                     [--chains-global on|off] [--chains-mem-mib 1024]
                     [--chains-snapshot off | PATH] [--chains-snapshot-s 1800]
                     [--chains-heat-half-life-s 21600] [--chains-prune-grace-s 7200] [--chains-prune-heat 0.25]
                     [--ngram-table NAME=GGUF (repeatable)] [--ngram-cache-gib 4] [--ngram-windows 16]
                     [--ngram-window-mib 16] [--ngram-idle-s 600] [--ngram-bins NAME=NGBINS (repeatable)]
                     [--ngram-rows NAME=FILE@OFFSET/ROW_BYTES (repeatable; rows of another codec, hashing from the GGUF)]
                     [--ngram-resident NAME (repeatable; whole table in RAM when memory allows, dropped under pressure)]
                     [--idle-exit-s 0 (exit after N s without connections; with systemd socket activation, LISTEN_FDS)]
                     [--pool-lazy on|off (on: memory only for slots in use; tmpfs only)]
                     [--pool-pin-safe on|off (on: on-demand pool never punches once a client asked for the pool state)]
                     [--pool-reuse-hold-s 30 (reused on-demand pool: no punching for this long after a restart)]
                     [--sweep-s 15] [--warm-mib 512] [--watchdog-s 20 (0 = off)] [--evict-age-s 3600 (0 = off)] [--evict-step-pct 25] [--idle-shrink-s 0 (0 = off)]
                     (on-demand pool: --idle-exit-s waits for an empty pool, at most --idle-exit-max-s 3900)
                     [--compact-mib 512 (per sweep; 0 = off)]
                     [--class NAME=SLOT_KIB[,ram=GiB][,ssd=GiB][,write=GiB][,strict=1] (repeatable: one broker for several slot sizes;
                      HELLO name picks the class; --pool-gib = one RAM ceiling)] [--inactive-age-div 4] [--tail-age-div 4]
                     [--low-evict-margin-s 1800 (value classes, RESERVE flags ENTRY_LOW / ENTRY_LOWEST: stored and aged like
                      any entry; under room pressure the lowest, then the low class is taken before the policy's victim
                      unless more than this much hotter; hot set after normal prefixes; 0 = no class order)]
                     [--shared-gib 0 (shared-prefix class per pool; 0 = off)] [--shared-age-mult 168 (1 week at --evict-age-s 3600)]
                     [--shared-min-cont 3 (different continuations)] [--shared-max-depth 19 (chunks from the root)]
                     [--weights-kv-slot-ms 1300 (weights entries, kind 5: one KV slot's recompute ms; a weights entry
                      ages KV_SLOT_MS x slots / cost_hint times faster than KV, so it stays while worth more per slot; 0 = plain recency)]
                     [--seg-evict on|off (default on: a GDN state and the attention only it makes usable leave together;
                      attention no state can use and states whose chain is gone go first; off = per-entry order)]
                     [--seg-grace-s 900 (uncovered attention / broken states kept this long after last use or store)]
                     [--disk-codec planes-zstd|none (SSD tier: 4-byte planes + zstd 1 per 4 MiB chunk, lossless)]
                     [--ram-compress-after-s 30] (class option rcodec=1[,chunk=BYTES]: compress committed entries in
                      place in their slot; readers must decompress: the Arc staging-ring worker does)
                     [--ram-compress-quiet-ms 2000] (compress only after this long without a request, stop at the next
                      one; 0 = any time)
                     [--cksum-mib-s 256] [--cksum-reverify-s 21600] (integrity checksums, no ECC: an idle thread
                      hashes each RAM entry (XXH3-64) and re-verifies it at this age, at most this many MiB/s and only
                      after --ram-compress-quiet-ms without a request; a mismatch evicts the entry and the rest of its
                      chain. 0 = off)
                     [--trace-mib 64] [--trace-dir ~/.cache/llmt-kv-trace] [--trace-dumps-day 12] (operation trace: every
                      request and answer, metadata only, in a ring of this size; written to the directory only on an
                      anomaly or TRACE_DUMP, at most this many files a day, the newest 10 kept; replay: kvbench replay.
                      --trace-mib 0 or --trace-dir off = off)
                     [--bg worker|internal] (checksums and in-place compression: in a background worker process,
                      `llmt-kvbroker --bg-worker SOCKET` in its own low-weight unit (default; without a worker they
                      pause), or inside the broker (rollback))
                     [--latency-slice-us 100] (scheduler slice of the request threads: they wake ahead of busy
                      threads of other cgroups under CPU load; background threads keep the default; 0 = default)
                     [--hint-mib 32] (prefix-affinity hints for routers, HINT / WHERE: side-table size, 0 = off)
                     (n-gram tables load on first use and unload after idle; the cache shrinks under the same
                      --mem-reserve-gib pressure as the pool, before it, and regrows after --pool-regrow-s calm)"
    );
    std::process::exit(2)
}

fn parse_opts() -> Opts {
    let mut o = Opts {
        pool_dir: "/dev/shm/llmt-kv".into(),
        pool_bytes: 24 << 30,
        slot_bytes: 2 << 20,
        socket: "/run/llmt-kv/broker.sock".into(),
        metrics: "127.0.0.1:9103".into(),
        policy: PolicyKind::Arc,
        prefault: false,
        reserve_timeout_ms: 300_000,
        lease_timeout_ms: 60_000,
        low_water_pct: 1.0,
        new_pool: false,
        reuse_grace_ms: 3_000,
        mlock: true,
        stats_log_s: 600,
        events_dir: "/var/lib/llmt-kv".into(),
        chains: "off".into(),
        chains_global: true,
        chains_mem_mib: 1024,
        // off by default (the only RAM-tier disk writes are the persistent hot set); enable deliberately
        chains_snapshot: "off".into(),
        chains_snapshot_s: 1800,
        chains_heat_half_life_s: kvchains::store::HALF_LIFE_S,
        chains_prune_grace_s: kvchains::store::PRUNE_GRACE_S,
        chains_prune_heat: kvchains::store::PRUNE_HEAT,
        stale_unhit_s: 0,
        ns_floor_pct: 0.0,
        ns_cap_pct: 0.0,
        scrub: "report".into(),
        scrub_mbps: 500,
        hotset_dir: "/var/lib/llmt-kv/hotset".into(),
        hotset_gib: 2.0,
        hotset_min_hits: 3,
        hotset_save_s: 1800,
        hotset_stable_saves: 3,
        hotset_rank_use: true,
        hotset_min_uses: 8.0,
        hotset_use_half_life_s: 86_400,
        disk_dir: "off".into(),
        disk_gib: 40.0,
        disk_ns_gib: 0.0,
        disk_flush_s: 5,
        disk_min_hits: 2,
        disk_state_min_hits: 1,
        disk_write_gib_day: 4.0,
        pool_min_gib: 6.0,
        mem_reserve_gib: 4.0,
        protect_gib: 0.0,
        protect_window_s: 900,
        protect_floor_gib: 2.0,
        disk_write_mode: "through".into(),
        disk_session_resumes: 3,
        disk_session_gap_s: 600,
        disk_session_span_s: 3600,
        pool_regrow_s: 60,
        pool_ack_timeout_s: 0,
        hugepages_ctl: true,
        ngram_tables: Vec::new(),
        ngram_bins: Vec::new(),
        ngram_rows: Vec::new(),
        ngram_resident: Vec::new(),
        ngram_cache_gib: 4.0,
        ngram_windows: 16,
        ngram_window_mib: 16,
        ngram_idle_s: 600,
        idle_exit_s: 0,
        pool_lazy: true,
        pool_pin_safe: true,
        pool_reuse_hold_s: 30,
        sweep_s: 15,
        warm_mib: 512,
        watchdog_s: 20,
        evict_age_s: 3600,
        evict_step_pct: 25.0,
        idle_shrink_s: 0,
        compact_mib: 512,
        idle_exit_max_s: 3900,
        classes: Vec::new(),
        inactive_div: 4,
        tail_age_div: 4,
        heat_hit_ms: 600_000,
        low_margin_s: 1800,
        shared_gib: 0.0,
        shared_age_mult: 168,
        shared_min_cont: 3,
        shared_max_depth: 19,
        seg_evict: true,
        seg_grace_s: 900,
        // one 56,100 KiB int4 slot = a 3,200-token block of Qwen3.8-27B; ~2,400 tok/s prefill under load on the Arc
        weights_kv_slot_ms: 1300,
        disk_codec: 1,
        ram_compress_after_s: 30,
        ram_compress_quiet_ms: 2000,
        cksum_mib_s: 256,
        cksum_reverify_s: 6 * 3600,
        trace_mib: 64,
        trace_dir: std::env::var("HOME").map(|h| format!("{h}/.cache/llmt-kv-trace")).unwrap_or_else(|_| "off".into()),
        trace_dumps_day: 12,
        latency_slice_us: 100,
        hint_mib: 32,
        bg_worker: true,
    };
    let mut a = std::env::args().skip(1);
    while let Some(f) = a.next() {
        let mut val = || a.next().unwrap_or_else(|| usage());
        match f.as_str() {
            "--pool-gib" => o.pool_bytes = (val().parse::<f64>().unwrap_or_else(|_| usage()) * (1u64 << 30) as f64) as u64,
            "--slot-kib" => o.slot_bytes = val().parse::<u64>().unwrap_or_else(|_| usage()) << 10,
            "--pool-dir" => o.pool_dir = val().into(),
            "--socket" => o.socket = val().into(),
            "--metrics" => o.metrics = val(),
            "--policy" => o.policy = val().parse().unwrap_or_else(|e| {
                eprintln!("{e}");
                usage()
            }),
            "--prefault" => o.prefault = true,
            "--reserve-timeout-ms" => o.reserve_timeout_ms = val().parse().unwrap_or_else(|_| usage()),
            "--lease-timeout-ms" => o.lease_timeout_ms = val().parse().unwrap_or_else(|_| usage()),
            "--low-water-pct" => o.low_water_pct = val().parse().unwrap_or_else(|_| usage()),
            "--new-pool" => o.new_pool = true,
            "--reuse-grace-ms" => o.reuse_grace_ms = val().parse().unwrap_or_else(|_| usage()),
            "--no-mlock" => o.mlock = false,
            "--stats-log-s" => o.stats_log_s = val().parse().unwrap_or_else(|_| usage()),
            "--events-dir" => o.events_dir = val(),
            "--chains" => o.chains = val(),
            "--chains-global" => o.chains_global = val() != "off",
            "--chains-mem-mib" => o.chains_mem_mib = val().parse().unwrap_or_else(|_| usage()),
            "--chains-snapshot" => o.chains_snapshot = val(),
            "--chains-snapshot-s" => o.chains_snapshot_s = val().parse().unwrap_or_else(|_| usage()),
            "--chains-heat-half-life-s" => o.chains_heat_half_life_s = val().parse().unwrap_or_else(|_| usage()),
            "--chains-prune-grace-s" => o.chains_prune_grace_s = val().parse().unwrap_or_else(|_| usage()),
            "--chains-prune-heat" => o.chains_prune_heat = val().parse().unwrap_or_else(|_| usage()),
            "--stale-unhit-s" => o.stale_unhit_s = val().parse().unwrap_or_else(|_| usage()),
            "--ns-floor-pct" => o.ns_floor_pct = val().parse().unwrap_or_else(|_| usage()),
            "--ns-cap-pct" => o.ns_cap_pct = val().parse().unwrap_or_else(|_| usage()),
            "--scrub" => o.scrub = val(),
            "--scrub-mbps" => o.scrub_mbps = val().parse().unwrap_or_else(|_| usage()),
            "--hotset-dir" => o.hotset_dir = val(),
            "--hotset-gib" => o.hotset_gib = val().parse().unwrap_or_else(|_| usage()),
            "--hotset-min-hits" => o.hotset_min_hits = val().parse().unwrap_or_else(|_| usage()),
            "--hotset-save-s" => o.hotset_save_s = val().parse().unwrap_or_else(|_| usage()),
            "--hotset-stable-saves" => o.hotset_stable_saves = val().parse().unwrap_or_else(|_| usage()),
            "--hotset-rank" => o.hotset_rank_use = match val().as_str() {
                "use" => true,
                "hits" => false,
                _ => usage(),
            },
            "--hotset-min-uses" => o.hotset_min_uses = val().parse().unwrap_or_else(|_| usage()),
            "--hotset-use-half-life-s" => o.hotset_use_half_life_s = val().parse().unwrap_or_else(|_| usage()),
            "--disk-dir" => o.disk_dir = val(),
            "--disk-gib" => o.disk_gib = val().parse().unwrap_or_else(|_| usage()),
            "--disk-ns-gib" => o.disk_ns_gib = val().parse().unwrap_or_else(|_| usage()),
            "--disk-flush-s" => o.disk_flush_s = val().parse().unwrap_or_else(|_| usage()),
            "--disk-min-hits" => o.disk_min_hits = val().parse().unwrap_or_else(|_| usage()),
            "--disk-state-min-hits" => o.disk_state_min_hits = val().parse().unwrap_or_else(|_| usage()),
            "--disk-write-gib-day" => o.disk_write_gib_day = val().parse().unwrap_or_else(|_| usage()),
            "--pool-min-gib" => o.pool_min_gib = val().parse().unwrap_or_else(|_| usage()),
            "--mem-reserve-gib" => o.mem_reserve_gib = val().parse().unwrap_or_else(|_| usage()),
            "--protect-gib" => o.protect_gib = val().parse().unwrap_or_else(|_| usage()),
            "--protect-window-s" => o.protect_window_s = val().parse().unwrap_or_else(|_| usage()),
            "--protect-floor-gib" => o.protect_floor_gib = val().parse().unwrap_or_else(|_| usage()),
            "--disk-write-mode" => {
                o.disk_write_mode = val();
                if o.disk_write_mode != "through" && o.disk_write_mode != "strict" {
                    usage()
                }
            }
            "--disk-session-resumes" => o.disk_session_resumes = val().parse().unwrap_or_else(|_| usage()),
            "--disk-session-gap-s" => o.disk_session_gap_s = val().parse().unwrap_or_else(|_| usage()),
            "--disk-session-span-s" => o.disk_session_span_s = val().parse().unwrap_or_else(|_| usage()),
            "--pool-regrow-s" => o.pool_regrow_s = val().parse().unwrap_or_else(|_| usage()),
            "--pool-ack-timeout-s" => o.pool_ack_timeout_s = val().parse().unwrap_or_else(|_| usage()),
            "--no-hugepages-ctl" => o.hugepages_ctl = false,
            "--ngram-table" => {
                let v = val();
                let (n, path) = v.split_once('=').unwrap_or_else(|| usage());
                o.ngram_tables.push((n.to_string(), path.into()));
            }
            "--ngram-bins" => {
                let v = val();
                let (n, path) = v.split_once('=').unwrap_or_else(|| usage());
                o.ngram_bins.push((n.to_string(), path.into()));
            }
            "--ngram-rows" => {
                let v = val();
                let (n, spec) = v.split_once('=').unwrap_or_else(|| usage());
                o.ngram_rows.push((n.to_string(), ngram::RowsSrc::parse(spec).unwrap_or_else(|| usage())));
            }
            "--ngram-resident" => o.ngram_resident.push(val()),
            "--ngram-cache-gib" => o.ngram_cache_gib = val().parse().unwrap_or_else(|_| usage()),
            "--ngram-windows" => o.ngram_windows = val().parse().unwrap_or_else(|_| usage()),
            "--ngram-window-mib" => o.ngram_window_mib = val().parse().unwrap_or_else(|_| usage()),
            "--idle-exit-s" => o.idle_exit_s = val().parse().unwrap_or_else(|_| usage()),
            "--pool-lazy" => o.pool_lazy = val() != "off",
            "--pool-pin-safe" => o.pool_pin_safe = val() != "off",
            "--pool-reuse-hold-s" => o.pool_reuse_hold_s = val().parse().unwrap_or_else(|_| usage()),
            "--sweep-s" => o.sweep_s = val().parse().unwrap_or_else(|_| usage()),
            "--warm-mib" => o.warm_mib = val().parse().unwrap_or_else(|_| usage()),
            "--watchdog-s" => o.watchdog_s = val().parse().unwrap_or_else(|_| usage()),
            "--evict-age-s" => o.evict_age_s = val().parse().unwrap_or_else(|_| usage()),
            "--evict-step-pct" => o.evict_step_pct = val().parse().unwrap_or_else(|_| usage()),
            "--idle-shrink-s" => o.idle_shrink_s = val().parse().unwrap_or_else(|_| usage()),
            "--compact-mib" => o.compact_mib = val().parse().unwrap_or_else(|_| usage()),
            "--idle-exit-max-s" => o.idle_exit_max_s = val().parse().unwrap_or_else(|_| usage()),
            "--class" => o.classes.push(parse_class(&val()).unwrap_or_else(|| usage())),
            "--inactive-age-div" => o.inactive_div = val().parse().unwrap_or_else(|_| usage()),
            "--tail-age-div" => o.tail_age_div = val().parse::<u64>().unwrap_or_else(|_| usage()).max(1),
            "--low-evict-margin-s" => o.low_margin_s = val().parse().unwrap_or_else(|_| usage()),
            "--heat-hit-s" => o.heat_hit_ms = val().parse::<u64>().unwrap_or_else(|_| usage()) * 1000,
            "--shared-gib" => o.shared_gib = val().parse().unwrap_or_else(|_| usage()),
            "--shared-age-mult" => o.shared_age_mult = val().parse::<u64>().unwrap_or_else(|_| usage()).max(1),
            "--shared-min-cont" => o.shared_min_cont = val().parse::<u8>().unwrap_or_else(|_| usage()).clamp(2, 4),
            "--shared-max-depth" => o.shared_max_depth = val().parse().unwrap_or_else(|_| usage()),
            "--seg-evict" => o.seg_evict = match val().as_str() { "on" | "1" => true, "off" | "0" => false, _ => usage() },
            "--seg-grace-s" => o.seg_grace_s = val().parse().unwrap_or_else(|_| usage()),
            "--weights-kv-slot-ms" => o.weights_kv_slot_ms = val().parse().unwrap_or_else(|_| usage()),
            "--ram-compress-after-s" => o.ram_compress_after_s = val().parse().unwrap_or_else(|_| usage()),
            "--ram-compress-quiet-ms" => o.ram_compress_quiet_ms = val().parse().unwrap_or_else(|_| usage()),
            "--cksum-mib-s" => o.cksum_mib_s = val().parse().unwrap_or_else(|_| usage()),
            "--cksum-reverify-s" => o.cksum_reverify_s = val().parse().unwrap_or_else(|_| usage()),
            "--trace-mib" => o.trace_mib = val().parse().unwrap_or_else(|_| usage()),
            "--trace-dir" => o.trace_dir = val(),
            "--trace-dumps-day" => o.trace_dumps_day = val().parse().unwrap_or_else(|_| usage()),
            "--latency-slice-us" => o.latency_slice_us = val().parse().unwrap_or_else(|_| usage()),
            "--hint-mib" => o.hint_mib = val().parse().unwrap_or_else(|_| usage()),
            "--bg" => o.bg_worker = match val().as_str() { "worker" => true, "internal" => false, _ => usage() },
            "--disk-codec" => o.disk_codec = match val().as_str() { "none" | "raw" => 0, "planes-zstd" => 1, _ => usage() },
            "--ngram-idle-s" => o.ngram_idle_s = val().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    if o.slot_bytes == 0 || o.slot_bytes % 4096 != 0 || o.pool_bytes < o.slot_bytes {
        eprintln!("slot size must be a positive multiple of 4 KiB and fit in the pool");
        usage()
    }
    o
}

fn random_u64() -> u64 {
    let mut b = [0u8; 8];
    if fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).is_err() {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
        return t.as_nanos() as u64 ^ (std::process::id() as u64) << 32;
    }
    u64::from_le_bytes(b)
}

/// Open the pool: reuse an existing `pool-*` file of exactly `bytes` (unless `new_pool`), so engines that still map
/// it after a broker restart keep their mapping and GPU registration and just reconnect; otherwise create
/// `pool-<epoch>`. Other `pool-*` files in the directory are removed (only the broker creates those). Returns the
/// path and whether it was reused. A reused pool's contents are stale: the index starts empty.
/// The whole pool is allocated up front (posix_fallocate): on a full tmpfs an engine writing a page that was never
/// allocated gets SIGBUS and dies without a traceback (e.g. two
/// pools larger than /dev/shm together), so a pool that doesn't fit fails here, at broker start, instead.
fn open_pool(dir: &Path, pool_bytes: u64, epoch: u64, prefault: bool, new_pool: bool, hp_ctl: bool, lazy: bool) -> std::io::Result<(PathBuf, bool)> {
    // file size rounded up to 2 MiB: required on hugetlbfs (a pool on 2 MB huge pages: never swapped or migrated);
    // harmless on tmpfs. Engines map only the pool bytes.
    let bytes = pool_bytes.div_ceil(2 << 20) * (2 << 20);
    fs::create_dir_all(dir)?;
    let mut reuse = None;
    for e in fs::read_dir(dir)?.flatten() {
        if !e.file_name().to_string_lossy().starts_with("pool-") {
            continue;
        }
        let fits = e.metadata().map(|m| m.is_file() && m.len() == bytes).unwrap_or(false);
        if !new_pool && fits && reuse.is_none() {
            reuse = Some(e.path());
            continue;
        }
        eprintln!("removing old pool {}", e.path().display());
        fs::remove_file(e.path())?;
    }
    if let Some(p) = reuse {
        if lazy {
            return Ok((p, true)); // lazy.rs punches it empty
        }
        let f = fs::OpenOptions::new().read(true).write(true).open(&p)?;
        ensure_hugepages(&f, dir, bytes, hp_ctl)?;
        allocate_file(&f, bytes, dir)?;
        if prefault {
            prefault_file(&f, bytes)?;
        }
        return Ok((p, true));
    }
    let path = dir.join(format!("pool-{epoch:016x}"));
    let f = fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path)?;
    f.set_permissions(fs::Permissions::from_mode(0o660))?;
    f.set_len(bytes)?;
    if lazy {
        eprintln!("pool: on demand (nothing allocated up front; ceiling {:.2} GiB); {} ({})", bytes as f64 / (1u64 << 30) as f64, fs_free(dir), dir.display());
        return Ok((path, false));
    }
    if let Err(e) = ensure_hugepages(&f, dir, bytes, hp_ctl).and_then(|_| allocate_file(&f, bytes, dir)) {
        let _ = fs::remove_file(&path);
        return Err(e);
    }
    if prefault {
        let t = Instant::now();
        prefault_file(&f, bytes)?;
        eprintln!("prefaulted {:.1} GiB in {:.1} s", bytes as f64 / (1u64 << 30) as f64, t.elapsed().as_secs_f64());
    }
    Ok((path, false))
}

/// Remove the pool / staging files in `dir` and in its class subdirectories (unified broker).
fn remove_pools(dir: &Path) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if e.file_type().is_ok_and(|t| t.is_dir()) {
            remove_pools(&e.path());
        } else if n.starts_with("pool-") || n.starts_with("ngram-") {
            let _ = fs::remove_file(e.path());
        }
    }
}

fn fs_free(dir: &Path) -> String {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap_or_default();
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return "unknown".into();
    }
    let gib = |n: u64| n as f64 * st.f_frsize as f64 / (1u64 << 30) as f64;
    format!("{:.1} of {:.1} GiB free", gib(st.f_bavail as u64), gib(st.f_blocks as u64))
}

/// Dynamic pool: a pool file on hugetlbfs may have holes (punched by the controller of a previous broker that was
/// killed before restoring the reservation, or shut down while shrunk). posix_fallocate needs a free huge page per hole
/// page, so raise vm.nr_hugepages by the deficit first (and say so); without the control flag only report.
fn ensure_hugepages(f: &fs::File, dir: &Path, bytes: u64, hp_ctl: bool) -> std::io::Result<()> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap_or_default();
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    let hugetlb = unsafe { libc::statfs(c.as_ptr(), &mut st) } == 0 && st.f_type as i64 == HUGETLBFS_MAGIC;
    if !hugetlb {
        return Ok(());
    }
    let allocated = f.metadata()?.blocks() * 512;
    let holes = bytes.saturating_sub(allocated);
    if holes == 0 {
        return Ok(());
    }
    let need = holes.div_ceil(HUGE);
    let free = hugepages("HugePages_Free:").unwrap_or(0);
    let total = hugepages("HugePages_Total:").unwrap_or(0);
    eprintln!("pool: the pool file has {:.2} GiB of holes ({} huge pages); {} of {} huge pages free", holes as f64 / (1u64 << 30) as f64, need, free, total);
    if free >= need {
        return Ok(());
    }
    if !hp_ctl {
        return Err(std::io::Error::other(format!("{} huge pages needed for the holes, {} free (--no-hugepages-ctl: not raising vm.nr_hugepages)", need, free)));
    }
    let want = total + (need - free);
    set_nr_hugepages(want).map_err(|e| std::io::Error::other(format!("cannot raise vm.nr_hugepages {total} -> {want}: {e}")))?;
    let free2 = hugepages("HugePages_Free:").unwrap_or(0);
    eprintln!("pool: raised vm.nr_hugepages {} -> {} ({} free now)", total, want, free2);
    if free2 < need {
        return Err(std::io::Error::other(format!("only {} of {} huge pages available after raising the reservation", free2, need)));
    }
    Ok(())
}

fn allocate_file(f: &fs::File, bytes: u64, dir: &Path) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let before = fs_free(dir);
    let r = unsafe { libc::posix_fallocate(f.as_raw_fd(), 0, bytes as libc::off_t) };
    if r != 0 {
        let e = std::io::Error::from_raw_os_error(r);
        return Err(std::io::Error::new(e.kind(), format!("cannot allocate {:.2} GiB ({e}; {} before)",
                                                         bytes as f64 / (1u64 << 30) as f64, before)));
    }
    eprintln!("pool allocated; {} ({})", fs_free(dir), dir.display());
    Ok(())
}

/// Lock the pool in RAM for the broker's lifetime (the mapping is never unmapped): mlock on a shared mapping keeps the
/// shmem pages resident, so they are never swapped out under the engines. The Arc's Level Zero import of the pool
/// (a userptr, not a long-term pin like cudaHostRegister) faulted the copy engine when pool pages were swapped
/// (xe "Engine memory CAT error, class=bcs", engine hung). Needs CAP_IPC_LOCK or RLIMIT_MEMLOCK >=
/// the pool size; on failure the broker warns and runs unlocked.
fn lock_pool(path: &Path, bytes: u64) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let f = fs::OpenOptions::new().read(true).write(true).open(path)?;
    unsafe {
        let p = libc::mmap(std::ptr::null_mut(), bytes as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, f.as_raw_fd(), 0);
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        if libc::mlock(p, bytes as usize) != 0 {
            let e = std::io::Error::last_os_error();
            libc::munmap(p, bytes as usize);
            return Err(e);
        }
    }
    Ok(())
}

fn prefault_file(f: &fs::File, bytes: u64) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    unsafe {
        let p = libc::mmap(std::ptr::null_mut(), bytes as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, f.as_raw_fd(), 0);
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        const MADV_POPULATE_WRITE: i32 = 23;
        if libc::madvise(p, bytes as usize, MADV_POPULATE_WRITE) != 0 {
            // older kernels: touch every page
            let s = std::slice::from_raw_parts_mut(p as *mut u8, bytes as usize);
            for i in (0..s.len()).step_by(4096) {
                std::ptr::write_volatile(&mut s[i], 0);
            }
        }
        libc::munmap(p, bytes as usize);
    }
    Ok(())
}

// ---------------------------------------------------------------- shared state

struct OpStats {
    count: AtomicU64,
    sum_us: AtomicU64,
    buckets: [AtomicU64; LAT_BUCKETS_US.len()],
}

impl OpStats {
    fn new() -> Self {
        OpStats { count: AtomicU64::new(0), sum_us: AtomicU64::new(0), buckets: Default::default() }
    }
    fn record(&self, us: u64) {
        self.count.fetch_add(1, Relaxed);
        self.sum_us.fetch_add(us, Relaxed);
        for (i, b) in LAT_BUCKETS_US.iter().enumerate() {
            if us <= *b {
                self.buckets[i].fetch_add(1, Relaxed);
            }
        }
    }
}

struct Shared {
    core: Mutex<Core>,
    writers: Mutex<HashMap<ConnId, Arc<Mutex<UnixStream>>>>,
    start: Instant,
    hello: HelloReply,
    ops: Vec<OpStats>, // indexed by op - 1
    protocol_errors: AtomicU64,
    /// Reused pool: no reservations before this time (ms since start), so a GPU copy still in flight from an
    /// engine that wrote into a slot under the previous broker cannot land in a slot handed out again.
    grace_until: u64,
    /// Client name per connection (HELLO) and per namespace (the first client that used it), for metric labels.
    conn_names: Mutex<HashMap<ConnId, String>>,
    /// FEATURES: per connection that sent it, its feature bits and the formats it reads; a connection
    /// without an entry is a legacy client (reads slot codec 1 only, sends plain TOUCH)
    conn_feats: Mutex<HashMap<ConnId, (u64, Vec<(String, u16)>)>>,
    ns_names: Mutex<HashMap<u32, String>>,
    /// Cache domains: domain id -> canonical descriptor; connection -> its domain.
    domains: Mutex<HashMap<u64, String>>,
    conn_domains: Mutex<HashMap<ConnId, u64>>,
    /// Wall-clock ms at `start` (event journal times = start_unix_ms + broker ms).
    start_unix_ms: u64,
    scrub: Mutex<ScrubStats>,
    /// lookup chains listener (set once it is up): journal line, metrics, shutdown snapshot
    chains: std::sync::OnceLock<kvchains::server::Chains>,
    /// dynamic pool controller (parked runs, hole punching, huge-page reservation, acks)
    pool: Mutex<PoolCtl>,
    /// on-demand pool (lazy.rs): slot backing, sweep statistics; time of the last request (compaction waits for quiet)
    backing: Option<Mutex<lazy::Backing>>,
    /// a client that may pin the pool asked for its state (POOL_STATE / POOL_ACK): on-demand pools stop punching
    pins: Arc<std::sync::atomic::AtomicBool>,
    pin_safe: bool,
    lazy: lazy::Stats,
    last_req_ms: AtomicU64,
    /// unified broker: the other slab classes (class 0 = core / hello / backing above, its SSD tier = disk::DISK),
    /// each connection's class, connections per (class, namespace) (domains without clients age faster), the ceiling
    extra: Vec<Class>,
    class0_name: String,
    conn_class: Mutex<HashMap<ConnId, usize>>,
    ns_conns: Mutex<HashMap<(usize, u32), HashSet<ConnId>>>,
    ceiling: u64,
    /// per class: in-place RAM compression chunk (0 = off), and the age after which an entry is compressed
    rchunk: Vec<u64>,
    rcompress_after_ms: u64,
    rcompress_quiet_ms: u64,
    /// integrity checksums: bytes per second (0 = off), re-verification age
    cksum_bytes_s: u64,
    cksum_reverify_ms: u64,
    latency_slice_us: u64,
    /// background worker: mode, its connections, and the rewrites each has begun (class, ns, key): a
    /// worker that disconnects mid-rewrite leaves torn slots, which are evicted
    bg_worker: bool,
    bg_conns: Mutex<HashSet<ConnId>>,
    /// open connections of background workers: not clients (idle exit, idle shrink and the trace ring count clients)
    bg_open: std::sync::atomic::AtomicUsize,
    bg_rewrites: Mutex<HashMap<ConnId, Vec<(usize, u32, kvproto::Key)>>>,
    /// prefix-affinity hints (docs/prefix-hints.md): anchor -> engine hints (bounded side table, never the index);
    /// observers (routers: HELLO engine id OBSERVER_ENGINE_ID) are not clients and not traffic, like the worker
    hints: Mutex<kvcore::hints::HintIndex>,
    hint_stats: HintStats,
    obs_conns: Mutex<HashSet<ConnId>>,
    obs_open: std::sync::atomic::AtomicUsize,
    /// compaction moves handed to the worker (class, ns, key, guard, new slots), and the bytes it may still move
    /// before the next sweep (set by the sweep: 0 unless quiet, without memory pressure, not idle)
    bg_compacts: Mutex<HashMap<ConnId, Vec<(usize, u32, kvproto::Key, (u64, Vec<kvproto::Range>, bool, bool), Vec<kvproto::Range>)>>>,
    compact_budget: AtomicU64,
    /// operation trace (None = off) and handler panics so far (an anomaly that dumps the trace)
    tracer: Option<trace::Tracer>,
    handler_panics: AtomicU64,
    /// a weak handle on itself (background jobs started from request handling), the SSD warm-up window
    self_arc: std::sync::OnceLock<std::sync::Weak<Shared>>,
    warm_within_ms: u64,
    /// n-gram table service (None without --ngram-table)
    ngram: Option<Arc<ngram::Ngram>>,
}

/// A slab class of the unified broker: its own pool file (slot size), index, on-demand backing and SSD tier.
struct Class {
    name: String,
    core: Mutex<Core>,
    hello: HelloReply,
    backing: Option<Mutex<lazy::Backing>>,
    disk: std::sync::OnceLock<Arc<disk::DiskTier>>,
}

struct ClassRef<'a> {
    pub name: &'a str,
    pub core: &'a Mutex<Core>,
    pub hello: &'a HelloReply,
    pub backing: &'a Option<Mutex<lazy::Backing>>,
    pub disk: Option<&'a Arc<disk::DiskTier>>,
}

impl Shared {
    fn n_classes(&self) -> usize {
        1 + self.extra.len()
    }
    fn cls(&self, i: usize) -> ClassRef<'_> {
        if i == 0 {
            ClassRef { name: &self.class0_name, core: &self.core, hello: &self.hello, backing: &self.backing, disk: disk::DISK.get() }
        } else {
            let c = &self.extra[i - 1];
            ClassRef { name: &c.name, core: &c.core, hello: &c.hello, backing: &c.backing, disk: c.disk.get() }
        }
    }
    /// Write the operation trace now (TRACE_DUMP or an anomaly); the file, or None (off / budget used up).
    fn trace_dump(&self, reason: &str, anomaly: bool) -> Option<PathBuf> {
        let t = self.tracer.as_ref()?;
        let h = kvtrace::Header {
            broker_version: VERSION.into(),
            epoch: self.cls(0).hello.epoch,
            unix_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0),
            reason: reason.into(),
            classes: (0..self.n_classes()).map(|i| (self.cls(i).name.to_string(), self.cls(i).hello.slot_bytes)).collect(),
        };
        t.dump(h, anomaly)
    }

    /// Whether every client connected to pool class `ci` reads format `name` version `v`: a format is only written
    /// for readers that declared it (FEATURES); a legacy client (no FEATURES) reads slot codec 1 only.
    fn class_reads(&self, ci: usize, name: &str, v: u16) -> bool {
        let cc = self.conn_class.lock().unwrap();
        let ff = self.conn_feats.lock().unwrap();
        cc.iter().filter(|(_, c)| **c == ci).all(|(conn, _)| match ff.get(conn) {
            Some((_, fmts)) => fmts.iter().any(|(n, x)| n == name && *x == v),
            None => name == kvproto::FMT_SLOT_CODEC && v == 1,
        })
    }

    fn class_of(&self, conn: ConnId) -> usize {
        self.conn_class.lock().unwrap().get(&conn).copied().unwrap_or(0)
    }
    fn class_by_name(&self, name: &str) -> usize {
        self.extra.iter().position(|c| c.name == name).map(|i| i + 1).unwrap_or(0)
    }
    /// (class, namespace) pairs with a connected client
    fn active_ns(&self) -> HashSet<(usize, u32)> {
        self.ns_conns.lock().unwrap().iter().filter(|(_, v)| !v.is_empty()).map(|(k, _)| *k).collect()
    }
}

// ---------------------------------------------------------------- dynamic pool

const HUGE: u64 = 2 << 20;
const HUGETLBFS_MAGIC: i64 = 0x958458f6;

struct PoolCtl {
    file: fs::File,
    len: u64,
    slot: u64,
    n_slots: u64,
    min_slots: u64,
    hugetlb: bool,
    hp_ctl: bool,
    /// bumped whenever the parked set changes; clients ack the generation they have unregistered. Only clients that
    /// pin the pool take part: they announce themselves with a PoolAck at connect (any generation) and the broker
    /// punches holes only once every announced client has acked the current generation (or after
    /// --pool-ack-timeout-s, 0 = never). Clients that never ack (the scheduler-side tier clients, userptr importers)
    /// do not hold up a punch.
    gen: u64,
    acks: HashMap<ConnId, u64>,
    /// parked, not yet punched (waiting for acks); the Instant is the oldest pending park
    pending: Vec<kvproto::Range>,
    pending_since: Option<Instant>,
    /// hole-punched runs: memory returned, re-allocated on regrow
    punched: Vec<kvproto::Range>,
    /// huge pages removed from the reservation (vm.nr_hugepages) and not yet restored
    freed_pages: u64,
    /// explicit target (bytes of usable pool), set over HTTP; pool size = no explicit limit
    target_bytes: u64,
    /// autonomous policy: keep MemAvailable above this (bytes; 0 = explicit only). Set at start by --mem-reserve-gib,
    /// changed at runtime with POST /pool?reserve_gib=N
    reserve: u64,
    /// slots the memory-pressure policy wants parked
    auto_parked: u64,
    calm_since: Option<Instant>,
    last: String,
    punches: u64,
    regrows: u64,
}

/// A client asked for the pool state: it may pin the pool (cuda_tier's SlotPinner and the Level Zero importers ask
/// before they register), so on-demand pools stop punching for this broker's lifetime (`--pool-pin-safe`).
fn note_pinner(sh: &Shared) {
    if sh.pin_safe && sh.backing.is_some() && !sh.pins.swap(true, Relaxed) {
        eprintln!("pool: a client asked for the pool state (it may pin the pool): on-demand pool hole punching off \
                   until the broker restarts; memory returns through the dynamic-pool controller only");
    }
}

fn mem_available() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()))
        .map(|kb| kb << 10)
        .unwrap_or(u64::MAX)
}

fn hugepages(field: &str) -> Option<u64> {
    let m = fs::read_to_string("/proc/meminfo").ok()?;
    m.lines().find(|l| l.starts_with(field)).and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
}

fn set_nr_hugepages(n: u64) -> std::io::Result<()> {
    fs::write("/proc/sys/vm/nr_hugepages", format!("{n}\n"))
}

impl PoolCtl {
    fn new(path: &Path, len: u64, slot: u64, n_slots: u64, o: &Opts) -> std::io::Result<PoolCtl> {
        use std::os::unix::ffi::OsStrExt;
        let file = fs::OpenOptions::new().read(true).write(true).open(path)?;
        let c = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap_or_default();
        let mut st: libc::statfs = unsafe { std::mem::zeroed() };
        let hugetlb = unsafe { libc::statfs(c.as_ptr(), &mut st) } == 0 && st.f_type as i64 == HUGETLBFS_MAGIC;
        let min_slots = ((o.pool_min_gib * (1u64 << 30) as f64) as u64 / slot).min(n_slots);
        Ok(PoolCtl {
            file, len, slot, n_slots, min_slots, hugetlb, hp_ctl: o.hugepages_ctl && hugetlb,
            gen: 0, acks: HashMap::new(), pending: Vec::new(), pending_since: None, punched: Vec::new(), freed_pages: 0,
            target_bytes: n_slots * slot, reserve: (o.mem_reserve_gib.max(0.0) * (1u64 << 30) as f64) as u64,
            auto_parked: 0, calm_since: None, last: "init".into(), punches: 0, regrows: 0,
        })
    }

    /// 2 MiB-aligned interior of a slot run (the edge pages are shared with the neighbouring slots and stay).
    fn interior(&self, r: &kvproto::Range) -> (u64, u64) {
        let a = (r.start as u64 * self.slot).div_ceil(HUGE) * HUGE;
        let b = ((r.start as u64 + r.len as u64) * self.slot / HUGE * HUGE).min(self.len);
        (a, b.max(a))
    }

    fn punch(&mut self, r: &kvproto::Range) -> std::io::Result<u64> {
        use std::os::fd::AsRawFd;
        let (a, b) = self.interior(r);
        if b <= a {
            return Ok(0);
        }
        let rc = unsafe {
            libc::fallocate(self.file.as_raw_fd(), libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, a as libc::off_t, (b - a) as libc::off_t)
        };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok((b - a) / HUGE)
    }

    fn refill(&mut self, r: &kvproto::Range) -> std::io::Result<u64> {
        use std::os::fd::AsRawFd;
        let (a, b) = self.interior(r);
        if b <= a {
            return Ok(0);
        }
        let rc = unsafe { libc::posix_fallocate(self.file.as_raw_fd(), a as libc::off_t, (b - a) as libc::off_t) };
        if rc != 0 {
            return Err(std::io::Error::from_raw_os_error(rc));
        }
        Ok((b - a) / HUGE)
    }

    fn status(&self, core_parked: u64, mem: u64) -> String {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        format!(
            "pool {:.1} GiB in {} slots of {:.1} MiB; usable {:.1} GiB ({} slots), parked {} (pending {} runs, punched {} runs), \
             target {:.1} GiB, mem reserve {:.1} GiB ({}), auto-parked {} slots, min {:.1} GiB; hugetlbfs {} ctl {} freed_pages {}; gen {} acks {:?}; \
             MemAvailable {:.1} GiB; punches {} regrows {}; last: {}\n",
            gib(self.n_slots * self.slot), self.n_slots, self.slot as f64 / (1u64 << 20) as f64,
            gib((self.n_slots - core_parked) * self.slot), self.n_slots - core_parked, core_parked, self.pending.len(), self.punched.len(),
            gib(self.target_bytes), gib(self.reserve), if self.reserve > 0 { "autonomous culling on" } else { "explicit only" },
            self.auto_parked, gib(self.min_slots * self.slot), self.hugetlb, self.hp_ctl, self.freed_pages,
            self.gen, self.acks.values().collect::<Vec<_>>(), gib(mem), self.punches, self.regrows, self.last
        )
    }
}

/// One control step (every second): memory-pressure policy, culling, hole punching after acks, regrowing.
fn pool_step(sh: &Shared, regrow: Duration, ack_timeout: Duration) {
    let mem = mem_available();
    let mut p = sh.pool.lock().unwrap();
    let max_parked = p.n_slots - p.min_slots;
    let o_reserve = p.reserve;
    if o_reserve == 0 {
        p.auto_parked = 0;          // policy off: whatever it parked grows back
    }
    // autonomous policy
    if o_reserve > 0 {
        if mem < o_reserve {
            let deficit = o_reserve - mem + (1u64 << 30);
            p.auto_parked = (p.auto_parked + deficit.div_ceil(p.slot)).min(max_parked);
            p.calm_since = None;
        } else if mem > o_reserve + (3u64 << 30) {
            match p.calm_since {
                None => p.calm_since = Some(Instant::now()),
                Some(t) if t.elapsed() >= regrow && p.auto_parked > 0 => {
                    let step = (2u64 << 30) / p.slot;
                    p.auto_parked -= step.min(p.auto_parked);
                    p.calm_since = Some(Instant::now());
                }
                _ => {}
            }
        } else {
            p.calm_since = None;
        }
    }
    let explicit = p.n_slots.saturating_sub(p.target_bytes / p.slot);
    let want = explicit.max(p.auto_parked).min(max_parked);
    let parked = sh.core.lock().unwrap().parked_slots();
    if want > parked {
        let new = sh.core.lock().unwrap().cull(want);
        if !new.is_empty() {
            let n: u64 = new.iter().map(|r| r.len as u64).sum();
            p.last = format!("culled {} slots in {} runs (want {} parked, had {})", n, new.len(), want, parked);
            eprintln!("pool: {}", p.last);
            p.pending.extend(new);
            p.gen += 1;
            p.pending_since.get_or_insert_with(Instant::now);
        }
    }
    if !p.pending.is_empty() {
        let gen = p.gen;
        let all_acked = p.acks.values().all(|&g| g >= gen);
        let timed_out = !ack_timeout.is_zero() && p.pending_since.is_some_and(|t| t.elapsed() >= ack_timeout);
        if all_acked || timed_out {
            if timed_out && !all_acked {
                eprintln!("pool: punching after the ack timeout (acks {:?}, gen {})", p.acks, gen);
            }
            let runs = std::mem::take(&mut p.pending);
            let mut pages = 0;
            for r in &runs {
                match p.punch(r) {
                    Ok(n) => pages += n,
                    Err(e) => eprintln!("pool: punch {}+{} failed: {e}", r.start, r.len),
                }
            }
            p.punched.extend(runs);
            p.punches += 1;
            p.pending_since = None;
            if p.hp_ctl && pages > 0 {
                let total = hugepages("HugePages_Total:").unwrap_or(0);
                match set_nr_hugepages(total.saturating_sub(pages)) {
                    Ok(()) => p.freed_pages += pages,
                    Err(e) => eprintln!("pool: nr_hugepages {} -> {} failed: {e}", total, total.saturating_sub(pages)),
                }
            }
            p.last = format!("punched {} huge pages ({:.2} GiB); MemAvailable {:.1} GiB", pages, pages as f64 * 2.0 / 1024.0, mem as f64 / (1u64 << 30) as f64);
            eprintln!("pool: {}", p.last);
        }
    } else if want < parked {
        // regrow: smallest punched runs first, one run per step
        p.punched.sort_by_key(|r| r.len);
        if let Some(r) = p.punched.first().copied() {
            let (a, b) = p.interior(&r);
            let pages = (b.saturating_sub(a)) / HUGE;
            let mut ok = true;
            if p.hp_ctl && pages > 0 {
                let total = hugepages("HugePages_Total:").unwrap_or(0);
                let free0 = hugepages("HugePages_Free:").unwrap_or(0);
                if let Err(e) = set_nr_hugepages(total + pages) {
                    eprintln!("pool: nr_hugepages {} -> {} failed: {e}", total, total + pages);
                    ok = false;
                } else if hugepages("HugePages_Free:").unwrap_or(0) < free0 + pages {
                    eprintln!("pool: only {} of {} huge pages came back; keeping the run parked", hugepages("HugePages_Free:").unwrap_or(0) - free0, pages);
                    let _ = set_nr_hugepages(total);
                    ok = false;
                }
            }
            if ok {
                match p.refill(&r) {
                    Ok(_) => {
                        p.punched.remove(0);
                        if p.hp_ctl {
                            p.freed_pages = p.freed_pages.saturating_sub(pages);
                        }
                        sh.core.lock().unwrap().unpark(&[r]);
                        p.gen += 1;
                        p.regrows += 1;
                        p.last = format!("regrew {} slots at {} ({} huge pages)", r.len, r.start, pages);
                        eprintln!("pool: {}", p.last);
                    }
                    Err(e) => {
                        eprintln!("pool: refill {}+{} failed: {e}", r.start, r.len);
                        if p.hp_ctl && pages > 0 {
                            let total = hugepages("HugePages_Total:").unwrap_or(0);
                            let _ = set_nr_hugepages(total.saturating_sub(pages));
                        }
                    }
                }
            }
        } else if p.pending.is_empty() && parked > 0 {
            // parked but never punched (no interior pages): hand straight back
            let runs = sh.core.lock().unwrap().parked_runs();
            sh.core.lock().unwrap().unpark(&runs);
            p.gen += 1;
        }
    }
}

/// Integrity scrub counters: entries checked / bad per kind, bad pages per (kind, page index) to validate the page
/// map on real data before eviction is turned on.
#[derive(Default, Clone)]
struct ScrubStats {
    /// (namespace, kind) -> entries checked / entries with a failing page
    checked: HashMap<(u32, u8), u64>,
    bad: HashMap<(u32, u8), u64>,
    /// (namespace, kind, check, page index) -> failures; check = "scale" (attention), "conv" / "ssm" (state)
    bad_pages: HashMap<(u32, u8, &'static str, usize), u64>,
    evicted: u64,
    passes: u64,
}

impl Shared {
    fn name_ns(&self, conn: ConnId, ns: u32) {
        let mut m = self.ns_names.lock().unwrap();
        if !m.contains_key(&ns) {
            let mut name = self.conn_names.lock().unwrap().get(&conn).cloned().unwrap_or_default();
            // per-domain metric labels: the domain's model and engine
            if let Some(d) = self.conn_domains.lock().unwrap().get(&conn).copied() {
                if let Some(c) = self.domains.lock().unwrap().get(&d) {
                    let v = |k: &str| c.lines().find_map(|l| l.strip_prefix(k)).unwrap_or("?").to_string();
                    name = format!("{}@{}#{:08x}", v("model="), v("engine="), d as u32);
                }
            }
            m.insert(ns, name);
        }
    }

    fn now(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// The index namespace a request of `conn` uses: its own namespace inside its cache domain (a hash with the top bit
    /// set), or the plain namespace for clients without a domain (engines that send no domain: their namespace + tier tag already
    /// scope their keys; they are the "legacy" domains `ns:<n>`).
    fn scoped_ns(&self, conn: ConnId, ns: u32) -> u32 {
        match self.conn_domains.lock().unwrap().get(&conn) {
            None => ns,
            Some(&d) => {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ d;
                for b in ns.to_le_bytes() {
                    h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
                }
                ((h ^ (h >> 32)) as u32) | 0x8000_0000
            }
        }
    }
}

/// Canonical descriptor: trimmed `key=value` lines, sorted, empty lines dropped; must name the model, its weights,
/// the engine and the cache layout. Domain id = FNV-1a 64 of the canonical text.
pub fn canonical_domain(desc: &str) -> Result<(String, u64), &'static str> {
    let mut lines: Vec<String> = desc.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
    if lines.iter().any(|l| !l.contains('=')) {
        return Err("descriptor lines must be key=value");
    }
    lines.sort();
    lines.dedup();
    for k in ["model=", "weights=", "engine=", "layout="] {
        if !lines.iter().any(|l| l.starts_with(k)) {
            return Err("descriptor needs model=, weights=, engine= and layout=");
        }
    }
    let c = lines.join("\n");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in c.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    Ok((c, h))
}

impl Shared {
    fn send(&self, conn: ConnId, frame: &[u8]) {
        let w = self.writers.lock().unwrap().get(&conn).cloned();
        if let Some(w) = w {
            let _ = w.lock().unwrap().write_all(frame);
        }
    }
    fn notify(&self, notes: Vec<Notify>) {
        for n in notes {
            self.send(n.conn, &reply_frame(Op::Wait as u16, n.req_id, n.status, &Reply::Status));
        }
    }
}

// ---------------------------------------------------------------- request handling

/// Run one request against the index. `None` = WAIT registered (answered later).
fn handle(sh: &Shared, conn: ConnId, req_id: u64, req: Request) -> (Option<(Status, Reply)>, Vec<Notify>) {
    let now = sh.now();
    if !sh.bg_conns.lock().unwrap().contains(&conn) && !sh.obs_conns.lock().unwrap().contains(&conn) {
        sh.last_req_ms.store(now, std::sync::atomic::Ordering::Relaxed);   // the worker's / a router's requests are not traffic
    }
    if let Request::Hello { name, .. } = &req {
        // unified broker: the HELLO name picks the slab class (pool) of this connection
        let ci = sh.class_by_name(name);
        sh.conn_class.lock().unwrap().insert(conn, ci);
    }
    let ci = sh.class_of(conn);
    let cl = sh.cls(ci);
    fp::hit("handle-unlocked");   // failpoints build: a handler panic outside the core lock (no poisoned state)
    let mut core = cl.core.lock().unwrap();
    fp::hit("handle-locked");   // failpoints build: a lock held long (sleep) or poisoned (panic)
    if !matches!(req, Request::Hello { .. }) && !core.has_hello(conn) {
        return (Some((Status::NoHello, Reply::Status)), vec![]);
    }
    let mut notes = vec![];
    // scope namespaces to the connection's cache domain
    let req = match req {
        Request::Lookup { namespace, acquire, keys } => Request::Lookup { namespace: sh.scoped_ns(conn, namespace), acquire, keys },
        Request::Reserve { namespace, items } => Request::Reserve { namespace: sh.scoped_ns(conn, namespace), items },
        Request::Commit { namespace, keys } => Request::Commit { namespace: sh.scoped_ns(conn, namespace), keys },
        Request::Abort { namespace, keys } => Request::Abort { namespace: sh.scoped_ns(conn, namespace), keys },
        Request::Touch { namespace, keys } => Request::Touch { namespace: sh.scoped_ns(conn, namespace), keys },
        Request::Heat { namespace, hot, keys } => Request::Heat { namespace: sh.scoped_ns(conn, namespace), hot, keys },
        Request::Invalidate { namespace, keys } => Request::Invalidate { namespace: sh.scoped_ns(conn, namespace), keys },
        Request::TouchRequest { namespace, groups } => Request::TouchRequest { namespace: sh.scoped_ns(conn, namespace), groups },
        Request::TouchTagged { namespace, session, parent, class, agent, flags, groups } =>
            Request::TouchTagged { namespace: sh.scoped_ns(conn, namespace), session, parent, class, agent, flags, groups },
        Request::Wait { namespace, key, timeout_ms } => Request::Wait { namespace: sh.scoped_ns(conn, namespace), key, timeout_ms },
        Request::Probe { namespace, keys } => Request::Probe { namespace: sh.scoped_ns(conn, namespace), keys },
        other => other,
    };
    let r = match req {
        Request::Domain { descriptor } => match canonical_domain(&descriptor) {
            Err(e) => {
                eprintln!("conn {conn}: domain refused: {e}");
                (Status::BadRequest, Reply::Status)
            }
            Ok((c, id)) => {
                let mut ds = sh.domains.lock().unwrap();
                let mut cd = sh.conn_domains.lock().unwrap();
                if ds.get(&id).is_some_and(|x| *x != c) {
                    eprintln!("conn {conn}: domain id collision {id:016x}: refused");
                    (Status::BadRequest, Reply::Status)
                } else if cd.get(&conn).is_some_and(|&x| x != id) {
                    eprintln!("conn {conn}: domain already set to another descriptor: refused");
                    (Status::BadRequest, Reply::Status)
                } else {
                    if !ds.contains_key(&id) {
                        let short: Vec<&str> = c.lines().filter(|l| l.starts_with("model=") || l.starts_with("engine=")).collect();
                        eprintln!("domain {id:016x}: {} (new)", short.join(" "));
                    }
                    ds.insert(id, c);
                    cd.insert(conn, id);
                    (Status::Ok, Reply::Domain { domain_id: id })
                }
            }
        },
        Request::Hello { engine_id, protocol_version, name } => {
            if protocol_version != PROTOCOL_VERSION {
                eprintln!("conn {conn} ({name}): protocol {protocol_version}, want {PROTOCOL_VERSION}");
                return (Some((Status::Version, Reply::Hello(cl.hello.clone()))), vec![]);
            }
            core.hello(conn, engine_id);
            eprintln!("conn {conn}: hello engine {engine_id} ({name}), pool class {}", cl.name);
            if engine_id == kvproto::OBSERVER_ENGINE_ID && sh.obs_conns.lock().unwrap().insert(conn) {
                sh.obs_open.fetch_add(1, std::sync::atomic::Ordering::SeqCst);   // a router: not a client
            }
            // on-demand pool: a model loading -> its recent SSD entries come back in the background (not for the
            // background worker: it connects with the broker, no model is loading)
            if let (Some(d), Some(_), true) = (cl.disk, cl.backing.as_ref(),
                                               engine_id != kvproto::BG_ENGINE_ID && engine_id != kvproto::OBSERVER_ENGINE_ID) {
                if let Some(arc) = sh.self_arc.get().and_then(|w| w.upgrade()) {
                    d.warm(arc, sh.warm_within_ms, cl.hello.pool_bytes / 2);
                }
            }
            sh.conn_names.lock().unwrap().insert(conn, name);
            (Status::Ok, Reply::Hello(cl.hello.clone()))
        }
        Request::Probe { namespace, keys } => {
            let mut v: Vec<u8> = keys.iter().map(|k| core.contains(namespace, k) as u8).collect();
            if let Some(d) = cl.disk {
                for (i, k) in keys.iter().enumerate() {
                    if v[i] == 0 && d.has(namespace, k) {
                        v[i] = 2;
                        d.request(namespace, &keys[i..i + 1]);
                    }
                }
            }
            (Status::Ok, Reply::Probe(v))
        }
        Request::Lookup { namespace, acquire, keys } => {
            sh.name_ns(conn, namespace);
            sh.ns_conns.lock().unwrap().entry((ci, namespace)).or_default().insert(conn);
            let (n_hit, lease_id, entries) = core.lookup(conn, namespace, &keys, acquire, now);
            if let Some(d) = cl.disk {
                d.note_hits(namespace, &keys[..n_hit as usize]);   // SSD admission: reuse evidence
                if (n_hit as usize) < keys.len() {
                    d.request(namespace, &keys[n_hit as usize..]);   // SSD tier: promote the rest of the run
                }
            }
            (Status::Ok, Reply::Lookup { n_hit, lease_id, entries })
        }
        Request::Reserve { items, .. } if now < sh.grace_until => {
            (Status::Ok, Reply::Reserve(items.iter().map(|_| (Status::Full, vec![])).collect()))
        }
        Request::Reserve { namespace, items } => {
            sh.name_ns(conn, namespace);
            sh.ns_conns.lock().unwrap().entry((ci, namespace)).or_default().insert(conn);
            let mut v = core.reserve(conn, namespace, &items, now);
            if let Some(bk) = cl.backing {
                bk.lock().unwrap().last_reserve_ms = now;
                // on-demand pool: the client's writes allocate the pages (an fallocate here took longer than the
                // a connector's 2 ms call timeout); refuse when the pool's file system could not take them
                // (a write into an unallocatable page would be SIGBUS in the engine)
                let mut b = bk.lock().unwrap();
                for (i, (st, ranges)) in v.iter_mut().enumerate() {
                    if *st == Status::Ok && !b.room_for(ranges) {
                        core.abort(conn, namespace, &[items[i].key]);
                        sh.lazy.backing_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        *st = Status::Full;
                        ranges.clear();
                    }
                }
            }
            (Status::Ok, Reply::Reserve(v))
        }
        Request::Commit { namespace, keys } => {
            let (st, n) = core.commit(conn, namespace, &keys, now);
            notes = n;
            (Status::Ok, Reply::Statuses(st))
        }
        Request::Abort { namespace, keys } => (Status::Ok, Reply::Statuses(core.abort(conn, namespace, &keys))),
        Request::Release { lease_ids } => (Status::Ok, Reply::Statuses(core.release(conn, &lease_ids))),
        Request::Touch { namespace, keys } => (Status::Ok, Reply::Statuses(core.touch_conn(conn, namespace, &keys, now))),
        Request::Heat { namespace, hot, keys } => (Status::Ok, Reply::Statuses(core.heat_keys(namespace, &keys, hot, now))),
        Request::Invalidate { namespace, keys } => {
            // known-bad content (an engine got NaN from a state it restored): RAM entries, the chain after them and
            // their SSD copies; the SSD files are removed outside the core lock
            // chat beginnings on the SSD's core stay: every session's prefix, computed from token 0
            let keep: Vec<bool> = keys.iter().map(|k| cl.disk.is_some_and(|d| d.is_core(namespace, k))).collect();
            let ask: Vec<kvproto::Key> = keys.iter().zip(&keep).filter(|(_, c)| !**c).map(|(k, _)| *k).collect();
            let (st_ask, gone) = core.invalidate(namespace, &ask);
            drop(core);
            let mut it = st_ask.into_iter();
            let st: Vec<Status> = keep.iter().map(|c| if *c { Status::Exists } else { it.next().unwrap_or(Status::NotFound) }).collect();
            if let Some(d) = cl.disk {
                for (ns, k) in &gone {
                    d.forget(*ns, k);
                }
            }
            eprintln!("conn {conn}: INVALIDATE {} key(s): {} entries removed (with the chain after them)", keys.len(), gone.len());
            (Status::Ok, Reply::Statuses(st))
        }
        Request::TouchRequest { namespace, groups } => {
            core.touch_request(conn, namespace, &groups, now);
            (Status::Ok, Reply::Status)
        }
        Request::TouchTagged { namespace, session, parent, class, agent: _, flags, groups } => {
            core.touch_tagged(conn, namespace, session, parent, class, flags, &groups, now);
            (Status::Ok, Reply::Status)
        }
        Request::TraceDump { .. } => (Status::BadRequest, Reply::Status),   // answered in serve_conn
        Request::Hint { .. } | Request::Where { .. } => (Status::BadRequest, Reply::Status),   // serve_conn
        Request::BgJobs { .. } | Request::BgChecksum { .. } | Request::BgRewrite { .. }
            if !sh.bg_conns.lock().unwrap().contains(&conn) => (Status::BadRequest, Reply::Status),
        Request::BgSsdJobs { .. } | Request::BgSsdDone { .. } | Request::BgCompactJobs { .. }
            | Request::BgCompactDone { .. } => (Status::BadRequest, Reply::Status),   // serve_conn
        Request::BgJobs { kind, max } => {
            let quiet_ms = if kind == kvproto::BG_CHECKSUM { sh.rcompress_quiet_ms.max(1000) } else { sh.rcompress_quiet_ms };
            let quiet = now.saturating_sub(sh.last_req_ms.load(Relaxed)) >= quiet_ms;
            let slot = cl.hello.slot_bytes;
            let chunk = sh.rchunk.get(ci).copied().unwrap_or(0);
            let guard = |core: &Core, ns: u32, key: &kvproto::Key| core.copy_guard(ns, key).map(|g| kvproto::Guard {
                entry_id: g.0, ranges: g.1, compressed: g.2, rewriting: g.3 });
            let max = (max as usize).min(256);
            let jobs: Vec<kvproto::BgJob> = if !quiet {
                Vec::new()
            } else if kind == kvproto::BG_CHECKSUM && sh.cksum_bytes_s > 0 {
                core.checksum_candidates(now.saturating_sub(sh.cksum_reverify_ms), max).into_iter()
                    .filter_map(|(ns, key, _, bytes, _, _)| guard(&core, ns, &key).map(|g| kvproto::BgJob { namespace: ns, key, bytes, guard: g }))
                    .collect()
            } else if kind == kvproto::BG_COMPRESS && chunk > 0 && now >= sh.rcompress_after_ms
                && sh.class_reads(ci, kvproto::FMT_SLOT_CODEC, 1) {
                core.compress_candidates(now - sh.rcompress_after_ms, max).into_iter()
                    .filter(|(_, _, r, b)| r.len() == 1 && r[0].len == 1 && *b <= slot)   // one-slot entries
                    .filter_map(|(ns, key, _, bytes)| guard(&core, ns, &key).map(|g| kvproto::BgJob { namespace: ns, key, bytes, guard: g }))
                    .collect()
            } else {
                Vec::new()
            };
            let classes = (0..sh.n_classes()).map(|i| sh.cls(i).name.to_string()).collect();
            (Status::Ok, Reply::BgJobs { cksum_bytes_s: sh.cksum_bytes_s, chunk, classes, jobs })
        }
        Request::BgChecksum { namespace, key, guard, sum } => {
            let g = (guard.entry_id, guard.ranges, guard.compressed, guard.rewriting);
            let r = match core.checksum_result(namespace, &key, &g, sum, now) {
                kvcore::ChecksumResult::Stored => 0,
                kvcore::ChecksumResult::Verified => 1,
                kvcore::ChecksumResult::Corrupt => {
                    eprintln!("integrity: an entry failed its checksum (class {ci}, ns {namespace}): evicted with the rest \
                               of its chain");
                    2
                }
                kvcore::ChecksumResult::Raced => 3,
                kvcore::ChecksumResult::Busy => 4,
            };
            if sum.is_some() {
                sh.lazy.cksum_bytes.fetch_add(core.entry_info(namespace, &key).map_or(0, |e| e.bytes), Relaxed);
            }
            (Status::Ok, Reply::BgResult(r))
        }
        Request::BgRewrite { phase, namespace, key, guard, .. } => {
            let same = core.copy_guard(namespace, &key).is_some_and(|g| g.0 == guard.entry_id && g.1 == guard.ranges
                                                                        && g.2 == guard.compressed && g.3 == guard.rewriting);
            let quiet = now.saturating_sub(sh.last_req_ms.load(Relaxed)) >= sh.rcompress_quiet_ms;
            let r = match phase {
                // only while quiet: a request that arrived meanwhile finds the entry readable
                kvproto::BG_BEGIN if same && quiet && core.begin_rewrite(namespace, &key, &guard.ranges) => {
                    sh.bg_rewrites.lock().unwrap().entry(conn).or_default().push((ci, namespace, key));
                    0
                }
                kvproto::BG_SKIP if same && core.begin_rewrite(namespace, &key, &guard.ranges) => {
                    core.end_rewrite(namespace, &key, false);   // not worth it: never tried again
                    sh.lazy.rcompress_skipped.fetch_add(1, Relaxed);
                    0
                }
                kvproto::BG_BEGIN | kvproto::BG_SKIP => {
                    sh.lazy.rcompress_raced.fetch_add(1, Relaxed);
                    3
                }
                _ => return (Some((Status::BadRequest, Reply::Status)), vec![]),   // BG_END: answered in serve_conn
            };
            (Status::Ok, Reply::BgResult(r))
        }
        Request::Features { features, formats } => {
            eprintln!("conn {conn}: features {features:#x}, reads {formats:?}");
            if features & kvproto::FEAT_BG_WORKER != 0 && sh.bg_worker {
                if sh.bg_conns.lock().unwrap().insert(conn) {
                    sh.bg_open.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
                eprintln!("conn {conn}: background worker (checksums, in-place compression)");
            }
            sh.conn_feats.lock().unwrap().insert(conn, (features, formats));
            (Status::Ok, Reply::Features { features: kvproto::FEAT_TOUCH_REQUEST | kvproto::FEAT_INVALIDATE | kvproto::FEAT_REQUEST_TAGS
                                                     | if sh.hint_mib_on() { kvproto::FEAT_PREFIX_HINTS } else { 0 }
                                                     | if sh.tracer.is_some() { kvproto::FEAT_TRACE_DUMP } else { 0 }
                                                     | if sh.bg_worker { kvproto::FEAT_BG_WORKER } else { 0 },
                                           formats: vec![(kvproto::FMT_SLOT_CODEC.into(), 1)] })
        }
        Request::Stats => {
            let mut v = core.gauges();
            v.extend(core.counters());
            {
                let h = sh.hints.lock().unwrap();
                v.push(("hint_entries".into(), h.len() as u64));
                v.push(("hint_bytes".into(), h.bytes() as u64));
                v.push(("hints_dropped".into(), h.dropped));
            }
            let hs = &sh.hint_stats;
            for (m, x) in [("hints", &hs.hints), ("hints_refused", &hs.hints_refused), ("where_requests", &hs.wheres),
                           ("where_places_resident", &hs.where_resident)] {
                v.push((m.into(), x.load(Relaxed)));
            }
            (Status::Ok, Reply::Stats(v))
        }
        Request::Wait { namespace, key, timeout_ms } => match core.wait(conn, req_id, namespace, key, timeout_ms, now) {
            Some(st) => (st, Reply::Status),
            None => return (None, vec![]),
        },
        Request::PoolState if cl.backing.is_some() => {
            note_pinner(sh);
            // on-demand pool: every free slot run is "parked" for the client (the connector fallocates the pool
            // around parked runs at connect; it would otherwise allocate the whole ceiling). The broker allocates a
            // run's memory itself before handing it out in a RESERVE.
            (Status::Ok, Reply::PoolState { gen: 0, parked: core.free_runs() })
        }
        Request::PoolState => {
            // lock order is pool -> core (pool_step, /pool): release core first, or a pinning client's poll
            // deadlocks against the controller (slot-pinning test: every broker thread stuck on a futex)
            drop(core);
            let p = sh.pool.lock().unwrap();
            let mut parked: Vec<kvproto::Range> = p.pending.iter().chain(p.punched.iter()).copied().collect();
            parked.sort_by_key(|r| r.start);
            (Status::Ok, Reply::PoolState { gen: p.gen, parked })
        }
        Request::PoolAck { gen } => {
            drop(core);
            note_pinner(sh);
            sh.pool.lock().unwrap().acks.insert(conn, gen);
            (Status::Ok, Reply::Status)
        }
        // n-gram service: its own locks, never the index lock
        Request::NgramOpen { table, rows, row_bytes } => {
            drop(core);
            match sh.ngram.as_ref().map(|n| n.open(conn, &table, rows, row_bytes)) {
                None => (Status::NotFound, Reply::Status),
                Some(Err(st)) => (st, Reply::Status),
                Some(Ok((staging_offset, staging_bytes, row_bytes, heads))) => (
                    Status::Ok,
                    Reply::NgramOpen {
                        path: sh.ngram.as_ref().unwrap().staging_path.to_string_lossy().into_owned(),
                        staging_offset,
                        staging_bytes,
                        row_bytes,
                        heads,
                    },
                ),
            }
        }
        Request::NgramGather { seqs } => {
            drop(core);
            match sh.ngram.as_ref().map(|n| n.gather(conn, &seqs)) {
                None => (Status::NotFound, Reply::Status),
                Some(Err(st)) => (st, Reply::Status),
                Some(Ok((rows, disk_rows))) => (Status::Ok, Reply::NgramGather { rows, disk_rows }),
            }
        }
        Request::NgramPrefetch { seqs } => {
            drop(core);
            (sh.ngram.as_ref().map(|n| n.prefetch(conn, seqs)).unwrap_or(Status::NotFound), Reply::Status)
        }
    };
    (Some(r), notes)
}

/// A connection's cleanup, however its thread ends (failpoint review): a panic in a handler used to
/// unwind past the cleanup at the end of serve_conn, so the socket stayed open (the registered writer half: the
/// client waited for its own timeout) and the connection's leases and reservations stayed held until they timed out
/// (leases: 30 min). The socket is shut down first (no lock needed: the client sees the close at once), then the
/// connection is dropped everywhere as on a normal close. Poisoned locks are skipped, never unwrapped: a panic in
/// drop while unwinding would abort the broker; a poisoned core lock is the watchdog's (exit 70).
struct ConnGuard<'a> {
    sh: &'a Shared,
    conn: ConnId,
    sock: Option<UnixStream>,
}

impl Drop for ConnGuard<'_> {
    fn drop(&mut self) {
        let (sh, conn) = (self.sh, self.conn);
        if let Some(s) = &self.sock {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
        if std::thread::panicking() {
            eprintln!("conn {conn}: handler panicked: connection closed, its leases and reservations released");
            sh.handler_panics.fetch_add(1, Relaxed);
        }
        if let Some(tr) = &sh.tracer {
            tr.record(&kvtrace::Record { t_us: tr.now_us(), conn: conn as u32, op: kvtrace::OP_CLOSE, status: 0, lat_us: 0,
                                         body: kvtrace::Body::None });
        }
        if let Ok(mut w) = sh.writers.lock() {
            w.remove(&conn);
        }
        // a background worker that leaves mid-compaction: its target slots are quarantined (it may still write)
        let moves = sh.bg_compacts.lock().ok().and_then(|mut m| m.remove(&conn)).unwrap_or_default();
        for (ci, _, _, _, new) in moves {
            if let Ok(mut core) = sh.cls(ci).core.lock() {
                core.abandon_relocation(new);
            }
        }
        // a background worker that leaves mid-rewrite: its slots may hold part of an image
        let torn = sh.bg_rewrites.lock().ok().and_then(|mut m| m.remove(&conn)).unwrap_or_default();
        for (ci, ns, key) in torn {
            if let Ok(mut core) = sh.cls(ci).core.lock() {
                if core.abort_torn_rewrite(ns, &key) {
                    eprintln!("conn {conn}: background worker left mid-rewrite: entry evicted (class {ci}, ns {ns})");
                }
            }
        }
        if let Ok(mut b) = sh.obs_conns.lock() {
            if b.remove(&conn) {
                sh.obs_open.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
        if let Ok(mut b) = sh.bg_conns.lock() {
            if b.remove(&conn) {
                sh.bg_open.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                for i in 0..sh.n_classes() {
                    if let Some(d) = sh.cls(i).disk {
                        d.worker_gone(conn);    // its SSD writes in flight: dropped (planned again)
                    }
                }
            }
        }
        for i in 0..sh.n_classes() {
            match sh.cls(i).core.lock() {
                Ok(mut core) => core.disconnect(conn),
                Err(_) => eprintln!("conn {conn}: core lock of class {i} poisoned: not cleaned up (watchdog)"),
            }
        }
        if let Ok(mut m) = sh.conn_class.lock() {
            m.remove(&conn);
        }
        if let Ok(mut m) = sh.ns_conns.lock() {
            for v in m.values_mut() {
                v.remove(&conn);
            }
        }
        if let Ok(mut p) = sh.pool.lock() {
            p.acks.remove(&conn);
        }
        if let Some(n) = &sh.ngram {
            n.disconnect(conn);
        }
        if let Ok(mut m) = sh.conn_names.lock() {
            m.remove(&conn);
        }
        if let Ok(mut m) = sh.conn_feats.lock() {
            m.remove(&conn);
        }
        if let Ok(mut m) = sh.conn_domains.lock() {
            m.remove(&conn);
        }
        eprintln!("conn {conn}: closed");
    }
}

/// Open client connections: all open connections minus the background workers' (idle exit, idle shrink and the trace
/// ring's release go by clients).
fn sh_clients(oc: &std::sync::atomic::AtomicUsize, sh: &Shared) -> usize {
    oc.load(std::sync::atomic::Ordering::SeqCst)
        .saturating_sub(sh.bg_open.load(std::sync::atomic::Ordering::SeqCst) + sh.obs_open.load(std::sync::atomic::Ordering::SeqCst))
}

/// Ask the scheduler for a short slice for the calling thread (EEVDF: an earlier deadline on every wakeup, so it
/// preempts busy threads sooner; unprivileged; the shortest slice also propagates up the cgroup tree). Measured:
/// with CPU load in a sibling cgroup (nice inside another cgroup does not count against this one), the engines'
/// 2 ms broker timeouts expired while the request threads waited to run.
fn latency_slice(us: u64) {
    if us == 0 {
        return;
    }
    #[repr(C)]
    #[derive(Default)]
    struct SchedAttr { size: u32, policy: u32, flags: u64, nice: i32, prio: u32, runtime: u64, deadline: u64, period: u64,
                       util_min: u32, util_max: u32 }
    let nice = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
    let mut a = SchedAttr { size: std::mem::size_of::<SchedAttr>() as u32, nice, runtime: us * 1000, ..Default::default() };
    let r = unsafe { libc::syscall(libc::SYS_sched_setattr, 0, &mut a as *mut SchedAttr, 0u32) };
    if r != 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| eprintln!("latency slice {us} us not set: {}", std::io::Error::last_os_error()));
    }
}

/// BG_REWRITE end: the worker wrote a compressed image of `len` bytes into the slot of a rewrite it began. The rest of
/// the slot is punched (outside the core lock: the entry is being rewritten, nobody else touches its slot), then the
/// entry is readable again, as compressed. Only the worker that began the rewrite can end it.
fn bg_end(sh: &Shared, conn: ConnId, ns: u32, key: kvproto::Key, len: u64) -> (Status, Reply) {
    let ci = {
        let mut m = sh.bg_rewrites.lock().unwrap();
        let Some(v) = m.get_mut(&conn) else { return (Status::BadRequest, Reply::Status) };
        let Some(i) = v.iter().position(|x| x.1 == ns && x.2 == key) else { return (Status::BadRequest, Reply::Status) };
        v.swap_remove(i).0
    };
    let c = sh.cls(ci);
    let slot = c.hello.slot_bytes;
    let (start, bytes) = {
        let core = c.core.lock().unwrap();
        match (core.copy_guard(ns, &key), core.entry_info(ns, &key)) {
            (Some(g), Some(e)) if g.3 => (g.1[0].start, e.bytes),
            _ => return (Status::BadRequest, Reply::Status),
        }
    };
    let freed = match c.backing {
        Some(bk) if len > 0 && len <= slot => lazy::punch_tail(bk, start, len),
        _ => 0,
    };
    c.core.lock().unwrap().end_rewrite(ns, &key, true);
    sh.lazy.rcompress_entries.fetch_add(1, Relaxed);
    sh.lazy.rcompress_raw.fetch_add(bytes, Relaxed);
    sh.lazy.rcompress_stored.fetch_add(len, Relaxed);
    let _ = freed;
    (Status::Ok, Reply::BgResult(0))
}

/// BG_COMPACT_JOBS: one compaction move for the worker (one at a time: an entry is never handed out twice), only
/// while the broker is quiet and within the sweep's budget. The candidate is chosen and its new slots allocated under
/// the core lock; the new slots are backed (fallocate) outside it, and copied by the worker outside it too: the entry
/// stays readable at its old slots meanwhile (the copy used to run under the core lock, several ms per
/// slot, past the engines' 2 ms).
fn bg_compact_job(sh: &Shared, conn: ConnId) -> (Status, Reply) {
    if !sh.bg_conns.lock().unwrap().contains(&conn) {
        return (Status::BadRequest, Reply::Status);
    }
    let ci = sh.class_of(conn);
    let c = sh.cls(ci);
    let none = (Status::Ok, Reply::CompactJobs(Vec::new()));
    let Some(bk) = c.backing else { return none };
    let slot = c.hello.slot_bytes;
    if sh.now().saturating_sub(sh.last_req_ms.load(Relaxed)) < 3000 || sh.compact_budget.load(Relaxed) < slot {
        return none;
    }
    let (ns, key, new, g) = {
        let mut core = c.core.lock().unwrap();
        let (live, hw) = core.live_and_high_water();
        if hw as u64 <= live {
            return none; // dense
        }
        let Some((ns, key, _old, new)) = core.relocation_candidate() else { return none };
        let g = core.copy_guard(ns, &key).unwrap();
        (ns, key, new, g)
    };
    if bk.lock().unwrap().ensure(&new).is_err() {
        c.core.lock().unwrap().finish_relocation(ns, &key, &[], new); // frees the new slots
        return none;
    }
    let bytes: u64 = g.1.iter().map(|r| r.len as u64).sum::<u64>() * slot;
    let _ = sh.compact_budget.fetch_update(Relaxed, Relaxed, |b| Some(b.saturating_sub(bytes)));
    sh.bg_compacts.lock().unwrap().entry(conn).or_default().push((ci, ns, key, g.clone(), new.clone()));
    let job = kvproto::CompactJob { namespace: ns, key, guard: kvproto::Guard { entry_id: g.0, ranges: g.1, compressed: g.2,
                                                                                rewriting: g.3 }, new };
    (Status::Ok, Reply::CompactJobs(vec![job]))
}

/// BG_COMPACT_DONE: the worker copied (or did not copy) a move it took. Copied and the entry unchanged and
/// unleased: the entry points at its new slots, the old ones are free (the next sweep punches them); otherwise the new
/// slots are freed.
fn bg_compact_done(sh: &Shared, conn: ConnId, ns: u32, key: kvproto::Key, copied: bool) -> (Status, Reply) {
    let job = {
        let mut m = sh.bg_compacts.lock().unwrap();
        let Some(v) = m.get_mut(&conn) else { return (Status::BadRequest, Reply::Status) };
        let Some(i) = v.iter().position(|x| x.1 == ns && x.2 == key) else { return (Status::BadRequest, Reply::Status) };
        v.swap_remove(i)
    };
    let (ci, _, _, g, new) = job;
    let c = sh.cls(ci);
    let slot = c.hello.slot_bytes;
    let mut core = c.core.lock().unwrap();
    let moved = if copied {
        core.finish_relocation_guarded(ns, &key, &g, new)
    } else {
        core.finish_relocation(ns, &key, &[], new);
        false
    };
    if moved {
        sh.lazy.compact_moves.fetch_add(1, Relaxed);
        sh.lazy.compact_bytes.fetch_add(g.1.iter().map(|r| r.len as u64).sum::<u64>() * slot, Relaxed);
    }
    // no trim here (punching every free run under the core lock made 39 of 53 finishes take > 2.5 ms, 9 >
    // 10 ms, in front of engine requests): the old slots are free now, and the sweep punches free runs on every pass
    (Status::Ok, Reply::BgResult(if moved { 0 } else { 3 }))
}

/// BG_SSD_JOBS: planned SSD writes of the worker connection's pool class (outside the core lock).
fn bg_ssd_jobs(sh: &Shared, conn: ConnId, max: u32) -> (Status, Reply) {
    if !sh.bg_conns.lock().unwrap().contains(&conn) {
        return (Status::BadRequest, Reply::Status);
    }
    let Some(d) = sh.cls(sh.class_of(conn)).disk else { return (Status::Ok, Reply::SsdJobs(Vec::new())) };
    let jobs = d.take_jobs(conn, (max as usize).min(64)).into_iter().map(|j| kvproto::SsdJob {
        namespace: j.ns, key: j.key, guard: j.guard, kind: j.kind, format: j.format, producer: j.producer, hits: j.hits,
        bytes: j.bytes, last_used_ms: j.last_used_ms, meta: j.meta, path: j.path.display().to_string(), codec: j.codec,
    }).collect();
    (Status::Ok, Reply::SsdJobs(jobs))
}

/// BG_SSD_DONE: the worker's outcome of an SSD write it took; the index learns of the file.
fn bg_ssd_done(sh: &Shared, conn: ConnId, ns: u32, key: kvproto::Key, outcome: u8, file_bytes: u64) -> (Status, Reply) {
    if !sh.bg_conns.lock().unwrap().contains(&conn) {
        return (Status::BadRequest, Reply::Status);
    }
    let Some(d) = sh.cls(sh.class_of(conn)).disk else { return (Status::BadRequest, Reply::Status) };
    let ok = d.job_done(conn, ns, &key, disk::WriteOutcome::from_code(outcome, file_bytes));
    (Status::Ok, Reply::BgResult(if ok { 0 } else { 3 }))
}

/// Prefix-hint counters (metrics, STATS).
#[derive(Default)]
struct HintStats {
    hints: AtomicU64,
    hints_refused: AtomicU64,
    wheres: AtomicU64,
    where_anchors: AtomicU64,
    where_places: AtomicU64,
    where_resident: AtomicU64,
    where_us: AtomicU64,
}

/// WHERE: anchors per request it looks at, hints per engine it checks residency for.
const WHERE_MAX_ANCHORS: usize = 256;
const WHERE_PER_ENGINE: usize = 4;

impl Shared {
    fn hint_mib_on(&self) -> bool {
        self.hints.lock().unwrap().max_bytes > 0
    }
}

/// HINT: store an engine's anchor -> keys hint in the side table (outside every core lock but the HELLO check).
fn hint(sh: &Shared, conn: ConnId, namespace: u32, anchor: kvproto::Anchor, chunk_tokens: u32, flags: u8,
        groups: Vec<(u8, Vec<kvproto::Key>)>) -> (Status, Reply) {
    let ci = sh.class_of(conn);
    let Some(engine_id) = sh.cls(ci).core.lock().unwrap().engine_id(conn) else { return (Status::NoHello, Reply::Status) };
    sh.last_req_ms.store(sh.now(), Relaxed);
    let ns = sh.scoped_ns(conn, namespace);
    let name = sh.conn_names.lock().unwrap().get(&conn).cloned().unwrap_or_default();
    let rec = kvcore::hints::HintRec { engine_id, name, class: ci, ns, chunk_tokens, flags, groups, t_ms: sh.now() };
    if sh.hints.lock().unwrap().insert(anchor, rec) {
        sh.hint_stats.hints.fetch_add(1, Relaxed);
        (Status::Ok, Reply::Status)
    } else {
        sh.hint_stats.hints_refused.fetch_add(1, Relaxed);
        (Status::BadRequest, Reply::Status)
    }
}

/// WHERE: per (engine, class, namespace) the hint with the most leading chunks in RAM now (ties: the longer anchor,
/// i.e. the earlier in the request). Read-only: candidates are copied out of the side table, then each class's index
/// is read under its core lock once (no lease, no hit, no heat, no SSD promotion).
fn where_(sh: &Shared, conn: ConnId, anchors: &[kvproto::Anchor]) -> (Status, Reply) {
    let t = Instant::now();
    if !sh.cls(sh.class_of(conn)).core.lock().unwrap().has_hello(conn) {
        return (Status::NoHello, Reply::Status);
    }
    let anchors = &anchors[..anchors.len().min(WHERE_MAX_ANCHORS)];
    let cands = sh.hints.lock().unwrap().candidates(anchors, WHERE_PER_ENGINE);
    let now = sh.now();
    let mut best: HashMap<(u16, usize, u32), kvproto::Place> = HashMap::new();
    let mut classes: Vec<usize> = cands.iter().map(|(_, r)| r.class).collect();
    classes.sort_unstable();
    classes.dedup();
    for ci in classes {
        let core = sh.cls(ci).core.lock().unwrap();
        for (idx, r) in cands.iter().filter(|(_, r)| r.class == ci) {
            let n_resident = core.resident_prefix(r.ns, &r.groups);
            let p = kvproto::Place { anchor_idx: *idx, engine_id: r.engine_id, name: r.name.clone(), namespace: r.ns,
                                     age_ms: now.saturating_sub(r.t_ms), chunk_tokens: r.chunk_tokens, n_chunks: r.n_chunks(),
                                     n_resident, flags: r.flags };
            let e = best.entry((r.engine_id, ci, r.ns)).or_insert_with(|| p.clone());
            if p.n_resident > e.n_resident || (p.n_resident == e.n_resident && p.anchor_idx < e.anchor_idx) {
                *e = p;
            }
        }
    }
    let mut v: Vec<kvproto::Place> = best.into_values().collect();
    v.sort_by_key(|p| (p.anchor_idx, std::cmp::Reverse(p.n_resident), p.engine_id));
    let hs = &sh.hint_stats;
    hs.wheres.fetch_add(1, Relaxed);
    hs.where_anchors.fetch_add(anchors.len() as u64, Relaxed);
    hs.where_places.fetch_add(v.len() as u64, Relaxed);
    hs.where_resident.fetch_add(v.iter().filter(|p| p.n_resident > 0).count() as u64, Relaxed);
    hs.where_us.fetch_add(t.elapsed().as_micros() as u64, Relaxed);
    (Status::Ok, Reply::Where(v))
}

fn serve_conn(sh: Arc<Shared>, conn: ConnId, stream: UnixStream) {
    latency_slice(sh.latency_slice_us);
    let writer = match stream.try_clone() {
        Ok(w) => Arc::new(Mutex::new(w)),
        Err(e) => {
            eprintln!("conn {conn}: {e}");
            return;
        }
    };
    let _guard = ConnGuard { sh: &sh, conn, sock: stream.try_clone().ok() };
    sh.writers.lock().unwrap().insert(conn, writer.clone());
    for i in 0..sh.n_classes() {
        sh.cls(i).core.lock().unwrap().connect(conn);
    }
    let mut rd = BufReader::with_capacity(64 << 10, stream);
    let mut buf = Vec::new();
    loop {
        let h = match read_frame(&mut rd, &mut buf) {
            Ok(Some(h)) => h,
            Ok(None) => break,
            Err(e) => {
                sh.protocol_errors.fetch_add(1, Relaxed);
                eprintln!("conn {conn}: protocol error: {e}");
                break;
            }
        };
        let t = Instant::now();
        let Some(op) = Op::from_u16(h.op) else {
            sh.protocol_errors.fetch_add(1, Relaxed);
            let _ = writer.lock().unwrap().write_all(&reply_frame(h.op, h.req_id, Status::BadRequest, &Reply::Status));
            continue;
        };
        let mut traced = None;
        let (reply, notes) = match Request::decode(op, &buf) {
            Ok(Request::BgRewrite { phase: kvproto::BG_END, namespace, key, len, .. }) => (Some(bg_end(&sh, conn, namespace, key, len)), vec![]),
            Ok(Request::BgSsdJobs { max }) => (Some(bg_ssd_jobs(&sh, conn, max)), vec![]),
            Ok(Request::BgCompactJobs { .. }) => (Some(bg_compact_job(&sh, conn)), vec![]),
            Ok(Request::BgCompactDone { namespace, key, copied }) => (Some(bg_compact_done(&sh, conn, namespace, key, copied)), vec![]),
            Ok(Request::BgSsdDone { namespace, key, outcome, file_bytes }) => {
                (Some(bg_ssd_done(&sh, conn, namespace, key, outcome, file_bytes)), vec![])
            }
            Ok(Request::Hint { namespace, anchor, chunk_tokens, flags, groups }) => {
                (Some(hint(&sh, conn, namespace, anchor, chunk_tokens, flags, groups)), vec![])
            }
            Ok(Request::Where { anchors }) => (Some(where_(&sh, conn, &anchors)), vec![]),
            Ok(Request::TraceDump { reason }) => {
                // outside the core lock: the file write takes a while
                let r = match sh.trace_dump(&format!("request-{reason}"), false) {
                    Some(p) => (Status::Ok, Reply::TraceDump { path: p.display().to_string() }),
                    None => (Status::Busy, Reply::Status),
                };
                (Some(r), vec![])
            }
            Ok(req) => {
                if let Some(tr) = &sh.tracer {
                    traced = Some((tr.now_us(), trace::body_of(&req)));
                }
                handle(&sh, conn, h.req_id, req)
            }
            Err(_) => {
                sh.protocol_errors.fetch_add(1, Relaxed);
                (Some((Status::BadRequest, Reply::Status)), vec![])
            }
        };
        if let (Some(tr), Some((t_us, mut body))) = (&sh.tracer, traced) {
            let status = match &reply {
                Some((st, r)) => {
                    trace::with_reply(&mut body, r);
                    *st as u16
                }
                None => 0xfffe,     // answered later (WAIT)
            };
            tr.record(&kvtrace::Record { t_us, conn: conn as u32, op: h.op, status,
                                         lat_us: t.elapsed().as_micros().min(u32::MAX as u128) as u32, body });
        }
        if let Some((st, r)) = reply {
            fp::hit("reply-delay");   // failpoints build: a slow reply (outside the core lock)
            let f = reply_frame(h.op, h.req_id, st, &r);
            if writer.lock().unwrap().write_all(&f).is_err() {
                break;
            }
        }
        sh.notify(notes);
        sh.ops[op as usize - 1].record(t.elapsed().as_micros() as u64);
    }
}   // _guard: socket shut down, the connection dropped everywhere

// ---------------------------------------------------------------- metrics

// ---------------------------------------------------------------- persistent hot set

/// Persistent hot set (what survives a shutdown, reboot and start is the really high-use part of the
/// cache): the most-used entries (--hotset-rank use, the default: distinct request chains that touched them, decayed,
/// >= --hotset-min-uses, as loadable units: a GDN state boundary with its attention chain; kvcore hot.rs) or the
/// most-hit ones (--hotset-rank hits: >= min_hits), hottest first, up to
/// max_bytes, are written to <dir>/<ns>-<key>.kv every --hotset-save-s and on SIGTERM (systemd stop at shutdown),
/// incrementally (entries are immutable: existing files are kept, dropped ones deleted), and restored at start
/// before any engine connects. Keys carry the engines' compat tag (model id, KV format, layout), so a restored entry
/// from another model or format can never match. File: magic "LKVH3\0", ns u32, key 32, kind u8, format u16,
/// producer u16, hits u64, bytes u64, meta_len u16, meta, use f64, relation u8 (0 none, 1 attention parent, 2 GDN
/// state anchor), relation key 16 B, GDN groups u8, data checksum u64, data (LKVH2/LKVH1: no use / relation fields).
struct HotSet {
    dir: PathBuf,
    pool: shm::Shm,
    pool_bytes: u64,
    slot_bytes: u64,
    max_bytes: u64,
    min_hits: u64,
    /// Some(min uses): rank by use (kvcore hot.rs: distinct request chains that touched an entry, decayed; loadable
    /// units: GDN state boundaries with their attention chain), None: by RAM-tier hits (min_hits)
    by_use: Option<f64>,
    /// saves in a row an entry must rank in the hot set before it is written (see save())
    stable_saves: u32,
    streak: Mutex<HashMap<String, u32>>,
    lock: Mutex<()>,
    saved: AtomicU64,
    restored: AtomicU64,
    /// exclusive flock on <dir>/LOCK for this broker's lifetime (test brokers sharing the default dir
    /// deleted production's files on every save, so production rewrote ~1.2 GB each time and its shutdown save ran past
    /// systemd's stop timeout)
    _lockf: fs::File,
}


/// hot-set file magic: v1 = the old 64-bit mix data checksum (still read), v2 = XXH3-64 (still read), v3 = v2 + the
/// entry's use and chain relation after the metadata (written now; an older broker drops v3 files as damaged)
const HOT_MAGIC_V1: &[u8; 6] = b"LKVH1\0";
const HOT_MAGIC_V2: &[u8; 6] = b"LKVH2\0";
const HOT_MAGIC: &[u8; 6] = b"LKVH3\0";

fn checksum(d: &[u8]) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for c in d.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0xff51_afd7_ed55_8ccd).rotate_left(29);
    }
    h
}

impl HotSet {
    /// None (hot set off) when another broker holds the directory's lock.
    #[allow(clippy::too_many_arguments)]
    fn new(dir: &str, pool: &Path, pool_bytes: u64, slot_bytes: u64, gib: f64, min_hits: u64, stable_saves: u32,
           by_use: Option<f64>) -> Option<HotSet> {
        use std::io::Write as _;
        use std::os::fd::AsRawFd;
        let _ = fs::create_dir_all(dir);
        let lp = Path::new(dir).join("LOCK");
        let mut lockf = match fs::OpenOptions::new().read(true).write(true).create(true).open(&lp) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("hot set off: cannot open {}: {e}", lp.display());
                return None;
            }
        };
        if unsafe { libc::flock(lockf.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let owner = fs::read_to_string(&lp).unwrap_or_default();
            eprintln!("hot set off: {dir} is in use by another broker ({}); use --hotset-dir off or another dir", owner.trim());
            return None;
        }
        let _ = lockf.set_len(0);
        let _ = writeln!(lockf, "pid {}", std::process::id());
        let f = fs::OpenOptions::new().read(true).write(true).open(pool).expect("hot set: pool");
        let pmap = shm::Shm::map(&f, pool_bytes as usize, true).expect("hot set: mmap");
        Some(HotSet {
            dir: dir.into(),
            pool: pmap,
            pool_bytes,
            slot_bytes,
            max_bytes: (gib * (1u64 << 30) as f64) as u64,
            min_hits,
            by_use,
            stable_saves,
            streak: Mutex::new(HashMap::new()),
            lock: Mutex::new(()),
            saved: AtomicU64::new(0),
            restored: AtomicU64::new(0),
            _lockf: lockf,
        })
    }

    fn name(ns: u32, key: &[u8; 32]) -> String {
        format!("{ns:08x}-{}.kv", hex(key))
    }

    /// `budget`: stop writing new files after this long (shutdown: systemd's stop timeout). An incomplete save keeps
    /// every existing file (no deletions): the previous periodic save already covers most of the set.
    fn save(&self, sh: &Shared, budget: Option<Duration>) {
        use std::io::Write as _;
        let _g = self.lock.lock().unwrap();
        let t = Instant::now();
        let mut incomplete = false;
        // each entry with its use and chain relation (written into the file: a restored prefix is linked as before)
        let (hot, stats) = {
            let mut core = sh.core.lock().unwrap();
            let v = match self.by_use {
                Some(m) => core.hot_entries_by_use(m, self.max_bytes),
                None => core.hot_entries(self.min_hits),
            };
            let v: Vec<(kvcore::HotEntry, f64, kvcore::HotRel)> =
                v.into_iter().map(|e| { let (u, r) = (core.use_of(e.ns, &e.key), core.hot_rel(e.ns, &e.key)); (e, u, r) }).collect();
            (v, core.hot_stats())
        };
        let mut keep = HashSet::new();
        let (mut total, mut written, mut waiting) = (0u64, 0u64, 0u64);
        // Stability (prompts should fall naturally into the hot set, without a hard-coded
        // rule): an entry not yet on disk is written only once it has ranked in the set for `stable_saves` saves in a
        // row. Shared prompt prefixes are hit by every turn of every conversation and stay; one conversation's entries
        // are hot for minutes and drop out before qualifying, so they no longer churn ~1-2 GB to disk every save.
        let old_streak = std::mem::take(&mut *self.streak.lock().unwrap());
        let mut new_streak = HashMap::new();
        for (e, use_now, rel) in hot {
            let size: u64 = e.ranges.iter().map(|r| r.len as u64 * self.slot_bytes).sum();
            if total + size > self.max_bytes {
                continue;
            }
            total += size;
            let name = HotSet::name(e.ns, &e.key);
            let streak = old_streak.get(&name).copied().unwrap_or(0) + 1;
            new_streak.insert(name.clone(), streak);
            let path = self.dir.join(&name);
            if path.exists() {
                keep.insert(name);
                continue;
            }
            if streak < self.stable_saves {
                waiting += 1;
                continue;
            }
            keep.insert(name.clone());
            if budget.is_some_and(|b| t.elapsed() >= b) {
                incomplete = true;
                continue;
            }
            let mut data = Vec::with_capacity(size as usize);
            let g0 = sh.core.lock().unwrap().copy_guard(e.ns, &e.key);
            for r in &e.ranges {
                let off = r.start as u64 * self.slot_bytes;
                let len = r.len as u64 * self.slot_bytes;
                if off + len > self.pool_bytes {
                    continue;
                }
                // unleased copy: kept only if the entry is the same one, unmoved and unrewritten, afterwards
                data.extend_from_slice(&unsafe { self.pool.span(off as usize, len as usize) }.to_vec(len as usize));
            }
            // evicted / stored again / moved / rewritten while copying (copy_guard, as the SSD writer), or not matching
            // the entry's RAM integrity checksum (never persist a flipped bit): drop the copy
            let file_sum = xxhash_rust::xxh3::xxh3_64(&data);
            let ok = g0.as_ref().is_some_and(|g| !g.3) && {
                let g = g0.as_ref().unwrap();
                let body = &data[..(e.bytes as usize).min(data.len())];
                let raw_sum = if g.2 { lazy::image_sum(body, e.bytes) } else { Some(xxhash_rust::xxh3::xxh3_64(body)) };
                matches!(sh.core.lock().unwrap().checksum_result(e.ns, &e.key, g, raw_sum, sh.now()),
                         kvcore::ChecksumResult::Stored | kvcore::ChecksumResult::Verified)
            };
            if !ok {
                keep.remove(&name);
                total -= size;
                continue;
            }
            let mut hdr = Vec::new();
            hdr.extend_from_slice(HOT_MAGIC);
            hdr.extend_from_slice(&e.ns.to_le_bytes());
            hdr.extend_from_slice(&e.key);
            hdr.push(e.kind);
            hdr.extend_from_slice(&e.format.to_le_bytes());
            hdr.extend_from_slice(&e.producer.to_le_bytes());
            hdr.extend_from_slice(&e.hits.to_le_bytes());
            hdr.extend_from_slice(&e.bytes.to_le_bytes());
            hdr.extend_from_slice(&(e.meta.len() as u16).to_le_bytes());
            hdr.extend_from_slice(&e.meta);
            // v3: use (f64), relation (0 none, 1 attention parent, 2 GDN state anchor), its 16 key bytes, group count
            hdr.extend_from_slice(&use_now.to_bits().to_le_bytes());
            let (rk, rp, rn) = match rel {
                kvcore::HotRel::None => (0u8, [0u8; 16], 0u8),
                kvcore::HotRel::Parent(p) => (1, p, 0),
                kvcore::HotRel::Anchor(a, n) => (2, a, n),
            };
            hdr.push(rk);
            hdr.extend_from_slice(&rp);
            hdr.push(rn);
            hdr.extend_from_slice(&file_sum.to_le_bytes());
            let tmp = self.dir.join(format!("{name}.tmp"));
            let ok = fs::File::create(&tmp)
                .and_then(|mut f| f.write_all(&hdr).and_then(|_| f.write_all(&data)).and_then(|_| f.sync_all()))
                .and_then(|_| fs::rename(&tmp, &path));
            match ok {
                Ok(()) => written += 1,
                Err(err) => {
                    eprintln!("hot set: cannot write {}: {err}", path.display());
                    let _ = fs::remove_file(&tmp);
                }
            }
        }
        let mut removed = 0;
        if incomplete {
            eprintln!("hot set: time budget used up: keeping all existing files");
        } else if let Ok(rd) = fs::read_dir(&self.dir) {
            for f in rd.flatten() {
                let n = f.file_name().to_string_lossy().into_owned();
                if (n.ends_with(".kv") && !keep.contains(&n)) || n.ends_with(".tmp") {   // LOCK is kept
                    let _ = fs::remove_file(f.path());
                    removed += 1;
                }
            }
        }
        *self.streak.lock().unwrap() = new_streak;
        self.saved.store(keep.len() as u64, Relaxed);
        eprintln!("hot set: {} entries ({:.2} GiB ranked) saved, {written} written, {waiting} waiting to prove stable, {removed} removed, {:.1} s",
                  keep.len(), total as f64 / (1u64 << 30) as f64, t.elapsed().as_secs_f64());
        if self.by_use.is_some() {
            // counts only (no keys): what ranked by use, what was left out and why
            eprintln!("hot set by use: {} units ({} entries), {} over budget, {} with use but not loadable from RAM (chain or \
                       state missing), {} used attention entries without a GDN state boundary in RAM, {} touched keys not in RAM \
                       (VRAM only)", stats.units, stats.entries, stats.over_budget, stats.unloadable, stats.stateless,
                      stats.missing_keys);
        }
    }

    fn restore(&self, sh: &Shared) {
        let t = Instant::now();
        let (mut ok, mut bad) = (0u64, 0u64);
        let Ok(rd) = fs::read_dir(&self.dir) else { return };
        for f in rd.flatten() {
            if !f.file_name().to_string_lossy().ends_with(".kv") {
                continue;
            }
            let Ok(b) = fs::read(f.path()) else { continue };
            #[allow(clippy::type_complexity)]
            let parse = || -> Option<(u32, [u8; 32], u8, u16, u16, u64, u64, Vec<u8>, f64, kvcore::HotRel, &[u8])> {
                if b.len() < 6 + 4 + 32 + 1 + 2 + 2 + 8 + 8 + 2
                    || (&b[..6] != HOT_MAGIC && &b[..6] != HOT_MAGIC_V2 && &b[..6] != HOT_MAGIC_V1) {
                    return None;
                }
                let v2 = &b[..6] != HOT_MAGIC_V1;
                let v3 = &b[..6] == HOT_MAGIC;
                let mut i = 6;
                let ns = u32::from_le_bytes(b[i..i + 4].try_into().ok()?);
                i += 4;
                let key: [u8; 32] = b[i..i + 32].try_into().ok()?;
                i += 32;
                let kind = b[i];
                i += 1;
                let format = u16::from_le_bytes(b[i..i + 2].try_into().ok()?);
                i += 2;
                let producer = u16::from_le_bytes(b[i..i + 2].try_into().ok()?);
                i += 2;
                let hits = u64::from_le_bytes(b[i..i + 8].try_into().ok()?);
                i += 8;
                let bytes = u64::from_le_bytes(b[i..i + 8].try_into().ok()?);
                i += 8;
                let ml = u16::from_le_bytes(b[i..i + 2].try_into().ok()?) as usize;
                i += 2;
                let meta = b.get(i..i + ml)?.to_vec();
                i += ml;
                let (mut use_saved, mut rel) = (hits as f64, kvcore::HotRel::None);   // v1/v2: no use saved, hits instead
                if v3 {
                    use_saved = f64::from_bits(u64::from_le_bytes(b.get(i..i + 8)?.try_into().ok()?));
                    let rk = *b.get(i + 8)?;
                    let rp: [u8; 16] = b.get(i + 9..i + 25)?.try_into().ok()?;
                    let rn = *b.get(i + 25)?;
                    rel = match rk {
                        1 => kvcore::HotRel::Parent(rp),
                        2 => kvcore::HotRel::Anchor(rp, rn),
                        _ => kvcore::HotRel::None,
                    };
                    i += 26;
                    if !use_saved.is_finite() {
                        use_saved = 0.0;
                    }
                }
                let sum = u64::from_le_bytes(b.get(i..i + 8)?.try_into().ok()?);
                i += 8;
                let data = &b[i..];
                let ok = if v2 { xxhash_rust::xxh3::xxh3_64(data) == sum } else { checksum(data) == sum };
                ok.then_some((ns, key, kind, format, producer, hits, bytes, meta, use_saved, rel, data))
            };
            let Some((ns, key, kind, format, producer, hits, bytes, meta, use_saved, rel, data)) = parse() else {
                bad += 1;
                eprintln!("hot set: {} is damaged, removed", f.path().display());
                let _ = fs::remove_file(f.path());
                continue;
            };
            if kind == kvcore::KIND_WEIGHTS {
                // weights are never persisted (the save skips them); a file of one (older build, or edited) is dropped
                eprintln!("hot set: {} is a weights entry (never restored), removed", f.path().display());
                let _ = fs::remove_file(f.path());
                continue;
            }
            let mut core = sh.core.lock().unwrap();
            let now = sh.now();
            let Some(ranges) = core.insert_committed(ns, key, kind, format, producer, meta, bytes, hits, now) else { continue };
            if let Some(bk) = &sh.backing {
                if bk.lock().unwrap().ensure(&ranges).is_err() {
                    core.remove_key(ns, &key);
                    continue;
                }
            }
            let mut off_d = 0usize;
            for r in &ranges {
                let off = r.start as u64 * self.slot_bytes;
                let len = (r.len as u64 * self.slot_bytes) as usize;
                let n = len.min(data.len() - off_d);
                // restore at start, before any engine connects: the slots were just reserved for this entry
                unsafe { self.pool.span(off as usize, n) }.write(0, &data[off_d..off_d + n]);
                off_d += n;
            }
            core.restore_hot(ns, &key, use_saved, rel);
            ok += 1;
        }
        self.restored.store(ok, Relaxed);
        eprintln!("hot set: {ok} entries restored from {} ({bad} damaged), {:.1} s", self.dir.display(), t.elapsed().as_secs_f64());
    }
}

// ---------------------------------------------------------------- integrity scrub

/// Page layout of FORMAT 1 (llmt_kv.vllm_tier, Qwen3.8-27B, int8_per_token_head KV): one slot = 17
/// canonical KV tensors' pages of slot/17 bytes. Attention entries (kind 0): every page is NHD rows of 520 B
/// [K int8 x256 | K scale f32 | V int8 x256 | V scale f32]. GDN state entries (kind 1), from vLLM's
/// gated_delta_net_state_shape for this model: a page holds one GDN layer's conv state, fp16
/// (kernel - 1 + num_spec) x conv_dim = (3 + 6 MTP scratch rows) x 10,240 = 184,320 B, of which only the first 3
/// rows (61,440 B) are state (the scratch rows keep a slot's old bytes), then the SSM state, fp32 48 x 128 x 128 =
/// 3,145,728 B, then 131,072 B padding. Pages 0..15 are the 16 layers of a GDN group; page 16 belongs to the
/// attention group's 17th (MTP) layer and is unused in state slots. Dead weight per state slot: 12.8%.
const F1_PAGES: u64 = 17;
const F1_ROW: usize = 520;
const F1_CONV: usize = 184_320;
const F1_CONV_USED: usize = 61_440;
const F1_SSM: usize = 3_145_728;
const SCALE_MAX: f32 = 1.0e4;
const STATE_MAX: f32 = 1.0e6;

fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Failed checks of one page (empty = plausible). State slots: page index 16 is not used by the GDN groups (found
/// on a live pool: it holds stale attention bytes) and is skipped.
fn page_fails(page: &[u8], kind: u8, index: usize) -> Vec<&'static str> {
    let mut f = Vec::new();
    match kind {
        0 => {
            if !page.chunks_exact(F1_ROW).all(|r| {
                let (k, v) = (f32_at(r, 256), f32_at(r, 516));
                k.is_finite() && v.is_finite() && (0.0..=SCALE_MAX).contains(&k) && (0.0..=SCALE_MAX).contains(&v)
            }) {
                f.push("scale");
            }
        }
        1 if index < 16 && page.len() >= F1_CONV + F1_SSM => {
            if !page[..F1_CONV_USED].chunks_exact(2).all(|h| (u16::from_le_bytes([h[0], h[1]]) & 0x7c00) != 0x7c00) {
                f.push("conv");
            }
            if !page[F1_CONV..F1_CONV + F1_SSM].chunks_exact(4).all(|x| {
                let v = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                v.is_finite() && v.abs() <= STATE_MAX
            }) {
                f.push("ssm");
            }
        }
        _ => {}
    }
    f
}

/// Idle-priority integrity scrub of committed entries. Entries are immutable, so each is checked
/// once. report mode counts; evict mode also removes entries whose attention pages fail (state entries only once
/// the page map is confirmed from the per-page counts).
fn scrubber(sh: &Shared, path: &Path, bytes: u64, slot_bytes: u64, evict: bool, mbps: u64) {
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, 19);
    }
    let Ok(f) = fs::File::open(path) else {
        eprintln!("integrity scrub: cannot open {}", path.display());
        return;
    };
    // read-only shared mapping, unmapped when the scrub ends (it never was before); slots copied out, never borrowed
    let Ok(pool) = shm::Shm::map(&f, bytes as usize, false) else {
        eprintln!("integrity scrub: mmap failed");
        return;
    };
    let mut done: HashSet<u64> = HashSet::new();
    let mut retry: Vec<u64> = Vec::new();
    loop {
        let cands = sh.core.lock().unwrap().scrub_candidates(&done);
        if cands.is_empty() {
            std::thread::sleep(Duration::from_secs(5));
            sh.scrub.lock().unwrap().passes += 1;
            // forget ids that no longer exist (bounded memory)
            let core = sh.core.lock().unwrap();
            done.retain(|id| core.has_entry(*id));
            continue;
        }
        for (id, ns, ranges, kind, format, key) in cands {
            done.insert(id);
            if format != 1 || kind > 1 {
                continue;
            }
            let mut page_hashes: Vec<u64> = Vec::new();
            let total: u64 = ranges.iter().map(|r| r.len as u64 * slot_bytes).sum();
            let t0 = Instant::now();
            let mut bad_pages = Vec::new();
            for r in &ranges {
                // a multi-slot entry (state positions: one slot per GDN group) is checked slot by slot
                for si in 0..r.len as u64 {
                    let off = ((r.start as u64 + si) * slot_bytes) as usize;
                    let slot_copy = unsafe { pool.span(off, slot_bytes as usize) }.to_vec(slot_bytes as usize);
                    let slot = &slot_copy[..];
                    if slot_bytes % F1_PAGES != 0 {
                        continue;
                    }
                    let pb = slot.len() / F1_PAGES as usize;
                    for (pi, page) in slot.chunks_exact(pb).enumerate() {
                        for what in page_fails(page, kind, pi) {
                            bad_pages.push((what, pi));
                        }
                        // content fingerprint of the bytes that carry data (state pages: real conv rows + SSM)
                        let used = if kind == 1 && page.len() >= F1_CONV + F1_SSM {
                            checksum(&page[..F1_CONV_USED]) ^ checksum(&page[F1_CONV..F1_CONV + F1_SSM]).rotate_left(17)
                        } else {
                            checksum(page)
                        };
                        page_hashes.push(used);
                    }
                }
            }
            // an entry evicted (and its slot reused) during the read is not judged
            let alive = sh.core.lock().unwrap().has_entry(id);
            if alive {
                let slot = ranges.first().map(|r| r.start).unwrap_or(0);
                sh.core.lock().unwrap().log_content(ns, &key, kind, slot, page_hashes);
                let mut st = sh.scrub.lock().unwrap();
                *st.checked.entry((ns, kind)).or_default() += 1;
                if !bad_pages.is_empty() {
                    *st.bad.entry((ns, kind)).or_default() += 1;
                    for &(what, p) in &bad_pages {
                        *st.bad_pages.entry((ns, kind, what, p)).or_default() += 1;
                    }
                }
                drop(st);
                // evict only attention entries (page map certain); state entries need a confirmed map first
                if evict && kind == 0 && bad_pages.iter().any(|(w, _)| *w == "scale") {
                    if sh.core.lock().unwrap().remove_corrupt(id) {
                        sh.scrub.lock().unwrap().evicted += 1;
                        eprintln!("integrity scrub: removed corrupt attention entry {id} (pages {bad_pages:?})");
                    } else {
                        retry.push(id);
                        done.remove(&id);
                    }
                }
            }
            // rate limit
            let want = Duration::from_secs_f64(total as f64 / (mbps.max(1) as f64 * 1e6));
            if let Some(rest) = want.checked_sub(t0.elapsed()) {
                std::thread::sleep(rest);
            }
        }
        retry.clear();
    }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

const EVENT_KEEP_DAYS: u64 = 14;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// UTC date YYYYMMDD of a unix time in ms (civil-from-days, Howard Hinnant).
fn utc_day(ms: u64) -> String {
    let z = (ms / 86_400_000) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}{m:02}{d:02}")
}

/// Event journal (to trace back which hits happened in what order): every store, lookup and removal
/// as one JSON line, appended every 2 s to <dir>/events-YYYYMMDD.jsonl. Hashes (8-byte key prefixes) and times only.
fn event_writer(sh: &Shared, dir: &Path) {
    use std::io::Write as _;
    let _ = fs::create_dir_all(dir);
    let mut last_day = String::new();
    loop {
        std::thread::sleep(Duration::from_secs(2));
        let evs = {
            let mut core = sh.core.lock().unwrap();
            core.records.clear(); // the journal carries the same records as Remove events
            std::mem::take(&mut core.events)
        };
        if evs.is_empty() {
            continue;
        }
        let names = sh.ns_names.lock().unwrap().clone();
        let conns = sh.conn_names.lock().unwrap().clone();
        let _ = conns;
        let ts = |t: u64| (sh.start_unix_ms + t) as f64 / 1000.0;
        let client = |n: &u32| names.get(n).cloned().unwrap_or_default();
        let mut out = String::new();
        for ev in &evs {
            use std::fmt::Write as _;
            let _ = match ev {
                kvcore::Event::Store { t, ns, kind, engine, key, bytes, meta } => {
                    // tier metadata: b"L1" + parent 16 + fingerprint 8 + model 6 (or the older b"P" + parent 16)
                    let (parent, fp, model) = if meta.starts_with(b"L1") && meta.len() >= 32 {
                        let p = if meta[2..18].iter().all(|&b| b == 0) { "null".to_string() } else { format!("\"{}\"", hex(&meta[2..18])) };
                        (p, format!("\"{}\"", hex(&meta[18..26])), format!("\"{}\"", hex(&meta[26..32])))
                    } else if meta.first() == Some(&b'P') && meta.len() >= 17 {
                        (format!("\"{}\"", hex(&meta[1..17])), "null".into(), "null".into())
                    } else {
                        ("null".into(), "null".into(), "null".into())
                    };
                    writeln!(out, r#"{{"t":{:.3},"ev":"store","ns":"{ns:08x}","client":"{}","kind":"{}","engine":{engine},"key":"{}","bytes":{bytes},"parent":{parent},"fp":{fp},"model":{model}}}"#,
                             ts(*t), client(ns), kind_name(*kind), hex(key))
                }
                kvcore::Event::Lookup { t, ns, engine, key, n_keys, n_hit, acquire } => writeln!(
                    out,
                    r#"{{"t":{:.3},"ev":"lookup","ns":"{ns:08x}","client":"{}","engine":{engine},"key":"{}","n_keys":{n_keys},"n_hit":{n_hit},"acquire":{acquire}}}"#,
                    ts(*t), client(ns), hex(key)
                ),
                kvcore::Event::Content { t, ns, key, kind, slot, pages } => {
                    let ph: Vec<String> = pages.iter().map(|h| format!("\"{h:016x}\"")).collect();
                    writeln!(out, r#"{{"t":{:.3},"ev":"content","ns":"{ns:08x}","client":"{}","kind":"{}","key":"{}","slot":{slot},"pages":[{}]}}"#,
                             ts(*t), client(ns), kind_name(*kind), hex(key), ph.join(","))
                }
                kvcore::Event::Remove(r) => {
                    let eng: Vec<String> = (0..32).filter(|b| r.hit_engines & (1 << b) != 0).map(|b| b.to_string()).collect();
                    let recent: Vec<String> = r.recent.iter().map(|(t, e)| format!("[{:.3},{e}]", ts(*t))).collect();
                    let opt = |x: u64| if x == 0 { "null".to_string() } else { format!("{:.3}", ts(x)) };
                    writeln!(out, r#"{{"t":{:.3},"ev":"remove","ns":"{:08x}","client":"{}","kind":"{}","producer":{},"key":"{}","bytes":{},"born":{:.3},"first_hit":{},"last_hit":{},"hits":{},"hit_engines":[{}],"recent":[{}],"reason":"{}","age_s":{:.1}}}"#,
                             ts(r.left), r.ns, client(&r.ns), kind_name(r.kind), r.producer, hex(&r.key_prefix), r.bytes,
                             ts(r.born), opt(r.first_hit), opt(r.last_hit), r.hits, eng.join(","), recent.join(","),
                             r.reason, r.left.saturating_sub(r.born) as f64 / 1000.0)
                }
            };
        }
        let now = unix_ms();
        let day = utc_day(now);
        let path = dir.join(format!("events-{day}.jsonl"));
        match fs::OpenOptions::new().create(true).append(true).open(&path) {
            Ok(mut f) => {
                let _ = f.write_all(out.as_bytes());
            }
            Err(e) => eprintln!("event journal: cannot write {}: {e}", path.display()),
        }
        if day != last_day {
            last_day = day;
            let cutoff = utc_day(now.saturating_sub(EVENT_KEEP_DAYS * 86_400_000));
            if let Ok(rd) = fs::read_dir(dir) {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().into_owned();
                    if let Some(d) = n.strip_prefix("events-").and_then(|x| x.strip_suffix(".jsonl")) {
                        if d < cutoff.as_str() {
                            let _ = fs::remove_file(e.path());
                        }
                    }
                }
            }
        }
    }
}

fn kind_name(k: u8) -> &'static str {
    match k {
        0 => "attn",
        1 => "state",
        2 => "attn+state",
        kvcore::KIND_TAIL => "tail",
        kvcore::KIND_WEIGHTS => "weights",
        kvcore::KIND_ANY => "any",
        _ => "other",
    }
}

/// One journal line per namespace and entry kind every --stats-log-s (10 min): entries now, and hits / commits / evictions (never
/// hit) since the last line.
fn log_summary(sh: &Shared, prev: HashMap<(u32, u8), Vec<(&'static str, u64)>>) -> HashMap<(u32, u8), Vec<(&'static str, u64)>> {
    let (ns, free) = {
        let core = sh.core.lock().unwrap();
        (core.ns_stats(), core.gauges().iter().find(|(k, _)| k == "slots_free").map(|x| x.1).unwrap_or(0))
    };
    let names = sh.ns_names.lock().unwrap().clone();
    let mut out = HashMap::new();
    for ((n, kind), g, c) in ns {
        let p = prev.get(&(n, kind));
        let d = |k: &str| {
            let v = c.iter().find(|(x, _)| *x == k).map(|x| x.1).unwrap_or(0);
            v - p.and_then(|p| p.iter().find(|(x, _)| *x == k)).map(|x| x.1).unwrap_or(0)
        };
        let entries = g.iter().find(|(x, _)| *x == "entries").map(|x| x.1).unwrap_or(0);
        eprintln!(
            "stats: ns {n:08x} ({}) {}: entries {entries}, lookups {} keys, hits {}, commits {}, exists {}, full {}, evicted {} ({} never hit); free slots {free}",
            names.get(&n).map(String::as_str).unwrap_or("?"),
            kind_name(kind),
            d("lookup_keys"), d("hit_entries"), d("committed"), d("exists"), d("full"), d("evicted"), d("evicted_unused")
        );
        out.insert((n, kind), c);
    }
    out
}

fn metrics_text(sh: &Shared) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let (g, c) = {
        // summed over the slab classes (one class: as before)
        let sum = |v: &mut Vec<(String, u64)>, w: Vec<(String, u64)>| {
            for (k, x) in w {
                match v.iter_mut().find(|(n, _)| *n == k) {
                    Some(e) => e.1 += x,
                    None => v.push((k, x)),
                }
            }
        };
        let (mut g, mut c) = (Vec::new(), Vec::new());
        for i in 0..sh.n_classes() {
            let core = sh.cls(i).core.lock().unwrap();
            sum(&mut g, core.gauges());
            sum(&mut c, core.counters());
        }
        (g, c)
    };
    let _ = writeln!(s, "# TYPE llmt_kv_info gauge\nllmt_kv_info{{version=\"{VERSION}\",epoch=\"{:016x}\"}} 1", sh.hello.epoch);
    let _ = writeln!(s, "llmt_kv_slot_bytes {}", sh.hello.slot_bytes);
    if let Some(n) = &sh.ngram {
        n.metrics(&mut s);
    }
    for i in 0..sh.n_classes() {
        if let Some(d) = sh.cls(i).disk {
            d.metrics(&mut s);
        }
    }
    {
        let p = sh.pool.lock().unwrap();
        let _ = writeln!(s, "# TYPE llmt_kv_pool_target_bytes gauge\nllmt_kv_pool_target_bytes {}", p.target_bytes);
        let _ = writeln!(s, "# TYPE llmt_kv_pool_auto_parked_slots gauge\nllmt_kv_pool_auto_parked_slots {}", p.auto_parked);
        lazy::metrics(&sh, &mut s);
        let _ = writeln!(s, "# TYPE llmt_kv_pool_punched_slots gauge\nllmt_kv_pool_punched_slots {}", p.punched.iter().map(|r| r.len as u64).sum::<u64>());
        let _ = writeln!(s, "# TYPE llmt_kv_pool_freed_hugepages gauge\nllmt_kv_pool_freed_hugepages {}", p.freed_pages);
        let _ = writeln!(s, "# TYPE llmt_kv_pool_gen gauge\nllmt_kv_pool_gen {}", p.gen);
        let _ = writeln!(s, "# TYPE llmt_kv_mem_available_bytes gauge\nllmt_kv_mem_available_bytes {}", mem_available());
    }
    for (k, v) in g {
        let _ = writeln!(s, "# TYPE llmt_kv_{k} gauge\nllmt_kv_{k} {v}");
    }
    for (k, v) in c {
        let _ = writeln!(s, "# TYPE llmt_kv_{k}_total counter\nllmt_kv_{k}_total {v}");
    }
    let names = sh.ns_names.lock().unwrap().clone();
    let ns = sh.core.lock().unwrap().ns_stats();
    if !ns.is_empty() {
        let (g0, c0) = (&ns[0].1, &ns[0].2);
        for (k, _) in g0 {
            let _ = writeln!(s, "# TYPE llmt_kv_ns_{k} gauge");
        }
        for (k, _) in c0 {
            let _ = writeln!(s, "# TYPE llmt_kv_ns_{k}_total counter");
        }
    }
    for ((n, kind), g, c) in &ns {
        let label = format!("ns=\"{n:08x}\",client=\"{}\",kind=\"{}\"", names.get(n).map(String::as_str).unwrap_or(""),
                            kind_name(*kind));
        for (k, v) in g {
            let _ = writeln!(s, "llmt_kv_ns_{k}{{{label}}} {v}");
        }
        for (k, v) in c {
            let _ = writeln!(s, "llmt_kv_ns_{k}_total{{{label}}} {v}");
        }
    }
    let life = sh.core.lock().unwrap().life.clone();
    if !life.is_empty() {
        let _ = writeln!(s, "# TYPE llmt_kv_life_seconds histogram");
    }
    let mut life: Vec<_> = life.into_iter().collect();
    life.sort_by_key(|((w, k), _)| (*w, *k));
    for ((what, kind), h) in &life {
        let l = format!("what=\"{what}\",kind=\"{}\"", kind_name(*kind));
        for (i, b) in kvcore::LIFE_BUCKETS_S.iter().enumerate() {
            let _ = writeln!(s, "llmt_kv_life_seconds_bucket{{{l},le=\"{b}\"}} {}", h.buckets[i]);
        }
        let _ = writeln!(s, "llmt_kv_life_seconds_bucket{{{l},le=\"+Inf\"}} {}", h.count);
        let _ = writeln!(s, "llmt_kv_life_seconds_sum{{{l}}} {}", h.sum_ms as f64 / 1000.0);
        let _ = writeln!(s, "llmt_kv_life_seconds_count{{{l}}} {}", h.count);
    }
    // heat: slots / entries by time since last use, per class and kind; HEAT hint counters per class
    let _ = writeln!(s, "# TYPE llmt_kv_heat_slots gauge\n# TYPE llmt_kv_heat_entries gauge\n# TYPE llmt_kv_heat_hints_total counter");
    for i in 0..sh.n_classes() {
        let c = sh.cls(i);
        let (h, hot, cold) = {
            let core = c.core.lock().unwrap();
            (core.heat_hist(), core.c.heat_hot, core.c.heat_cold)
        };
        for (kind, b, slots, n) in h {
            let l = format!("class=\"{}\",kind=\"{}\",age=\"{}\"", c.name, kind_name(kind), kvcore::HEAT_AGE[b].1);
            let _ = writeln!(s, "llmt_kv_heat_slots{{{l}}} {slots}\nllmt_kv_heat_entries{{{l}}} {n}");
        }
        let _ = writeln!(s, "llmt_kv_heat_hints_total{{class=\"{}\",hint=\"hot\"}} {hot}\nllmt_kv_heat_hints_total{{class=\"{}\",hint=\"cold\"}} {cold}",
                         c.name, c.name);
    }
    // reuse histograms: counts only, per class
    let _ = writeln!(s, "# TYPE llmt_kv_reuse_hits_total counter\n# TYPE llmt_kv_reuse_evictions_total counter");
    for i in 0..sh.n_classes() {
        let c = sh.cls(i);
        let (mut h, mut e): (Vec<_>, Vec<_>) = {
            let core = c.core.lock().unwrap();
            (core.reuse_hits.iter().map(|(k, v)| (*k, *v)).collect(), core.reuse_evict.iter().map(|(k, v)| (*k, *v)).collect())
        };
        h.sort();
        e.sort();
        for ((kind, d, a), v) in h {
            let _ = writeln!(s, "llmt_kv_reuse_hits_total{{class=\"{}\",kind=\"{}\",depth=\"{}\",age=\"{}\"}} {v}", c.name,
                             kind_name(kind), kvcore::REUSE_DEPTH[d as usize].1, kvcore::REUSE_AGE[a as usize].1);
        }
        for ((kind, d, a, hb, r), v) in e {
            let _ = writeln!(s, "llmt_kv_reuse_evictions_total{{class=\"{}\",kind=\"{}\",depth=\"{}\",age=\"{}\",hits=\"{}\",reason=\"{}\"}} {v}",
                             c.name, kind_name(kind), kvcore::REUSE_DEPTH[d as usize].1, kvcore::REUSE_AGE[a as usize].1,
                             kvcore::REUSE_HITS[hb as usize], kvcore::REUSE_REASON[r as usize]);
        }
    }
    let sc = sh.scrub.lock().unwrap().clone();
    let names = sh.ns_names.lock().unwrap().clone();
    let nm = |n: &u32| names.get(n).cloned().unwrap_or_default();
    let _ = writeln!(s, "# TYPE llmt_kv_scrub_checked_total counter\n# TYPE llmt_kv_scrub_bad_total counter");
    let mut ck: Vec<_> = sc.checked.iter().collect();
    ck.sort();
    for ((n, k), c) in ck {
        let l = format!("ns=\"{n:08x}\",client=\"{}\",kind=\"{}\"", nm(n), kind_name(*k));
        let _ = writeln!(s, "llmt_kv_scrub_checked_total{{{l}}} {c}");
        let _ = writeln!(s, "llmt_kv_scrub_bad_total{{{l}}} {}", sc.bad.get(&(*n, *k)).copied().unwrap_or(0));
    }
    let mut bp: Vec<_> = sc.bad_pages.iter().collect();
    bp.sort();
    let _ = writeln!(s, "# TYPE llmt_kv_scrub_bad_pages_total counter");
    for ((n, k, w, p), c) in bp {
        let _ = writeln!(s, "llmt_kv_scrub_bad_pages_total{{ns=\"{n:08x}\",client=\"{}\",kind=\"{}\",check=\"{w}\",page=\"{p}\"}} {c}",
                         nm(n), kind_name(*k));
    }
    let _ = writeln!(s, "# TYPE llmt_kv_scrub_evicted_total counter\nllmt_kv_scrub_evicted_total {}", sc.evicted);
    let _ = writeln!(s, "# TYPE llmt_kv_scrub_idle_passes_total counter\nllmt_kv_scrub_idle_passes_total {}", sc.passes);
    let _ = writeln!(s, "# TYPE llmt_kv_protocol_errors_total counter\nllmt_kv_protocol_errors_total {}", sh.protocol_errors.load(Relaxed));
    {
        let hs = &sh.hint_stats;
        for (m, v) in [("hints", &hs.hints), ("hints_refused", &hs.hints_refused), ("where_requests", &hs.wheres),
                       ("where_anchors", &hs.where_anchors), ("where_places", &hs.where_places),
                       ("where_places_resident", &hs.where_resident), ("where_micros", &hs.where_us)] {
            let _ = writeln!(s, "# TYPE llmt_kv_{m}_total counter\nllmt_kv_{m}_total {}", v.load(Relaxed));
        }
        let h = sh.hints.lock().unwrap();
        let _ = writeln!(s, "# TYPE llmt_kv_hint_entries gauge\nllmt_kv_hint_entries {}", h.len());
        let _ = writeln!(s, "# TYPE llmt_kv_hint_bytes gauge\nllmt_kv_hint_bytes {}", h.bytes());
        let _ = writeln!(s, "# TYPE llmt_kv_hints_dropped_total counter\nllmt_kv_hints_dropped_total {}", h.dropped);
    }
    let _ = writeln!(s, "# TYPE llmt_kv_handler_panics_total counter\nllmt_kv_handler_panics_total {}", sh.handler_panics.load(Relaxed));
    if let Some(t) = &sh.tracer {
        for (m, v) in [("trace_records", &t.records), ("trace_dumps", &t.dumps), ("trace_dumps_refused", &t.refused)] {
            let _ = writeln!(s, "# TYPE llmt_kv_{m}_total counter\nllmt_kv_{m}_total {}", v.load(Relaxed));
        }
        let _ = writeln!(s, "# TYPE llmt_kv_trace_ring_bytes gauge\nllmt_kv_trace_ring_bytes {}", t.ring_bytes());
    }
    let _ = writeln!(s, "# TYPE llmt_kv_request_seconds histogram");
    for op in Op::ALL {
        let st = &sh.ops[op as usize - 1];
        let n = op.name();
        for (i, b) in LAT_BUCKETS_US.iter().enumerate() {
            let _ = writeln!(s, "llmt_kv_request_seconds_bucket{{op=\"{n}\",le=\"{}\"}} {}", *b as f64 / 1e6, st.buckets[i].load(Relaxed));
        }
        let cnt = st.count.load(Relaxed);
        let _ = writeln!(s, "llmt_kv_request_seconds_bucket{{op=\"{n}\",le=\"+Inf\"}} {cnt}");
        let _ = writeln!(s, "llmt_kv_request_seconds_sum{{op=\"{n}\"}} {}", st.sum_us.load(Relaxed) as f64 / 1e6);
        let _ = writeln!(s, "llmt_kv_request_seconds_count{{op=\"{n}\"}} {cnt}");
    }
    if let Some(c) = sh.chains.get() {
        s.push_str(&c.metrics());
    }
    s
}

fn serve_metrics(sh: Arc<Shared>, addr: String) {
    let l = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("metrics: cannot bind {addr}: {e}");
            return;
        }
    };
    for s in l.incoming().flatten() {
        let mut s = s;
        let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
        let mut req = [0u8; 1024];
        let n = s.read(&mut req).unwrap_or(0);
        let line = String::from_utf8_lossy(&req[..n]).lines().next().unwrap_or("").to_string();
        let (ok, body) = if line.starts_with("GET /metrics") {
            (true, metrics_text(&sh))
        } else if line.starts_with("GET /pool") || line.starts_with("POST /pool") {
            // POST /pool?target_gib=N (or max): explicit usable-pool target; ?reserve_gib=N: autonomous policy (0 = off);
            // GET /pool: status
            let q = line.split_whitespace().nth(1).unwrap_or("");
            let mut p = sh.pool.lock().unwrap();
            if let Some(v) = q.split('?').nth(1).and_then(|qs| qs.split('&').find_map(|kv| kv.strip_prefix("reserve_gib="))) {
                if let Ok(g) = v.parse::<f64>() {
                    p.reserve = (g.max(0.0) * (1u64 << 30) as f64) as u64;
                    p.calm_since = None;
                    p.last = format!("mem reserve set to {:.1} GiB ({})", g.max(0.0), if p.reserve > 0 { "autonomous culling on" } else { "explicit only" });
                    eprintln!("pool: {}", p.last);
                }
            }
            if let Some(v) = q.split('?').nth(1).and_then(|qs| qs.split('&').find_map(|kv| kv.strip_prefix("target_gib="))) {
                let full = p.n_slots * p.slot;
                let t = if v == "max" { full } else { v.parse::<f64>().map(|g| (g * (1u64 << 30) as f64) as u64).unwrap_or(full) };
                p.target_bytes = t.clamp(p.min_slots * p.slot, full);
                p.last = format!("target set to {:.1} GiB", p.target_bytes as f64 / (1u64 << 30) as f64);
                eprintln!("pool: {}", p.last);
            }
            let parked = sh.core.lock().unwrap().parked_slots();
            (true, p.status(parked, mem_available()))
        } else {
            (false, "see /metrics or /pool\n".into())
        };
        let status = if ok { "200 OK" } else { "404 Not Found" };
        let _ = write!(
            s,
            "HTTP/1.1 {status}\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    }
}

// ---------------------------------------------------------------- main

fn main() {
    // background worker mode (bg.rs): the same binary, its own process and unit
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("--bg-worker") {
        fp::setup();
        bg::run(a.get(2).map(PathBuf::from).unwrap_or_else(|| usage()));
    }
    // SIGTERM/SIGINT are handled by one thread (hot-set save, then exit): block them in every thread
    let mut sigs: libc::sigset_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::sigemptyset(&mut sigs);
        libc::sigaddset(&mut sigs, libc::SIGTERM);
        libc::sigaddset(&mut sigs, libc::SIGINT);
        libc::pthread_sigmask(libc::SIG_BLOCK, &sigs, std::ptr::null_mut());
    }
    fp::setup();
    let mut o = parse_opts();
    // unified broker: class 0 = the first --class (its pool in <pool-dir>/<name>); the others are built below
    let base_pool_dir = o.pool_dir.clone();
    let base_disk_dir = o.disk_dir.clone();
    let ceiling = o.pool_bytes;
    let class_cap = |c: &ClassSpec| c.ram.map(|g| ((g * (1u64 << 30) as f64) as u64).min(ceiling)).unwrap_or(ceiling);
    let class0_name = match o.classes.first().cloned() {
        Some(c0) => {
            if !o.pool_lazy {
                eprintln!("--class needs the on-demand pool (--pool-lazy on)");
                std::process::exit(2);
            }
            o.slot_bytes = c0.slot;
            o.pool_bytes = class_cap(&c0);
            o.pool_dir = base_pool_dir.join(&c0.name);
            if base_disk_dir != "off" {
                o.disk_dir = format!("{}/{}", base_disk_dir, c0.name);
                o.disk_gib = c0.ssd.unwrap_or(o.disk_gib);
                o.disk_write_gib_day = c0.write.unwrap_or(o.disk_write_gib_day);
            }
            c0.name
        }
        None => "default".to_string(),
    };
    let n_slots = o.pool_bytes / o.slot_bytes;
    if n_slots > u32::MAX as u64 {
        eprintln!("too many slots");
        std::process::exit(2);
    }
    let pool_bytes = n_slots * o.slot_bytes;
    let epoch = random_u64();
    // on-demand pool on tmpfs only (huge-page pools keep the eager reservation logic)
    let lazy_pool = o.pool_lazy && {
        use std::os::unix::ffi::OsStrExt;
        let _ = fs::create_dir_all(&o.pool_dir);
        let c = std::ffi::CString::new(o.pool_dir.as_os_str().as_bytes()).unwrap_or_default();
        let mut st: libc::statfs = unsafe { std::mem::zeroed() };
        !(unsafe { libc::statfs(c.as_ptr(), &mut st) } == 0 && st.f_type as i64 == HUGETLBFS_MAGIC)
    };
    let (pool, reused) = open_pool(&o.pool_dir, pool_bytes, epoch, o.prefault && !lazy_pool, o.new_pool, o.hugepages_ctl, lazy_pool).unwrap_or_else(|e| {
        eprintln!("cannot create pool in {}: {e}", o.pool_dir.display());
        std::process::exit(1)
    });
    if o.mlock && lazy_pool {
        eprintln!("pool: on demand, so not mlocked (mlock would fault in the whole ceiling)");
    }
    if o.mlock && !lazy_pool {
        let t = Instant::now();
        match lock_pool(&pool, pool_bytes) {
            Ok(()) => eprintln!("pool locked in RAM (mlock, {:.1} s)", t.elapsed().as_secs_f64()),
            Err(e) => eprintln!("WARNING: cannot mlock the pool ({e}); pool pages may be swapped out (--no-mlock silences this)"),
        }
    }
    if let Some(d) = o.socket.parent() {
        fs::create_dir_all(d).expect("socket directory");
    }
    // systemd socket activation (LISTEN_PID / LISTEN_FDS, fd 3): the listener is inherited, the socket file is systemd's
    let activated = std::env::var("LISTEN_PID").ok().and_then(|v| v.parse::<u32>().ok()) == Some(std::process::id())
        && std::env::var("LISTEN_FDS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0) >= 1;
    let listener = if activated {
        use std::os::fd::FromRawFd;
        unsafe { libc::fcntl(3, libc::F_SETFD, libc::FD_CLOEXEC) };
        eprintln!("socket activation: listening on the inherited fd 3");
        unsafe { UnixListener::from_raw_fd(3) }
    } else {
        let _ = fs::remove_file(&o.socket);
        let l = UnixListener::bind(&o.socket).unwrap_or_else(|e| {
            eprintln!("cannot bind {}: {e}", o.socket.display());
            std::process::exit(1)
        });
        fs::set_permissions(&o.socket, fs::Permissions::from_mode(0o660)).expect("socket permissions");
        l
    };
    let cfg = kvcore::Config {
        shared_slots: (o.shared_gib * (1u64 << 30) as f64 / o.slot_bytes as f64) as u64,
        shared_age_mult: o.shared_age_mult,
        shared_min_cont: o.shared_min_cont,
        shared_max_depth: o.shared_max_depth,
        n_slots: n_slots as u32,
        slot_bytes: o.slot_bytes,
        reserve_timeout_ms: o.reserve_timeout_ms,
        lease_timeout_ms: o.lease_timeout_ms,
        low_water_slots: (n_slots as f64 * o.low_water_pct / 100.0) as u64,
        policy: o.policy,
        stale_unhit_ms: o.stale_unhit_s * 1000,
        ns_floor_slots: (n_slots as f64 * o.ns_floor_pct / 100.0) as u64,
        ns_cap_slots: (n_slots as f64 * o.ns_cap_pct / 100.0) as u64,
        tail_age_div: o.tail_age_div,
        heat_hit_ms: o.heat_hit_ms,
        seg_evict: o.seg_evict,
        seg_grace_ms: o.seg_grace_s * 1000,
        weights_kv_slot_ms: o.weights_kv_slot_ms,
    };
    eprintln!(
        "llmt-kvbroker {VERSION}: protocol {PROTOCOL_VERSION}, epoch {epoch:016x}, pool {} ({:.2} GiB, {} slots of {} KiB), \
         socket {}, policy {:?}, metrics {}, timeouts reserve {} ms / lease {} ms, low water {} slots",
        pool.display(),
        pool_bytes as f64 / (1u64 << 30) as f64,
        n_slots,
        o.slot_bytes >> 10,
        o.socket.display(),
        o.policy,
        o.metrics,
        cfg.reserve_timeout_ms,
        cfg.lease_timeout_ms,
        cfg.low_water_slots
    );
    eprintln!("segment eviction {} (grace {} s)", if o.seg_evict { "on" } else { "off" }, o.seg_grace_s);
    let mk_cfg = |n: u64, slot: u64| kvcore::Config {
        shared_slots: (o.shared_gib * (1u64 << 30) as f64 / slot as f64) as u64,
        shared_age_mult: o.shared_age_mult,
        shared_min_cont: o.shared_min_cont,
        shared_max_depth: o.shared_max_depth,
        n_slots: n as u32,
        slot_bytes: slot,
        reserve_timeout_ms: o.reserve_timeout_ms,
        lease_timeout_ms: o.lease_timeout_ms,
        low_water_slots: (n as f64 * o.low_water_pct / 100.0) as u64,
        policy: o.policy,
        stale_unhit_ms: o.stale_unhit_s * 1000,
        ns_floor_slots: (n as f64 * o.ns_floor_pct / 100.0) as u64,
        ns_cap_slots: (n as f64 * o.ns_cap_pct / 100.0) as u64,
        tail_age_div: o.tail_age_div,
        heat_hit_ms: o.heat_hit_ms,
        seg_evict: o.seg_evict,
        seg_grace_ms: o.seg_grace_s * 1000,
        weights_kv_slot_ms: (o.weights_kv_slot_ms as u128 * slot as u128 / o.slot_bytes.max(1) as u128) as u64,
    };
    // pinning guard of the on-demand pool (lazy.rs notes): one flag for every class; the reuse hold per pool file
    let pins = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reuse_hold = |reused: bool| (reused && o.pool_reuse_hold_s > 0).then(|| Duration::from_secs(o.pool_reuse_hold_s));
    if lazy_pool && reused {
        eprintln!("on-demand pool reused: backed slots read back from the file; no hole punching for {} s", o.pool_reuse_hold_s);
    }
    let extra: Vec<Class> = o.classes.iter().skip(1).map(|c| {
        let n = class_cap(c) / c.slot;
        let bytes = n * c.slot;
        let dir = base_pool_dir.join(&c.name);
        let (path, reused) = open_pool(&dir, bytes, epoch, false, o.new_pool, o.hugepages_ctl, true).unwrap_or_else(|e| {
            eprintln!("cannot create pool in {}: {e}", dir.display());
            std::process::exit(1)
        });
        eprintln!("pool class {}: {} ({:.2} GiB, {} slots of {} KiB)", c.name, path.display(), bytes as f64 / (1u64 << 30) as f64, n, c.slot >> 10);
        Class {
            name: c.name.clone(),
            core: Mutex::new({
                let mut x = Core::new(mk_cfg(n, c.slot));
                x.sess_gap_ms = o.disk_session_gap_s * 1000;
                x.use_half_life_ms = o.hotset_use_half_life_s * 1000;
                x.hot_min_use = if o.hotset_rank_use { o.hotset_min_uses } else { 0.0 };
                x.low_margin_ms = o.low_margin_s * 1000;
                x
            }),
            hello: HelloReply {
                protocol_version: PROTOCOL_VERSION,
                broker_version: VERSION.into(),
                epoch,
                pool_path: path.to_string_lossy().into_owned(),
                pool_bytes: bytes,
                slot_bytes: c.slot,
                n_slots: n as u32,
            },
            backing: Some(Mutex::new(lazy::Backing::new(&path, bytes, c.slot, pins.clone(), reuse_hold(reused)).unwrap_or_else(|e| {
                eprintln!("cannot open the on-demand pool {}: {e}", path.display());
                std::process::exit(1)
            }))),
            disk: std::sync::OnceLock::new(),
        }
    }).collect();
    let sh = Arc::new(Shared {
        core: Mutex::new({
            let mut x = Core::new(cfg);
            x.sess_gap_ms = o.disk_session_gap_s * 1000;
            x.use_half_life_ms = o.hotset_use_half_life_s * 1000;
            x.hot_min_use = if o.hotset_rank_use { o.hotset_min_uses } else { 0.0 };
            x.low_margin_ms = o.low_margin_s * 1000;
            x
        }),
        writers: Mutex::new(HashMap::new()),
        start: Instant::now(),
        hello: HelloReply {
            protocol_version: PROTOCOL_VERSION,
            broker_version: VERSION.into(),
            epoch,
            pool_path: pool.to_string_lossy().into_owned(),
            pool_bytes,
            slot_bytes: o.slot_bytes,
            n_slots: n_slots as u32,
        },
        ops: Op::ALL.iter().map(|_| OpStats::new()).collect(),
        protocol_errors: AtomicU64::new(0),
        grace_until: if reused { o.reuse_grace_ms } else { 0 },
        conn_names: Mutex::new(HashMap::new()),
        conn_feats: Mutex::new(HashMap::new()),
        ns_names: Mutex::new(HashMap::new()),
        domains: Mutex::new(HashMap::new()),
        conn_domains: Mutex::new(HashMap::new()),
        start_unix_ms: unix_ms(),
        scrub: Mutex::new(ScrubStats::default()),
        chains: std::sync::OnceLock::new(),
        pool: Mutex::new(match PoolCtl::new(&pool, pool_bytes.div_ceil(2 << 20) * (2 << 20), o.slot_bytes, n_slots as u64, &o) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("cannot open the pool for the dynamic-pool controller: {e}");
                std::process::exit(1)
            }
        }),
        backing: if lazy_pool {
            match lazy::Backing::new(&pool, pool_bytes, o.slot_bytes, pins.clone(), reuse_hold(reused)) {
                Ok(b) => Some(Mutex::new(b)),
                Err(e) => {
                    eprintln!("cannot open the on-demand pool: {e}");
                    std::process::exit(1)
                }
            }
        } else {
            None
        },
        pins: pins.clone(),
        pin_safe: o.pool_pin_safe,
        lazy: lazy::Stats::default(),
        last_req_ms: AtomicU64::new(0),
        extra,
        class0_name,
        conn_class: Mutex::new(HashMap::new()),
        ns_conns: Mutex::new(HashMap::new()),
        ceiling,
        rchunk: if o.classes.is_empty() { vec![0] } else { o.classes.iter().map(|c| c.rchunk).collect() },
        rcompress_after_ms: o.ram_compress_after_s * 1000,
        rcompress_quiet_ms: o.ram_compress_quiet_ms,
        cksum_bytes_s: o.cksum_mib_s << 20,
        cksum_reverify_ms: o.cksum_reverify_s * 1000,
        tracer: (o.trace_mib > 0 && o.trace_dir != "off")
            .then(|| trace::Tracer::new(o.trace_mib, PathBuf::from(&o.trace_dir), o.trace_dumps_day)),
        handler_panics: AtomicU64::new(0),
        latency_slice_us: o.latency_slice_us,
        bg_worker: o.bg_worker,
        bg_conns: Mutex::new(HashSet::new()),
        bg_open: std::sync::atomic::AtomicUsize::new(0),
        bg_rewrites: Mutex::new(HashMap::new()),
        hints: Mutex::new(kvcore::hints::HintIndex::new((o.hint_mib << 20) as usize)),
        hint_stats: HintStats::default(),
        obs_conns: Mutex::new(HashSet::new()),
        obs_open: std::sync::atomic::AtomicUsize::new(0),
        bg_compacts: Mutex::new(HashMap::new()),
        compact_budget: AtomicU64::new(0),
        self_arc: std::sync::OnceLock::new(),
        warm_within_ms: o.evict_age_s * 1000,
        ngram: if o.ngram_tables.is_empty() {
            None
        } else {
            match ngram::Ngram::new(&o.ngram_tables, &o.ngram_bins, &o.ngram_rows, &o.ngram_resident, &o.pool_dir, epoch, (o.ngram_cache_gib * (1u64 << 30) as f64) as u64,
                                    o.ngram_windows, o.ngram_window_mib << 20, Duration::from_secs(o.ngram_idle_s),
                                    (o.mem_reserve_gib * (1u64 << 30) as f64) as u64, Duration::from_secs(o.pool_regrow_s)) {
                Ok(n) => Some(n),
                Err(e) => {
                    eprintln!("ngram: {e}");
                    std::process::exit(1)
                }
            }
        },
    });
    let _ = sh.self_arc.set(Arc::downgrade(&sh));
    {
        let p = sh.pool.lock().unwrap();
        eprintln!("dynamic pool: min {:.1} GiB, mem reserve {:.1} GiB ({}), regrow after {} s calm, ack timeout {} s, hugetlbfs {} (reservation control {})",
                  p.min_slots as f64 * p.slot as f64 / (1u64 << 30) as f64, o.mem_reserve_gib,
                  if o.mem_reserve_gib > 0.0 { "autonomous culling on" } else { "explicit only" }, o.pool_regrow_s, o.pool_ack_timeout_s, p.hugetlb, p.hp_ctl);
    }
    if !lazy_pool {
        let sh = sh.clone();
        let regrow = Duration::from_secs(o.pool_regrow_s);
        let ack = Duration::from_secs(o.pool_ack_timeout_s);
        std::thread::Builder::new()
            .name("poolctl".into())
            .spawn(move || loop {
                std::thread::sleep(Duration::from_secs(1));
                pool_step(&sh, regrow, ack);
            })
            .unwrap();
    }
    if o.scrub != "off" {
        let sh = sh.clone();
        let (evict, mbps, path, bytes, sb) = (o.scrub == "evict", o.scrub_mbps, pool.clone(), pool_bytes, o.slot_bytes);
        std::thread::Builder::new()
            .name("scrub".into())
            .spawn(move || scrubber(&sh, &path, bytes, sb, evict, mbps))
            .unwrap();
        eprintln!("integrity scrub: {} mode, {} MB/s", o.scrub, o.scrub_mbps);
    }
    if o.chains != "off" {
        // Lookup chains: own socket and threads, no state shared with the KV core. A failure here
        // is logged and the KV broker runs on.
        let mut cc = kvchains::server::Config::new(&o.chains);
        cc.global = o.chains_global;
        cc.store_mem = o.chains_mem_mib << 20;
        cc.snapshot = (o.chains_snapshot != "off").then(|| PathBuf::from(&o.chains_snapshot));
        cc.snapshot_s = o.chains_snapshot_s;
        cc.heat_half_life_s = o.chains_heat_half_life_s;
        cc.prune_grace_s = o.chains_prune_grace_s;
        cc.prune_heat = o.chains_prune_heat;
        match kvchains::server::serve(cc) {
            Ok(c) => {
                let _ = sh.chains.set(c);
                eprintln!("lookup chains: listening on {} (global store {}, {} MiB, snapshot {} every {} s, heat half-life \
                           {} s, prune below {} after {} s)", o.chains, if o.chains_global { "on" } else { "off" },
                          o.chains_mem_mib, o.chains_snapshot, o.chains_snapshot_s, o.chains_heat_half_life_s,
                          o.chains_prune_heat, o.chains_prune_grace_s)
            }
            Err(e) => eprintln!("lookup chains: cannot listen on {}: {e} (chains off)", o.chains),
        }
    }
    if o.events_dir != "off" {
        sh.core.lock().unwrap().log_events = true;
        let sh = sh.clone();
        let dir = PathBuf::from(&o.events_dir);
        std::thread::Builder::new()
            .name("events".into())
            .spawn(move || event_writer(&sh, &dir))
            .unwrap();
        eprintln!("event journal: {}/events-YYYYMMDD.jsonl (UTC days, kept {EVENT_KEEP_DAYS} days)", o.events_dir);
    }
    if reused {
        eprintln!("reusing pool {} (index empty; no reservations for {} ms)", pool.display(), o.reuse_grace_ms);
    }
    {
        let sh = sh.clone();
        let every = Duration::from_secs(o.stats_log_s);
        std::thread::Builder::new()
            .name("timer".into())
            .spawn(move || {
                let mut last_log = Instant::now();
                let mut prev: HashMap<(u32, u8), Vec<(&'static str, u64)>> = HashMap::new();
                loop {
                    std::thread::sleep(Duration::from_millis(10));
                    for i in 0..sh.n_classes() {
                        let notes = sh.cls(i).core.lock().unwrap().tick(sh.now());
                        sh.notify(notes);
                    }
                    if !every.is_zero() && last_log.elapsed() >= every {
                        last_log = Instant::now();
                        prev = log_summary(&sh, prev);
                        if let Some(c) = sh.chains.get() {
                            eprintln!("{}", c.summary());
                        }
                    }
                }
            })
            .unwrap();
    }
    if o.metrics != "off" {
        let sh = sh.clone();
        let addr = o.metrics.clone();
        std::thread::Builder::new().name("metrics".into()).spawn(move || serve_metrics(sh, addr)).unwrap();
    }
    if o.disk_dir != "off" {
        let ns_gib = if o.disk_ns_gib > 0.0 { o.disk_ns_gib } else { o.disk_gib };
        if let Some(d) = disk::DiskTier::new(&o.disk_dir, &pool, pool_bytes, o.slot_bytes, o.disk_gib, ns_gib, o.disk_min_hits,
                                             o.disk_state_min_hits, (o.hotset_gib, o.hotset_min_hits, o.hotset_stable_saves),
                                             o.disk_write_gib_day) {
            let mut d = d;
            d.codec = o.disk_codec;
            d.set_strict(o.classes.first().and_then(|c| c.strict).unwrap_or(o.disk_write_mode == "strict"), o.disk_session_resumes,
                         o.disk_session_span_s * 1000);
            if !o.classes.is_empty() {
                d.label = sh.class0_name.clone();
            }
            let d = Arc::new(d);
            d.start_load(sh.clone());
            d.flush_loop(sh.clone(), Duration::from_secs(o.disk_flush_s.max(1)), Duration::from_secs(o.hotset_save_s));
            let dd = d.clone();
            sh.core.lock().unwrap().set_elsewhere(Box::new(move |ns, k| dd.has(ns, k)));
            let _ = disk::DISK.set(d);
        }
        for (k, c) in o.classes.iter().enumerate().skip(1) {
            let gib = c.ssd.unwrap_or(o.disk_gib);
            let ns_gib = if o.disk_ns_gib > 0.0 { o.disk_ns_gib.min(gib) } else { gib };
            let x = &sh.extra[k - 1];
            if let Some(mut d) = disk::DiskTier::new(&format!("{}/{}", base_disk_dir, c.name), Path::new(&x.hello.pool_path), x.hello.pool_bytes,
                                                     c.slot, gib, ns_gib, o.disk_min_hits, o.disk_state_min_hits,
                                                     (o.hotset_gib, o.hotset_min_hits, o.hotset_stable_saves), c.write.unwrap_or(o.disk_write_gib_day)) {
                d.cls = k;
                d.codec = o.disk_codec;
                d.label = c.name.clone();
                d.set_strict(c.strict.unwrap_or(o.disk_write_mode == "strict"), o.disk_session_resumes, o.disk_session_span_s * 1000);
                let d = Arc::new(d);
                d.start_load(sh.clone());
                d.flush_loop(sh.clone(), Duration::from_secs(o.disk_flush_s.max(1)), Duration::from_secs(o.hotset_save_s));
                let dd = d.clone();
                x.core.lock().unwrap().set_elsewhere(Box::new(move |ns, k| dd.has(ns, k)));
                let _ = x.disk.set(d);
            }
        }
    }
    // the disk tier's core class is the hot set (same rule, same files as every other SSD entry)
    let hot = if o.hotset_dir != "off" && disk::DISK.get().is_none() {
        HotSet::new(&o.hotset_dir, &pool, pool_bytes, o.slot_bytes, o.hotset_gib, o.hotset_min_hits, o.hotset_stable_saves,
                    o.hotset_rank_use.then_some(o.hotset_min_uses)).map(Arc::new)
    } else {
        None
    };
    if let Some(h) = &hot {
        h.restore(&sh);
        if o.hotset_save_s > 0 {
            let (h, sh, every) = (h.clone(), sh.clone(), o.hotset_save_s);
            std::thread::Builder::new()
                .name("hotset".into())
                .spawn(move || loop {
                    std::thread::sleep(Duration::from_secs(every));
                    h.save(&sh, None);
                })
                .unwrap();
        }
    }
    {
        let (hot, sh, ondemand_dir) = (hot.clone(), sh.clone(), (o.idle_exit_s > 0).then(|| base_pool_dir.clone()));
        std::thread::Builder::new()
            .name("signals".into())
            .spawn(move || {
                let mut sig: libc::c_int = 0;
                unsafe { libc::sigwait(&sigs, &mut sig) };
                eprintln!("signal {sig}: shutting down");
                for i in 0..sh.n_classes() {
                    if let Some(d) = sh.cls(i).disk {
                        d.flush(&sh);   // admitted entries reach the SSD tier before exit
                    }
                }
                // on-demand instances (--idle-exit-s): remove the pool / staging files. Never punch them (found by
                // kvbench race with SIGTERM restarts: 179 foreign reads): engines still map them and may be copying
                // from them; an unlinked file keeps its pages for every mapping, its memory goes with the last one.
                if let Some(dir) = &ondemand_dir {
                    remove_pools(dir);
                }
                // dynamic pool: a restart re-allocates the whole reused pool, which on hugetlbfs
                // needs the pages the controller returned to the kernel; restore the reservation first (the next start
                // fills the holes). A crash skips this: ensure_hugepages() at startup covers that case.
                {
                    let p = sh.pool.lock().unwrap();
                    if p.hp_ctl && p.freed_pages > 0 {
                        let total = hugepages("HugePages_Total:").unwrap_or(0);
                        match set_nr_hugepages(total + p.freed_pages) {
                            Ok(()) => eprintln!("pool: restored {} huge pages to the reservation ({} -> {})", p.freed_pages, total, total + p.freed_pages),
                            Err(e) => eprintln!("pool: could not restore {} huge pages ({e}); the next start will try", p.freed_pages),
                        }
                    }
                }
                // chains snapshot in parallel with the hot-set save; both well inside TimeoutStopSec=180
                let chains = sh.chains.get().filter(|c| c.snapshot_path().is_some()).cloned().map(|c| {
                    let (tx, rx) = std::sync::mpsc::channel();
                    std::thread::spawn(move || {
                        let _ = tx.send(c.save_now());
                    });
                    rx
                });
                let t = Instant::now();
                if let Some(h) = hot {
                    h.save(&sh, Some(Duration::from_secs(60)));
                }
                if let Some(rx) = chains {
                    // an unfinished snapshot leaves the previous file in place (atomic rename)
                    match rx.recv_timeout(Duration::from_secs(90).saturating_sub(t.elapsed()).max(Duration::from_secs(5))) {
                        Ok(Some(Ok(d))) => eprintln!("chains: snapshot saved at shutdown ({:.2} s)", d.as_secs_f64()),
                        Ok(Some(Err(e))) => eprintln!("chains: shutdown snapshot failed: {e}"),
                        Ok(None) => {}
                        Err(_) => eprintln!("chains: shutdown snapshot not finished in time; keeping the previous one"),
                    }
                }
                std::process::exit(0)
            })
            .unwrap();
    }
    // --idle-exit-s N: exit after N s without any connection (socket activation restarts it on the next connect); the
    // pool and staging files in --pool-dir are removed so an idle dev broker holds no memory
    let open_conns = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let last_close = Arc::new(Mutex::new(Instant::now()));
    if sh.tracer.is_some() {
        // operation trace (trace.rs): dump on an anomaly; drop the ring a minute after the last connection closed
        let (oc, lc, shx) = (open_conns.clone(), last_close.clone(), sh.clone());
        std::thread::Builder::new().name("trace-watch".into()).spawn(move || {
            let sums = |sh: &Shared| (0..sh.n_classes()).fold((0u64, 0u64), |a, i| match sh.cls(i).core.lock() {
                Ok(c) => (a.0 + c.c.seg_runs_skipped, a.1 + c.c.cksum_corrupt),
                Err(_) => a,
            });
            let (mut skipped, mut corrupt) = sums(&shx);
            let mut panics = 0u64;
            let mut window: std::collections::VecDeque<(Instant, u64)> = std::collections::VecDeque::new();
            loop {
                std::thread::sleep(Duration::from_secs(10));
                let tr = shx.tracer.as_ref().unwrap();
                if sh_clients(&oc, &shx) == 0
                    && lc.lock().unwrap_or_else(|e| e.into_inner()).elapsed() >= Duration::from_secs(60) {
                    tr.clear();
                }
                let (s, c) = sums(&shx);
                let p = shx.handler_panics.load(Relaxed);
                window.push_back((Instant::now(), s.saturating_sub(skipped)));
                while window.front().is_some_and(|w| w.0.elapsed() > Duration::from_secs(60)) {
                    window.pop_front();
                }
                let burst: u64 = window.iter().map(|w| w.1).sum();
                let reason = if c > corrupt {
                    Some("cksum-corrupt")
                } else if p > panics {
                    Some("handler-panic")
                } else if burst >= 64 {
                    window.clear();
                    Some("seg-skip-burst")
                } else {
                    None
                };
                (skipped, corrupt, panics) = (s, c, p);
                if let Some(r) = reason {
                    let _ = shx.trace_dump(r, true);
                }
            }
        }).unwrap();
        let t = sh.tracer.as_ref().unwrap();
        eprintln!("operation trace: ring {} MiB, dumps to {} on an anomaly or TRACE_DUMP (at most {} a day)",
                  o.trace_mib, o.trace_dir, o.trace_dumps_day);
        let _ = t;
    }
    if o.idle_exit_s > 0 {
        let (oc, lc, dir, idle, shx) = (open_conns.clone(), last_close.clone(), base_pool_dir.clone(), Duration::from_secs(o.idle_exit_s), sh.clone());
        // --idle-exit-max-s 0 (heat policy): never exit while the pool holds entries (they stay until RAM is
        // wanted); otherwise exit at that age at the latest
        let idle_max = (o.idle_exit_max_s > 0).then(|| Duration::from_secs(o.idle_exit_max_s.max(o.idle_exit_s)));
        std::thread::Builder::new().name("idle-exit".into()).spawn(move || loop {
            std::thread::sleep(Duration::from_secs(5));
            let since = lc.lock().unwrap().elapsed();
            // on-demand pool (shrinks over about an hour): exit once the gradual eviction emptied the pool, or
            // after --idle-exit-max-s at the latest
            let empty = (0..shx.n_classes()).all(|i| { let c = shx.cls(i); c.backing.is_none() || c.core.lock().unwrap().live_and_high_water().0 == 0 });
            if sh_clients(&oc, &shx) == 0 && since >= idle && (empty || idle_max.is_some_and(|m| since >= m)) {
                eprintln!("idle for {} s without connections: exiting", idle.as_secs());
                for i in 0..shx.n_classes() {
                    if let Some(d) = shx.cls(i).disk {
                        d.flush(&shx);   // the cache lives on on the SSD
                    }
                }
                remove_pools(&dir);
                std::process::exit(0);
            }
        }).unwrap();
    }
    if o.watchdog_s > 0 {
        // lock watchdog (after a lock-order deadlock stalled the engine): a prober takes and releases every
        // core, pool-backing and pool-controller lock once a second; if a round has not finished for --watchdog-s (a
        // deadlock, or a lock poisoned by a panic), the broker exits non-zero and systemd restarts it. The pool file is
        // reused at the same size, so engines reconnect by themselves (the RAM index starts empty).
        let last_ok = Arc::new(AtomicU64::new(unix_ms()));
        let (shp, lp) = (sh.clone(), last_ok.clone());
        std::thread::Builder::new().name("lock-probe".into()).spawn(move || loop {
            for i in 0..shp.n_classes() {
                let c = shp.cls(i);
                drop(c.core.lock().unwrap());
                if let Some(bk) = c.backing {
                    drop(bk.lock().unwrap());
                }
            }
            drop(shp.pool.lock().unwrap());
            lp.store(unix_ms(), Relaxed);
            std::thread::sleep(Duration::from_secs(1));
        }).unwrap();
        let limit_ms = o.watchdog_s * 1000;
        std::thread::Builder::new().name("watchdog".into()).spawn(move || loop {
            std::thread::sleep(Duration::from_secs(1));
            let age = unix_ms().saturating_sub(last_ok.load(Relaxed));
            if age > limit_ms {
                eprintln!("watchdog: an internal lock has been held for {:.0} s (deadlock or poisoned lock): exiting so \
                           systemd restarts the broker", age as f64 / 1000.0);
                std::process::exit(70);
            }
        }).unwrap();
        eprintln!("lock watchdog: exit after {} s without a full lock round", o.watchdog_s);
    }
    if let Ok(ms) = std::env::var("LLMT_KV_TEST_HANG_MS").map(|v| v.parse::<u64>().unwrap_or(0)) {
        // test hook (watchdog / client tests): after `ms`, hold the first pool's core lock for good, as a deadlock does
        // LLMT_KV_TEST_HANG_ONCE=<file>: only if the file does not exist yet (created now), so a restarted broker runs
        let once = std::env::var("LLMT_KV_TEST_HANG_ONCE").ok();
        let armed = match &once {
            Some(f) => fs::OpenOptions::new().write(true).create_new(true).open(f).is_ok(),
            None => true,
        };
        if ms > 0 && armed {
            let shh = sh.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(ms));
                eprintln!("TEST: holding the core lock (LLMT_KV_TEST_HANG_MS)");
                let _g = shh.core.lock().unwrap();
                loop {
                    std::thread::sleep(Duration::from_secs(3600));
                }
            });
        }
    }
    if lazy_pool {
        let (oc, lc, shx) = (open_conns.clone(), last_close.clone(), sh.clone());
        let cfg = lazy::SweepCfg {
            every: Duration::from_secs(o.sweep_s.max(1)),
            age_ms: o.evict_age_s * 1000,
            step_pct: o.evict_step_pct,
            idle_shrink: Duration::from_secs(o.idle_shrink_s),
            reserve: (o.mem_reserve_gib * (1u64 << 30) as f64) as u64,
            compact_bytes: if o.scrub != "off" { 0 } else { o.compact_mib << 20 },
            inactive_div: o.inactive_div,
            protect_bytes: (o.protect_gib.max(0.0) * (1u64 << 30) as f64) as u64,
            protect_window_ms: o.protect_window_s * 1000,
            protect_floor: (o.protect_floor_gib.max(0.0) * (1u64 << 30) as f64) as u64,
        };
        eprintln!("pool sweep: every {} s; evict entries unused for {} s ({}% of live per sweep), all after {} s without clients, \
                   under memory pressure below {:.1} GiB; compaction {} MiB per sweep",
                  o.sweep_s, o.evict_age_s, o.evict_step_pct, o.idle_shrink_s, o.mem_reserve_gib, cfg.compact_bytes >> 20);
        if cfg.protect_bytes > 0 {
            eprintln!("pressure protection: sessions used in the last {} s + the shared class, up to {:.1} GiB; gives way \
                       below {:.1} GiB MemAvailable", o.protect_window_s, o.protect_gib, o.protect_floor_gib);
        }
        if o.cksum_mib_s > 0 && !o.bg_worker {
            let shc = sh.clone();
            std::thread::Builder::new().name("kv-cksum".into()).spawn(move || lazy::checksum_loop(&shc)).unwrap();
            eprintln!("integrity checksums: idle, at most {} MiB/s, re-verified after {} s", o.cksum_mib_s, o.cksum_reverify_s);
        }
        if o.warm_mib > 0 {
            // allocation ahead: the broker's own memory for new entries (see lazy::Backing::keep_warm)
            for i in 0..sh.n_classes() {
                if sh.cls(i).backing.is_some() {
                    sh.cls(i).core.lock().unwrap().set_alloc_low_first(true);
                }
            }
            let (shw, wb, act, res) = (sh.clone(), o.warm_mib << 20, o.evict_age_s.max(600) * 1000, cfg.reserve);
            eprintln!("warm margin: {} MiB per pool allocated ahead by the broker while a client stores; lowest slots first",
                      o.warm_mib);
            std::thread::Builder::new().name("warm".into()).spawn(move || loop {
                std::thread::sleep(Duration::from_millis(200));
                lazy::warm(&shw, wb, act, res);
            }).unwrap();
        }
        std::thread::Builder::new().name("sweep".into()).spawn(move || {
            unsafe {
                libc::setpriority(libc::PRIO_PROCESS, 0, 10);
            }
            let mut last = Instant::now();
            loop {
                std::thread::sleep(Duration::from_secs(1));
                // memory pressure is checked every second, everything else every sweep period
                let idle = !cfg.idle_shrink.is_zero() && sh_clients(&oc, &shx) == 0
                    && lc.lock().unwrap().elapsed() >= cfg.idle_shrink;
                if last.elapsed() >= cfg.every || (cfg.reserve > 0 && mem_available() < cfg.reserve) {
                    last = Instant::now();
                    lazy::sweep(&shx, &cfg, idle);
                }
            }
        }).unwrap();
    }
    let mut next_conn: ConnId = 1;
    for s in listener.incoming() {
        match s {
            Ok(s) => {
                let sh = sh.clone();
                let conn = next_conn;
                next_conn += 1;
                let (oc, lc) = (open_conns.clone(), last_close.clone());
                oc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                std::thread::Builder::new().name(format!("conn{conn}")).spawn(move || {
                    // a handler panic must not skip this bookkeeping: the connection would count as open
                    // forever and the idle exit never come (ConnGuard has already cleaned the connection up)
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| serve_conn(sh, conn, s)));
                    *lc.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
                    oc.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                }).unwrap();
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
}
