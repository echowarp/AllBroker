//! Broker core: index, slot allocation, reservations, leases, chain-aware eviction, waiters.
//! No I/O and no clock: callers pass `now` in milliseconds, so every race is a deterministic test.

pub mod alloc;
pub mod hints;
pub mod policy;
mod seg;
mod hot;
pub use hot::{HotRel, HotStats};

use kvproto::{EntryDesc, Key, Range, ReserveItem, Status, CLASS_LOW, CLASS_LOWEST, ENTRY_EPHEMERAL, ENTRY_LOW, ENTRY_LOWEST,
              MAX_META, TAG_TERMINAL};

/// Value class rank of RESERVE flags: 0 normal, 1 low (ENTRY_LOW), 2 lowest (ENTRY_LOWEST).
pub(crate) fn rank_of(flags: u8) -> u8 {
    if flags & ENTRY_LOWEST != 0 { 2 } else if flags & ENTRY_LOW != 0 { 1 } else { 0 }
}

fn with_rank(flags: u8, r: u8) -> u8 {
    (flags & !(ENTRY_LOW | ENTRY_LOWEST)) | match r { 0 => 0, 1 => ENTRY_LOW, _ => ENTRY_LOWEST }
}

/// Rank a TOUCH_TAGGED class raises entries to (None = a plain TOUCH / TOUCH_REQUEST: normal).
fn class_rank(class: Option<u8>) -> u8 {
    match class { Some(CLASS_LOWEST) => 2, Some(CLASS_LOW) => 1, _ => 0 }
}
use policy::{EntryId, Policy, PolicyKind};
use std::collections::{HashMap, HashSet};

pub type ConnId = u64;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_slots: u32,
    pub slot_bytes: u64,
    pub reserve_timeout_ms: u64,
    pub lease_timeout_ms: u64,
    /// Background eviction keeps at least this many slots free.
    pub low_water_slots: u64,
    pub policy: PolicyKind,
    /// Shared-pool fairness (several engines on one pool; 0 = off):
    /// entries never hit and older than this go first, oldest first, from any namespace ...
    pub stale_unhit_ms: u64,
    /// ... then policy order, but a namespace holding <= this many slots is not evicted from ...
    pub ns_floor_slots: u64,
    /// ... and a namespace holding >= this many slots evicts from itself first when it stores (0 = no cap).
    pub ns_cap_slots: u64,
    /// Partial-tail entries (KIND_TAIL, Arc tail-hold spill): they age this many times faster than other
    /// entries in every age / heat order (gradual sweep, RAM ceiling, memory pressure), and RESERVE evicts them before
    /// anything else. No TTL: they leave only by that order. 1 = treated like any entry.
    pub tail_age_div: u64,
    /// heat (cull only when RAM must be cleared, by heat): each hit counts as
    /// this many ms of extra recency, up to HEAT_MAX_HITS hits (0 = plain recency)
    pub heat_hit_ms: u64,
    /// Shared-prefix class (system prompts / tool definitions reused across sessions are held long):
    /// an entry that two different chains continue from (the entry-metadata parent link of their next chunks), its
    /// ancestors, and entries reserved with the ENTRY_SHARED_HINT flag, up to this many slots (0 = off; the most
    /// recently used first beyond it), are evicted only after every other entry and age `shared_age_mult` times
    /// slower.
    pub shared_slots: u64,
    pub shared_age_mult: u64,
    /// a prefix is shared once this many different chains continue from it (a single regenerate or
    /// retry of one chat must not qualify; default 3)
    pub shared_min_cont: u8,
    /// shared prefixes are at most this many chained entries deep from their root (the chat beginning cap, ~32k
    /// tokens = 19 chunks of 1,664; 0 = no cap)
    pub shared_max_depth: u32,
    /// Segment-wise eviction (seg.rs): a GDN state leaves together with the attention only it makes
    /// usable; attention no state can use and states whose chain is gone are evicted first. false = per-entry order.
    pub seg_evict: bool,
    /// ... uncovered attention is kept this long after its last use / store (an engine stores attention ahead of the
    /// prompt-end state); orphan states likewise after their chain broke.
    pub seg_grace_ms: u64,
    /// Weights entries (KIND_WEIGHTS) are valued per slot against KV: one slot of KV is taken to cost this many ms to
    /// recompute, a weights entry `cost_hint` ms (its reload from disk) for all its slots. Its idle time counts
    /// `weights_kv_slot_ms x n_slots / cost_hint` times in every age / heat order (faster when it is worth less per
    /// slot than KV, slower when more), and RESERVE / pressure take it before the policy's KV victim once it is
    /// colder than that victim. 0, or a cost_hint of 0, = plain recency like KV.
    pub weights_kv_slot_ms: u64,
}

/// ReserveItem flag bit 1: the client knows this entry is a shared prefix (e.g. a junction several sessions use).
/// Clients without a flags field (the Arc's PyO3 build) send the same hint as entry metadata byte 32 bit 0 of an
/// "L1" record (META_V1 + 16 B parent + 8 B fingerprint + 6 B model id + this byte). A reserve that finds the entry
/// already committed applies the hint to it.
pub const ENTRY_SHARED_HINT: u8 = 2;

fn shared_hint(flags: u8, meta: &[u8]) -> bool {
    flags & ENTRY_SHARED_HINT != 0 || (meta.len() > 32 && &meta[..2] == b"L1" && meta[32] & 1 != 0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Reserved { conn: ConnId, deadline: u64 },
    Committed,
}

struct Entry {
    ns: u32,
    key: Key,
    parent: Option<EntryId>,
    children: HashSet<EntryId>,
    state: State,
    ranges: Vec<Range>,
    n_slots: u32,
    bytes: u64,
    kind: u8,
    format: u16,
    producer: u16,
    cost_hint: u32,
    flags: u8,
    ttl_ms: u32,
    /// Ephemeral entries: evict first after this time (set at commit) or once a lease on them ends.
    expires: Option<u64>,
    meta: Vec<u8>,
    leases: u32,
    hits: u64,
    last_used: u64,
    /// Lifecycle (ms on the broker clock): committed at, first and last lookup hit (0 = never), engines that hit it.
    born: u64,
    first_hit: u64,
    last_hit: u64,
    hit_engines: u32,
    /// The RECENT_HITS most recent lookup hits: (ms, engine id), oldest first.
    recent: std::collections::VecDeque<(u64, u16)>,
    /// In-place slot compression: the broker is rewriting the slot (lookups miss it meanwhile, it is not
    /// evictable); the slot holds the compressed image.
    rewriting: bool,
    compressed: bool,
    /// shared-prefix class member (counted in `Core::shared_now`)
    shared: bool,
    /// ghost keys of the first different chains continuing from this entry (0 = none yet)
    succ: [u64; 4],
    /// chunks from the chain's root (parent depth + 1; 0 = no parent): the reuse histograms' depth bucket
    depth: u16,
    /// session: the touch-batch chain this entry belongs to (0 = none known)
    sess: u64,
    /// Integrity checksum (for hosts without ECC): XXH3-64 of the entry's raw bytes (decompressed if the slot is
    /// compressed), computed and re-verified by the broker as idle work; None until first computed. `sum_at`: when
    /// it was computed / last verified (broker clock).
    sum: Option<u64>,
    sum_at: u64,
    /// use by distinct chains (TOUCH), decayed: the persistent hot set's rank (hot.rs)
    use_: hot::UseStat,
}

/// Lookup hits kept per entry (a running list of the most recent ones).
pub const RECENT_HITS: usize = 10;

impl Entry {
    fn evictable(&self) -> bool {
        self.state == State::Committed && self.leases == 0 && self.children.is_empty() && !self.rewriting
    }
}

struct Lease {
    conn: ConnId,
    entries: Vec<EntryId>,
    deadline: u64,
}

struct Waiter {
    conn: ConnId,
    req_id: u64,
    deadline: u64,
}

#[derive(Default)]
struct Conn {
    engine_id: u16,
    hello: bool,
    leases: HashSet<u64>,
    reservations: HashSet<EntryId>,
}

/// A deferred reply (WAIT) the server must send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Notify {
    pub conn: ConnId,
    pub req_id: u64,
    pub status: Status,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Removal {
    Evicted,     // to make room in RESERVE
    LowWater,    // background eviction
    Expired,     // ephemeral TTL / released
    Aborted,     // ABORT (the engine's copy is over: failed store / NaN scan)
    WriterGone,  // the writer's socket closed with the reservation open (its copy may still be running)
    ResTimeout,  // reservation timed out
    Corrupt,     // integrity scrub found bad values
    Culled,      // dynamic pool: evicted to park slots and return their memory to the kernel
    Orphan,      // segment eviction: attention no state can use, a state whose chain is gone
    TornRewrite, // the background worker went away mid-rewrite (its slot quarantined: it may still write)
    Cascade,     // the chain after an attention block evicted for any reason, which no lookup can reach any
                 // more
    Invalidated, // INVALIDATE: an engine found the content bad (NaN from a state it restored)
}

/// Hits that count towards heat (Config::heat_hit_ms each).
pub const HEAT_MAX_HITS: u64 = 8;

/// How long a timed-out / aborted reservation's slots stay out of the allocator.
pub const QUARANTINE_MS: u64 = 60_000;

/// Heat buckets: time since last use (ms upper bound, label); the last one is "demoted" (HEAT cold).
pub const HEAT_AGE: [(u64, &str); 7] = [(60_000, "1m"), (600_000, "10m"), (3_600_000, "1h"), (21_600_000, "6h"),
    (86_400_000, "1d"), (u64::MAX, "older"), (0, "demoted")];

/// Outcome of [`Core::checksum_result`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChecksumResult {
    Stored,
    Verified,
    /// mismatch / undecodable: evicted as corrupt
    Corrupt,
    /// the entry changed while it was copied
    Raced,
    /// mismatch but leased or being rewritten: next pass
    Busy,
}

#[derive(Default, Clone, Debug)]
pub struct Counters {
    /// integrity checksums: stored, re-verified, corrupt entries evicted
    pub cksum_stored: u64,
    pub cksum_verified: u64,
    pub cksum_corrupt: u64,
    /// entries evicted because an attention block before them in their chain was
    pub cascade_evicted: u64,
    /// in-place rewrites whose writer went away mid-rewrite: entry evicted
    pub torn_rewrites: u64,
    /// HEAT entries refreshed / demoted
    pub heat_hot: u64,
    pub heat_cold: u64,
    pub lookups: u64,
    pub lookup_keys: u64,
    pub hit_entries: u64,
    pub hit_bytes: u64,
    pub cross_engine_hit_entries: u64,
    pub full_hits: u64,
    pub partial_hits: u64,
    pub misses: u64,
    pub reserved: u64,
    pub committed: u64,
    pub committed_bytes: u64,
    pub exists: u64,
    pub busy: u64,
    pub full: u64,
    pub no_parent: u64,
    pub evict_room: u64,
    pub evict_low_water: u64,
    pub evict_expired: u64,
    pub evict_culled: u64,
    pub aborted: u64,
    pub reservation_timeouts: u64,
    pub lease_timeouts: u64,
    pub leases_closed_by_disconnect: u64,
    pub wait_timeouts: u64,
    /// Entries the integrity scrub removed.
    pub corrupt: u64,
    /// entries removed by INVALIDATE (with the chain after them)
    pub invalidated: u64,
    /// Compaction moves (lazily backed pool).
    pub relocations: u64,
    /// Entries compressed in place.
    pub compressed: u64,
    /// Shared-prefix class members demoted to normal entries by its slot budget.
    pub shared_demoted: u64,
    /// Segment eviction: orphans evicted (attention entries / GDN state entries), segments of 2+ entries
    /// evicted and their entries, touch runs linked / skipped.
    pub evict_orphan_attn: u64,
    pub evict_orphan_state: u64,
    pub seg_units: u64,
    pub seg_unit_entries: u64,
    pub seg_runs_linked: u64,
    pub seg_runs_skipped: u64,
    /// why: run too long / empty, group count not learned yet, length not a multiple of it, a batch
    /// naming the other kind (cut at the wrong place), no single attention batch / a state batch longer than it
    pub seg_skip_over: u64,
    pub seg_skip_shape: u64,
    pub seg_skip_len: u64,
    pub seg_skip_kind: u64,
    pub seg_skip_layout: u64,
    /// TOUCH_REQUEST: requests linked from their group tags / only touched (not a hybrid layout)
    pub seg_tagged_linked: u64,
    pub seg_tagged_unlinked: u64,
    /// TOUCH keys / of them not in RAM (a block only in an engine's VRAM, or not stored); GDN states of a hot-set
    /// boundary refreshed to their anchor block's last use
    pub touch_keys: u64,
    pub touch_missing_keys: u64,
    pub hot_state_refreshed: u64,
    /// Request classes (TOUCH_TAGGED, RESERVE flag ENTRY_LOW): low-class entries committed, raised to normal by a
    /// non-low use, evicted (room / low water) and other entries evicted likewise; victim picks that took a low-class
    /// unit instead of the policy's choice; tagged touches (of them low), sessions named by a tag
    pub low_committed: u64,
    pub low_promoted: u64,
    pub low_evicted: u64,
    pub normal_evicted: u64,
    pub low_picked: std::cell::Cell<u64>,
    pub tagged_touches: u64,
    pub tagged_low_touches: u64,
    pub tagged_sessions: u64,
    /// lowest-class entries committed / evicted, entries demoted to lowest by a session's terminal request
    pub lowest_committed: u64,
    pub lowest_evicted: u64,
    pub terminal_demoted: u64,
}

/// One line per committed entry when it leaves the pool (the broker writes them to a JSONL file): hashes and times
/// only. Times in ms on the broker clock (since its start); 0 = never.
#[derive(Clone, Debug)]
pub struct EntryRecord {
    pub ns: u32,
    pub kind: u8,
    pub producer: u16,
    pub bytes: u64,
    pub key_prefix: [u8; 8],
    pub meta: Vec<u8>,
    pub born: u64,
    pub first_hit: u64,
    pub last_hit: u64,
    pub left: u64,
    pub hits: u64,
    pub hit_engines: u32,
    /// Up to RECENT_HITS most recent hits: (ms, engine id), oldest first.
    pub recent: Vec<(u64, u16)>,
    pub reason: &'static str,
}

/// One line of the event journal (to trace back which hits happened in what order): every store,
/// lookup and removal, in order. Times in ms on the broker clock.
#[derive(Clone, Debug)]
pub enum Event {
    /// Committed entry.
    Store { t: u64, ns: u32, kind: u8, engine: u16, key: [u8; 8], bytes: u64, meta: Vec<u8> },
    /// Lookup: asked `n_keys` (first key's prefix), found the first `n_hit` of them.
    Lookup { t: u64, ns: u32, engine: u16, key: [u8; 8], n_keys: u32, n_hit: u32, acquire: bool },
    /// A committed entry left the pool (with its life record).
    Remove(EntryRecord),
    /// Content fingerprint of a committed entry, one hash per page (per KV layer tensor), from the integrity scrub:
    /// two runs of the same prompt can be compared block by block and layer by layer.
    Content { t: u64, ns: u32, key: [u8; 8], kind: u8, slot: u32, pages: Vec<u64> },
}

fn kp(k: &Key) -> [u8; 8] {
    let mut x = [0u8; 8];
    x.copy_from_slice(&k[..8]);
    x
}

/// An entry selected for the persistent hot set (bytes are read from `ranges` by the caller).
#[derive(Clone, Debug)]
pub struct HotEntry {
    pub ns: u32,
    pub key: Key,
    pub kind: u8,
    pub format: u16,
    pub producer: u16,
    pub meta: Vec<u8>,
    pub bytes: u64,
    pub hits: u64,
    pub last_used: u64,
    pub ranges: Vec<Range>,
    /// member of the shared-prefix class (SSD: admitted at once, exempt from the daily write budget)
    pub shared: bool,
}

/// Seconds buckets for the lifecycle histograms (store -> first hit, hit -> next hit, age at eviction).
pub const LIFE_BUCKETS_S: [u64; 10] = [1, 5, 15, 60, 300, 900, 3600, 4 * 3600, 24 * 3600, 7 * 24 * 3600];

/// Reuse histograms (counts only, nothing that identifies tokens or
/// blocks). Depth = chunks from the chain root (Arc: 1,664 tokens per chunk), age = since the entry was stored.
pub const REUSE_DEPTH: [(u16, &str); 5] = [(4, "0-4"), (9, "5-9"), (19, "10-19"), (39, "20-39"), (u16::MAX, "40+")];
pub const REUSE_AGE: [(u64, &str); 6] =
    [(10_000, "10s"), (60_000, "1m"), (600_000, "10m"), (3_600_000, "1h"), (86_400_000, "1d"), (u64::MAX, "1d+")];
pub const REUSE_HITS: [&str; 3] = ["0", "1", "2+"];
pub const REUSE_REASON: [&str; 6] = ["evicted", "low_water", "culled", "expired", "orphan", "cascade"];

pub fn reuse_depth_b(d: u16) -> u8 {
    REUSE_DEPTH.iter().position(|(m, _)| d <= *m).unwrap_or(4) as u8
}

pub fn reuse_age_b(ms: u64) -> u8 {
    REUSE_AGE.iter().position(|(m, _)| ms < *m).unwrap_or(5) as u8
}

#[derive(Default, Clone, Debug)]
pub struct Hist {
    pub buckets: [u64; LIFE_BUCKETS_S.len()],
    pub count: u64,
    pub sum_ms: u64,
}

impl Hist {
    fn obs(&mut self, ms: u64) {
        self.count += 1;
        self.sum_ms += ms;
        for (i, b) in LIFE_BUCKETS_S.iter().enumerate() {
            if ms <= b * 1000 {
                self.buckets[i] += 1;
            }
        }
    }
}

/// `ns_c` kind for counters that have no entry kind (lookups of keys that may not exist).
pub const KIND_ANY: u8 = 255;
/// Entry kind of a partial tail (attention pages past the last full block + GDN state at the resume point, written by
/// the Arc engine when a VRAM tail hold is released): evicted first, ages `tail_age_div` times faster.
pub const KIND_TAIL: u8 = 4;
/// Entry kind of cached model weights (a multimodal model's vision tower / mmproj, ~16 slots, shared by every engine
/// of the model): an ordinary RAM entry for eviction (same orders, no pinning, no TTL), valued per slot against KV
/// (`Config::weights_kv_slot_ms`, see `Core::heat`), and never persisted: not in the hot set, never written to the SSD
/// tier, never restored from either (the weights file is on disk already; the engine reloads it on a miss).
pub const KIND_WEIGHTS: u8 = 5;

/// Bucket of an entry kind in the per-kind gauges / heat histogram: kinds 0..=5 their own, any other kind counts with
/// kind 4 as before (kinds 0-4 unchanged).
fn kind_bucket(kind: u8) -> usize {
    if kind == KIND_WEIGHTS { 5 } else { (kind as usize).min(4) }
}

/// Per-namespace counters (one namespace per engine today), for the RAM-cache dashboard.
#[derive(Default, Clone, Debug)]
pub struct NsCounters {
    pub lookup_keys: u64,
    pub hit_entries: u64,
    pub hit_bytes: u64,
    pub committed: u64,
    pub committed_bytes: u64,
    pub exists: u64,
    pub full: u64,
    /// Removed to make room or by the low-water mark.
    pub evicted: u64,
    /// ... of those, never hit while stored (a wasted store).
    pub evicted_unused: u64,
    /// Sum of the hit counts of evicted entries (mean hits per stored entry = this / evicted).
    pub evicted_hits: u64,
}

/// A session (SSD writes only for sessions with proven repeated use): one chain of touch batches.
/// A request's TOUCH (its whole chain, root to end, one KV group per call) extends the session whose chain end it
/// contains; its last key becomes the new end. Counts and broker times only: no token ids or text.
#[derive(Clone, Debug, Default)]
pub struct Session {
    /// first and last use (broker ms)
    pub first_use: u64,
    pub last_use: u64,
    /// continuations after at least `sess_gap_ms` without use (a real resume, not a lookup probe)
    pub resumes: u32,
    /// the most recently committed GDN-state (kind 1) entry of the session: the only state worth keeping on SSD
    pub newest_state: Option<EntryId>,
}

/// Sessions remembered (least recently used dropped beyond).
pub const SESSIONS_MAX: usize = 8192;

pub struct Core {
    cfg: Config,
    alloc: alloc::Alloc,
    policy: Box<dyn Policy>,
    index: HashMap<(u32, Key), EntryId>,
    entries: HashMap<EntryId, Entry>,
    leases: HashMap<u64, Lease>,
    conns: HashMap<ConnId, Conn>,
    waiters: HashMap<(u32, Key), Vec<Waiter>>,
    ephemeral: HashSet<EntryId>,
    next_id: EntryId,
    next_lease: u64,
    pub c: Counters,
    /// Per (namespace, entry kind); kind KIND_ANY for lookups (a missing key has no kind).
    pub ns_c: HashMap<(u32, u8), NsCounters>,
    /// Lifecycle histograms per (name, kind): "first_hit" (commit -> first hit), "rehit" (hit -> next hit),
    /// "age" (commit -> eviction).
    pub life: HashMap<(&'static str, u8), Hist>,
    /// Reuse histograms: hits[(kind, depth bucket, age bucket)], evictions[(kind, depth, age, hits bucket, reason)].
    pub reuse_hits: HashMap<(u8, u8, u8), u64>,
    pub reuse_evict: HashMap<(u8, u8, u8, u8, u8), u64>,
    /// Records of entries that left; drained by the server (bounded: oldest dropped beyond 100k).
    pub records: Vec<EntryRecord>,
    /// Event journal, drained by the server; only filled when `log_events` (bounded: oldest dropped beyond 1M).
    pub events: Vec<Event>,
    pub log_events: bool,
    now: u64,
    /// (namespace, first 16 key bytes) -> committed entry: resolves the parent prefix in entry metadata ("L1" + 16 B)
    pfx: HashMap<(u32, [u8; 16]), EntryId>,
    /// slots of the shared-prefix class
    shared_now: u64,
    /// sessions: records, chain end -> session, chain keys -> session (for entries committed later)
    pub sessions: HashMap<u64, Session>,
    sess_by_end: HashMap<(u32, Key), u64>,
    key_sess: HashMap<(u32, Key), u64>,
    next_sess: u64,
    /// a continuation after at least this long without use counts as a resume
    pub sess_gap_ms: u64,
    /// Slots of reservations that timed out or were aborted (incl. the writer's socket closing), held back from the
    /// allocator until the given time (the engine's copy into a reserved slot can
    /// still be running when the reservation goes; reusing the slot at once let a late copy overwrite the next owner)
    quarantine: std::collections::VecDeque<(u64, Vec<Range>)>,
    quarantined: u64,
    /// free slots the broker is hole-punching outside the lock (begin_freeing): out of the allocator until the punch is done
    freeing: Vec<Range>,
    /// segment eviction state (seg.rs)
    seg: seg::Seg,
    /// "the entry is on the SSD tier" (the broker's disk index; called with the core lock held, takes the index lock:
    /// the documented lock order)
    elsewhere: Option<Box<dyn Fn(u32, &Key) -> bool + Send>>,
    /// persistent hot set by use (hot.rs): state, the use half-life, and the rank threshold (retention refresh and
    /// the touched-but-not-resident gauge use it too)
    hot: hot::Hot,
    pub use_half_life_ms: u64,
    pub hot_min_use: f64,
    /// Class order: under room pressure a low-class unit (ENTRY_LOW) goes before the policy's victim unless it is more
    /// than this much hotter (ms of heat; 0 = no class order). Ageing with room is the same for every class.
    pub low_margin_ms: u64,
    /// TOUCH_TAGGED sessions: (namespace, router session tag) -> broker session
    tag_sess: HashMap<(u32, u64), u64>,
    /// the class and session tag of the TOUCH_TAGGED being handled (None: plain TOUCH / TOUCH_REQUEST)
    touch_class: Option<u8>,
    touch_tag: Option<u64>,
}

/// Parent key prefix in the clients' entry metadata: b"L1" + 16 bytes of the previous stored chunk's key in the same
/// KV group (zeros = none); see llmt_kv.vllm_tier META_V1.
fn meta_parent(meta: &[u8]) -> Option<[u8; 16]> {
    if meta.len() < 18 || &meta[..2] != b"L1" {
        return None;
    }
    let mut p = [0u8; 16];
    p.copy_from_slice(&meta[2..18]);
    (p != [0u8; 16]).then_some(p)
}

fn ghost(ns: u32, k: &Key) -> u64 {
    // FNV-1a over namespace + key; collisions only blur ARC's ghost hits.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in ns.to_le_bytes().iter().chain(k.iter()) {
        h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

impl Core {
    pub fn new(cfg: Config) -> Core {
        Core {
            alloc: alloc::Alloc::new(cfg.n_slots),
            policy: policy::make(cfg.policy, cfg.n_slots as u64),
            cfg,
            index: HashMap::new(),
            entries: HashMap::new(),
            leases: HashMap::new(),
            conns: HashMap::new(),
            waiters: HashMap::new(),
            ephemeral: HashSet::new(),
            next_id: 1,
            next_lease: 1,
            c: Counters::default(),
            ns_c: HashMap::new(),
            life: HashMap::new(),
            reuse_hits: HashMap::new(),
            reuse_evict: HashMap::new(),
            records: Vec::new(),
            events: Vec::new(),
            log_events: false,
            now: 0,
            pfx: HashMap::new(),
            shared_now: 0,
            sessions: HashMap::new(),
            sess_by_end: HashMap::new(),
            key_sess: HashMap::new(),
            next_sess: 1,
            sess_gap_ms: 600_000,
            quarantine: std::collections::VecDeque::new(),
            quarantined: 0,
            freeing: Vec::new(),
            seg: seg::Seg::default(),
            elsewhere: None,
            hot: hot::Hot::default(),
            use_half_life_ms: 24 * 3_600_000,
            hot_min_use: 8.0,
            low_margin_ms: 1_800_000,
            tag_sess: HashMap::new(),
            touch_class: None,
            touch_tag: None,
        }
    }

    /// The SSD tier's presence check (segment eviction: a chain block or state on the SSD still counts).
    pub fn set_elsewhere(&mut self, f: Box<dyn Fn(u32, &Key) -> bool + Send>) {
        self.elsewhere = Some(f);
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    // ------------------------------------------------------------ connections

    pub fn connect(&mut self, conn: ConnId) {
        self.conns.insert(conn, Conn::default());
    }

    pub fn hello(&mut self, conn: ConnId, engine_id: u16) {
        let c = self.conns.entry(conn).or_default();
        c.engine_id = engine_id;
        c.hello = true;
    }

    pub fn has_hello(&self, conn: ConnId) -> bool {
        self.conns.get(&conn).is_some_and(|c| c.hello)
    }

    /// The socket closed: end its leases, abort its reservations, drop its waiters.
    pub fn disconnect(&mut self, conn: ConnId) {
        self.end_run(conn);
        let Some(c) = self.conns.remove(&conn) else { return };
        for l in c.leases {
            if self.end_lease(l) {
                self.c.leases_closed_by_disconnect += 1;
            }
        }
        for id in c.reservations {
            if self.entries.get(&id).is_some_and(|e| matches!(e.state, State::Reserved { conn: o, .. } if o == conn)) {
                self.remove_tree(id, Removal::WriterGone);
            }
        }
        for ws in self.waiters.values_mut() {
            ws.retain(|w| w.conn != conn);
        }
        self.waiters.retain(|_, ws| !ws.is_empty());
    }

    fn engine(&self, conn: ConnId) -> u16 {
        self.conns.get(&conn).map_or(0, |c| c.engine_id)
    }

    // ------------------------------------------------------------ lookup / touch

    /// Longest committed prefix of `keys`. With `acquire`, lease it and describe it.
    pub fn lookup(&mut self, conn: ConnId, ns: u32, keys: &[Key], acquire: bool, now: u64) -> (u32, u64, Vec<EntryDesc>) {
        self.end_run(conn);
        self.c.lookups += 1;
        self.c.lookup_keys += keys.len() as u64;
        let reader = self.engine(conn);
        let mut hit = Vec::new();
        for k in keys {
            match self.index.get(&(ns, *k)) {
                Some(&id) if self.entries[&id].state == State::Committed && !self.entries[&id].rewriting => hit.push(id),
                _ => break,
            }
        }
        self.now = self.now.max(now);
        self.ns_c.entry((ns, KIND_ANY)).or_default().lookup_keys += keys.len() as u64;
        if self.log_events && !keys.is_empty() {
            let ev = Event::Lookup { t: now, ns, engine: reader, key: kp(&keys[0]), n_keys: keys.len() as u32,
                                     n_hit: hit.len() as u32, acquire };
            self.push_event(ev);
        }
        for id in &hit {
            let e = self.entries.get_mut(id).unwrap();
            if e.first_hit == 0 {
                e.first_hit = now.max(1);
                self.life.entry(("first_hit", e.kind)).or_default().obs(now.saturating_sub(e.born));
            } else {
                self.life.entry(("rehit", e.kind)).or_default().obs(now.saturating_sub(e.last_hit));
            }
            e.last_hit = now.max(1);
            e.hit_engines |= 1u32 << (reader % 32);
            *self.reuse_hits.entry((e.kind, reuse_depth_b(e.depth), reuse_age_b(now.saturating_sub(e.born)))).or_default() += 1;
            if e.recent.len() == RECENT_HITS {
                e.recent.pop_front();
            }
            e.recent.push_back((now, reader));
            let e = &self.entries[id];
            let n = self.ns_c.entry((ns, e.kind)).or_default();
            n.hit_entries += 1;
            n.hit_bytes += e.bytes;
        }
        match hit.len() {
            0 => self.c.misses += 1,
            n if n == keys.len() => self.c.full_hits += 1,
            _ => self.c.partial_hits += 1,
        }
        let mut descs = Vec::new();
        for &id in &hit {
            let e = self.entries.get_mut(&id).unwrap();
            e.hits += 1;
            e.last_used = now;
            self.c.hit_entries += 1;
            self.c.hit_bytes += e.bytes;
            if e.producer != reader {
                self.c.cross_engine_hit_entries += 1;
            }
            if acquire {
                e.leases += 1;
                descs.push(EntryDesc {
                    bytes: e.bytes,
                    kind: e.kind,
                    format: e.format,
                    producer: e.producer,
                    flags: e.flags,
                    meta: e.meta.clone(),
                    ranges: e.ranges.clone(),
                });
            }
            self.policy.access(id);
        }
        self.note_hit_use(conn, ns, keys, &hit, now);
        let mut lease_id = 0;
        if acquire && !hit.is_empty() {
            lease_id = self.next_lease;
            self.next_lease += 1;
            self.leases.insert(lease_id, Lease { conn, entries: hit.clone(), deadline: now + self.cfg.lease_timeout_ms });
            if let Some(c) = self.conns.get_mut(&conn) {
                c.leases.insert(lease_id);
            }
        }
        (hit.len() as u32, lease_id, descs)
    }

    /// HEAT: retention only, no session bookkeeping. hot: refresh as TOUCH does; cold: least recently used
    /// and first in the policy order (shared-class entries are left alone: other sessions start with them).
    pub fn heat_keys(&mut self, ns: u32, keys: &[Key], hot: bool, now: u64) -> Vec<Status> {
        keys.iter()
            .map(|k| match self.index.get(&(ns, *k)) {
                Some(&id) if self.entries[&id].state == State::Committed => {
                    let e = self.entries.get_mut(&id).unwrap();
                    if hot {
                        e.last_used = now;
                        self.policy.access(id);
                        self.c.heat_hot += 1;
                    } else if !e.shared {
                        e.last_used = 0;
                        self.policy.demote(id);
                        self.c.heat_cold += 1;
                    }
                    Status::Ok
                }
                _ => Status::NotFound,
            })
            .collect()
    }

    pub fn touch(&mut self, ns: u32, keys: &[Key], now: u64) -> Vec<Status> {
        // a use raises an entry of a lower class to the request's class (normal for plain TOUCH)
        let rank = class_rank(self.touch_class);
        let st = keys.iter()
            .map(|k| match self.index.get(&(ns, *k)) {
                Some(&id) if self.entries[&id].state == State::Committed => {
                    let e = self.entries.get_mut(&id).unwrap();
                    e.last_used = now;
                    if rank_of(e.flags) > rank {
                        e.flags = with_rank(e.flags, rank);
                        self.c.low_promoted += 1;
                    }
                    self.policy.access(id);
                    Status::Ok
                }
                _ => Status::NotFound,
            })
            .collect();
        self.note_session(ns, keys, now);
        self.note_use(ns, keys, now);
        st
    }

    /// Session bookkeeping for one TOUCH batch (a request's chain in one KV group, root to end): the deepest key that
    /// ended an earlier batch names the session it continues (a resume if it was unused for `sess_gap_ms`); else a new
    /// session. The batch's last key becomes a chain end of the session (earlier ends stay: a branch from an older
    /// point carries the counts too); every key of the batch is mapped to the session for entries committed later.
    fn note_session(&mut self, ns: u32, keys: &[Key], now: u64) {
        let Some(end) = keys.last().copied() else { return };
        // a tagged request names its session (TOUCH_TAGGED): no chain-end inference; its first request opens it
        let tagged = self.touch_tag.map(|t| (t, self.tag_sess.get(&(ns, t)).copied().filter(|s| self.sessions.contains_key(s))));
        let prev = match tagged {
            Some((_, s)) => s,
            None => keys.iter().rev().find_map(|k| self.sess_by_end.get(&(ns, *k)).copied())
                .filter(|s| self.sessions.contains_key(s)),
        };
        let sid = match prev {
            Some(s) => {
                let gap = self.sess_gap_ms;
                let r = self.sessions.get_mut(&s).unwrap();
                if now.saturating_sub(r.last_use) >= gap {
                    r.resumes += 1;
                }
                r.last_use = r.last_use.max(now);
                s
            }
            None => {
                let s = self.next_sess;
                self.next_sess += 1;
                self.sessions.insert(s, Session { first_use: now, last_use: now, ..Default::default() });
                if let Some((t, _)) = tagged {
                    self.tag_sess.insert((ns, t), s);
                    self.c.tagged_sessions += 1;
                }
                s
            }
        };
        self.sess_by_end.insert((ns, end), sid);
        for k in keys {
            self.key_sess.insert((ns, *k), sid);
            if let Some(&id) = self.index.get(&(ns, *k)) {
                self.assign_session(id, sid);
            }
        }
        if self.sessions.len() > SESSIONS_MAX {
            let mut v: Vec<(u64, u64)> = self.sessions.iter().map(|(s, r)| (r.last_use, *s)).collect();
            v.sort_unstable();
            for (_, s) in v.iter().take(self.sessions.len() - SESSIONS_MAX * 3 / 4) {
                self.sessions.remove(s);
            }
            let live: HashSet<u64> = self.sessions.keys().copied().collect();
            self.sess_by_end.retain(|_, s| live.contains(s));
            self.key_sess.retain(|_, s| live.contains(s));
            self.tag_sess.retain(|_, s| live.contains(s));
        }
    }

    fn assign_session(&mut self, id: EntryId, sid: u64) {
        let (kind, born) = {
            let e = self.entries.get_mut(&id).unwrap();
            e.sess = sid;
            (e.kind, e.born)
        };
        if kind == 1 {
            let newer = match self.sessions.get(&sid).and_then(|r| r.newest_state) {
                Some(o) => self.entries.get(&o).is_none_or(|x| x.born <= born),
                None => true,
            };
            if newer {
                if let Some(r) = self.sessions.get_mut(&sid) {
                    r.newest_state = Some(id);
                }
            }
        }
    }

    /// The session of entry `id` has proven repeated use: at least `min_resumes` resumes and an active span (first
    /// to last use) of at least `min_span_ms`.
    pub fn session_proven(&self, id: EntryId, min_resumes: u32, min_span_ms: u64) -> bool {
        let Some(e) = self.entries.get(&id) else { return false };
        self.sessions.get(&e.sess).is_some_and(|r| r.resumes >= min_resumes && r.last_use.saturating_sub(r.first_use) >= min_span_ms)
    }

    /// Strict SSD rule: the entries about to leave RAM by age (the next evictable leaves of the
    /// gradual sweep, least recently used first, at most `max_slots`) that are worth an SSD copy: their session has
    /// proven repeated use, and a GDN state only if it is its session's newest one (superseded states never). Shared
    /// beginnings are excluded here (the regular pass writes them). Returns (ns, key) of each.
    pub fn strict_age_candidates(&self, before: &dyn Fn(u32) -> u64, max_slots: u64, min_resumes: u32, min_span_ms: u64)
                                 -> Vec<(u32, Key)> {
        if self.cfg.seg_evict {
            return self.seg_strict_age_candidates(before, max_slots, min_resumes, min_span_ms);
        }
        let mut v: Vec<(i64, EntryId, u64)> = self.entries.iter()
            .filter(|(_, e)| e.evictable() && self.heat(e) < i64::try_from(before(e.ns)).unwrap_or(i64::MAX))
            .map(|(id, e)| (self.heat(e), *id, e.n_slots as u64))
            .collect();
        v.sort_unstable();
        let mut out = Vec::new();
        let mut slots = 0u64;
        for (_, id, n) in v {
            if max_slots > 0 && slots >= max_slots {
                break;
            }
            slots += n;
            if let Some(x) = self.strict_ok(id, min_resumes, min_span_ms) {
                out.push(x);
            }
        }
        out
    }

    fn strict_ok(&self, id: EntryId, min_resumes: u32, min_span_ms: u64) -> Option<(u32, Key)> {
        let e = &self.entries[&id];
        if e.shared || e.kind == KIND_TAIL || e.kind == KIND_WEIGHTS || !self.session_proven(id, min_resumes, min_span_ms) {
            return None;
        }
        if e.kind == 1 && self.sessions.get(&e.sess).and_then(|r| r.newest_state) != Some(id) {
            return None;
        }
        Some((e.ns, e.key))
    }

    /// `strict_age_candidates` on segment units: the units the age pass is about to evict, never orphans.
    fn seg_strict_age_candidates(&self, before: &dyn Fn(u32) -> u64, max_slots: u64, min_resumes: u32, min_span_ms: u64)
                                 -> Vec<(u32, Key)> {
        let plan = self.seg_plan();
        let mut out = Vec::new();
        let mut slots = 0u64;
        for (h, ns, ids, o) in self.units_by_heat(&plan, None) {
            if o > 0 || h >= i64::try_from(before(ns)).unwrap_or(i64::MAX) {
                continue;
            }
            if max_slots > 0 && slots >= max_slots {
                break;
            }
            for id in ids {
                slots += self.entries[&id].n_slots as u64;
                if let Some(x) = self.strict_ok(id, min_resumes, min_span_ms) {
                    out.push(x);
                }
            }
        }
        out
    }

    // ------------------------------------------------------------ memory-pressure protection

    /// Protection candidates under memory pressure: committed entries used since `since` (sessions active in the
    /// protection window: a session's touch refreshes its whole chain) or in the shared-prefix class, as
    /// (shared, last use, id, slots).
    pub fn protect_candidates(&self, since: u64) -> Vec<(bool, u64, EntryId, u64)> {
        self.entries.iter()
            .filter(|(_, e)| e.state == State::Committed && (e.shared || e.last_used >= since))
            .map(|(id, e)| (e.shared, e.last_used, *id, e.n_slots as u64))
            .collect()
    }

    /// `oldest_evictable` skipping the protected entries.
    pub fn oldest_evictable_except(&self, skip: &HashSet<EntryId>) -> Option<u64> {
        if self.cfg.seg_evict {
            if let Some((h, _, _)) = self.seg_oldest_except(Some(skip)) {
                return Some(h.max(0) as u64);
            }
        }
        self.entries.iter().filter(|(id, e)| e.evictable() && !skip.contains(id)).map(|(_, e)| self.heat(e).max(0) as u64).min()
    }

    /// `evict_oldest` skipping the protected entries; slots freed (0 = none).
    pub fn evict_oldest_except(&mut self, skip: &HashSet<EntryId>) -> u64 {
        if self.cfg.seg_evict {
            if let Some((_, ids, o)) = self.seg_oldest_except(Some(skip)) {
                return self.evict_unit(&ids, o, Removal::LowWater);
            }
        }
        let v = self.entries.iter().filter(|(id, e)| e.evictable() && !skip.contains(id))
            .min_by_key(|(id, e)| (self.heat(e), **id)).map(|(id, e)| (*id, e.n_slots as u64));
        match v {
            Some((id, _)) => self.evict_one(id, Removal::LowWater),
            None => 0,
        }
    }

    pub fn release(&mut self, conn: ConnId, lease_ids: &[u64]) -> Vec<Status> {
        self.end_run(conn);
        lease_ids
            .iter()
            .map(|l| match self.leases.get(l) {
                Some(x) if x.conn == conn => {
                    self.end_lease(*l);
                    Status::Ok
                }
                Some(_) => Status::NotOwner,
                None => Status::NotFound,
            })
            .collect()
    }

    fn end_lease(&mut self, lease_id: u64) -> bool {
        let Some(l) = self.leases.remove(&lease_id) else { return false };
        if let Some(c) = self.conns.get_mut(&l.conn) {
            c.leases.remove(&lease_id);
        }
        for id in l.entries {
            if let Some(e) = self.entries.get_mut(&id) {
                e.leases -= 1;
                if e.flags & ENTRY_EPHEMERAL != 0 {
                    e.expires = Some(0); // read once: evict first
                }
            }
        }
        true
    }

    // ------------------------------------------------------------ store

    pub fn reserve(&mut self, conn: ConnId, ns: u32, items: &[ReserveItem], now: u64) -> Vec<(Status, Vec<Range>)> {
        self.now = self.now.max(now);
        self.end_run(conn);
        items.iter().map(|it| self.reserve_one(conn, ns, it, now)).collect()
    }

    fn reserve_one(&mut self, conn: ConnId, ns: u32, it: &ReserveItem, now: u64) -> (Status, Vec<Range>) {
        if let Some(&id) = self.index.get(&(ns, it.key)) {
            if self.cfg.shared_slots > 0 && self.entries[&id].state == State::Committed && shared_hint(it.flags, &it.meta) {
                self.mark_shared(id);
            }
            return match self.entries[&id].state {
                State::Committed => {
                    let e = self.entries.get_mut(&id).unwrap();
                    if rank_of(e.flags) > rank_of(it.flags) {
                        e.flags = with_rank(e.flags, rank_of(it.flags));   // a higher class stores the same block
                        self.c.low_promoted += 1;
                    }
                    self.c.exists += 1;
                    self.ns_c.entry((ns, it.kind)).or_default().exists += 1;
                    (Status::Exists, vec![])
                }
                State::Reserved { .. } => {
                    self.c.busy += 1;
                    (Status::Busy, vec![])
                }
            };
        }
        if it.meta.len() > MAX_META || it.parent == Some(it.key) {
            return (Status::BadRequest, vec![]);
        }
        let n = it.bytes.div_ceil(self.cfg.slot_bytes).max(1);
        if n > self.cfg.n_slots as u64 - self.alloc.parked() {
            return (Status::TooBig, vec![]);
        }
        let n = n as u32;
        let parent = match it.parent {
            None => None,
            Some(pk) => match self.index.get(&(ns, pk)) {
                Some(&p) => Some(p),
                None => {
                    self.c.no_parent += 1;
                    return (Status::NoParent, vec![]);
                }
            },
        };
        let id = self.next_id;
        self.next_id += 1;
        // Link to the parent before evicting, so making room can't take the parent.
        if let Some(p) = parent {
            self.entries.get_mut(&p).unwrap().children.insert(id);
        }
        if !self.make_room(n as u64, Removal::Evicted, Some(ns)) {
            if let Some(p) = parent {
                self.entries.get_mut(&p).unwrap().children.remove(&id);
            }
            self.c.full += 1;
            self.ns_c.entry((ns, it.kind)).or_default().full += 1;
            return (Status::Full, vec![]);
        }
        let ranges = self.alloc.alloc(n);
        let producer = self.engine(conn);
        let depth = parent.and_then(|p| self.entries.get(&p)).map(|p| p.depth.saturating_add(1)).unwrap_or(0);
        self.entries.insert(
            id,
            Entry {
                ns,
                key: it.key,
                parent,
                children: HashSet::new(),
                state: State::Reserved { conn, deadline: now + self.cfg.reserve_timeout_ms },
                ranges: ranges.clone(),
                n_slots: n,
                bytes: it.bytes,
                kind: it.kind,
                format: it.format,
                producer,
                cost_hint: it.cost_hint,
                flags: it.flags,
                ttl_ms: it.ttl_ms,
                expires: None,
                meta: it.meta.clone(),
                leases: 0,
                hits: 0,
                last_used: now,
                born: now,
                first_hit: 0,
                last_hit: 0,
                hit_engines: 0,
                recent: std::collections::VecDeque::new(),
                rewriting: false,
                sum: None,
                sum_at: 0,
                compressed: false,
                shared: false,
                succ: [0; 4],
                depth,
                sess: 0,
                use_: Default::default(),
            },
        );
        self.index.insert((ns, it.key), id);
        if let Some(c) = self.conns.get_mut(&conn) {
            c.reservations.insert(id);
        }
        self.c.reserved += 1;
        (Status::Ok, ranges)
    }

    pub fn commit(&mut self, conn: ConnId, ns: u32, keys: &[Key], now: u64) -> (Vec<Status>, Vec<Notify>) {
        self.end_run(conn);
        let mut notes = Vec::new();
        let st = keys
            .iter()
            .map(|k| {
                let Some(&id) = self.index.get(&(ns, *k)) else { return Status::NotFound };
                let e = self.entries.get_mut(&id).unwrap();
                match e.state {
                    State::Reserved { conn: o, .. } if o == conn => {
                        e.state = State::Committed;
                        e.last_used = now;
                        e.born = now;
                        if e.flags & ENTRY_EPHEMERAL != 0 {
                            e.expires = Some(now + e.ttl_ms as u64);
                            self.ephemeral.insert(id);
                        }
                        let (n, bytes, kind) = (e.n_slots as u64, e.bytes, e.kind);
                        // weights stay out of the policy (ARC's lists and target): they compete with its pick by
                        // heat (colder_weights), so KV's order is the same with or without them
                        if kind != KIND_WEIGHTS {
                            self.policy.insert(id, ghost(ns, k), n);
                        }
                        if let Some(c) = self.conns.get_mut(&conn) {
                            c.reservations.remove(&id);
                        }
                        self.c.committed += 1;
                        self.c.committed_bytes += bytes;
                        match rank_of(self.entries[&id].flags) {
                            0 => {}
                            1 => self.c.low_committed += 1,
                            _ => self.c.lowest_committed += 1,
                        }
                        if self.log_events {
                            let e = &self.entries[&id];
                            let ev = Event::Store { t: now, ns, kind, engine: e.producer, key: kp(k), bytes,
                                                    meta: e.meta.clone() };
                            self.push_event(ev);
                        }
                        let nc = self.ns_c.entry((ns, kind)).or_default();
                        nc.committed += 1;
                        nc.committed_bytes += bytes;
                        self.on_committed(id, ns, k);
                        self.seg_on_commit(id);
                        self.hot_on_commit(id);
                        if let Some(&sid) = self.key_sess.get(&(ns, *k)) {
                            if self.sessions.contains_key(&sid) {
                                self.assign_session(id, sid);
                            }
                        }
                        if let Some(ws) = self.waiters.remove(&(ns, *k)) {
                            notes.extend(ws.into_iter().map(|w| Notify { conn: w.conn, req_id: w.req_id, status: Status::Ok }));
                        }
                        Status::Ok
                    }
                    State::Reserved { .. } => Status::NotOwner,
                    State::Committed => Status::Exists,
                }
            })
            .collect();
        (st, notes)
    }

    /// Shared-prefix bookkeeping for a newly committed entry: index its key prefix, note it as a continuation of
    /// its metadata parent (a second, different continuation makes the parent shared), honour the client's hint.
    fn on_committed(&mut self, id: EntryId, ns: u32, k: &Key) {
        let mut p16 = [0u8; 16];
        p16.copy_from_slice(&k[..16]);
        self.pfx.insert((ns, p16), id);
        if self.cfg.shared_slots == 0 {
            return;
        }
        if shared_hint(self.entries[&id].flags, &self.entries[&id].meta) {
            self.mark_shared(id);
        }
        let Some(pp) = meta_parent(&self.entries[&id].meta) else { return };
        let Some(&pid) = self.pfx.get(&(ns, pp)) else { return };
        if pid == id || !self.entries.contains_key(&pid) {
            return;
        }
        let g = ghost(ns, k).max(1);
        let need = (self.cfg.shared_min_cont.max(2) as usize).min(4);
        let pe = self.entries.get_mut(&pid).unwrap();
        if pe.succ.contains(&g) {
            return;
        }
        let Some(slot) = pe.succ.iter().position(|x| *x == 0) else { return };
        pe.succ[slot] = g;
        if slot + 1 == need && self.depth(pid) <= self.cfg.shared_max_depth.max(1) as usize
            || slot + 1 == need && self.cfg.shared_max_depth == 0 {
            self.mark_shared(pid);
        }
    }

    /// chained entries from the root to `id` (1 = a root), along the entry-metadata parent links
    fn depth(&self, id: EntryId) -> usize {
        let mut d = 0;
        let mut cur = Some(id);
        while let Some(x) = cur {
            d += 1;
            if d > 4096 {
                break;
            }
            let Some(e) = self.entries.get(&x) else { break };
            let ns = e.ns;
            cur = meta_parent(&e.meta).and_then(|pp| self.pfx.get(&(ns, pp)).copied()).filter(|p| *p != x);
        }
        d
    }

    /// Put an entry and its metadata ancestors into the shared-prefix class (a chain is useless without its root),
    /// then keep the class within its budget (least recently used members leave first).
    #[cfg(test)]
    pub fn note_continuation_for_test(&mut self, ns: u32, key: &Key, ghost: u64) {
        let id = self.index[&(ns, *key)];
        let e = self.entries.get_mut(&id).unwrap();
        if let Some(slot) = e.succ.iter().position(|x| *x == 0) {
            e.succ[slot] = ghost;
        }
    }

    #[cfg(test)]
    pub fn mark_shared_for_test(&mut self, ns: u32, key: &Key) {
        let id = self.index[&(ns, *key)];
        self.mark_shared(id);
    }

    fn mark_shared(&mut self, id: EntryId) {
        let mut cur = Some(id);
        let mut hops = 0;
        while let Some(x) = cur {
            let Some(e) = self.entries.get_mut(&x) else { break };
            if !e.shared {
                e.shared = true;
                self.shared_now += e.n_slots as u64;
            }
            hops += 1;
            let ns = e.ns;
            cur = if hops > 4096 { None } else { meta_parent(&e.meta).and_then(|pp| self.pfx.get(&(ns, pp)).copied()) };
            if cur == Some(x) {
                break;
            }
        }
        while self.shared_now > self.cfg.shared_slots {
            let v = self.entries.iter().filter(|(_, e)| e.shared).min_by_key(|(i, e)| (e.last_used, **i)).map(|(i, _)| *i);
            let Some(v) = v else { break };
            let e = self.entries.get_mut(&v).unwrap();
            e.shared = false;
            self.shared_now -= e.n_slots as u64;
            self.c.shared_demoted += 1;
        }
    }

    pub fn abort(&mut self, conn: ConnId, ns: u32, keys: &[Key]) -> Vec<Status> {
        self.end_run(conn);
        keys.iter()
            .map(|k| {
                let Some(&id) = self.index.get(&(ns, *k)) else { return Status::NotFound };
                match self.entries[&id].state {
                    State::Reserved { conn: o, .. } if o == conn => {
                        self.remove_tree(id, Removal::Aborted);
                        Status::Ok
                    }
                    State::Reserved { .. } => Status::NotOwner,
                    State::Committed => Status::Exists,
                }
            })
            .collect()
    }

    /// WAIT: `Some(status)` to answer now, `None` if registered (answered later through `Notify`).
    pub fn wait(&mut self, conn: ConnId, req_id: u64, ns: u32, key: Key, timeout_ms: u32, now: u64) -> Option<Status> {
        self.end_run(conn);
        if let Some(&id) = self.index.get(&(ns, key)) {
            if self.entries[&id].state == State::Committed {
                return Some(Status::Ok);
            }
        }
        if timeout_ms == 0 {
            return Some(Status::Timeout);
        }
        self.waiters.entry((ns, key)).or_default().push(Waiter { conn, req_id, deadline: now + timeout_ms as u64 });
        None
    }

    // ------------------------------------------------------------ removal and eviction

    /// Remove an entry and, for a reservation, its reserved descendants. Leased or committed descendants of a
    /// removed reservation are detached (their parent link dropped) and age out normally.
    fn remove_tree(&mut self, id: EntryId, why: Removal) {
        let mut stack = vec![id];
        while let Some(x) = stack.pop() {
            let Some(e) = self.entries.remove(&x) else { continue };
            for ch in &e.children {
                let reserved_child = self.entries.get(ch).is_some_and(|c| matches!(c.state, State::Reserved { .. }));
                if reserved_child && matches!(why, Removal::Aborted | Removal::WriterGone | Removal::ResTimeout) {
                    stack.push(*ch);
                } else if let Some(c) = self.entries.get_mut(ch) {
                    c.parent = None;
                }
            }
            self.finish_removal(x, e, why);
        }
    }

    fn finish_removal(&mut self, id: EntryId, e: Entry, why: Removal) {
        if e.state == State::Committed && matches!(why, Removal::Evicted | Removal::LowWater | Removal::Culled | Removal::Cascade) {
            match rank_of(e.flags) {
                0 => self.c.normal_evicted += 1,
                1 => self.c.low_evicted += 1,
                _ => self.c.lowest_evicted += 1,
            }
        }
        self.seg_on_remove(&e);
        self.hot_on_remove(&e);
        if let Some(p) = e.parent.and_then(|p| self.entries.get_mut(&p)) {
            p.children.remove(&id);
        }
        if let State::Reserved { conn, .. } = e.state {
            if let Some(c) = self.conns.get_mut(&conn) {
                c.reservations.remove(&id);
            }
        } else {
            self.policy.remove(id, matches!(why, Removal::Evicted | Removal::LowWater | Removal::Culled));
        }
        // an INVALIDATEd entry left the index early, and a fresh entry may hold its key by now
        if self.index.get(&(e.ns, e.key)) == Some(&id) {
            self.index.remove(&(e.ns, e.key));
        }
        let mut p16 = [0u8; 16];
        p16.copy_from_slice(&e.key[..16]);
        if self.pfx.get(&(e.ns, p16)) == Some(&id) {
            self.pfx.remove(&(e.ns, p16));
        }
        if e.shared {
            self.shared_now -= e.n_slots as u64;
        }
        self.ephemeral.remove(&id);
        if matches!(e.state, State::Reserved { .. }) && matches!(why, Removal::WriterGone | Removal::ResTimeout)
            || why == Removal::TornRewrite {
            self.quarantined += e.n_slots as u64;
            self.quarantine.push_back((self.now + QUARANTINE_MS, e.ranges.clone()));
        } else {
            self.alloc.release(&e.ranges);
        }
        if e.state == State::Committed {
            let now = self.now;
            self.life.entry(("age", e.kind)).or_default().obs(now.saturating_sub(e.born));
            let r = match why {
                Removal::Evicted => Some(0u8),
                Removal::LowWater => Some(1),
                Removal::Culled => Some(2),
                Removal::Expired => Some(3),
                Removal::Orphan => Some(4),
                Removal::Cascade => Some(5),
                _ => None,
            };
            if let Some(r) = r {
                let hb = e.hits.min(2) as u8;
                *self.reuse_evict.entry((e.kind, reuse_depth_b(e.depth), reuse_age_b(now.saturating_sub(e.born)), hb, r))
                    .or_default() += 1;
            }
            if self.records.len() >= 100_000 {
                self.records.drain(..10_000);
            }
            let rec = EntryRecord {
                ns: e.ns,
                kind: e.kind,
                producer: e.producer,
                bytes: e.bytes,
                key_prefix: kp(&e.key),
                meta: e.meta.clone(),
                born: e.born,
                first_hit: e.first_hit,
                last_hit: e.last_hit,
                left: now,
                hits: e.hits,
                hit_engines: e.hit_engines,
                recent: e.recent.iter().copied().collect(),
                reason: match why {
                    Removal::Evicted => "evicted",
                    Removal::LowWater => "low_water",
                    Removal::Expired => "expired",
                    Removal::Aborted | Removal::WriterGone => "aborted",
                    Removal::ResTimeout => "res_timeout",
                    Removal::Corrupt | Removal::TornRewrite => "corrupt",
                    Removal::Culled => "culled",
                    Removal::Orphan => "orphan",
                    Removal::Cascade => "cascade",
                    Removal::Invalidated => "invalidated",
                },
            };
            if self.log_events {
                self.push_event(Event::Remove(rec.clone()));
            }
            self.records.push(rec);
        }
        if matches!(why, Removal::Evicted | Removal::LowWater | Removal::Culled | Removal::Orphan | Removal::Cascade) {
            let nc = self.ns_c.entry((e.ns, e.kind)).or_default();
            nc.evicted += 1;
            nc.evicted_hits += e.hits;
            if e.hits == 0 {
                nc.evicted_unused += 1;
            }
        }
        match why {
            Removal::Evicted => self.c.evict_room += 1,
            Removal::LowWater => self.c.evict_low_water += 1,
            Removal::Expired => self.c.evict_expired += 1,
            Removal::Aborted | Removal::WriterGone => self.c.aborted += 1,
            Removal::ResTimeout => self.c.reservation_timeouts += 1,
            Removal::Corrupt | Removal::TornRewrite => self.c.corrupt += 1,
            Removal::Culled => self.c.evict_culled += 1,
            Removal::Orphan => {}   // counted per kind in evict_unit
            Removal::Cascade => self.c.cascade_evicted += 1,
            Removal::Invalidated => self.c.invalidated += 1,
        }
    }

    /// Journal a content fingerprint (from the server's scrub thread).
    pub fn log_content(&mut self, ns: u32, key: &Key, kind: u8, slot: u32, pages: Vec<u64>) {
        if self.log_events {
            let t = self.now;
            self.push_event(Event::Content { t, ns, key: kp(key), kind, slot, pages });
        }
    }

    fn push_event(&mut self, ev: Event) {
        if self.events.len() >= 1_000_000 {
            self.events.drain(..100_000);
        }
        self.events.push(ev);
    }

    /// Remove an entry; slots freed, counting the chain it takes with it.
    fn evict_one(&mut self, id: EntryId, why: Removal) -> u64 {
        // An attention block thrown out takes the rest of its chain with it (no lookup can reach
        // past the gap). Not when the block is still on the SSD: the chain stays reachable through it. A leased
        // entry of the chain cannot go now: it expires when its leases end.
        let e = &self.entries[&id];
        let after = if e.kind == 0 && !self.on_disk(e.ns, &e.key) { self.chain_after(id) } else { Vec::new() };
        let mut freed = 0;
        for x in after.into_iter().rev() {                    // deepest first: children before their parents
            let Some(d) = self.entries.get_mut(&x) else { continue };
            if d.state != State::Committed {
                continue;
            }
            if d.leases > 0 || d.rewriting {
                d.expires = Some(self.now);
                self.ephemeral.insert(x);       // the expiry sweep takes it once its leases end
                continue;
            }
            let kids: Vec<EntryId> = d.children.iter().copied().collect();
            if kids.iter().any(|c| self.entries.contains_key(c)) {
                continue;
            }
            let d = self.entries.remove(&x).unwrap();
            freed += d.n_slots as u64;
            self.finish_removal(x, d, Removal::Cascade);
        }
        let e = self.entries.remove(&id).unwrap();
        debug_assert!(e.leases == 0);
        for ch in &e.children {
            if let Some(d) = self.entries.get_mut(ch) {
                d.parent = None;                // a leased child the cascade left (it expires with its leases)
            }
        }
        freed += e.n_slots as u64;
        self.finish_removal(id, e, why);
        freed
    }

    /// What an entry continues from, by key (the dependencies explicit parent links do not carry): an attention
    /// block's predecessor in its chain (learned segment links; zeros = a chain root), the attention block a GDN state
    /// is anchored at, the block a tail's metadata names. None = not known.
    fn seg_dep(&self, x: &Entry) -> Option<seg::P16> {
        let k = (x.ns, seg::p16(&x.key));
        match x.kind {
            0 => self.seg.link.get(&k).map(|(p, _)| p),
            1 => self.seg.anchor.get(&k).map(|(a, _)| a),
            KIND_TAIL => meta_parent(&x.meta),
            _ => None,
        }
    }

    /// The dependency index at a glance (exported to Prometheus). Per committed entry, what it
    /// continues from: `chain_roots` (attention blocks that start a chain), `chain_linked_entries` (predecessor
    /// known), `chain_unknown_entries` (no explicit parent, no link learned yet), `chain_dangling_entries`
    /// (predecessor neither in RAM nor known on the SSD: unreachable; the cascade keeps this at 0, so anything else
    /// is a bug or a store after its predecessor left), `chain_dangling_ssd_entries` (predecessor on the SSD only:
    /// reachable through it). Shape: `chain_chains` (roots with at least one block after them), `chain_depth_max`
    /// (attention blocks in the longest chain). Integrity: `cksum_unchecked_entries` / `_bytes` (not hashed yet).
    fn chain_gauges(&self) -> Vec<(String, u64)> {
        let committed = || self.entries.iter().filter(|(_, e)| e.state == State::Committed);
        let present: HashSet<(u32, seg::P16)> = committed().map(|(_, e)| (e.ns, seg::p16(&e.key))).collect();
        let (mut roots, mut linked, mut unknown, mut dangling, mut dangling_ssd) = (Vec::new(), 0u64, 0u64, 0u64, 0u64);
        let (mut unchecked, mut unchecked_bytes) = (0u64, 0u64);
        let mut next: HashMap<(u32, seg::P16), Vec<EntryId>> = HashMap::new();
        for (id, e) in committed() {
            if e.sum.is_none() {
                unchecked += 1;
                unchecked_bytes += e.bytes;
            }
            if e.parent.is_some() {
                linked += 1;                                    // explicit: present by construction (check())
                continue;
            }
            match self.seg_dep(e) {
                Some(p) if p == [0u8; 16] => {
                    if e.kind == 0 {
                        roots.push(*id);
                    }
                }
                Some(p) => {
                    linked += 1;
                    if e.kind == 0 {
                        next.entry((e.ns, p)).or_default().push(*id);
                    }
                    if !present.contains(&(e.ns, p)) {
                        if self.seg.gone.get(&(e.ns, p)).is_some_and(|(k, _)| self.on_disk(e.ns, &k)) {
                            dangling_ssd += 1;
                        } else {
                            dangling += 1;
                        }
                    }
                }
                None => unknown += 1,
            }
        }
        // longest chain: breadth-first from the roots over explicit children and learned links (attention only)
        let (mut chains, mut depth_max) = (0u64, 0u64);
        for r in &roots {
            let mut depth = 0u64;
            let mut level = vec![*r];
            let mut seen: HashSet<EntryId> = HashSet::from([*r]);
            while !level.is_empty() {
                depth += 1;
                let mut nxt = Vec::new();
                for x in &level {
                    let e = &self.entries[x];
                    let more = e.children.iter().chain(next.get(&(e.ns, seg::p16(&e.key))).into_iter().flatten());
                    for c in more {
                        if self.entries.get(c).is_some_and(|d| d.kind == 0) && seen.insert(*c) {
                            nxt.push(*c);
                        }
                    }
                }
                level = nxt;
            }
            chains += (depth > 1) as u64;
            depth_max = depth_max.max(depth);
        }
        vec![
            ("chain_roots".into(), roots.len() as u64),
            ("chain_linked_entries".into(), linked),
            ("chain_unknown_entries".into(), unknown),
            ("chain_dangling_entries".into(), dangling),
            ("chain_dangling_ssd_entries".into(), dangling_ssd),
            ("chain_chains".into(), chains),
            ("chain_depth_max".into(), depth_max),
            ("cksum_unchecked_entries".into(), unchecked),
            ("cksum_unchecked_bytes".into(), unchecked_bytes),
        ]
    }

    /// Entries a removal of attention block `id` leaves unreachable, in discovery order (shallow to deep): its
    /// explicit children (recursively), the deeper attention blocks of its chain (segment links), GDN states anchored
    /// at a removed block, tails whose metadata parent is a removed block. One pass builds the dependency index.
    fn chain_after(&self, id: EntryId) -> Vec<EntryId> {
        let mut deps: HashMap<(u32, seg::P16), Vec<EntryId>> = HashMap::new();
        if self.cfg.seg_evict {
            for (xid, x) in &self.entries {
                if x.state != State::Committed || *xid == id {
                    continue;
                }
                if let Some(p) = self.seg_dep(x).filter(|p| *p != [0u8; 16]) {
                    deps.entry((x.ns, p)).or_default().push(*xid);
                }
            }
        }
        let mut seen: HashSet<EntryId> = HashSet::from([id]);
        let mut out = Vec::new();
        let mut queue = std::collections::VecDeque::from([id]);
        while let Some(x) = queue.pop_front() {
            let Some(e) = self.entries.get(&x) else { continue };
            let more = e.children.iter().copied().chain(deps.get(&(e.ns, seg::p16(&e.key))).into_iter().flatten().copied());
            for c in more.collect::<Vec<_>>() {
                if seen.insert(c) && self.entries.contains_key(&c) {
                    out.push(c);
                    if self.entries[&c].kind == 0 {
                        queue.push_back(c);                       // only attention blocks extend the chain
                    }
                }
            }
        }
        out
    }

    /// Evict until `need` slots are free. Expired/read ephemerals go first, then policy order (unleased leaves
    /// only).
    /// Committed entries for the integrity scrub: (id, byte ranges, kind, format), skipping ids in `done`.
    #[allow(clippy::type_complexity)]
    pub fn scrub_candidates(&self, done: &HashSet<EntryId>) -> Vec<(EntryId, u32, Vec<Range>, u8, u16, Key)> {
        // newest first: a new, never-hit entry is the first to be evicted from a full pool
        let mut v: Vec<_> = self
            .entries
            .iter()
            .filter(|(id, e)| e.state == State::Committed && !done.contains(id))
            .map(|(id, e)| (*id, e.ns, e.ranges.clone(), e.kind, e.format, e.key))
            .collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        v
    }

    /// Whether entry `id` still exists (an entry evicted during a scrub read is not judged).
    pub fn has_entry(&self, id: EntryId) -> bool {
        self.entries.contains_key(&id)
    }

    /// INVALIDATE: the entries for `keys` and the chain after each are known bad (an engine got NaN from a
    /// state it restored; the RAM copy would poison every later restore, and a recompute's fresh copy is refused as
    /// EXISTS while it is there). They leave the index at once (no lookup finds them, a fresh entry can take the key);
    /// unleased ones are freed now, leased ones when their leases end. Per key: OK, or NOT_FOUND (no committed entry).
    pub fn invalidate(&mut self, ns: u32, keys: &[Key]) -> (Vec<Status>, Vec<(u32, Key)>) {
        let mut out = Vec::with_capacity(keys.len());
        let mut gone = Vec::new();
        for k in keys {
            let id = match self.index.get(&(ns, *k)) {
                Some(&id) if self.entries[&id].state == State::Committed => id,
                _ => {
                    out.push(Status::NotFound);
                    continue;
                }
            };
            if self.entries[&id].shared {
                out.push(Status::Exists);         // a shared beginning (computed from token 0, every session's prefix)
                continue;
            }
            let mut all = self.chain_after(id);
            all.insert(0, id);
            for x in all.into_iter().rev() {                          // deepest first: children before parents
                let Some(d) = self.entries.get(&x) else { continue };
                if d.state != State::Committed {
                    continue;
                }
                let key = (d.ns, d.key);
                gone.push(key);
                if self.index.get(&key) == Some(&x) {
                    self.index.remove(&key);
                }
                let d = self.entries.get_mut(&x).unwrap();
                if d.leases > 0 || d.rewriting {
                    d.expires = Some(self.now);
                    self.ephemeral.insert(x);                         // the expiry sweep takes it once its leases end
                    continue;
                }
                let d = self.entries.remove(&x).unwrap();
                for ch in &d.children {
                    if let Some(c) = self.entries.get_mut(ch) {
                        c.parent = None;                              // a leased child left behind
                    }
                }
                self.finish_removal(x, d, Removal::Invalidated);
            }
            out.push(Status::Ok);
        }
        (out, gone)
    }

    /// Remove an entry the scrub found corrupt. False if it is leased or has children right now (retry later).
    pub fn remove_corrupt(&mut self, id: EntryId) -> bool {
        match self.entries.get(&id) {
            Some(e) if e.evictable() => {
                self.evict_one(id, Removal::Corrupt);
                true
            }
            _ => false,
        }
    }

    // ------------------------------------------------------------ lazily backed pool

    /// On-demand pools: allocate the lowest free slots first (see `Alloc::low_first`).
    pub fn set_alloc_low_first(&mut self, on: bool) {
        self.alloc.low_first = on;
    }

    /// Free slot runs (address order): the broker returns their memory to the kernel.
    pub fn free_runs(&self) -> Vec<Range> {
        self.alloc.free_runs()
    }

    /// Take these free slots out of the allocator while the broker hole-punches them outside the lock: no RESERVE
    /// can hand them out until `end_freeing`. Slots that are not free any more (the free list was read before this
    /// lock) are skipped; returns the ones taken.
    pub fn begin_freeing(&mut self, ranges: &[Range]) -> Vec<Range> {
        let took: Vec<Range> = ranges.iter().copied().filter(|r| self.alloc.take_range(*r)).collect();
        self.freeing.extend(took.iter().copied());
        took
    }

    /// The punch of `begin_freeing`'s slots is done: they are free again.
    pub fn end_freeing(&mut self, ranges: &[Range]) {
        for r in ranges {
            if let Some(i) = self.freeing.iter().position(|x| x == r) {
                self.freeing.swap_remove(i);
                self.alloc.release(&[*r]);
            }
        }
    }

    /// Slots held by entries (committed + reserved) and the highest slot in use + 1.
    pub fn live_and_high_water(&self) -> (u64, u32) {
        let live = self.cfg.n_slots as u64 - self.alloc.free() - self.alloc.parked() - self.freeing.iter().map(|r| r.len as u64).sum::<u64>();
        let hw = self.entries.values().flat_map(|e| e.ranges.iter()).map(|r| r.start + r.len).max().unwrap_or(0);
        (live, hw)
    }

    /// Drop an entry whose slots could not be backed (or whose bytes could not be loaded).
    pub fn remove_key(&mut self, ns: u32, key: &Key) -> bool {
        match self.index.get(&(ns, *key)).copied() {
            Some(id) if self.entries[&id].evictable() => {
                self.evict_one(id, Removal::Aborted);
                true
            }
            _ => false,
        }
    }

    /// Gradual eviction: committed, unleased leaves last used before `before` (broker ms), least recently used first,
    /// until `max_slots` slots are freed (0 = no limit). Parents become leaves as their children go. Returns slots freed.
    pub fn evict_aged(&mut self, before: u64, max_slots: u64) -> u64 {
        let mut freed = 0u64;
        if self.cfg.seg_evict {
            freed = self.seg_evict_aged_by(&|_| before, max_slots);
            if before != u64::MAX {
                return freed;
            }
            // idle flush (everything): the held rest goes too
        }
        loop {
            let v = self
                .entries
                .iter()
                .filter(|(_, e)| e.evictable() && self.heat(e) < i64::try_from(before).unwrap_or(i64::MAX))
                .min_by_key(|(id, e)| (self.heat(e), **id))
                .map(|(id, e)| (*id, e.n_slots as u64));
            let Some((id, _)) = v else { break };
            if max_slots > 0 && freed >= max_slots {
                break;
            }
            freed += self.evict_one(id, Removal::LowWater);
        }
        freed
    }

    /// Gradual eviction with a per-namespace age: `before(ns)` = evict entries of `ns` last used before this time.
    /// Least recently used first (relative to their threshold), at most `max_slots` (0 = no limit).
    pub fn evict_aged_by(&mut self, before: &dyn Fn(u32) -> u64, max_slots: u64) -> u64 {
        if self.cfg.seg_evict {
            return self.seg_evict_aged_by(before, max_slots);
        }
        let mut freed = 0u64;
        loop {
            if max_slots > 0 && freed >= max_slots {
                break;
            }
            let v = self
                .entries
                .iter()
                .filter(|(_, e)| e.evictable() && self.heat(e) < i64::try_from(before(e.ns)).unwrap_or(i64::MAX))
                .min_by_key(|(id, e)| (self.heat(e), **id))
                .map(|(id, e)| (*id, e.n_slots as u64));
            let Some((id, _)) = v else { break };
            freed += self.evict_one(id, Removal::LowWater);
        }
        freed
    }

    /// In-place compression candidates: committed, unleased, not compressed, last used before `before`; up to `max`,
    /// least recently used first: (ns, key, ranges, bytes).
    pub fn compress_candidates(&self, before: u64, max: usize) -> Vec<(u32, Key, Vec<Range>, u64)> {
        let mut v: Vec<_> = self.entries.values()
            .filter(|e| e.state == State::Committed && e.leases == 0 && !e.compressed && !e.rewriting && e.last_used < before
                    && e.kind != KIND_WEIGHTS)   // weights readers copy the raw slots (no decompressing reader)
            .map(|e| (e.last_used, e.ns, e.key, e.ranges.clone(), e.bytes))
            .collect();
        v.sort_by_key(|x| x.0);
        v.into_iter().take(max).map(|(_, ns, k, r, b)| (ns, k, r, b)).collect()
    }

    /// Start rewriting an entry's slot: only if it is still committed, unleased, uncompressed and at `ranges`; while
    /// rewriting, lookups miss it (no lease can start) and it is not evictable.
    pub fn begin_rewrite(&mut self, ns: u32, key: &Key, ranges: &[Range]) -> bool {
        let Some(&id) = self.index.get(&(ns, *key)) else { return false };
        let e = self.entries.get_mut(&id).unwrap();
        if e.state != State::Committed || e.leases != 0 || e.compressed || e.rewriting || e.ranges != ranges {
            return false;
        }
        e.rewriting = true;
        true
    }

    /// End a rewrite; `compressed`: the slot now holds the compressed image.
    pub fn end_rewrite(&mut self, ns: u32, key: &Key, compressed: bool) {
        if let Some(&id) = self.index.get(&(ns, *key)) {
            let e = self.entries.get_mut(&id).unwrap();
            e.rewriting = false;
            e.compressed = true; // done: compressed, or not worth it (never tried again)
            if compressed {
                self.c.compressed += 1;
            }
        }
    }

    /// A rewrite that will never finish (the background worker doing it disconnected): the slot may hold
    /// part of an image, so the entry is evicted (with the rest of its chain), never readable again. True if it was
    /// being rewritten.
    pub fn abort_torn_rewrite(&mut self, ns: u32, key: &Key) -> bool {
        let Some(&id) = self.index.get(&(ns, *key)) else { return false };
        let e = self.entries.get_mut(&id).unwrap();
        if !e.rewriting {
            return false;
        }
        e.rewriting = false;
        if e.leases > 0 {
            // cannot happen (lookups miss a rewriting entry); never evict under a lease: expire with it
            e.expires = Some(self.now);
            self.ephemeral.insert(id);
            return true;
        }
        self.c.torn_rewrites += 1;
        self.evict_one(id, Removal::TornRewrite);
        true
    }

    /// Finish a relocation whose copy ran outside the core lock (background worker): only if the entry is
    /// still exactly the one copied (`guard`: id, old ranges, compressed, not rewriting) and unleased; otherwise the
    /// new slots are freed (the copier is done with them).
    pub fn finish_relocation_guarded(&mut self, ns: u32, key: &Key, guard: &(EntryId, Vec<Range>, bool, bool), new: Vec<Range>)
                                     -> bool {
        if self.copy_guard(ns, key).as_ref() == Some(guard) && !guard.3 {
            return self.finish_relocation(ns, key, &guard.1.clone(), new);
        }
        self.alloc.release(&new);
        false
    }

    /// A relocation whose copier went away (the worker's connection closed): its new slots go to quarantine, not the
    /// free list (the process may still be writing into them).
    pub fn abandon_relocation(&mut self, new: Vec<Range>) {
        self.quarantined += new.iter().map(|r| r.len as u64).sum::<u64>();
        self.quarantine.push_back((self.now + QUARANTINE_MS, new));
    }

    /// Identity of a committed entry's bytes: entry id, slot ranges, compressed,
    /// rewriting. A copier (the SSD writer) takes it before and after copying the slots without a lease; any
    /// difference (evicted and stored again, moved, rewritten) means the copy may be torn.
    /// Integrity checksums: entries to checksum or re-verify, as idle work. Never-checksummed entries
    /// first (oldest first), then those last verified before `reverify_before` (oldest first). Committed entries
    /// not being rewritten: (ns, key, ranges, bytes, compressed, stored sum).
    pub fn checksum_candidates(&self, reverify_before: u64, max: usize) -> Vec<(u32, Key, Vec<Range>, u64, bool, Option<u64>)> {
        let mut v: Vec<_> = self.entries.values()
            .filter(|e| e.state == State::Committed && !e.rewriting && (e.sum.is_none() || e.sum_at < reverify_before))
            .map(|e| ((e.sum.is_some(), if e.sum.is_some() { e.sum_at } else { e.born }), e.ns, e.key, e.ranges.clone(),
                      e.bytes, e.compressed, e.sum))
            .collect();
        v.sort_by_key(|x| x.0);
        v.into_iter().take(max).map(|(_, ns, k, r, b, c, s)| (ns, k, r, b, c, s)).collect()
    }

    /// The stored checksum of an entry's raw bytes, if computed yet.
    pub fn checksum_of(&self, ns: u32, key: &Key) -> Option<u64> {
        self.index.get(&(ns, *key)).and_then(|id| self.entries[id].sum)
    }

    /// Result of an idle checksum pass over an entry whose bytes were copied while it matched `guard`
    /// (copy_guard). `computed`: XXH3-64 of its raw bytes, or None if the bytes could not even be decoded (a
    /// compressed slot whose chunk checksums fail). A mismatch evicts the entry as corrupt (the engines recompute it);
    /// a leased entry is left for the next pass (it cannot be evicted under a lease).
    pub fn checksum_result(&mut self, ns: u32, key: &Key, guard: &(EntryId, Vec<Range>, bool, bool), computed: Option<u64>,
                           now: u64) -> ChecksumResult {
        if self.copy_guard(ns, key).as_ref() != Some(guard) {
            return ChecksumResult::Raced;           // changed while copied: the copy says nothing about it
        }
        let id = guard.0;
        let e = self.entries.get_mut(&id).unwrap();
        match (e.sum, computed) {
            (None, Some(s)) => {
                e.sum = Some(s);
                e.sum_at = now;
                self.c.cksum_stored += 1;
                ChecksumResult::Stored
            }
            (Some(a), Some(b)) if a == b => {
                e.sum_at = now;
                self.c.cksum_verified += 1;
                ChecksumResult::Verified
            }
            _ => {
                if e.leases > 0 || e.rewriting {
                    return ChecksumResult::Busy;
                }
                self.c.cksum_corrupt += 1;
                self.evict_one(id, Removal::Corrupt);
                ChecksumResult::Corrupt
            }
        }
    }

    pub fn copy_guard(&self, ns: u32, key: &Key) -> Option<(EntryId, Vec<Range>, bool, bool)> {
        let id = *self.index.get(&(ns, *key))?;
        let e = &self.entries[&id];
        (e.state == State::Committed).then(|| (id, e.ranges.clone(), e.compressed, e.rewriting))
    }

    pub fn is_compressed(&self, ns: u32, key: &Key) -> bool {
        self.index.get(&(ns, *key)).is_some_and(|id| self.entries[id].compressed)
    }

    /// Global heat order across pools (unified broker): last use of the least recently used evictable entry.
    pub fn oldest_evictable(&self) -> Option<u64> {
        if self.cfg.seg_evict {
            if let Some((h, _, _)) = self.seg_oldest_except(None) {
                return Some(h.max(0) as u64);
            }
        }
        self.entries.values().filter(|e| e.evictable()).map(|e| self.heat(e).max(0) as u64).min()
    }

    /// Evict the least recently used evictable entry; slots freed (0 = none).
    pub fn evict_oldest(&mut self) -> u64 {
        if self.cfg.seg_evict {
            if let Some((_, ids, o)) = self.seg_oldest_except(None) {
                return self.evict_unit(&ids, o, Removal::LowWater);
            }
        }
        let v = self.entries.iter().filter(|(_, e)| e.evictable()).min_by_key(|(id, e)| (self.heat(e), **id)).map(|(id, e)| (*id, e.n_slots as u64));
        match v {
            Some((id, _)) => self.evict_one(id, Removal::LowWater),
            None => 0,
        }
    }

    /// Memory pressure: evict in policy order until `want_slots` slots are freed (or nothing is evictable).
    pub fn evict_coldest(&mut self, want_slots: u64) -> u64 {
        let mut freed = 0u64;
        while freed < want_slots {
            let Some((ids, o)) = self.pick_unit(None) else { break };
            let n = self.evict_unit(&ids, o, Removal::LowWater);
            if n == 0 {
                break;
            }
            freed += n;
        }
        freed
    }

    /// Compaction step: the unleased committed entry reaching highest in the pool, if all its slots can move to free
    /// slots below its lowest one. Returns (namespace, key, old ranges, new ranges); the new slots are taken from the
    /// allocator. The caller copies the bytes (holding the core lock), then calls `finish_relocation`.
    pub fn relocation_candidate(&mut self) -> Option<(u32, Key, Vec<Range>, Vec<Range>)> {
        let mut v: Vec<(EntryId, u32, u32)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.state == State::Committed && e.leases == 0 && !e.rewriting)
            .map(|(id, e)| (*id, e.ranges.iter().map(|r| r.start + r.len).max().unwrap_or(0), e.ranges.iter().map(|r| r.start).min().unwrap_or(0)))
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        for (id, _, low) in v.into_iter().take(8) {
            let e = &self.entries[&id];
            if let Some(new) = self.alloc.alloc_below(e.n_slots, low) {
                let e = &self.entries[&id];
                return Some((e.ns, e.key, e.ranges.clone(), new));
            }
        }
        None
    }

    /// Point the entry at its new slots (bytes copied) and free the old ones; if the entry changed meanwhile (cannot
    /// happen under one core lock, but cheap to check) the new slots are freed instead.
    pub fn finish_relocation(&mut self, ns: u32, key: &Key, old: &[Range], new: Vec<Range>) -> bool {
        match self.index.get(&(ns, *key)).copied() {
            Some(id) if self.entries[&id].ranges == old && self.entries[&id].leases == 0 => {
                self.entries.get_mut(&id).unwrap().ranges = new;
                self.alloc.release(old);
                self.c.relocations += 1;
                true
            }
            _ => {
                self.alloc.release(&new);
                false
            }
        }
    }

    /// Hot-set candidates: committed entries with at least `min_hits` hits, hottest first (hits, then last use).
    pub fn hot_entries(&self, min_hits: u64) -> Vec<HotEntry> {
        let mut v: Vec<HotEntry> = self
            .entries
            .values()
            .filter(|e| e.state == State::Committed && e.hits >= min_hits && e.kind != KIND_WEIGHTS)
            .map(|e| HotEntry {
                ns: e.ns,
                key: e.key,
                kind: e.kind,
                format: e.format,
                producer: e.producer,
                meta: e.meta.clone(),
                bytes: e.bytes,
                hits: e.hits,
                last_used: e.last_used,
                ranges: e.ranges.clone(),
                shared: e.shared,
            })
            .collect();
        // shared-prefix entries first (system prompts: the SSD hot set's core, loaded first at start)
        let shared: HashSet<(u32, Key)> = self.entries.values().filter(|e| e.shared).map(|e| (e.ns, e.key)).collect();
        v.sort_by(|a, b| shared.contains(&(b.ns, b.key)).cmp(&shared.contains(&(a.ns, a.key)))
            .then(b.hits.cmp(&a.hits)).then(b.last_used.cmp(&a.last_used)));
        v
    }

    /// Insert an entry restored from the hot set as committed (its bytes are copied into `ranges` by the caller
    /// right after, before any engine can look it up: the caller holds the core lock across both). None if it
    /// exists already, does not fit without evicting, or is a weights entry (KIND_WEIGHTS: never restored).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_committed(&mut self, ns: u32, key: Key, kind: u8, format: u16, producer: u16, meta: Vec<u8>,
                            bytes: u64, hits: u64, now: u64) -> Option<Vec<Range>> {
        if kind == KIND_WEIGHTS || self.index.contains_key(&(ns, key)) {
            // weights are never restored from a persisted copy (the engine reloads them from the model file)
            return None;
        }
        let n = bytes.div_ceil(self.cfg.slot_bytes).max(1);
        if n > self.alloc.free() {
            return None;
        }
        let n = n as u32;
        let ranges = self.alloc.alloc(n);
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(
            id,
            Entry {
                ns,
                key,
                parent: None,
                children: HashSet::new(),
                state: State::Committed,
                ranges: ranges.clone(),
                n_slots: n,
                bytes,
                kind,
                format,
                producer,
                cost_hint: 0,
                flags: 0,
                ttl_ms: 0,
                expires: None,
                meta,
                leases: 0,
                hits,
                last_used: now,
                born: now,
                first_hit: 0,
                last_hit: 0,
                hit_engines: 0,
                recent: std::collections::VecDeque::new(),
                rewriting: false,
                sum: None,
                sum_at: 0,
                compressed: false,
                shared: false,
                succ: [0; 4],
                depth: 0,
                sess: 0,
                use_: Default::default(),
            },
        );
        self.index.insert((ns, key), id);
        let mut p16 = [0u8; 16];
        p16.copy_from_slice(&key[..16]);
        self.pfx.insert((ns, p16), id);
        self.seg_on_commit(id);
        self.policy.insert(id, ghost(ns, &key), n as u64);
        if hits > 1 {
            self.policy.access(id); // ARC: a restored hot entry starts among the frequent ones
        }
        Some(ranges)
    }

    /// Free room for an entry of `bytes` (evicting by policy), for the disk tier's promotion of an entry back into the
    /// pool. False if nothing evictable frees enough.
    pub fn make_room_for(&mut self, ns: u32, bytes: u64) -> bool {
        let n = bytes.div_ceil(self.cfg.slot_bytes).max(1);
        self.make_room(n, Removal::Evicted, Some(ns))
    }

    /// A committed entry's description and slot ranges (the disk tier copies its bytes out of the pool).
    pub fn entry_info(&self, ns: u32, key: &Key) -> Option<HotEntry> {
        let e = &self.entries[self.index.get(&(ns, *key))?];
        (e.state == State::Committed).then(|| HotEntry {
            ns: e.ns,
            key: e.key,
            kind: e.kind,
            format: e.format,
            producer: e.producer,
            meta: e.meta.clone(),
            bytes: e.bytes,
            hits: e.hits,
            last_used: e.last_used,
            ranges: e.ranges.clone(),
            shared: e.shared,
        })
    }

    /// RESERVE flags of a committed entry as they stand now (ENTRY_LOW cleared once a normal request used it).
    pub fn entry_flags(&self, ns: u32, key: &Key) -> Option<u8> {
        let e = &self.entries[self.index.get(&(ns, *key))?];
        (e.state == State::Committed).then_some(e.flags)
    }

    /// Keys of the entry's committed ancestors (parent first), for the disk tier's prefix-closed admission.
    pub fn ancestors(&self, ns: u32, key: &Key) -> Vec<Key> {
        let mut out = Vec::new();
        let mut cur = self.index.get(&(ns, *key)).and_then(|id| self.entries[id].parent);
        while let Some(id) = cur {
            let Some(e) = self.entries.get(&id) else { break };
            if e.state == State::Committed {
                out.push(e.key);
            }
            cur = e.parent;
            if out.len() > 4096 {
                break;
            }
        }
        out
    }

    /// (ns, key, last_used) of every committed entry: the disk tier's write pass and its recency updates.
    pub fn committed_keys(&self) -> Vec<(u32, Key, u64)> {
        self.entries.values().filter(|e| e.state == State::Committed).map(|e| (e.ns, e.key, e.last_used)).collect()
    }

    /// Read-only presence check (committed entries), for offline simulation: no hit, no policy access.
    pub fn contains(&self, ns: u32, key: &Key) -> bool {
        self.index.get(&(ns, *key)).is_some_and(|id| self.entries[id].state == State::Committed)
    }

    /// Records and events accumulated so far are dropped (offline simulation with log_events off).
    pub fn drain_records(&mut self) -> Vec<EntryRecord> {
        std::mem::take(&mut self.records)
    }

    /// Last use as the age orders see it: a tail entry looks `tail_age_div` times longer unused than it is
    /// (last_used - (div - 1) x idle; may go below 0, so signed: no saturation at a young broker clock); a
    /// shared-prefix entry `shared_age_mult` times shorter; a weights entry `weights_kv_slot_ms x n_slots / cost_hint`
    /// times longer (its value per slot relative to KV's).
    fn heat(&self, e: &Entry) -> i64 {
        // heat = now - idle / decay + reuse bonus (reused prefixes are hotter and decay slower):
        // the bonus is HEAT_MAX_HITS hits at most, heat_hit_ms each; the decay divisor is shared_age_mult for a shared
        // beginning, 1 + its distinct continuations for a prefix several chains continue from, 1 otherwise; a tail
        // decays tail_age_div times faster instead
        let idle = self.now.saturating_sub(e.last_used) as i64;
        let bonus = (e.hits.min(HEAT_MAX_HITS) * self.cfg.heat_hit_ms) as i64;
        let decayed = if e.kind == KIND_TAIL && self.cfg.tail_age_div > 1 {
            idle * self.cfg.tail_age_div as i64
        } else if e.kind == KIND_WEIGHTS && self.cfg.weights_kv_slot_ms > 0 && e.cost_hint > 0 {
            // per-slot value: KV weights_kv_slot_ms per slot vs these weights cost_hint / n_slots
            let r = idle as i128 * self.cfg.weights_kv_slot_ms as i128 * e.n_slots.max(1) as i128 / e.cost_hint as i128;
            r.min(i64::MAX as i128 / 2) as i64
        } else if e.shared && self.cfg.shared_age_mult > 1 {
            idle / self.cfg.shared_age_mult as i64
        } else {
            idle / (1 + e.succ.iter().filter(|x| **x != 0).count() as i64)
        };
        self.now as i64 - decayed + bonus
    }

    /// Slots held by committed and reserved entries of `ns`.
    fn ns_slots(&self, ns: u32) -> u64 {
        self.entries.values().filter(|e| e.ns == ns).map(|e| e.n_slots as u64).sum()
    }

    /// Next victim. Order: (0) partial tails (KIND_TAIL), least recently used first; (1) never hit and older than stale_unhit_ms, oldest first, any namespace; (2) if the
    /// storing namespace is at its cap, policy order within it; (3) policy order, skipping namespaces at or below
    /// their floor; (4) plain policy order.
    fn pick_victim(&self, for_ns: Option<u32>) -> Option<EntryId> {
        if self.shared_now > 0 {
            if let Some(v) = self.pick_victim_in(for_ns, false) {
                return Some(v);
            }
        }
        self.pick_victim_in(for_ns, true)
    }

    /// `pick_victim` without (`with_shared` false) or with the shared-prefix class.
    fn pick_victim_in(&self, for_ns: Option<u32>, with_shared: bool) -> Option<EntryId> {
        let now = self.now;
        let ev = |e: &Entry| e.evictable() && (with_shared || !e.shared);
        // (0) partial tails first, least recently used first (KIND_TAIL)
        if let Some((id, _)) = self.entries.iter().filter(|(_, e)| e.kind == KIND_TAIL && ev(e))
            .min_by_key(|(id, e)| (e.last_used, **id)) {
            return Some(*id);
        }
        if self.cfg.stale_unhit_ms > 0 {
            let stale = self
                .entries
                .iter()
                .filter(|(_, e)| ev(e) && e.hits == 0 && now.saturating_sub(e.born) >= self.cfg.stale_unhit_ms)
                .min_by_key(|(id, e)| (e.born, **id));
            if let Some((id, _)) = stale {
                return Some(*id);
            }
        }
        if self.cfg.ns_cap_slots > 0 {
            if let Some(ns) = for_ns {
                if self.ns_slots(ns) >= self.cfg.ns_cap_slots {
                    if let Some(id) = self.policy.candidates().find(|id| ev(&self.entries[id]) && self.entries[id].ns == ns) {
                        return Some(id);
                    }
                }
            }
        }
        // weights entries (KIND_WEIGHTS) are not in the policy: they compete with its pick by heat (colder_weights)
        if self.cfg.ns_floor_slots > 0 {
            let mut held: HashMap<u32, u64> = HashMap::new();
            for e in self.entries.values() {
                *held.entry(e.ns).or_default() += e.n_slots as u64;
            }
            let above = |e: &Entry| held[&e.ns] > self.cfg.ns_floor_slots;
            let p = self.policy.candidates().find(|id| ev(&self.entries[id]) && above(&self.entries[id]));
            if let Some(id) = self.colder_weights(p.map(|x| vec![x]).as_deref(), &|e| ev(e) && above(e)).or(p) {
                return Some(id);
            }
        }
        let p = self.policy.candidates().find(|id| ev(&self.entries[id]));
        let v = self.colder_weights(p.map(|x| vec![x]).as_deref(), &ev).or(p);
        for r in [2u8, 1] {
            let low = self.entries.iter().filter(|(_, e)| rank_of(e.flags) >= r && ev(e)).map(|(id, e)| (self.heat(e), vec![*id]));
            if let Some(u) = self.colder_low(v.map(|x| vec![x]).as_deref(), r, low) {
                return Some(u[0]);
            }
        }
        v
    }

    /// Class order (ENTRY_LOW): the coldest low-class unit of `low` (heat, members) if it is at most `low_margin_ms`
    /// hotter than `victim` (the unit the policy order picked) or there is no victim; None = keep the victim. A
    /// low-class prefix reused often enough (heat bonus per hit, recent use) still outlives a cold normal entry.
    pub(crate) fn colder_low(&self, victim: Option<&[EntryId]>, rank: u8, low: impl Iterator<Item = (i64, Vec<EntryId>)>)
                             -> Option<Vec<EntryId>> {
        if self.low_margin_ms == 0 {
            return None;
        }
        let (h, ids) = low.min_by_key(|(h, ids)| (*h, ids[0]))?;
        if victim.is_some_and(|v| v.iter().all(|i| rank_of(self.entries[i].flags) >= rank)) {
            return None;                                // the policy picked a unit of this class (or lower) itself
        }
        let ok = match victim {
            None => true,
            Some(v) => {
                let hv = v.iter().map(|i| self.heat(&self.entries[i])).max().unwrap_or(i64::MIN);
                h < hv.saturating_add(self.low_margin_ms as i64)
            }
        };
        if ok {
            self.c.low_picked.set(self.c.low_picked.get() + 1);
        }
        ok.then_some(ids)
    }

    /// TOUCH_TAGGED: TOUCH_REQUEST with the request's tags. A session tag names the broker session (its first request
    /// is already in it, no chain-end inference); the touch raises entries of a lower class to the request's class.
    /// TAG_TERMINAL (the session's last request): afterwards the touched entries only this session used (not in the
    /// shared-prefix class, fewer than 2 chains' use, at most one continuation) drop to the lowest class: still cached,
    /// first to go under room pressure. `parent` is accepted for the wire (the router places subagents with it).
    #[allow(clippy::too_many_arguments)]
    pub fn touch_tagged(&mut self, conn: ConnId, ns: u32, session: u64, _parent: u64, class: u8, flags: u8,
                        groups: &[(u8, Vec<Key>)], now: u64) {
        self.c.tagged_touches += 1;
        if class == CLASS_LOW {
            self.c.tagged_low_touches += 1;
        }
        self.touch_class = Some(class);
        self.touch_tag = (session != 0).then_some(session);
        self.touch_request(conn, ns, groups, now);
        self.touch_class = None;
        self.touch_tag = None;
        if flags & TAG_TERMINAL != 0 && session != 0 {
            let Some(&sid) = self.tag_sess.get(&(ns, session)) else { return };
            let half = self.use_half_life_ms;
            for (_, ks) in groups {
                for k in ks {
                    let Some(&id) = self.index.get(&(ns, *k)) else { continue };
                    let e = self.entries.get_mut(&id).unwrap();
                    let own = e.sess == sid && !e.shared && e.use_.at(now, half) < 2.0
                        && e.succ.iter().filter(|x| **x != 0).count() <= 1;
                    if own && e.state == State::Committed && rank_of(e.flags) < 2 {
                        e.flags = with_rank(e.flags, 2);
                        self.c.terminal_demoted += 1;
                    }
                }
            }
        }
    }

    /// The coldest weights entry (KIND_WEIGHTS) accepted by `ok`, by heat, if it is colder than every member of
    /// `victim` (the unit the policy would evict instead), or there is no such unit. Weights stay out of the policy
    /// order and leave once they are worth less than the KV they hold the room of (`heat`, weights_kv_slot_ms). No
    /// weights entry: None, so kinds 0-4 keep the policy order exactly.
    pub(crate) fn colder_weights(&self, victim: Option<&[EntryId]>, ok: &dyn Fn(&Entry) -> bool) -> Option<EntryId> {
        let (id, w) = self.entries.iter().filter(|(_, e)| e.kind == KIND_WEIGHTS && ok(e))
            .min_by_key(|(id, e)| (self.heat(e), **id))?;
        let hw = self.heat(w);
        match victim {
            None => Some(*id),
            Some(v) => (hw < v.iter().map(|i| self.heat(&self.entries[i])).max().unwrap_or(i64::MIN)).then_some(*id),
        }
    }

    fn make_room(&mut self, need: u64, why: Removal, for_ns: Option<u32>) -> bool {
        if self.alloc.free() >= need {
            return true;
        }
        let now_expired: Vec<EntryId> = self
            .ephemeral
            .iter()
            .copied()
            .filter(|id| self.entries[id].evictable() && self.entries[id].expires == Some(0))
            .collect();
        for id in now_expired {
            if self.alloc.free() >= need {
                return true;
            }
            self.evict_one(id, Removal::Expired);
        }
        // One victim per scan: evicting a leaf can make its parent the next candidate in policy order.
        // Cost per eviction = the leased/interior entries skipped at the head of the order.
        while self.alloc.free() < need {
            let Some((ids, o)) = self.pick_unit(for_ns) else {
                return false;
            };
            if self.evict_unit(&ids, o, why) == 0 {
                return false;
            }
        }
        true
    }

    // ------------------------------------------------------------ dynamic pool

    /// Park slots until `want_parked` slots are parked: evict the policy's worst entries (the same order as making
    /// room, leased/reserved/interior entries never) until enough slots are free, then take the largest free runs out of
    /// the allocator. Returns the newly parked runs (the broker hole-punches their interior and lowers the huge-page
    /// reservation once every pinning client has unregistered them). Parks fewer than asked when nothing more is
    /// evictable.
    pub fn cull(&mut self, want_parked: u64) -> Vec<Range> {
        let parked = self.alloc.parked();
        if want_parked <= parked {
            return Vec::new();
        }
        let need = want_parked - parked;
        let _ = self.make_room(need, Removal::Culled, None);
        let take = self.alloc.free().min(need) as u32;
        if take == 0 {
            return Vec::new();
        }
        self.alloc.park(take)
    }

    /// Give parked runs back to the allocator (after the broker re-allocated their memory).
    pub fn unpark(&mut self, ranges: &[Range]) {
        self.alloc.unpark(ranges);
    }

    pub fn parked_slots(&self) -> u64 {
        self.alloc.parked()
    }

    pub fn parked_runs(&self) -> Vec<Range> {
        self.alloc.parked_runs()
    }

    /// Timers: reservation and lease timeouts, WAIT timeouts, ephemeral expiry, low-water eviction.
    pub fn tick(&mut self, now: u64) -> Vec<Notify> {
        self.now = self.now.max(now);
        while self.quarantine.front().is_some_and(|(t, _)| *t <= self.now) {
            let (_, r) = self.quarantine.pop_front().unwrap();
            self.quarantined -= r.iter().map(|x| x.len as u64).sum::<u64>();
            self.alloc.release(&r);
        }
        let expired_res: Vec<EntryId> = self
            .entries
            .iter()
            .filter(|(_, e)| matches!(e.state, State::Reserved { deadline, .. } if deadline <= now))
            .map(|(id, _)| *id)
            .collect();
        for id in expired_res {
            if self.entries.contains_key(&id) {
                self.remove_tree(id, Removal::ResTimeout);
            }
        }
        let expired_leases: Vec<u64> = self.leases.iter().filter(|(_, l)| l.deadline <= now).map(|(id, _)| *id).collect();
        for l in expired_leases {
            self.end_lease(l);
            self.c.lease_timeouts += 1;
        }
        let ttl_done: Vec<EntryId> = self
            .ephemeral
            .iter()
            .copied()
            .filter(|id| self.entries[id].evictable() && self.entries[id].expires.is_some_and(|t| t <= now))
            .collect();
        for id in ttl_done {
            self.evict_one(id, Removal::Expired);
        }
        let mut notes = Vec::new();
        for ws in self.waiters.values_mut() {
            ws.retain(|w| {
                if w.deadline <= now {
                    notes.push(Notify { conn: w.conn, req_id: w.req_id, status: Status::Timeout });
                    false
                } else {
                    true
                }
            });
        }
        self.c.wait_timeouts += notes.len() as u64;
        self.waiters.retain(|_, ws| !ws.is_empty());
        self.seg_scan();
        self.hot_refresh();
        let lw = self.cfg.low_water_slots;
        self.make_room(lw, Removal::LowWater, None);
        notes
    }

    // ------------------------------------------------------------ stats

    /// Heat of the committed entries (heat metrics for the whole cache): per kind, slots and entries
    /// by time since last use (HEAT_AGE buckets; "demoted" = HEAT cold, a dead chain waiting to go).
    pub fn heat_hist(&self) -> Vec<(u8, usize, u64, u64)> {
        let mut m: HashMap<(u8, usize), (u64, u64)> = HashMap::new();
        for e in self.entries.values() {
            if e.state != State::Committed {
                continue;
            }
            let b = if e.last_used == 0 {
                HEAT_AGE.len() - 1
            } else {
                let idle = self.now.saturating_sub(e.last_used);
                HEAT_AGE[..HEAT_AGE.len() - 1].iter().position(|(ms, _)| idle < *ms).unwrap_or(HEAT_AGE.len() - 2)
            };
            let x = m.entry((kind_bucket(e.kind) as u8, b)).or_default();
            x.0 += e.n_slots as u64;
            x.1 += 1;
        }
        let mut v: Vec<_> = m.into_iter().map(|((k, b), (s, n))| (k, b, s, n)).collect();
        v.sort();
        v
    }

    pub fn gauges(&self) -> Vec<(String, u64)> {
        let mut reserved_slots = 0u64;
        let mut leased_slots = 0u64;
        let mut committed_bytes = 0u64;
        let mut by_kind = [0u64; 6];
        for e in self.entries.values() {
            match e.state {
                State::Reserved { .. } => reserved_slots += e.n_slots as u64,
                State::Committed => {
                    committed_bytes += e.bytes;
                    by_kind[kind_bucket(e.kind)] += 1;
                    if e.leases > 0 {
                        leased_slots += e.n_slots as u64;
                    }
                }
            }
        }
        let mut v = vec![
            ("slots_total".into(), self.cfg.n_slots as u64),
            ("slots_free".into(), self.alloc.free()),
            ("slots_quarantined".into(), self.quarantined),
            ("slots_parked".into(), self.alloc.parked()),
            ("slots_reserved".into(), reserved_slots),
            ("slots_leased".into(), leased_slots),
            ("free_runs".into(), self.alloc.runs() as u64),
            ("entries".into(), self.entries.len() as u64),
            ("bytes_committed_now".into(), committed_bytes),
            ("leases_active".into(), self.leases.len() as u64),
            ("connections".into(), self.conns.len() as u64),
            ("waiters".into(), self.waiters.values().map(|w| w.len() as u64).sum()),
        ];
        for (k, n) in by_kind.iter().enumerate() {
            v.push((format!("entries_kind{k}"), *n));
        }
        v.push(("shared_slots".into(), self.shared_now));
        for (r, name) in [(1u8, "low"), (2, "lowest")] {
            let low: Vec<&Entry> = self.entries.values().filter(|e| e.state == State::Committed && rank_of(e.flags) == r).collect();
            v.push((format!("{name}_entries"), low.len() as u64));
            v.push((format!("{name}_slots"), low.iter().map(|e| e.n_slots as u64).sum()));
        }
        v.push(("shared_entries".into(), self.entries.values().filter(|e| e.shared).count() as u64));
        v.extend(self.seg_gauges());
        v.extend(self.chain_gauges());
        v.extend(self.hot_gauges());
        v
    }

    /// Per (namespace, entry kind): gauges (entries, bytes, slots now held by committed entries) and counters.
    /// Kind KIND_ANY carries the lookup counter.
    #[allow(clippy::type_complexity)]
    pub fn ns_stats(&self) -> Vec<((u32, u8), Vec<(&'static str, u64)>, Vec<(&'static str, u64)>)> {
        let mut g: HashMap<(u32, u8), [u64; 3]> = HashMap::new();
        for e in self.entries.values() {
            if e.state == State::Committed {
                let v = g.entry((e.ns, e.kind)).or_default();
                v[0] += 1;
                v[1] += e.bytes;
                v[2] += e.n_slots as u64;
            }
        }
        let mut nss: Vec<(u32, u8)> = g.keys().chain(self.ns_c.keys()).copied().collect();
        nss.sort_unstable();
        nss.dedup();
        nss.into_iter()
            .map(|ns| {
                let v = g.get(&ns).copied().unwrap_or_default();
                let c = self.ns_c.get(&ns).cloned().unwrap_or_default();
                (
                    ns,
                    vec![("entries", v[0]), ("bytes", v[1]), ("slots", v[2])],
                    vec![
                        ("lookup_keys", c.lookup_keys),
                        ("hit_entries", c.hit_entries),
                        ("hit_bytes", c.hit_bytes),
                        ("committed", c.committed),
                        ("committed_bytes", c.committed_bytes),
                        ("exists", c.exists),
                        ("full", c.full),
                        ("evicted", c.evicted),
                        ("evicted_unused", c.evicted_unused),
                        ("evicted_hits", c.evicted_hits),
                    ],
                )
            })
            .collect()
    }

    pub fn counters(&self) -> Vec<(String, u64)> {
        let c = &self.c;
        vec![
            ("cksum_stored".into(), c.cksum_stored),
            ("cksum_verified".into(), c.cksum_verified),
            ("cksum_corrupt".into(), c.cksum_corrupt),
            ("cascade_evicted".into(), c.cascade_evicted),
            ("invalidated".into(), c.invalidated),
            ("torn_rewrites".into(), c.torn_rewrites),
            ("heat_hot".into(), c.heat_hot),
            ("heat_cold".into(), c.heat_cold),
            ("lookups".into(), c.lookups),
            ("lookup_keys".into(), c.lookup_keys),
            ("hit_entries".into(), c.hit_entries),
            ("hit_bytes".into(), c.hit_bytes),
            ("cross_engine_hit_entries".into(), c.cross_engine_hit_entries),
            ("full_hits".into(), c.full_hits),
            ("partial_hits".into(), c.partial_hits),
            ("misses".into(), c.misses),
            ("reserved".into(), c.reserved),
            ("committed".into(), c.committed),
            ("committed_bytes".into(), c.committed_bytes),
            ("exists".into(), c.exists),
            ("busy".into(), c.busy),
            ("full".into(), c.full),
            ("no_parent".into(), c.no_parent),
            ("evict_room".into(), c.evict_room),
            ("evict_low_water".into(), c.evict_low_water),
            ("evict_expired".into(), c.evict_expired),
            ("evict_culled".into(), c.evict_culled),
            ("relocations".into(), c.relocations),
            ("shared_demoted".into(), c.shared_demoted),
            ("compressed_in_place".into(), c.compressed),
            ("aborted".into(), c.aborted),
            ("reservation_timeouts".into(), c.reservation_timeouts),
            ("lease_timeouts".into(), c.lease_timeouts),
            ("leases_closed_by_disconnect".into(), c.leases_closed_by_disconnect),
            ("wait_timeouts".into(), c.wait_timeouts),
            ("corrupt".into(), c.corrupt),
            ("evict_orphan_attn".into(), c.evict_orphan_attn),
            ("evict_orphan_state".into(), c.evict_orphan_state),
            ("seg_units_evicted".into(), c.seg_units),
            ("seg_unit_entries_evicted".into(), c.seg_unit_entries),
            ("seg_runs_linked".into(), c.seg_runs_linked),
            ("seg_runs_skipped".into(), c.seg_runs_skipped),
            ("seg_skip_over".into(), c.seg_skip_over),
            ("seg_skip_shape".into(), c.seg_skip_shape),
            ("seg_skip_len".into(), c.seg_skip_len),
            ("seg_skip_kind".into(), c.seg_skip_kind),
            ("seg_skip_layout".into(), c.seg_skip_layout),
            ("seg_tagged_linked".into(), c.seg_tagged_linked),
            ("seg_tagged_unlinked".into(), c.seg_tagged_unlinked),
            ("touch_keys".into(), c.touch_keys),
            ("touch_missing_keys".into(), c.touch_missing_keys),
            ("hotset_state_refreshed".into(), c.hot_state_refreshed),
            ("low_committed".into(), c.low_committed),
            ("low_promoted".into(), c.low_promoted),
            ("low_evicted".into(), c.low_evicted),
            ("normal_evicted".into(), c.normal_evicted),
            ("low_picked".into(), c.low_picked.get()),
            ("tagged_touches".into(), c.tagged_touches),
            ("tagged_low_touches".into(), c.tagged_low_touches),
            ("tagged_sessions".into(), c.tagged_sessions),
            ("lowest_committed".into(), c.lowest_committed),
            ("lowest_evicted".into(), c.lowest_evicted),
            ("terminal_demoted".into(), c.terminal_demoted),
        ]
    }

    /// Internal consistency check for tests: slot accounting, links, lease counts.
    pub fn check(&self) {
        let used: u64 = self.entries.values().map(|e| e.n_slots as u64).sum();
        assert_eq!(used + self.alloc.free() + self.alloc.parked() + self.quarantined + self.freeing.iter().map(|r| r.len as u64).sum::<u64>(), self.cfg.n_slots as u64,
                   "slot accounting");
        assert_eq!(self.quarantine.iter().flat_map(|q| q.1.iter()).map(|r| r.len as u64).sum::<u64>(), self.quarantined,
                   "quarantine count");
        let mut lease_count: HashMap<EntryId, u32> = HashMap::new();
        for l in self.leases.values() {
            for id in &l.entries {
                *lease_count.entry(*id).or_default() += 1;
            }
        }
        for (id, e) in &self.entries {
            assert_eq!(self.index.get(&(e.ns, e.key)), Some(id));
            assert_eq!(e.leases, lease_count.get(id).copied().unwrap_or(0), "lease count");
            if let Some(p) = e.parent {
                assert!(self.entries[&p].children.contains(id), "parent link");
            }
            for ch in &e.children {
                assert_eq!(self.entries[ch].parent, Some(*id), "child link");
            }
        }
        // every slot exactly once: entries' ranges, free runs, parked runs, quarantined ranges (a slot
        // both free and held, or held twice, is the failure that serves another key's bytes)
        let mut slots: Vec<(u32, u32)> = self.entries.values().flat_map(|e| e.ranges.iter().map(|r| (r.start, r.len)))
            .chain(self.alloc.free_runs().iter().map(|r| (r.start, r.len)))
            .chain(self.alloc.parked_runs().iter().map(|r| (r.start, r.len)))
            .chain(self.quarantine.iter().flat_map(|q| q.1.iter()).map(|r| (r.start, r.len)))
            .chain(self.freeing.iter().map(|r| (r.start, r.len)))
            .collect();
        slots.sort();
        for w in slots.windows(2) {
            assert!(w[0].0 + w[0].1 <= w[1].0, "overlapping ranges {:?} {:?}", w[0], w[1]);
        }
        assert!(slots.iter().all(|r| r.1 > 0 && r.0 + r.1 <= self.cfg.n_slots), "empty or out-of-pool range");
        assert_eq!(slots.iter().map(|r| r.1 as u64).sum::<u64>(), self.cfg.n_slots as u64, "slots covered once");
    }

    pub fn cost_hint_sum(&self) -> u64 {
        self.entries.values().map(|e| e.cost_hint as u64).sum()
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod model_tests;
