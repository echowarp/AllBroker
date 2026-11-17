//! llmt-kvbroker wire protocol v1 (see `docs/protocol.md`).
//!
//! Frame: `u32 len | u16 op | u16 flags | u64 req_id | payload`, little-endian; `len` counts the bytes after itself.
//! Request and response payloads are typed here so the broker and the client share one codec.

use std::io::{self, Read, Write};

pub const PROTOCOL_VERSION: u16 = 1;
pub const MAX_FRAME: usize = 4 << 20;
pub const HEADER: usize = 12; // op + flags + req_id
pub const FLAG_RESP: u16 = 1;
pub const MAX_META: usize = 1024;
pub const ENTRY_EPHEMERAL: u8 = 1;
/// ReserveItem flags bit 2: the block was stored by a low-priority request (bookkeeping / background traffic). The
/// broker stores and ages it like any entry, but under room pressure takes it before other entries of comparable heat
/// (`--low-evict-margin-s`) and ranks it after other prefixes in the hot set. A later non-low use (a TOUCH, or a
/// RESERVE of the same key without the bit) clears it. An older broker ignores the bit (an ordinary entry).
pub const ENTRY_LOW: u8 = 4;
/// ReserveItem flags bit 3: lowest retention (e.g. background batch items, entries of a session that ended): like
/// ENTRY_LOW, taken before low-class entries of comparable heat. Both bits set count as lowest.
pub const ENTRY_LOWEST: u8 = 8;
/// TOUCH_TAGGED flags bit 0: the session's terminal request (its write-up / last pass): afterwards the entries only this
/// session used (not shared prefixes) drop to the lowest class (still cached, first to go under room pressure).
pub const TAG_TERMINAL: u8 = 1;

/// TOUCH_TAGGED request classes.
pub const CLASS_UNKNOWN: u8 = 0;
pub const CLASS_LOW: u8 = 1;
pub const CLASS_NORMAL: u8 = 2;
pub const CLASS_AGENT: u8 = 3;
pub const CLASS_LOWEST: u8 = 4;

pub type Key = [u8; 32];

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Hello = 1,
    Lookup = 2,
    Reserve = 3,
    Commit = 4,
    Abort = 5,
    Release = 6,
    Touch = 7,
    Stats = 8,
    Wait = 9,
    /// Dynamic pool: parked slot runs whose memory the broker returns to the kernel; clients that pin the
    /// pool (cudaHostRegister) unregister them and acknowledge the generation before the broker punches the holes.
    PoolState = 10,
    PoolAck = 11,
    /// N-gram (PLE) table service (crates/ngram-store/DESIGN.md): rows of a model's n-gram
    /// embedding table, served from a row cache in the broker into a per-connection staging area of a shared file.
    NgramOpen = 12,
    NgramGather = 13,
    NgramPrefetch = 14,
    /// Cache domain: the connection's model / weights / engine /
    /// layout / numerics descriptor; afterwards its namespaces are scoped to that domain (other domains never hit).
    Domain = 15,
    /// Presence of keys without leasing (for whole-state snapshot clients such as gufo, which probe the
    /// candidate checkpoint lengths of a prompt, longest first): per key 0 = absent, 1 = in RAM, 2 = on the SSD tier
    /// only (its promotion is queued, so a LOOKUP shortly after hits).
    Probe = 16,
    /// Retention hint without session bookkeeping: hot = refresh like TOUCH (a request waiting behind
    /// another session's hold keeps its chain), cold = coldest in the eviction order (a compacted session's dead chain).
    Heat = 17,
    /// Feature negotiation (docs/protocol.md "Versioning"): the client's feature bits and the formats it reads; the
    /// reply carries the broker's. Sent by new clients right after HELLO; a broker without it answers BAD_REQUEST
    /// (= no features), and a client that never sends it is served as before.
    Features = 18,
    /// One request's TOUCH with every KV group in one message, each batch tagged with its kind: the broker
    /// links the chain from the tags instead of inferring request boundaries and group kinds from TOUCH runs.
    TouchRequest = 19,
    /// Write the broker's operation trace (crates/trace) to a file now: request = a reason, reply = the
    /// file's path. BUSY: tracing off, or the day's dump budget used up.
    TraceDump = 20,
    /// Background worker (docs/protocol.md "Background worker"): a separate low-priority process does the broker's
    /// CPU-heavy idle work (integrity checksums, in-place compression) so the broker itself only answers requests.
    /// Only on a connection that declared FEAT_BG_WORKER. BG_JOBS: work to do now (empty unless the broker is quiet).
    BgJobs = 21,
    /// A checksum the worker computed for a job (the broker stores / verifies it; a mismatch evicts the entry).
    BgChecksum = 22,
    /// In-place rewrite of a slot: begin (the entry stops being readable), end (the image is written: the broker
    /// punches the rest of the slot), or skip (not worth compressing: never tried again).
    BgRewrite = 23,
    /// SSD tier writes for the background worker: copy an entry's slots, check them against its RAM
    /// checksum (BG_CHECKSUM), compress, write the file; the broker plans them (admission, budget) and records them.
    BgSsdJobs = 24,
    /// The outcome of an SSD write (0 written, 1 raced, 2 corrupt, 3 failed) and the file's bytes.
    BgSsdDone = 25,
    /// Compaction moves for the background worker: copy an entry's slots to lower ones the broker
    /// allocated; the entry stays readable at its old slots meanwhile. Only while quiet and not under memory pressure.
    BgCompactJobs = 26,
    /// A compaction copy done (1) or not done (0): the broker moves the entry if it is unchanged and unleased.
    BgCompactDone = 27,
    /// INVALIDATE (FEAT_INVALIDATE): drop committed entries and the chain after them, in RAM and on the
    /// SSD, because their content is known bad (an engine got NaN from a state it restored). A leased entry leaves when
    /// its leases end and no longer answers lookups meanwhile. Reply: one status per key (OK, NOT_FOUND).
    Invalidate = 28,
    /// TOUCH_TAGGED (FEAT_REQUEST_TAGS): TOUCH_REQUEST plus the request's tags: an opaque session id and its parent
    /// session (0 = none; the router's keyed hashes), the request class (CLASS_*) and the agent letter (0 = none). The
    /// session id names the broker session (no chain-end inference); a touch raises entries of a lower class to the
    /// request's class (CLASS_LOWEST < CLASS_LOW < the rest); flags TAG_TERMINAL ends the session's own entries' class.
    TouchTagged = 29,
    /// HINT (FEAT_PREFIX_HINTS, docs/prefix-hints.md): an engine tells the broker that a router anchor (16 opaque bytes,
    /// the router's hash of a request's message prefix) covers these keys of its own, per KV group in token order.
    /// Kept in a bounded side table, never in the entry index; reply = status only.
    Hint = 30,
    /// WHERE (FEAT_PREFIX_HINTS): for a request's candidate anchors (longest prefix first), which engines hinted
    /// one of them and how many leading chunks of it are committed in RAM right now. Read-only: no lease, no hit, no
    /// heat, no SSD promotion.
    Where = 31,
}

/// Feature bits (FEATURES). A bit is only ever added, never reused for another meaning.
pub const FEAT_TOUCH_REQUEST: u64 = 1 << 0;
pub const FEAT_TRACE_DUMP: u64 = 1 << 1;
/// Broker: accepts a background worker. Client: this connection is the background worker (its requests do not count
/// as traffic for the broker's quiet-time rules).
pub const FEAT_BG_WORKER: u64 = 1 << 2;

/// Broker: understands INVALIDATE (a client must not send it otherwise).
pub const FEAT_INVALIDATE: u64 = 1 << 3;

/// Broker: understands TOUCH_TAGGED and the ENTRY_LOW reserve flag's eviction order.
pub const FEAT_REQUEST_TAGS: u64 = 1 << 4;
/// Broker: understands HINT and WHERE (prefix-affinity hints for routers). A client must not send them otherwise.
pub const FEAT_PREFIX_HINTS: u64 = 1 << 5;

/// The engine id a router (or another observer that only asks WHERE / STATS) sends in HELLO: it is not a client (no SSD
/// warm-up, its requests are not traffic for the quiet-time rules, it does not keep the broker from its idle exit).
pub const OBSERVER_ENGINE_ID: u16 = 0xbb01;

/// A router's prefix anchor (opaque to the broker).
pub type Anchor = [u8; 16];
/// HINT flags: the engine held the prefix in VRAM when it sent the hint (it had just served it).
pub const HINT_VRAM: u8 = 1;

/// The engine id a background worker sends in HELLO (it is not a client: no SSD warm-up for it, and it does not keep
/// the broker from its idle exit).
pub const BG_ENGINE_ID: u16 = 0xbb00;

/// BG_JOBS kinds and BG_REWRITE phases / results.
pub const BG_CHECKSUM: u8 = 1;
pub const BG_COMPRESS: u8 = 2;
pub const BG_BEGIN: u8 = 1;
pub const BG_END: u8 = 2;
pub const BG_SKIP: u8 = 3;

/// The identity of an entry's bytes when a job was handed out (kvcore copy_guard): a result only counts while the entry
/// is still exactly this one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Guard {
    pub entry_id: u64,
    pub ranges: Vec<Range>,
    pub compressed: bool,
    pub rewriting: bool,
}

/// A compaction move (BG_COMPACT_JOBS): the entry (its guard has the old ranges) and the slots to copy it to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactJob {
    pub namespace: u32,
    pub key: Key,
    pub guard: Guard,
    pub new: Vec<Range>,
}

/// An SSD tier write (BG_SSD_JOBS): the entry, its guard, the file header's fields, the file and codec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SsdJob {
    pub namespace: u32,
    pub key: Key,
    pub guard: Guard,
    pub kind: u8,
    pub format: u16,
    pub producer: u16,
    pub hits: u64,
    pub bytes: u64,
    pub last_used_ms: u64,
    pub meta: Vec<u8>,
    pub path: String,
    pub codec: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgJob {
    pub namespace: u32,
    pub key: Key,
    pub bytes: u64,
    pub guard: Guard,
}
/// Format names in FEATURES (name, version). Readers accept the versions they know explicitly and refuse any other.
/// `slot_codec`: compressed RAM slot images (kvslotz LKVZ<version>); a broker writes a version only if every client
/// of the pool class declared it (clients without FEATURES read version 1 only).
pub const FMT_SLOT_CODEC: &str = "slot_codec";

impl Op {
    pub fn from_u16(v: u16) -> Option<Op> {
        use Op::*;
        Some(match v {
            1 => Hello,
            2 => Lookup,
            3 => Reserve,
            4 => Commit,
            5 => Abort,
            6 => Release,
            7 => Touch,
            8 => Stats,
            9 => Wait,
            10 => PoolState,
            11 => PoolAck,
            12 => NgramOpen,
            13 => NgramGather,
            14 => NgramPrefetch,
            15 => Domain,
            16 => Probe,
            17 => Heat,
            18 => Features,
            19 => TouchRequest,
            20 => TraceDump,
            21 => BgJobs,
            22 => BgChecksum,
            23 => BgRewrite,
            24 => BgSsdJobs,
            25 => BgSsdDone,
            26 => BgCompactJobs,
            27 => BgCompactDone,
            28 => Invalidate,
            29 => TouchTagged,
            30 => Hint,
            31 => Where,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        use Op::*;
        match self {
            Hello => "hello",
            Lookup => "lookup",
            Reserve => "reserve",
            Commit => "commit",
            Abort => "abort",
            Release => "release",
            Touch => "touch",
            Stats => "stats",
            Wait => "wait",
            PoolState => "pool_state",
            PoolAck => "pool_ack",
            NgramOpen => "ngram_open",
            NgramGather => "ngram_gather",
            NgramPrefetch => "ngram_prefetch",
            Domain => "domain",
            Probe => "probe",
            Heat => "heat",
            Features => "features",
            TouchRequest => "touch_request",
            TraceDump => "trace_dump",
            BgJobs => "bg_jobs",
            BgChecksum => "bg_checksum",
            BgRewrite => "bg_rewrite",
            BgSsdJobs => "bg_ssd_jobs",
            BgSsdDone => "bg_ssd_done",
            BgCompactJobs => "bg_compact_jobs",
            BgCompactDone => "bg_compact_done",
            Invalidate => "invalidate",
            TouchTagged => "touch_tagged",
            Hint => "hint",
            Where => "where",
        }
    }
    pub const ALL: [Op; 31] = [
        Op::Hello,
        Op::Lookup,
        Op::Reserve,
        Op::Commit,
        Op::Abort,
        Op::Release,
        Op::Touch,
        Op::Stats,
        Op::Wait,
        Op::PoolState,
        Op::PoolAck,
        Op::NgramOpen,
        Op::NgramGather,
        Op::NgramPrefetch,
        Op::Domain,
        Op::Probe,
        Op::Heat,
        Op::Features,
        Op::TouchRequest,
        Op::TraceDump,
        Op::BgJobs,
        Op::BgChecksum,
        Op::BgRewrite,
        Op::BgSsdJobs,
        Op::BgSsdDone,
        Op::BgCompactJobs,
        Op::BgCompactDone,
        Op::Invalidate,
        Op::TouchTagged,
        Op::Hint,
        Op::Where,
    ];
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok = 0,
    Exists = 1,
    Busy = 2,
    Full = 3,
    NotFound = 4,
    Timeout = 5,
    BadRequest = 6,
    NotOwner = 7,
    NoParent = 8,
    TooBig = 9,
    NoHello = 10,
    Version = 11,
}

impl Status {
    pub fn from_u16(v: u16) -> Status {
        use Status::*;
        match v {
            0 => Ok,
            1 => Exists,
            2 => Busy,
            3 => Full,
            4 => NotFound,
            5 => Timeout,
            7 => NotOwner,
            8 => NoParent,
            9 => TooBig,
            10 => NoHello,
            11 => Version,
            _ => BadRequest,
        }
    }
}

/// A run of contiguous slots in the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReserveItem {
    pub key: Key,
    pub parent: Option<Key>,
    pub bytes: u64,
    pub kind: u8,
    pub format: u16,
    pub cost_hint: u32,
    pub flags: u8,
    pub ttl_ms: u32,
    pub meta: Vec<u8>,
}

/// An entry as returned by an acquiring LOOKUP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryDesc {
    pub bytes: u64,
    pub kind: u8,
    pub format: u16,
    pub producer: u16,
    pub flags: u8,
    pub meta: Vec<u8>,
    pub ranges: Vec<Range>,
}

/// One engine's answer for a WHERE: the anchor (index into the request's list) it hinted, and how many leading chunks
/// of it are committed in RAM now (`n_resident` of `n_chunks`, `chunk_tokens` tokens each).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub anchor_idx: u32,
    pub engine_id: u16,
    pub name: String,
    pub namespace: u32,
    pub age_ms: u64,
    pub chunk_tokens: u32,
    pub n_chunks: u32,
    pub n_resident: u32,
    pub flags: u8,
}

/// A token run for the n-gram service: the hash history before it (newest first, -1 = none) and its tokens. The
/// broker hashes rows exactly as gufo does (EOS rules included) and returns `heads` rows per token, head-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NgramSeq {
    pub prev: [i32; 2],
    pub tokens: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    Hello { engine_id: u16, protocol_version: u16, name: String },
    Lookup { namespace: u32, acquire: bool, keys: Vec<Key> },
    Reserve { namespace: u32, items: Vec<ReserveItem> },
    Commit { namespace: u32, keys: Vec<Key> },
    Abort { namespace: u32, keys: Vec<Key> },
    Release { lease_ids: Vec<u64> },
    Touch { namespace: u32, keys: Vec<Key> },
    Stats,
    Wait { namespace: u32, key: Key, timeout_ms: u32 },
    /// Parked runs and the parking generation (dynamic pool).
    PoolState,
    /// "I no longer touch the runs parked as of generation `gen`."
    PoolAck { gen: u64 },
    /// Attach to a configured table (`--ngram-table NAME=PATH`); `rows` / `row_bytes` must match (identity check).
    NgramOpen { table: String, rows: u64, row_bytes: u32 },
    /// Rows of the runs, copied into this connection's staging area in request order.
    NgramGather { seqs: Vec<NgramSeq> },
    /// Warm the cache for these runs; nothing is copied (reply = status only).
    NgramPrefetch { seqs: Vec<NgramSeq> },
    /// `key=value` lines (model, weights, engine, layout, kv_format, numerics knobs); canonicalized by the broker.
    Domain { descriptor: String },
    Probe { namespace: u32, keys: Vec<Key> },
    Heat { namespace: u32, hot: bool, keys: Vec<Key> },
    /// The client's feature bits and the formats (name, version) it reads.
    Features { features: u64, formats: Vec<(String, u16)> },
    /// One request's keys per KV group, in group order: (kind 0 attention / 1 GDN state / other = unknown, keys).
    TouchRequest { namespace: u32, groups: Vec<(u8, Vec<Key>)> },
    TraceDump { reason: String },
    BgJobs { kind: u8, max: u32 },
    /// `sum`: XXH3-64 of the entry's raw bytes, None = the bytes do not decode
    BgChecksum { namespace: u32, key: Key, guard: Guard, sum: Option<u64> },
    /// `len`: the image's bytes (BG_END)
    BgRewrite { phase: u8, namespace: u32, key: Key, guard: Guard, len: u64 },
    BgSsdJobs { max: u32 },
    BgSsdDone { namespace: u32, key: Key, outcome: u8, file_bytes: u64 },
    BgCompactJobs { max: u32 },
    BgCompactDone { namespace: u32, key: Key, copied: bool },
    Invalidate { namespace: u32, keys: Vec<Key> },
    /// TOUCH_REQUEST with the request's tags (see Op::TouchTagged).
    TouchTagged { namespace: u32, session: u64, parent: u64, class: u8, agent: u8, flags: u8, groups: Vec<(u8, Vec<Key>)> },
    /// One request's keys per KV group in token order (same group encoding as TOUCH_REQUEST), under a router anchor.
    Hint { namespace: u32, anchor: Anchor, chunk_tokens: u32, flags: u8, groups: Vec<(u8, Vec<Key>)> },
    Where { anchors: Vec<Anchor> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelloReply {
    pub protocol_version: u16,
    pub broker_version: String,
    pub epoch: u64,
    pub pool_path: String,
    pub pool_bytes: u64,
    pub slot_bytes: u64,
    pub n_slots: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    /// Any request that failed as a whole (bad frame, no HELLO, version mismatch), and WAIT.
    Status,
    Hello(HelloReply),
    Lookup { n_hit: u32, lease_id: u64, entries: Vec<EntryDesc> },
    Reserve(Vec<(Status, Vec<Range>)>),
    Statuses(Vec<Status>),
    Stats(Vec<(String, u64)>),
    /// Dynamic pool: generation (bumped whenever the parked set changes) and the parked runs (slot ranges).
    PoolState { gen: u64, parked: Vec<Range> },
    /// N-gram staging: shared file to map, this connection's staging window in it, row format.
    NgramOpen { path: String, staging_offset: u64, staging_bytes: u64, row_bytes: u32, heads: u32 },
    /// Rows written to staging (tokens x heads), and how many of them came from disk.
    NgramGather { rows: u32, disk_rows: u32 },
    /// The domain id the broker derived (hash of the canonical descriptor).
    Domain { domain_id: u64 },
    /// Per probed key: 0 absent, 1 RAM, 2 SSD tier only.
    Probe(Vec<u8>),
    /// The broker's feature bits and the formats (name, version) it writes / serves.
    Features { features: u64, formats: Vec<(String, u16)> },
    /// TRACE_DUMP: the trace file written
    TraceDump { path: String },
    /// Jobs and the broker's settings for them: checksum bytes per second, compression chunk bytes (0 = class off),
    /// the broker's pool classes (the worker connects to each by name).
    BgJobs { cksum_bytes_s: u64, chunk: u64, classes: Vec<String>, jobs: Vec<BgJob> },
    /// BG_CHECKSUM: 0 stored, 1 verified, 2 corrupt (evicted), 3 raced, 4 busy; BG_REWRITE: 0 ok, 3 raced / refused.
    BgResult(u8),
    SsdJobs(Vec<SsdJob>),
    CompactJobs(Vec<CompactJob>),
    /// WHERE: at most one place per (engine, namespace)
    Where(Vec<Place>),
}

impl Request {
    pub fn op(&self) -> Op {
        match self {
            Request::Hello { .. } => Op::Hello,
            Request::Lookup { .. } => Op::Lookup,
            Request::Reserve { .. } => Op::Reserve,
            Request::Commit { .. } => Op::Commit,
            Request::Abort { .. } => Op::Abort,
            Request::Release { .. } => Op::Release,
            Request::Touch { .. } => Op::Touch,
            Request::Stats => Op::Stats,
            Request::Wait { .. } => Op::Wait,
            Request::PoolState => Op::PoolState,
            Request::PoolAck { .. } => Op::PoolAck,
            Request::NgramOpen { .. } => Op::NgramOpen,
            Request::NgramGather { .. } => Op::NgramGather,
            Request::NgramPrefetch { .. } => Op::NgramPrefetch,
            Request::Domain { .. } => Op::Domain,
            Request::Probe { .. } => Op::Probe,
            Request::Heat { .. } => Op::Heat,
            Request::Features { .. } => Op::Features,
            Request::TouchRequest { .. } => Op::TouchRequest,
            Request::TraceDump { .. } => Op::TraceDump,
            Request::BgJobs { .. } => Op::BgJobs,
            Request::BgChecksum { .. } => Op::BgChecksum,
            Request::BgRewrite { .. } => Op::BgRewrite,
            Request::BgSsdJobs { .. } => Op::BgSsdJobs,
            Request::BgSsdDone { .. } => Op::BgSsdDone,
            Request::BgCompactJobs { .. } => Op::BgCompactJobs,
            Request::BgCompactDone { .. } => Op::BgCompactDone,
            Request::Invalidate { .. } => Op::Invalidate,
            Request::TouchTagged { .. } => Op::TouchTagged,
            Request::Hint { .. } => Op::Hint,
            Request::Where { .. } => Op::Where,
        }
    }
}

// ------------------------------------------------------------------ encoding

#[derive(Default)]
pub struct Enc(pub Vec<u8>);

impl Enc {
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    pub fn u16(&mut self, v: u16) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn key(&mut self, k: &Key) -> &mut Self {
        self.0.extend_from_slice(k);
        self
    }
    pub fn blob(&mut self, b: &[u8]) -> &mut Self {
        self.u16(b.len() as u16);
        self.0.extend_from_slice(b);
        self
    }
    pub fn anchor(&mut self, a: &Anchor) -> &mut Self {
        self.0.extend_from_slice(a);
        self
    }
    pub fn keys(&mut self, ks: &[Key]) -> &mut Self {
        self.u32(ks.len() as u32);
        for k in ks {
            self.key(k);
        }
        self
    }
    pub fn ranges(&mut self, rs: &[Range]) -> &mut Self {
        self.u32(rs.len() as u32);
        for r in rs {
            self.u32(r.start).u32(r.len);
        }
        self
    }
}

pub struct Dec<'a> {
    b: &'a [u8],
    p: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Truncated;

type R<T> = Result<T, Truncated>;

impl<'a> Dec<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Dec { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        if self.b.len() - self.p < n {
            return Err(Truncated);
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> R<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> R<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> R<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn key(&mut self) -> R<Key> {
        Ok(self.take(32)?.try_into().unwrap())
    }
    pub fn blob(&mut self) -> R<Vec<u8>> {
        let n = self.u16()? as usize;
        Ok(self.take(n)?.to_vec())
    }
    pub fn string(&mut self) -> R<String> {
        Ok(String::from_utf8_lossy(&self.blob()?).into_owned())
    }
    /// A u32 count, sanity-checked against the bytes left (each element is at least `min_size` bytes).
    pub fn count(&mut self, min_size: usize) -> R<usize> {
        let n = self.u32()? as usize;
        if n.saturating_mul(min_size) > self.b.len() - self.p {
            return Err(Truncated);
        }
        Ok(n)
    }
    pub fn anchor(&mut self) -> R<Anchor> {
        Ok(self.take(16)?.try_into().unwrap())
    }
    pub fn keys(&mut self) -> R<Vec<Key>> {
        let n = self.count(32)?;
        (0..n).map(|_| self.key()).collect()
    }
    pub fn ranges(&mut self) -> R<Vec<Range>> {
        let n = self.count(8)?;
        (0..n).map(|_| Ok(Range { start: self.u32()?, len: self.u32()? })).collect()
    }
    pub fn done(&self) -> bool {
        self.p == self.b.len()
    }
}

impl Request {
    pub fn encode_payload(&self, e: &mut Enc) {
        match self {
            Request::Hello { engine_id, protocol_version, name } => {
                e.u16(*engine_id).u16(*protocol_version).blob(name.as_bytes());
            }
            Request::Lookup { namespace, acquire, keys } => {
                e.u32(*namespace).u8(*acquire as u8).keys(keys);
            }
            Request::Reserve { namespace, items } => {
                e.u32(*namespace).u32(items.len() as u32);
                for it in items {
                    e.key(&it.key);
                    match &it.parent {
                        Some(p) => e.u8(1).key(p),
                        None => e.u8(0),
                    };
                    e.u64(it.bytes).u8(it.kind).u16(it.format).u32(it.cost_hint).u8(it.flags).u32(it.ttl_ms);
                    e.blob(&it.meta);
                }
            }
            Request::Commit { namespace, keys } | Request::Abort { namespace, keys } | Request::Touch { namespace, keys } => {
                e.u32(*namespace).keys(keys);
            }
            Request::Release { lease_ids } => {
                e.u32(lease_ids.len() as u32);
                for l in lease_ids {
                    e.u64(*l);
                }
            }
            Request::Stats => {}
            Request::Wait { namespace, key, timeout_ms } => {
                e.u32(*namespace).key(key).u32(*timeout_ms);
            }
            Request::PoolState => {}
            Request::PoolAck { gen } => {
                e.u64(*gen);
            }
            Request::NgramOpen { table, rows, row_bytes } => {
                e.blob(table.as_bytes()).u64(*rows).u32(*row_bytes);
            }
            Request::NgramGather { seqs } | Request::NgramPrefetch { seqs } => enc_seqs(e, seqs),
            Request::Domain { descriptor } => {
                e.blob(descriptor.as_bytes());
            }
            Request::Probe { namespace, keys } => {
                e.u32(*namespace).keys(keys);
            }
            Request::Heat { namespace, hot, keys } => {
                e.u32(*namespace).u8(*hot as u8).keys(keys);
            }
            Request::Features { features, formats } => enc_features(e, *features, formats),
            Request::TouchRequest { namespace, groups } => {
                e.u32(*namespace).u32(groups.len() as u32);
                for (kind, keys) in groups {
                    e.u8(*kind).keys(keys);
                }
            }
            Request::TraceDump { reason } => {
                e.blob(reason.as_bytes());
            }
            Request::BgJobs { kind, max } => {
                e.u8(*kind).u32(*max);
            }
            Request::BgChecksum { namespace, key, guard, sum } => {
                e.u32(*namespace).key(key);
                enc_guard(e, guard);
                match sum {
                    Some(s) => e.u8(1).u64(*s),
                    None => e.u8(0).u64(0),
                };
            }
            Request::BgRewrite { phase, namespace, key, guard, len } => {
                e.u8(*phase).u32(*namespace).key(key);
                enc_guard(e, guard);
                e.u64(*len);
            }
            Request::BgSsdJobs { max } => {
                e.u32(*max);
            }
            Request::BgSsdDone { namespace, key, outcome, file_bytes } => {
                e.u32(*namespace).key(key).u8(*outcome).u64(*file_bytes);
            }
            Request::BgCompactJobs { max } => {
                e.u32(*max);
            }
            Request::BgCompactDone { namespace, key, copied } => {
                e.u32(*namespace).key(key).u8(*copied as u8);
            }
            Request::Invalidate { namespace, keys } => {
                e.u32(*namespace).keys(keys);
            }
            Request::TouchTagged { namespace, session, parent, class, agent, flags, groups } => {
                e.u32(*namespace).u64(*session).u64(*parent).u8(*class).u8(*agent).u8(*flags).u32(groups.len() as u32);
                for (kind, keys) in groups {
                    e.u8(*kind).keys(keys);
                }
            }
            Request::Hint { namespace, anchor, chunk_tokens, flags, groups } => {
                e.u32(*namespace).anchor(anchor).u32(*chunk_tokens).u8(*flags).u32(groups.len() as u32);
                for (kind, keys) in groups {
                    e.u8(*kind).keys(keys);
                }
            }
            Request::Where { anchors } => {
                e.u32(anchors.len() as u32);
                for a in anchors {
                    e.anchor(a);
                }
            }
        }
    }

    pub fn decode(op: Op, b: &[u8]) -> R<Request> {
        let mut d = Dec::new(b);
        let r = match op {
            Op::Hello => Request::Hello { engine_id: d.u16()?, protocol_version: d.u16()?, name: d.string()? },
            Op::Lookup => Request::Lookup { namespace: d.u32()?, acquire: d.u8()? != 0, keys: d.keys()? },
            Op::Reserve => {
                let namespace = d.u32()?;
                let n = d.count(32 + 1 + 8 + 1 + 2 + 4 + 1 + 4 + 2)?;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    let key = d.key()?;
                    let parent = if d.u8()? != 0 { Some(d.key()?) } else { None };
                    items.push(ReserveItem {
                        key,
                        parent,
                        bytes: d.u64()?,
                        kind: d.u8()?,
                        format: d.u16()?,
                        cost_hint: d.u32()?,
                        flags: d.u8()?,
                        ttl_ms: d.u32()?,
                        meta: d.blob()?,
                    });
                }
                Request::Reserve { namespace, items }
            }
            Op::Commit => Request::Commit { namespace: d.u32()?, keys: d.keys()? },
            Op::Abort => Request::Abort { namespace: d.u32()?, keys: d.keys()? },
            Op::Touch => Request::Touch { namespace: d.u32()?, keys: d.keys()? },
            Op::Release => {
                let n = d.count(8)?;
                Request::Release { lease_ids: (0..n).map(|_| d.u64()).collect::<R<_>>()? }
            }
            Op::Stats => Request::Stats,
            Op::Wait => Request::Wait { namespace: d.u32()?, key: d.key()?, timeout_ms: d.u32()? },
            Op::PoolState => Request::PoolState,
            Op::PoolAck => Request::PoolAck { gen: d.u64()? },
            Op::NgramOpen => Request::NgramOpen { table: d.string()?, rows: d.u64()?, row_bytes: d.u32()? },
            Op::NgramGather => Request::NgramGather { seqs: dec_seqs(&mut d)? },
            Op::NgramPrefetch => Request::NgramPrefetch { seqs: dec_seqs(&mut d)? },
            Op::Domain => Request::Domain { descriptor: d.string()? },
            Op::Probe => Request::Probe { namespace: d.u32()?, keys: d.keys()? },
            Op::Heat => Request::Heat { namespace: d.u32()?, hot: d.u8()? != 0, keys: d.keys()? },
            Op::Features => {
                let (features, formats) = dec_features(&mut d)?;
                Request::Features { features, formats }
            }
            Op::TouchRequest => {
                let namespace = d.u32()?;
                let n = d.count(1 + 4)?;
                let groups = (0..n).map(|_| Ok((d.u8()?, d.keys()?))).collect::<R<_>>()?;
                Request::TouchRequest { namespace, groups }
            }
            Op::TraceDump => Request::TraceDump { reason: d.string()? },
            Op::BgJobs => Request::BgJobs { kind: d.u8()?, max: d.u32()? },
            Op::BgChecksum => {
                let (namespace, key, guard) = (d.u32()?, d.key()?, dec_guard(&mut d)?);
                let has = d.u8()? != 0;
                let s = d.u64()?;
                Request::BgChecksum { namespace, key, guard, sum: has.then_some(s) }
            }
            Op::BgRewrite => {
                let (phase, namespace, key) = (d.u8()?, d.u32()?, d.key()?);
                Request::BgRewrite { phase, namespace, key, guard: dec_guard(&mut d)?, len: d.u64()? }
            }
            Op::BgSsdJobs => Request::BgSsdJobs { max: d.u32()? },
            Op::BgSsdDone => Request::BgSsdDone { namespace: d.u32()?, key: d.key()?, outcome: d.u8()?, file_bytes: d.u64()? },
            Op::BgCompactJobs => Request::BgCompactJobs { max: d.u32()? },
            Op::BgCompactDone => Request::BgCompactDone { namespace: d.u32()?, key: d.key()?, copied: d.u8()? != 0 },
            Op::Invalidate => Request::Invalidate { namespace: d.u32()?, keys: d.keys()? },
            Op::TouchTagged => {
                let (namespace, session, parent, class, agent, flags) = (d.u32()?, d.u64()?, d.u64()?, d.u8()?, d.u8()?, d.u8()?);
                let n = d.count(1 + 4)?;
                let groups = (0..n).map(|_| Ok((d.u8()?, d.keys()?))).collect::<R<_>>()?;
                Request::TouchTagged { namespace, session, parent, class, agent, flags, groups }
            }
            Op::Hint => {
                let (namespace, anchor, chunk_tokens, flags) = (d.u32()?, d.anchor()?, d.u32()?, d.u8()?);
                let n = d.count(1 + 4)?;
                let groups = (0..n).map(|_| Ok((d.u8()?, d.keys()?))).collect::<R<_>>()?;
                Request::Hint { namespace, anchor, chunk_tokens, flags, groups }
            }
            Op::Where => {
                let n = d.count(16)?;
                Request::Where { anchors: (0..n).map(|_| d.anchor()).collect::<R<_>>()? }
            }
        };
        if !d.done() {
            return Err(Truncated);
        }
        Ok(r)
    }
}

/// `u32 n_seqs | n_seqs x (i32 prev0, i32 prev1, u32 n, n x i32 token)`
fn enc_seqs(e: &mut Enc, seqs: &[NgramSeq]) {
    e.u32(seqs.len() as u32);
    for s in seqs {
        e.u32(s.prev[0] as u32).u32(s.prev[1] as u32).u32(s.tokens.len() as u32);
        for &t in &s.tokens {
            e.u32(t as u32);
        }
    }
}

fn dec_seqs(d: &mut Dec) -> R<Vec<NgramSeq>> {
    let n = d.count(12)?;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let prev = [d.u32()? as i32, d.u32()? as i32];
        let m = d.count(4)?;
        v.push(NgramSeq { prev, tokens: (0..m).map(|_| Ok(d.u32()? as i32)).collect::<R<_>>()? });
    }
    Ok(v)
}

fn enc_entry(e: &mut Enc, x: &EntryDesc) {
    e.u64(x.bytes).u8(x.kind).u16(x.format).u16(x.producer).u8(x.flags).blob(&x.meta).ranges(&x.ranges);
}

fn enc_guard(e: &mut Enc, g: &Guard) {
    e.u64(g.entry_id).ranges(&g.ranges).u8(g.compressed as u8).u8(g.rewriting as u8);
}

fn dec_guard(d: &mut Dec) -> R<Guard> {
    Ok(Guard { entry_id: d.u64()?, ranges: d.ranges()?, compressed: d.u8()? != 0, rewriting: d.u8()? != 0 })
}

fn enc_features(e: &mut Enc, features: u64, formats: &[(String, u16)]) {
    e.u64(features).u32(formats.len() as u32);
    for (name, v) in formats {
        e.blob(name.as_bytes()).u16(*v);
    }
}

fn dec_features(d: &mut Dec) -> R<(u64, Vec<(String, u16)>)> {
    let features = d.u64()?;
    let n = d.count(2 + 2)?;
    let formats = (0..n).map(|_| Ok((d.string()?, d.u16()?))).collect::<R<_>>()?;
    Ok((features, formats))
}

fn dec_entry(d: &mut Dec) -> R<EntryDesc> {
    Ok(EntryDesc {
        bytes: d.u64()?,
        kind: d.u8()?,
        format: d.u16()?,
        producer: d.u16()?,
        flags: d.u8()?,
        meta: d.blob()?,
        ranges: d.ranges()?,
    })
}

impl Reply {
    pub fn encode_payload(&self, status: Status, e: &mut Enc) {
        e.u16(status as u16);
        match self {
            Reply::Status => {}
            Reply::Hello(h) => {
                e.u16(h.protocol_version).blob(h.broker_version.as_bytes()).u64(h.epoch).blob(h.pool_path.as_bytes());
                e.u64(h.pool_bytes).u64(h.slot_bytes).u32(h.n_slots);
            }
            Reply::Lookup { n_hit, lease_id, entries } => {
                e.u32(*n_hit).u64(*lease_id);
                for x in entries {
                    enc_entry(e, x);
                }
            }
            Reply::Reserve(v) => {
                e.u32(v.len() as u32);
                for (s, rs) in v {
                    e.u16(*s as u16).ranges(rs);
                }
            }
            Reply::Statuses(v) => {
                e.u32(v.len() as u32);
                for s in v {
                    e.u16(*s as u16);
                }
            }
            Reply::Stats(v) => {
                e.u32(v.len() as u32);
                for (k, x) in v {
                    e.blob(k.as_bytes()).u64(*x);
                }
            }
            Reply::PoolState { gen, parked } => {
                e.u64(*gen).ranges(parked);
            }
            Reply::NgramOpen { path, staging_offset, staging_bytes, row_bytes, heads } => {
                e.blob(path.as_bytes()).u64(*staging_offset).u64(*staging_bytes).u32(*row_bytes).u32(*heads);
            }
            Reply::NgramGather { rows, disk_rows } => {
                e.u32(*rows).u32(*disk_rows);
            }
            Reply::Domain { domain_id } => {
                e.u64(*domain_id);
            }
            Reply::Probe(v) => {
                e.blob(v);
            }
            Reply::Features { features, formats } => enc_features(e, *features, formats),
            Reply::TraceDump { path } => {
                e.blob(path.as_bytes());
            }
            Reply::BgJobs { cksum_bytes_s, chunk, classes, jobs } => {
                e.u64(*cksum_bytes_s).u64(*chunk).u32(classes.len() as u32);
                for c in classes {
                    e.blob(c.as_bytes());
                }
                e.u32(jobs.len() as u32);
                for j in jobs {
                    e.u32(j.namespace).key(&j.key).u64(j.bytes);
                    enc_guard(e, &j.guard);
                }
            }
            Reply::BgResult(r) => {
                e.u8(*r);
            }
            Reply::CompactJobs(v) => {
                e.u32(v.len() as u32);
                for j in v {
                    e.u32(j.namespace).key(&j.key);
                    enc_guard(e, &j.guard);
                    e.ranges(&j.new);
                }
            }
            Reply::Where(v) => {
                e.u32(v.len() as u32);
                for p in v {
                    e.u32(p.anchor_idx).u16(p.engine_id).blob(p.name.as_bytes()).u32(p.namespace).u64(p.age_ms);
                    e.u32(p.chunk_tokens).u32(p.n_chunks).u32(p.n_resident).u8(p.flags);
                }
            }
            Reply::SsdJobs(v) => {
                e.u32(v.len() as u32);
                for j in v {
                    e.u32(j.namespace).key(&j.key);
                    enc_guard(e, &j.guard);
                    e.u8(j.kind).u16(j.format).u16(j.producer).u64(j.hits).u64(j.bytes).u64(j.last_used_ms);
                    e.blob(&j.meta).blob(j.path.as_bytes()).u8(j.codec);
                }
            }
        }
    }

    /// Decode a reply to request `op`. A non-OK whole-request status yields `Reply::Status`.
    pub fn decode(op: Op, b: &[u8]) -> R<(Status, Reply)> {
        let mut d = Dec::new(b);
        let status = Status::from_u16(d.u16()?);
        if d.done() {
            return Ok((status, Reply::Status));
        }
        let r = match op {
            Op::Hello => Reply::Hello(HelloReply {
                protocol_version: d.u16()?,
                broker_version: d.string()?,
                epoch: d.u64()?,
                pool_path: d.string()?,
                pool_bytes: d.u64()?,
                slot_bytes: d.u64()?,
                n_slots: d.u32()?,
            }),
            Op::Lookup => {
                let n_hit = d.u32()?;
                let lease_id = d.u64()?;
                let mut entries = Vec::new();
                while !d.done() {
                    entries.push(dec_entry(&mut d)?);
                }
                Reply::Lookup { n_hit, lease_id, entries }
            }
            Op::Reserve => {
                let n = d.count(6)?;
                Reply::Reserve((0..n).map(|_| Ok((Status::from_u16(d.u16()?), d.ranges()?))).collect::<R<_>>()?)
            }
            Op::Commit | Op::Abort | Op::Release | Op::Touch | Op::Heat | Op::Invalidate => {
                let n = d.count(2)?;
                Reply::Statuses((0..n).map(|_| Ok(Status::from_u16(d.u16()?))).collect::<R<_>>()?)
            }
            Op::Stats => {
                let n = d.count(10)?;
                Reply::Stats((0..n).map(|_| Ok((d.string()?, d.u64()?))).collect::<R<_>>()?)
            }
            Op::Wait => Reply::Status,
            Op::PoolState => Reply::PoolState { gen: d.u64()?, parked: d.ranges()? },
            Op::PoolAck => Reply::Status,
            Op::NgramOpen => Reply::NgramOpen {
                path: d.string()?,
                staging_offset: d.u64()?,
                staging_bytes: d.u64()?,
                row_bytes: d.u32()?,
                heads: d.u32()?,
            },
            Op::NgramGather => Reply::NgramGather { rows: d.u32()?, disk_rows: d.u32()? },
            Op::NgramPrefetch => Reply::Status,
            Op::Domain => Reply::Domain { domain_id: d.u64()? },
            Op::Probe => Reply::Probe(d.blob()?),
            Op::Features => {
                let (features, formats) = dec_features(&mut d)?;
                Reply::Features { features, formats }
            }
            Op::TouchRequest | Op::TouchTagged => Reply::Status,
            Op::TouchRequest | Op::Hint => Reply::Status,
            Op::Where => {
                let n = d.count(4 + 2 + 2 + 4 + 8 + 4 * 3 + 1)?;
                Reply::Where((0..n).map(|_| Ok(Place { anchor_idx: d.u32()?, engine_id: d.u16()?, name: d.string()?,
                                                         namespace: d.u32()?, age_ms: d.u64()?, chunk_tokens: d.u32()?,
                                                         n_chunks: d.u32()?, n_resident: d.u32()?, flags: d.u8()? }))
                    .collect::<R<_>>()?)
            }
            Op::TraceDump => Reply::TraceDump { path: d.string()? },
            Op::BgJobs => {
                let (cksum_bytes_s, chunk) = (d.u64()?, d.u64()?);
                let nc = d.count(2)?;
                let classes = (0..nc).map(|_| d.string()).collect::<R<_>>()?;
                let n = d.count(4 + 32 + 8 + 8 + 4 + 2)?;
                let jobs = (0..n).map(|_| Ok(BgJob { namespace: d.u32()?, key: d.key()?, bytes: d.u64()?, guard: dec_guard(&mut d)? }))
                    .collect::<R<_>>()?;
                Reply::BgJobs { cksum_bytes_s, chunk, classes, jobs }
            }
            Op::BgChecksum | Op::BgRewrite | Op::BgSsdDone | Op::BgCompactDone => Reply::BgResult(d.u8()?),
            Op::BgCompactJobs => {
                let n = d.count(4 + 32 + 8 + 4 + 2 + 4)?;
                let jobs = (0..n).map(|_| Ok(CompactJob { namespace: d.u32()?, key: d.key()?, guard: dec_guard(&mut d)?, new: d.ranges()? }))
                    .collect::<R<_>>()?;
                Reply::CompactJobs(jobs)
            }
            Op::BgSsdJobs => {
                let n = d.count(4 + 32 + 8 + 4 + 2 + 1 + 2 + 2 + 8 * 3 + 2 + 2 + 1)?;
                let jobs = (0..n).map(|_| {
                    let (namespace, key, guard) = (d.u32()?, d.key()?, dec_guard(&mut d)?);
                    Ok(SsdJob { namespace, key, guard, kind: d.u8()?, format: d.u16()?, producer: d.u16()?, hits: d.u64()?,
                                bytes: d.u64()?, last_used_ms: d.u64()?, meta: d.blob()?, path: d.string()?, codec: d.u8()? })
                }).collect::<R<_>>()?;
                Reply::SsdJobs(jobs)
            }
        };
        if !d.done() {
            return Err(Truncated);
        }
        Ok((status, r))
    }
}

// ------------------------------------------------------------------ framing

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub op: u16,
    pub flags: u16,
    pub req_id: u64,
}

/// Build one complete frame: header + payload.
pub fn frame(op: u16, flags: u16, req_id: u64, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + HEADER + payload.len());
    v.extend_from_slice(&((HEADER + payload.len()) as u32).to_le_bytes());
    v.extend_from_slice(&op.to_le_bytes());
    v.extend_from_slice(&flags.to_le_bytes());
    v.extend_from_slice(&req_id.to_le_bytes());
    v.extend_from_slice(payload);
    v
}

pub fn request_frame(req_id: u64, r: &Request) -> Vec<u8> {
    let mut e = Enc::default();
    r.encode_payload(&mut e);
    frame(r.op() as u16, 0, req_id, &e.0)
}

pub fn reply_frame(op: u16, req_id: u64, status: Status, r: &Reply) -> Vec<u8> {
    let mut e = Enc::default();
    r.encode_payload(status, &mut e);
    frame(op, FLAG_RESP, req_id, &e.0)
}

/// Read one frame. `Ok(None)` on a clean EOF before the length prefix.
pub fn read_frame<Rd: Read>(r: &mut Rd, buf: &mut Vec<u8>) -> io::Result<Option<FrameHeader>> {
    let mut lb = [0u8; 4];
    match r.read_exact(&mut lb) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(lb) as usize;
    if !(HEADER..=MAX_FRAME).contains(&len) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("bad frame length {len}")));
    }
    buf.resize(len, 0);
    r.read_exact(buf)?;
    let h = FrameHeader {
        op: u16::from_le_bytes([buf[0], buf[1]]),
        flags: u16::from_le_bytes([buf[2], buf[3]]),
        req_id: u64::from_le_bytes(buf[4..12].try_into().unwrap()),
    };
    buf.drain(..HEADER);
    Ok(Some(h))
}

pub fn write_all<W: Write>(w: &mut W, b: &[u8]) -> io::Result<()> {
    w.write_all(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(n: u8) -> Key {
        [n; 32]
    }

    #[test]
    fn request_roundtrip() {
        let reqs = vec![
            Request::Hello { engine_id: 7, protocol_version: 1, name: "arc".into() },
            Request::Lookup { namespace: 3, acquire: true, keys: vec![k(1), k(2)] },
            Request::Reserve {
                namespace: 3,
                items: vec![ReserveItem {
                    key: k(1),
                    parent: Some(k(9)),
                    bytes: 123456,
                    kind: 2,
                    format: 5,
                    cost_hint: 40,
                    flags: ENTRY_EPHEMERAL,
                    ttl_ms: 1000,
                    meta: vec![1, 2, 3],
                }],
            },
            Request::Commit { namespace: 1, keys: vec![k(4)] },
            Request::Abort { namespace: 1, keys: vec![] },
            Request::Release { lease_ids: vec![1, 2, 3] },
            Request::Touch { namespace: 0, keys: vec![k(5)] },
            Request::Heat { namespace: 3, hot: false, keys: vec![k(6), k(7)] },
            Request::Stats,
            Request::PoolState,
            Request::PoolAck { gen: 9 },
            Request::Wait { namespace: 2, key: k(8), timeout_ms: 500 },
            Request::NgramOpen { table: "flash-next".into(), rows: 320_001_536, row_bytes: 90 },
            Request::NgramGather {
                seqs: vec![
                    NgramSeq { prev: [-1, -1], tokens: vec![1, 248044, 7] },
                    NgramSeq { prev: [5, 6], tokens: vec![] },
                ],
            },
            Request::NgramPrefetch { seqs: vec![NgramSeq { prev: [3, -1], tokens: vec![9] }] },
            Request::Domain { descriptor: "model=qwen3.8-27b\nengine=example-arc".into() },
            Request::Features { features: FEAT_TOUCH_REQUEST, formats: vec![(FMT_SLOT_CODEC.into(), 1)] },
            Request::Features { features: 0, formats: vec![] },
            Request::TouchRequest { namespace: 4, groups: vec![(1, vec![k(9)]), (0, vec![k(1), k(2)]), (3, vec![])] },
            Request::TouchTagged { namespace: 4, session: u64::MAX, parent: 7, class: CLASS_AGENT, agent: 0, flags: 0, groups: vec![(0, vec![k(1)]), (1, vec![])] },
            Request::BgJobs { kind: BG_CHECKSUM, max: 16 },
            Request::BgChecksum { namespace: 2, key: k(3), guard: Guard { entry_id: 9, ranges: vec![], compressed: false, rewriting: true }, sum: None },
            Request::BgRewrite { phase: BG_SKIP, namespace: 2, key: k(3), guard: Guard { entry_id: 9, ranges: vec![Range { start: 1, len: 2 }], compressed: false, rewriting: false }, len: 0 },
            Request::Hint { namespace: 5, anchor: [7; 16], chunk_tokens: 3200, flags: 0, groups: vec![(0, vec![k(1)]), (1, vec![])] },
            Request::Where { anchors: vec![] },
            Request::Where { anchors: vec![[1; 16], [2; 16], [3; 16]] },
        ];
        for r in reqs {
            let f = request_frame(42, &r);
            let mut buf = Vec::new();
            let h = read_frame(&mut &f[..], &mut buf).unwrap().unwrap();
            assert_eq!(h.req_id, 42);
            let d = Request::decode(Op::from_u16(h.op).unwrap(), &buf).unwrap();
            assert_eq!(d, r);
        }
    }

    #[test]
    fn reply_roundtrip() {
        let e = EntryDesc {
            bytes: 9,
            kind: 1,
            format: 2,
            producer: 3,
            flags: 0,
            meta: vec![7],
            ranges: vec![Range { start: 4, len: 2 }],
        };
        let cases = vec![
            (Op::Lookup, Reply::Lookup { n_hit: 2, lease_id: 5, entries: vec![e.clone(), e] }),
            (Op::Where, Reply::Where(vec![Place { anchor_idx: 0, engine_id: 1, name: "rtx5090".into(), namespace: 0x8000_0001,
                                                  age_ms: 9, chunk_tokens: 3200, n_chunks: 8, n_resident: 0, flags: 0 },
                                          Place { anchor_idx: 2, engine_id: 2, name: String::new(), namespace: 1, age_ms: 0,
                                                  chunk_tokens: 1, n_chunks: 1, n_resident: 1, flags: HINT_VRAM }])),
            (Op::Reserve, Reply::Reserve(vec![(Status::Ok, vec![Range { start: 1, len: 1 }]), (Status::Busy, vec![])])),
            (Op::Commit, Reply::Statuses(vec![Status::Ok, Status::NotOwner])),
            (Op::Stats, Reply::Stats(vec![("a".into(), 1)])),
            (Op::PoolState, Reply::PoolState { gen: 7, parked: vec![Range { start: 3, len: 2 }, Range { start: 40, len: 1 }] }),
            (Op::PoolAck, Reply::Status),
            (
                Op::Hello,
                Reply::Hello(HelloReply {
                    protocol_version: 1,
                    broker_version: "x".into(),
                    epoch: 9,
                    pool_path: "/p".into(),
                    pool_bytes: 1,
                    slot_bytes: 1,
                    n_slots: 1,
                }),
            ),
            (
                Op::NgramOpen,
                Reply::NgramOpen { path: "/dev/shm/x".into(), staging_offset: 1 << 24, staging_bytes: 1 << 24, row_bytes: 90, heads: 16 },
            ),
            (Op::NgramGather, Reply::NgramGather { rows: 32, disk_rows: 3 }),
            (Op::Domain, Reply::Domain { domain_id: 0x1234_5678_9abc }),
            (Op::Features, Reply::Features { features: 3, formats: vec![("slot_codec".into(), 1), ("trace".into(), 2)] }),
            (Op::BgJobs, Reply::BgJobs { cksum_bytes_s: 1 << 28, chunk: 32768, classes: vec!["default".into(), "arc-27b".into()], jobs: vec![] }),
            (Op::BgRewrite, Reply::BgResult(0)),
        ];
        for (op, r) in cases {
            let mut enc = Enc::default();
            r.encode_payload(Status::Ok, &mut enc);
            assert_eq!(Reply::decode(op, &enc.0).unwrap(), (Status::Ok, r));
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(Request::decode(Op::Lookup, &[1, 2, 3]).is_err());
        // count larger than the payload
        let mut e = Enc::default();
        e.u32(0).u8(0).u32(1_000_000);
        assert!(Request::decode(Op::Lookup, &e.0).is_err());
        // trailing bytes
        let mut e = Enc::default();
        e.u32(0).key(&[0; 32]).u32(5).u8(0);
        assert!(Request::decode(Op::Wait, &e.0).is_err());
    }

    /// Wire-format pinning (docs/protocol.md "Versioning"): the exact bytes of one canonical request and reply per
    /// op. Other clients (gufo's C++ client, other connectors) speak these bytes; a change to an existing encoding must fail
    /// here. Adding an op or a feature changes nothing pinned; changing one needs a new op or PROTOCOL_VERSION.
    #[test]
    fn wire_format_pinned() {
        fn fnv(h: &mut u64, b: &[u8]) {
            for x in b {
                *h = (*h ^ *x as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        let k = |n: u8| [n; 32];
        let reqs = vec![
            Request::Hello { engine_id: 7, protocol_version: 1, name: "e".into() },
            Request::Lookup { namespace: 1, acquire: true, keys: vec![k(1)] },
            Request::Reserve { namespace: 1, items: vec![ReserveItem { key: k(2), parent: Some(k(1)), bytes: 5, kind: 1,
                                                                       format: 2, cost_hint: 3, flags: 1, ttl_ms: 4,
                                                                       meta: vec![9] }] },
            Request::Commit { namespace: 1, keys: vec![k(2)] },
            Request::Abort { namespace: 1, keys: vec![k(2)] },
            Request::Release { lease_ids: vec![5] },
            Request::Touch { namespace: 1, keys: vec![k(3)] },
            Request::Stats,
            Request::Wait { namespace: 1, key: k(4), timeout_ms: 6 },
            Request::PoolState,
            Request::PoolAck { gen: 8 },
            Request::NgramOpen { table: "t".into(), rows: 9, row_bytes: 10 },
            Request::NgramGather { seqs: vec![NgramSeq { prev: [1, -1], tokens: vec![2] }] },
            Request::NgramPrefetch { seqs: vec![] },
            Request::Domain { descriptor: "m=x".into() },
            Request::Probe { namespace: 1, keys: vec![k(5)] },
            Request::Heat { namespace: 1, hot: true, keys: vec![k(6)] },
            Request::Features { features: FEAT_TOUCH_REQUEST, formats: vec![("slot_codec".into(), 1)] },
            Request::TouchRequest { namespace: 1, groups: vec![(1, vec![k(7)]), (0, vec![k(8)])] },
            Request::TraceDump { reason: "r".into() },
            Request::BgJobs { kind: BG_COMPRESS, max: 8 },
            Request::BgChecksum { namespace: 1, key: k(9), guard: Guard { entry_id: 3, ranges: vec![Range { start: 4, len: 1 }],
                                                                          compressed: true, rewriting: false }, sum: Some(5) },
            Request::BgRewrite { phase: BG_END, namespace: 1, key: k(9), guard: Guard { entry_id: 3, ranges: vec![Range { start: 4, len: 1 }],
                                                                                        compressed: false, rewriting: false }, len: 77 },
            Request::BgSsdJobs { max: 4 },
            Request::BgSsdDone { namespace: 1, key: k(9), outcome: 0, file_bytes: 8192 },
            Request::BgCompactJobs { max: 2 },
            Request::BgCompactDone { namespace: 1, key: k(9), copied: true },
            Request::Invalidate { namespace: 1, keys: vec![k(9), k(10)] },
            Request::TouchTagged { namespace: 1, session: 0x1234, parent: 0, class: CLASS_LOW, agent: b'm', flags: TAG_TERMINAL, groups: vec![(0, vec![k(8)]), (1, vec![k(7)])] },
            Request::Hint { namespace: 1, anchor: [3; 16], chunk_tokens: 1664, flags: HINT_VRAM,
                            groups: vec![(0, vec![k(1), k(2)]), (1, vec![k(3)])] },
            Request::Where { anchors: vec![[4; 16], [5; 16]] },
        ];
        assert_eq!(reqs.len(), Op::ALL.len(), "a canonical request per op");
        let e = EntryDesc { bytes: 1, kind: 0, format: 1, producer: 2, flags: 0, meta: vec![3], ranges: vec![Range { start: 4, len: 1 }] };
        let reps = vec![
            (Op::Hello, Reply::Hello(HelloReply { protocol_version: 1, broker_version: "b".into(), epoch: 2, pool_path: "/p".into(),
                                                  pool_bytes: 3, slot_bytes: 4, n_slots: 5 })),
            (Op::Lookup, Reply::Lookup { n_hit: 1, lease_id: 2, entries: vec![e] }),
            (Op::Reserve, Reply::Reserve(vec![(Status::Ok, vec![Range { start: 1, len: 2 }])])),
            (Op::Commit, Reply::Statuses(vec![Status::Ok, Status::Busy])),
            (Op::Stats, Reply::Stats(vec![("s".into(), 1)])),
            (Op::PoolState, Reply::PoolState { gen: 1, parked: vec![Range { start: 2, len: 3 }] }),
            (Op::NgramOpen, Reply::NgramOpen { path: "/n".into(), staging_offset: 1, staging_bytes: 2, row_bytes: 3, heads: 4 }),
            (Op::NgramGather, Reply::NgramGather { rows: 1, disk_rows: 2 }),
            (Op::Domain, Reply::Domain { domain_id: 7 }),
            (Op::Probe, Reply::Probe(vec![0, 1, 2])),
            (Op::Features, Reply::Features { features: 1, formats: vec![("slot_codec".into(), 1)] }),
            (Op::TraceDump, Reply::TraceDump { path: "/t".into() }),
            (Op::BgJobs, Reply::BgJobs { cksum_bytes_s: 1, chunk: 2, classes: vec!["c".into()], jobs: vec![BgJob { namespace: 3, key: k(4), bytes: 5,
                guard: Guard { entry_id: 6, ranges: vec![Range { start: 7, len: 1 }], compressed: false, rewriting: false } }] }),
            (Op::BgChecksum, Reply::BgResult(2)),
            (Op::BgSsdJobs, Reply::SsdJobs(vec![SsdJob { namespace: 1, key: k(2), guard: Guard { entry_id: 3, ranges: vec![Range { start: 4, len: 2 }],
                compressed: true, rewriting: false }, kind: 1, format: 2, producer: 3, hits: 4, bytes: 5, last_used_ms: 6,
                meta: vec![7], path: "/d/f.kvd".into(), codec: 1 }])),
            (Op::BgCompactJobs, Reply::CompactJobs(vec![CompactJob { namespace: 1, key: k(2), guard: Guard { entry_id: 3,
                ranges: vec![Range { start: 9, len: 1 }], compressed: false, rewriting: false }, new: vec![Range { start: 1, len: 1 }] }])),
            (Op::Where, Reply::Where(vec![Place { anchor_idx: 1, engine_id: 2, name: "arc".into(), namespace: 3, age_ms: 4,
                                                  chunk_tokens: 1664, n_chunks: 6, n_resident: 5, flags: HINT_VRAM }])),
        ];
        let digest = |b: &[u8]| {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            fnv(&mut h, b);
            h
        };
        let mut got: Vec<String> = reqs.iter().map(|r| format!("req {} {:016x}", r.op().name(), digest(&request_frame(1, r)))).collect();
        for (op, r) in &reps {
            let mut enc = Enc::default();
            r.encode_payload(Status::Ok, &mut enc);
            got.push(format!("rep {} {:016x}", op.name(), digest(&enc.0)));
        }
        let pinned: Vec<&str> = WIRE_PINNED.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        for p in &pinned {
            assert!(got.iter().any(|g| g == p), "wire format of `{p}` changed: an existing op's bytes must never change; \
                     add an op / feature instead, or bump PROTOCOL_VERSION.\nnow:\n{}", got.join("\n"));
        }
        for g in &got {
            assert!(pinned.contains(&g.as_str()), "`{g}` is not pinned: add it to WIRE_PINNED (new ops only)\nnow:\n{}",
                    got.join("\n"));
        }
    }
}

/// Pinned wire bytes per op (tests::wire_format_pinned): FNV-1a 64 of one canonical frame each. Lines are only ever
/// added (a new op); an existing line never changes.
#[cfg(test)]
const WIRE_PINNED: &str = "
    req hello 56bfaf4ba9683b1c
    req lookup 17b79dc6f0932288
    req reserve 2e7409db177fd3ba
    req commit a1c6777c429db5c4
    req abort d03a3987a31bc895
    req release 5ca9fae84f1101ee
    req touch a962148f0fcfc2d7
    req stats fb0192cdcbf2b5c0
    req wait d826a214b283c62e
    req pool_state e5889a0e818f2722
    req pool_ack 9c68a71706c12623
    req ngram_open 556edd1186505c51
    req ngram_gather 0d5c78f56db15c16
    req ngram_prefetch f602b82fd072ef1a
    req domain 7bc190c2650f6ee9
    req probe ab82a349f65cf1b0
    req heat 09be89b29d87ee71
    req features f7c62f6968fcde6a
    req touch_request 93a2ccd58f6456b1
    rep hello 97dc152ff214b470
    rep lookup fb9cbb8765509264
    rep reserve 34c51d843e30eb7e
    rep commit a7c1c9de36cfbd7d
    rep stats 45d8e70aaca4fed3
    rep pool_state 6964931c6a6c96ec
    rep ngram_open ff4b3c1ceeec8ca6
    rep ngram_gather 8acd98d3c2731e8e
    rep domain 0ce2b2b10029112a
    rep probe 843d653f33d9bb99
    rep features c55a0009142c857f
    req trace_dump 6241bd8a14e495f4
    rep trace_dump 184a8f0ad66854a4
    req bg_jobs 563f8449e839661e
    req bg_checksum 13b35f4bc01b7fc4
    req bg_rewrite 96766c30d3a10f11
    req bg_ssd_jobs 03f5901fcf17b168
    req bg_ssd_done 1e152ba8e8ec813f
    rep bg_ssd_jobs 59d7c5da3da9cdf5
    req bg_compact_jobs e1e713c0a1ad5b0c
    req bg_compact_done 5e174e7d05d9104a
    rep bg_compact_jobs 8a6a58ae6dc06d3e
    rep bg_jobs c154ff69cba75509
    rep bg_checksum d94d14186c0f331d
    req invalidate aba1b7e936ca6d4f
    req touch_tagged 26668b360d373fa7
    req hint 66a767269f8511bf
    req where d934744c63815259
    rep where ced3f3fbaa9960cd
";
