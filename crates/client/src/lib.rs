//! Blocking client for llmt-kvbroker. One request in flight per `Client`; replies to earlier requests that timed
//! out are discarded by req_id, so a client stays usable after a timeout. Any I/O error means the broker is gone:
//! drop the client, run without the shared tier, reconnect later (a new epoch means an empty tier).

pub use kvproto::{BgJob, CompactJob, SsdJob, EntryDesc, Guard, HelloReply, Key, NgramSeq, Range, ReserveItem, Status, BG_BEGIN, BG_CHECKSUM,
                  BG_COMPRESS, BG_END, BG_ENGINE_ID, BG_SKIP, FEAT_BG_WORKER, FEAT_INVALIDATE, FEAT_REQUEST_TAGS, FEAT_TOUCH_REQUEST, ENTRY_LOW, ENTRY_LOWEST, TAG_TERMINAL, CLASS_LOWEST, CLASS_LOW, CLASS_NORMAL, CLASS_AGENT, CLASS_UNKNOWN, FMT_SLOT_CODEC, PROTOCOL_VERSION,
                  Anchor, Place, FEAT_PREFIX_HINTS, HINT_VRAM, OBSERVER_ENGINE_ID};
use kvproto::{read_frame, request_frame, Op, Reply, Request};
use std::io::{self, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Timeout,
    Status(Status),
    Protocol(&'static str),
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        match e.kind() {
            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => Error::Timeout,
            _ => Error::Io(e),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// When a connection with timeouts counts as hung. In production, 16 consecutive 2 ms timeouts
/// while the machine was loaded dropped the engine's connection to an alive broker, and the tier stayed off for hours
/// (dropping releases the engine's leases, so it may not come back to the same broker). The broker is only hung if it
/// also sent nothing at all for SILENT_DROP: late replies to timed-out requests (read and discarded by the next call)
/// prove it alive.
#[derive(Debug, Default)]
pub struct Liveness {
    timeouts: u32,
}

pub const TIMEOUTS_DROP: u32 = 16;
/// Unanswered requests a connection may have before new calls wait for the broker to catch up (flow control).
pub const MAX_OUTSTANDING: u32 = 8;
/// A reply frame that has started arriving is read to its end within this (the broker writes whole frames).
const FRAME_FINISH: Duration = Duration::from_secs(1);
pub const SILENT_DROP: Duration = Duration::from_secs(5);

impl Liveness {
    /// A call that got its reply (or any error other than a timeout).
    pub fn ok(&mut self) {
        self.timeouts = 0;
    }
    /// A call that timed out on a connection that last received a frame `silent` ago: true = treat as hung (drop).
    pub fn timed_out(&mut self, silent: Duration) -> bool {
        self.timeouts += 1;
        self.timeouts >= TIMEOUTS_DROP && silent >= SILENT_DROP
    }
    pub fn timeouts(&self) -> u32 {
        self.timeouts
    }
}

pub struct Client {
    rd: BufReader<UnixStream>,
    wr: UnixStream,
    next_id: u64,
    timeout: Duration,
    buf: Vec<u8>,
    /// when a frame from the broker was last received (any reply, also a late one): liveness
    last_rx: Instant,
    /// requests sent whose reply has not been read (timed out; flow control), acquiring LOOKUPs among them (their
    /// late leases are released by the client), leases to release
    outstanding: u32,
    late_lookups: Vec<u64>,
    late_leases: Vec<u64>,
    /// Testing only: sleep this long between sending a request and reading its reply (a caller thread descheduled
    /// under CPU load).
    pub test_pause: Duration,
    pub hello: HelloReply,
    /// The broker's feature bits and formats (FEATURES, negotiated on every connection; 0 / empty: an older broker).
    pub features: u64,
    pub formats: Vec<(String, u16)>,
}

/// What this client build reads (sent in FEATURES): compressed RAM slots of codec version 1 (kvslotz LKVZ1).
pub fn client_formats() -> Vec<(String, u16)> {
    vec![(FMT_SLOT_CODEC.into(), 1)]
}

pub struct Lookup {
    pub n_hit: u32,
    pub lease_id: u64,
    pub entries: Vec<EntryDesc>,
}

impl Client {
    pub fn connect(socket: &Path, engine_id: u16, name: &str, timeout: Duration) -> Result<Client> {
        // HELLO gets a generous timeout: the first connection may race the broker's start-up.
        Self::connect_hello(socket, engine_id, name, timeout, timeout.max(Duration::from_secs(2)))
    }

    /// `connect` with an explicit HELLO timeout (an engine reconnecting to a hung broker must not wait 2 s
    /// per attempt in its scheduler thread).
    pub fn connect_hello(socket: &Path, engine_id: u16, name: &str, timeout: Duration, hello_timeout: Duration)
                         -> Result<Client> {
        let s = UnixStream::connect(socket)?;
        let wr = s.try_clone()?;
        let mut c = Client {
            rd: BufReader::with_capacity(64 << 10, s),
            wr,
            next_id: 1,
            timeout,
            buf: Vec::new(),
            last_rx: Instant::now(),
            outstanding: 0,
            late_lookups: Vec::new(),
            late_leases: Vec::new(),
            test_pause: Duration::ZERO,
            hello: HelloReply {
                protocol_version: 0,
                broker_version: String::new(),
                epoch: 0,
                pool_path: String::new(),
                pool_bytes: 0,
                slot_bytes: 0,
                n_slots: 0,
            },
            features: 0,
            formats: Vec::new(),
        };
        let t = c.timeout;
        c.timeout = hello_timeout;
        let r = c.call(&Request::Hello { engine_id, protocol_version: PROTOCOL_VERSION, name: name.into() });
        c.timeout = t;
        match r? {
            (Status::Ok, Reply::Hello(h)) => c.hello = h,
            (s, _) => return Err(Error::Status(s)),
        }
        // feature negotiation on every connection (a reconnect may reach another broker build); hello timeout
        c.timeout = hello_timeout;
        let r = c.negotiate(FEAT_TOUCH_REQUEST | FEAT_REQUEST_TAGS, client_formats());
        c.timeout = t;
        r?;
        Ok(c)
    }

    /// FEATURES: declare this client's feature bits and the formats it reads; learn the broker's. An older broker
    /// answers BAD_REQUEST (unknown op): no features, and nothing new is used with it.
    pub fn negotiate(&mut self, features: u64, formats: Vec<(String, u16)>) -> Result<()> {
        match self.call(&Request::Features { features, formats })? {
            (Status::Ok, Reply::Features { features, formats }) => {
                self.features = features;
                self.formats = formats;
            }
            (Status::BadRequest, _) => {
                self.features = 0;
                self.formats.clear();
            }
            (s, _) => return Err(Error::Status(s)),
        }
        Ok(())
    }

    /// One request's TOUCH with its KV groups tagged by kind (0 attention, 1 GDN state, other unknown), in group
    /// order. An older broker gets the groups as plain TOUCH batches in the same order (it infers the request shape).
    pub fn touch_request(&mut self, namespace: u32, groups: &[(u8, Vec<Key>)]) -> Result<()> {
        if self.features & FEAT_TOUCH_REQUEST != 0 {
            return match self.call(&Request::TouchRequest { namespace, groups: groups.to_vec() })? {
                (Status::Ok, _) => Ok(()),
                (s, _) => Err(Error::Status(s)),
            };
        }
        for (_, keys) in groups.iter().filter(|g| !g.1.is_empty()) {
            self.touch(namespace, keys)?;
        }
        Ok(())
    }

    /// TOUCH_TAGGED: `touch_request` with the request's session tags and class (CLASS_*). A broker without
    /// FEAT_REQUEST_TAGS gets a plain `touch_request` (tags dropped; it falls back to TOUCH batches in turn).
    #[allow(clippy::too_many_arguments)]
    pub fn touch_tagged(&mut self, namespace: u32, groups: &[(u8, Vec<Key>)], session: u64, parent: u64, class: u8,
                        agent: u8, flags: u8) -> Result<()> {
        if self.features & FEAT_REQUEST_TAGS != 0 {
            return match self.call(&Request::TouchTagged { namespace, session, parent, class, agent, flags,
                                                           groups: groups.to_vec() })? {
                (Status::Ok, _) => Ok(()),
                (s, _) => Err(Error::Status(s)),
            };
        }
        self.touch_request(namespace, groups)
    }

    /// Time since the broker last sent anything on this connection (Liveness).
    pub fn silent_for(&self) -> Duration {
        self.last_rx.elapsed()
    }

    pub fn set_timeout(&mut self, t: Duration) {
        self.timeout = t;
    }

    fn call(&mut self, req: &Request) -> Result<(Status, Reply)> {
        let deadline = Instant::now() + self.timeout;
        // flow control: a broker that has not answered MAX_OUTSTANDING earlier (timed-out) requests gets
        // no new ones until it catches up: each timed-out request is still queued at the broker, and a client that kept
        // sending to a slow broker filled the socket until a write blocked (the connection is then lost). Its late
        // replies are drained here (they also prove the broker alive: Liveness).
        while self.outstanding >= MAX_OUTSTANDING {
            self.read_one(None, deadline)?;
        }
        // leases a late LOOKUP reply granted after its caller gave up: released before anything else
        if !self.late_leases.is_empty() {
            let ids = std::mem::take(&mut self.late_leases);
            let rid = self.next_id;
            self.next_id += 1;
            self.send(rid, &Request::Release { lease_ids: ids })?;
            self.outstanding += 1;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.send(id, req)?;
        if !self.test_pause.is_zero() {
            std::thread::sleep(self.test_pause);
        }
        let r = self.read_one(Some(id), deadline);
        if matches!(r, Err(Error::Timeout)) {
            self.outstanding += 1;      // its reply comes later (or never: Liveness)
            if matches!(req, Request::Lookup { acquire: true, .. }) {
                self.late_lookups.push(id);
            }
        }
        r.map(|x| x.expect("a reply to the call"))
    }

    /// Write one request frame. Any write error is Io, also a timeout: a frame cut by the write timeout must not stay
    /// on a connection that is kept (it was mapped to Timeout).
    fn send(&mut self, id: u64, req: &Request) -> Result<()> {
        // writes have the same deadline as the reply: a broker thread that stops reading (deadlock) left
        // timed-out requests in the socket until its buffer was full, and the next write blocked the engine for good.
        self.wr.set_write_timeout(Some(self.timeout.max(Duration::from_millis(1)))).map_err(Error::Io)?;
        self.wr.write_all(&request_frame(id, req)).map_err(Error::Io)
    }

    /// Read frames until the reply to `want` (Some: returned) or, with `want` None, one late reply (Ok(None));
    /// Timeout when nothing arrives before `deadline`. A frame that has started arriving is read to its end with
    /// FRAME_FINISH (a timeout inside a frame would leave the stream misaligned: Io).
    fn read_one(&mut self, want: Option<u64>, deadline: Instant) -> Result<Option<(Status, Reply)>> {
        use std::io::BufRead;
        loop {
            if self.rd.buffer().is_empty() {
                // past the deadline, still look at what has arrived (a caller thread descheduled for the
                // whole timeout, e.g. under CPU load, gave up on a reply that was already in the socket)
                let left = deadline.saturating_duration_since(Instant::now()).max(Duration::from_micros(1));
                let past = Instant::now() >= deadline;
                self.rd.get_ref().set_nonblocking(past).map_err(Error::Io)?;
                if !past {
                    self.rd.get_ref().set_read_timeout(Some(left))?;
                }
                let r = self.rd.fill_buf().map(|b| b.is_empty());
                if past {
                    self.rd.get_ref().set_nonblocking(false).map_err(Error::Io)?;
                }
                if r? {
                    return Err(Error::Protocol("broker closed the connection"));
                }
            }
            self.rd.get_ref().set_read_timeout(Some(FRAME_FINISH)).map_err(Error::Io)?;
            let h = read_frame(&mut self.rd, &mut self.buf).map_err(Error::Io)?
                .ok_or(Error::Protocol("broker closed the connection"))?;
            self.last_rx = Instant::now();
            let op = Op::from_u16(h.op).ok_or(Error::Protocol("unknown op in reply"))?;
            if Some(h.req_id) == want {
                return Reply::decode(op, &self.buf).map(Some).map_err(|_| Error::Protocol("bad reply"));
            }
            // a late reply to a request that timed out (or to a lease release this client sent by itself)
            self.outstanding = self.outstanding.saturating_sub(1);
            if let Some(i) = self.late_lookups.iter().position(|x| *x == h.req_id) {
                self.late_lookups.swap_remove(i);
                if let Ok((Status::Ok, Reply::Lookup { lease_id, .. })) = Reply::decode(op, &self.buf) {
                    if lease_id != 0 {
                        self.late_leases.push(lease_id);    // nobody will load it: release it (next call)
                    }
                }
            }
            if want.is_none() {
                return Ok(None);
            }
        }
    }

    fn statuses(&mut self, req: Request) -> Result<Vec<Status>> {
        match self.call(&req)? {
            (Status::Ok, Reply::Statuses(v)) => Ok(v),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Longest committed prefix of `keys`. With `acquire`, the hit is leased and described (ranges + meta).
    pub fn lookup(&mut self, namespace: u32, keys: &[Key], acquire: bool) -> Result<Lookup> {
        match self.call(&Request::Lookup { namespace, acquire, keys: keys.to_vec() })? {
            (Status::Ok, Reply::Lookup { n_hit, lease_id, entries }) => Ok(Lookup { n_hit, lease_id, entries }),
            (s, _) => Err(Error::Status(s)),
        }
    }

    pub fn reserve(&mut self, namespace: u32, items: Vec<ReserveItem>) -> Result<Vec<(Status, Vec<Range>)>> {
        match self.call(&Request::Reserve { namespace, items })? {
            (Status::Ok, Reply::Reserve(v)) => Ok(v),
            (s, _) => Err(Error::Status(s)),
        }
    }

    pub fn commit(&mut self, namespace: u32, keys: &[Key]) -> Result<Vec<Status>> {
        self.statuses(Request::Commit { namespace, keys: keys.to_vec() })
    }

    pub fn abort(&mut self, namespace: u32, keys: &[Key]) -> Result<Vec<Status>> {
        self.statuses(Request::Abort { namespace, keys: keys.to_vec() })
    }

    pub fn release(&mut self, lease_ids: &[u64]) -> Result<Vec<Status>> {
        self.statuses(Request::Release { lease_ids: lease_ids.to_vec() })
    }

    pub fn touch(&mut self, namespace: u32, keys: &[Key]) -> Result<Vec<Status>> {
        self.statuses(Request::Touch { namespace, keys: keys.to_vec() })
    }

    /// INVALIDATE: the entries for `keys` and the chain after them are known bad; the broker drops them
    /// (RAM and SSD). Per key OK / NOT_FOUND. A broker without FEAT_INVALIDATE: Err(BadRequest), nothing sent.
    pub fn invalidate(&mut self, namespace: u32, keys: &[Key]) -> Result<Vec<Status>> {
        if self.features & FEAT_INVALIDATE == 0 {
            return Err(Error::Status(Status::BadRequest));
        }
        self.statuses(Request::Invalidate { namespace, keys: keys.to_vec() })
    }

    /// Retention hint: hot = refresh, cold = evict first; no session bookkeeping.
    pub fn heat(&mut self, namespace: u32, keys: &[Key], hot: bool) -> Result<Vec<Status>> {
        self.statuses(Request::Heat { namespace, hot, keys: keys.to_vec() })
    }

    /// Scope this connection to a cache domain (`key=value` lines naming model, weights,
    /// engine, layout and numerics knobs). Returns the domain id; entries of other domains are never visible.
    pub fn domain(&mut self, descriptor: &str) -> Result<u64> {
        match self.call(&Request::Domain { descriptor: descriptor.into() })? {
            (Status::Ok, Reply::Domain { domain_id }) => Ok(domain_id),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// HINT (FEAT_PREFIX_HINTS, docs/prefix-hints.md): `anchor` (a router's prefix hash) covers these keys of this
    /// engine, per KV group in token order, `chunk_tokens` tokens per key. Refused (BAD_REQUEST, nothing sent) by a broker
    /// without the feature.
    pub fn hint(&mut self, namespace: u32, anchor: &Anchor, chunk_tokens: u32, flags: u8, groups: &[(u8, Vec<Key>)]) -> Result<()> {
        if self.features & FEAT_PREFIX_HINTS == 0 {
            return Err(Error::Status(Status::BadRequest));
        }
        match self.call(&Request::Hint { namespace, anchor: *anchor, chunk_tokens, flags, groups: groups.to_vec() })? {
            (Status::Ok, _) => Ok(()),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// WHERE: which engines hinted one of `anchors` (longest prefix first) and how much of it is in RAM now.
    /// Read-only on the broker. Refused (BAD_REQUEST, nothing sent) by a broker without FEAT_PREFIX_HINTS.
    pub fn where_(&mut self, anchors: &[Anchor]) -> Result<Vec<Place>> {
        if self.features & FEAT_PREFIX_HINTS == 0 {
            return Err(Error::Status(Status::BadRequest));
        }
        match self.call(&Request::Where { anchors: anchors.to_vec() })? {
            (Status::Ok, Reply::Where(v)) => Ok(v),
            (Status::Ok, Reply::Status) => Ok(Vec::new()),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Per key: 0 absent, 1 in RAM, 2 on the SSD tier only (promotion queued). No lease.
    pub fn probe(&mut self, namespace: u32, keys: &[Key]) -> Result<Vec<u8>> {
        match self.call(&Request::Probe { namespace, keys: keys.to_vec() })? {
            (Status::Ok, Reply::Probe(v)) => Ok(v),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Background worker (a connection that declared FEAT_BG_WORKER): jobs of `kind` to do now, with the broker's
    /// settings (checksum bytes / s, compression chunk) and its pool classes.
    pub fn bg_jobs(&mut self, kind: u8, max: u32) -> Result<(u64, u64, Vec<String>, Vec<BgJob>)> {
        match self.call(&Request::BgJobs { kind, max })? {
            (Status::Ok, Reply::BgJobs { cksum_bytes_s, chunk, classes, jobs }) => Ok((cksum_bytes_s, chunk, classes, jobs)),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Background worker: the checksum of a job's bytes (None: they do not decode). 0 stored, 1 verified, 2 corrupt
    /// (evicted), 3 raced, 4 busy.
    pub fn bg_checksum(&mut self, namespace: u32, key: Key, guard: Guard, sum: Option<u64>) -> Result<u8> {
        match self.call(&Request::BgChecksum { namespace, key, guard, sum })? {
            (Status::Ok, Reply::BgResult(r)) => Ok(r),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Background worker: begin / end / skip an in-place rewrite. 0 ok, 3 refused (raced, or not quiet).
    pub fn bg_rewrite(&mut self, phase: u8, namespace: u32, key: Key, guard: Guard, len: u64) -> Result<u8> {
        match self.call(&Request::BgRewrite { phase, namespace, key, guard, len })? {
            (Status::Ok, Reply::BgResult(r)) => Ok(r),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Background worker: one compaction move (none unless the broker is quiet and within its budget).
    pub fn bg_compact_jobs(&mut self) -> Result<Vec<CompactJob>> {
        match self.call(&Request::BgCompactJobs { max: 1 })? {
            (Status::Ok, Reply::CompactJobs(v)) => Ok(v),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Background worker: a compaction copy done (or not). 0 = the entry moved, 3 = not moved (changed / leased).
    pub fn bg_compact_done(&mut self, namespace: u32, key: Key, copied: bool) -> Result<u8> {
        match self.call(&Request::BgCompactDone { namespace, key, copied })? {
            (Status::Ok, Reply::BgResult(r)) => Ok(r),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Background worker: planned SSD tier writes of this connection's pool class (at most `max`).
    pub fn bg_ssd_jobs(&mut self, max: u32) -> Result<Vec<SsdJob>> {
        match self.call(&Request::BgSsdJobs { max })? {
            (Status::Ok, Reply::SsdJobs(v)) => Ok(v),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Background worker: the outcome of an SSD write (0 written, 1 raced, 2 corrupt, 3 failed). 0 = recorded.
    pub fn bg_ssd_done(&mut self, namespace: u32, key: Key, outcome: u8, file_bytes: u64) -> Result<u8> {
        match self.call(&Request::BgSsdDone { namespace, key, outcome, file_bytes })? {
            (Status::Ok, Reply::BgResult(r)) => Ok(r),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// TRACE_DUMP: the broker writes its operation trace to a file now; the file's path. Status BUSY: tracing off or
    /// the day's dump budget used up.
    pub fn trace_dump(&mut self, reason: &str) -> Result<String> {
        match self.call(&Request::TraceDump { reason: reason.into() })? {
            (Status::Ok, Reply::TraceDump { path }) => Ok(path),
            (s, _) => Err(Error::Status(s)),
        }
    }

    pub fn stats(&mut self) -> Result<Vec<(String, u64)>> {
        match self.call(&Request::Stats)? {
            (Status::Ok, Reply::Stats(v)) => Ok(v),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Dynamic pool: (generation, parked slot runs). Pinning clients unregister the parked runs and `pool_ack`.
    pub fn pool_state(&mut self) -> Result<(u64, Vec<Range>)> {
        match self.call(&Request::PoolState)? {
            (Status::Ok, Reply::PoolState { gen, parked }) => Ok((gen, parked)),
            (s, _) => Err(Error::Status(s)),
        }
    }

    pub fn pool_ack(&mut self, gen: u64) -> Result<Status> {
        Ok(self.call(&Request::PoolAck { gen })?.0)
    }

    /// Attach to the broker's n-gram table `table` (identity: rows x row_bytes) and map this connection's staging window.
    pub fn ngram_open(&mut self, table: &str, rows: u64, row_bytes: u32) -> Result<NgramStaging> {
        match self.call(&Request::NgramOpen { table: table.into(), rows, row_bytes })? {
            (Status::Ok, Reply::NgramOpen { path, staging_offset, staging_bytes, row_bytes, heads }) => {
                Ok(NgramStaging::map(&path, staging_offset, staging_bytes, row_bytes, heads)?)
            }
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Rows of the runs into the staging window (tokens x heads rows, request order). Returns (rows, rows from disk).
    pub fn ngram_gather(&mut self, seqs: Vec<NgramSeq>) -> Result<(u32, u32)> {
        match self.call(&Request::NgramGather { seqs })? {
            (Status::Ok, Reply::NgramGather { rows, disk_rows }) => Ok((rows, disk_rows)),
            (s, _) => Err(Error::Status(s)),
        }
    }

    /// Queue a cache warm-up for the runs (the broker answers at once; `Busy` = its queue was full, dropped).
    pub fn ngram_prefetch(&mut self, seqs: Vec<NgramSeq>) -> Result<Status> {
        Ok(self.call(&Request::NgramPrefetch { seqs })?.0)
    }

    /// Block until `key` is committed (`Ok`) or `timeout_ms` passes (`Timeout`). Uses its own deadline.
    pub fn wait(&mut self, namespace: u32, key: Key, timeout_ms: u32) -> Result<Status> {
        let t = self.timeout;
        self.timeout = t + Duration::from_millis(timeout_ms as u64 + 100);
        let r = self.call(&Request::Wait { namespace, key, timeout_ms });
        self.timeout = t;
        Ok(r?.0)
    }
}

/// The pool, mapped shared read/write (an owned `memmap2::MmapRaw`, unmapped exactly once when the
/// PoolMap goes; `MmapRaw` because other processes write this memory, so no Rust reference to it is ever formed).
/// Slots are reached through [`SlotRange`]s that borrow the PoolMap: using one after the map is dropped does not
/// compile. Engines that must hand the address to a GPU runtime use [`PoolMap::address`] and keep the PoolMap.
pub struct PoolMap {
    map: memmap2::MmapRaw,
    pub len: usize,
    pub slot_bytes: usize,
}

impl PoolMap {
    pub fn open(h: &HelloReply) -> io::Result<PoolMap> {
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&h.pool_path)?;
        let len = h.pool_bytes as usize;
        // MAP_NORESERVE (no_reserve_swap): on hugetlbfs a shared mapping otherwise reserves a huge page for every hole
        // of the file (a shrunk pool's parked slots) -> ENOSPC. Clients only touch broker-assigned, allocated slots.
        let map = memmap2::MmapOptions::new().len(len).no_reserve_swap().map_raw(&f)?;
        Ok(PoolMap { map, len, slot_bytes: h.slot_bytes as usize })
    }

    /// Start address of the mapping, valid while this PoolMap lives (GPU registration, the Python engines).
    pub fn address(&self) -> usize {
        self.map.as_ptr() as usize
    }

    /// The bytes of one slot range.
    ///
    /// # Safety
    /// The caller must hold a reservation (writing) or lease (reading) covering `r`.
    pub unsafe fn range(&self, r: Range) -> SlotRange<'_> {
        self.span(r.start as usize * self.slot_bytes, r.len as usize * self.slot_bytes)
    }

    /// `n` bytes at byte offset `off` of the pool (bounds-checked).
    ///
    /// # Safety
    /// As [`PoolMap::range`]: the caller must hold a reservation or lease covering the bytes.
    pub unsafe fn span(&self, off: usize, n: usize) -> SlotRange<'_> {
        assert!(off.checked_add(n).is_some_and(|e| e <= self.len), "pool span {off}+{n} outside {}", self.len);
        SlotRange { ptr: self.map.as_mut_ptr().add(off), len: n, _map: std::marker::PhantomData }
    }
}

/// Bytes of the pool borrowed from a [`PoolMap`]: copies in and out, bounds-checked. Other processes may write
/// the same memory, so it is only ever copied (never handed out as a `&[u8]`).
///
/// A SlotRange cannot outlive its PoolMap (the kvbench race tool's first version kept writing through a
/// raw pointer after dropping the map and segfaulted; the same code does not compile now):
/// ```compile_fail,E0505
/// # fn late(h: &kvproto::HelloReply) {
/// let pool = kvclient::PoolMap::open(h).unwrap();
/// let dst = unsafe { pool.range(kvproto::Range { start: 0, len: 1 }) };
/// drop(pool);                  // error[E0505]: cannot move out of `pool` because it is borrowed
/// dst.fill(0, 1, 0xff);
/// # }
/// ```
pub struct SlotRange<'a> {
    ptr: *mut u8,
    len: usize,
    _map: std::marker::PhantomData<&'a PoolMap>,
}

impl SlotRange<'_> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Copy `out.len()` bytes starting at `off` into `out`.
    pub fn read(&self, off: usize, out: &mut [u8]) {
        assert!(off.checked_add(out.len()).is_some_and(|e| e <= self.len), "read outside the slot range");
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.add(off), out.as_mut_ptr(), out.len()) }
    }
    /// The first `n` bytes (all of them if `n` is larger).
    pub fn to_vec(&self, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n.min(self.len)];
        self.read(0, &mut v);
        v
    }
    /// Copy `src` into the range at `off`.
    pub fn write(&self, off: usize, src: &[u8]) {
        assert!(off.checked_add(src.len()).is_some_and(|e| e <= self.len), "write outside the slot range");
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(off), src.len()) }
    }
    /// Set `n` bytes at `off` to `byte`.
    pub fn fill(&self, off: usize, n: usize, byte: u8) {
        assert!(off.checked_add(n).is_some_and(|e| e <= self.len), "fill outside the slot range");
        unsafe { std::ptr::write_bytes(self.ptr.add(off), byte, n) }
    }
}

/// This connection's n-gram staging window, mapped read-only. Valid until the next gather on the same connection.
pub struct NgramStaging {
    base: *mut u8,
    map_len: usize,
    offset: usize,
    pub bytes: usize,
    pub row_bytes: usize,
    pub heads: usize,
}

unsafe impl Send for NgramStaging {}

impl NgramStaging {
    fn map(path: &str, offset: u64, bytes: u64, row_bytes: u32, heads: u32) -> io::Result<NgramStaging> {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open(path)?;
        let map_len = f.metadata()?.len() as usize;
        let p = unsafe { libc::mmap(std::ptr::null_mut(), map_len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0) };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(NgramStaging { base: p as *mut u8, map_len, offset: offset as usize, bytes: bytes as usize,
                          row_bytes: row_bytes as usize, heads: heads as usize })
    }

    /// The first `rows` rows of the last gather.
    pub fn rows(&self, rows: usize) -> &[u8] {
        let n = (rows * self.row_bytes).min(self.bytes);
        unsafe { std::slice::from_raw_parts(self.base.add(self.offset), n) }
    }
}

impl Drop for NgramStaging {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base as *mut libc::c_void, self.map_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn liveness_drops_only_silent_brokers() {
        let mut l = Liveness::default();
        for _ in 0..1000 {
            assert!(!l.timed_out(Duration::from_millis(3)), "a slow broker that still answers is never dropped");
        }
        let mut l = Liveness::default();
        for i in 1..TIMEOUTS_DROP {
            assert!(!l.timed_out(SILENT_DROP), "only after {TIMEOUTS_DROP} timeouts (at {i})");
        }
        assert!(l.timed_out(SILENT_DROP));
        let mut l = Liveness::default();
        for _ in 0..TIMEOUTS_DROP * 2 {
            l.timed_out(SILENT_DROP);
        }
        l.ok();
        assert_eq!(l.timeouts(), 0);
        assert!(!l.timed_out(SILENT_DROP), "an answer resets the count");
    }
}
