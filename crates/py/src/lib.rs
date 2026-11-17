//! Python bindings for the llmt-kvbroker client (module `llmt_kv._llmt_kv`, re-exported by `llmt_kv`).
//! Every broker call releases the GIL. Errors: `BrokerTimeout` (no reply in time: run without the tier this step),
//! `BrokerGone` (connection lost; the client reconnects by itself, at most every `reconnect_ms`),
//! `BrokerReset` (this call reconnected: the broker forgot this engine's reservations and leases, and after a
//! broker restart its whole index; drop local transfer bookkeeping and go on), `BrokerPoolChanged` (reconnected to
//! a broker with a different pool file: mappings and GPU registrations of the old one are stale, so stop using the
//! tier until the engine restarts), `BrokerError` (base class; also refused whole requests).

use kvclient::{Client as RsClient, EntryDesc, Error, Key, PoolMap, ReserveItem, Status};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::{Duration, Instant};

create_exception!(_llmt_kv, BrokerError, PyException);
create_exception!(_llmt_kv, BrokerTimeout, BrokerError);
create_exception!(_llmt_kv, BrokerGone, BrokerError);
create_exception!(_llmt_kv, BrokerReset, BrokerError);
create_exception!(_llmt_kv, BrokerPoolChanged, BrokerError);

fn err(e: Error) -> PyErr {
    match e {
        Error::Timeout => BrokerTimeout::new_err("broker did not answer in time"),
        Error::Io(e) => BrokerGone::new_err(format!("broker connection lost: {e}")),
        Error::Protocol(m) => BrokerGone::new_err(format!("broker protocol error: {m}")),
        Error::Status(s) => BrokerError::new_err(format!("broker refused the request: {s:?}")),
    }
}

fn key(b: &[u8]) -> PyResult<Key> {
    b.try_into().map_err(|_| PyValueError::new_err(format!("keys must be 32 bytes, got {}", b.len())))
}

fn keys(v: Vec<Vec<u8>>) -> PyResult<Vec<Key>> {
    v.iter().map(|k| key(k)).collect()
}

/// One leased entry from `lookup(..., acquire=True)`.
#[pyclass(frozen, get_all, module = "llmt_kv")]
struct Entry {
    nbytes: u64,
    kind: u8,
    format: u16,
    producer: u16,
    flags: u8,
    meta: Py<PyBytes>,
    /// Slot ranges: [(start_slot, n_slots)].
    ranges: Vec<(u32, u32)>,
    /// The same ranges in bytes from the start of the pool: [(offset, length)]; the entry's data is the first
    /// `nbytes` of their concatenation.
    byte_ranges: Vec<(u64, u64)>,
}

#[pymethods]
impl Entry {
    fn __repr__(&self) -> String {
        format!(
            "Entry(nbytes={}, kind={}, format={}, producer={}, ranges={:?})",
            self.nbytes, self.kind, self.format, self.producer, self.ranges
        )
    }
}

fn entry(py: Python, e: EntryDesc, slot: u64) -> Entry {
    Entry {
        nbytes: e.bytes,
        kind: e.kind,
        format: e.format,
        producer: e.producer,
        flags: e.flags,
        meta: PyBytes::new(py, &e.meta).unbind(),
        ranges: e.ranges.iter().map(|r| (r.start, r.len)).collect(),
        byte_ranges: e.ranges.iter().map(|r| (r.start as u64 * slot, r.len as u64 * slot)).collect(),
    }
}

struct Conn {
    c: Option<RsClient>,
    last_attempt: Instant,
    pool_changed: Option<String>,
    /// consecutive timeouts on this connection, and whether it counts as hung (kvclient::Liveness)
    live: kvclient::Liveness,
    /// current reconnect interval: doubles after each failed attempt up to RECONNECT_MAX, reset on success
    backoff: Duration,
}

/// A hung broker (TIMEOUTS_DROP timeouts in a row and nothing at all received for SILENT_DROP, kvclient::Liveness)
/// is dropped: it must cost the engine a few seconds of timed-out calls, never a stalled step; a slow but alive one is
/// kept.
use kvclient::{SILENT_DROP, TIMEOUTS_DROP};
const RECONNECT_MAX: Duration = Duration::from_secs(10);
/// HELLO timeout on reconnects (the first connect at engine start keeps the generous one).
const RECONNECT_HELLO: Duration = Duration::from_millis(100);

/// Connection to llmt-kvbroker. Calls are serialised by an internal lock. Reconnects by itself after the
/// connection is lost (broker restart, or a protocol error that made the broker drop it); the pool fields describe
/// the pool of the first connection, which `pool_address()` maps.
#[pyclass(module = "llmt_kv")]
struct Client {
    inner: Mutex<Conn>,
    pool: Mutex<Option<PoolMap>>,
    socket: PathBuf,
    engine_id: u16,
    name: String,
    timeout: Mutex<Duration>,
    reconnect: Duration,
    epoch: AtomicU64,
    reconnects: AtomicU64,
    /// connections dropped after TIMEOUTS_DROP consecutive timeouts
    drops: AtomicU64,
    #[pyo3(get)]
    pool_path: String,
    #[pyo3(get)]
    pool_bytes: u64,
    #[pyo3(get)]
    slot_bytes: u64,
    #[pyo3(get)]
    n_slots: u32,
    #[pyo3(get)]
    broker_version: String,
}

enum CallErr {
    E(Error),
    Gone,
    Reset(u64),
    PoolChanged(String),
}

impl Client {
    fn with<T: Send>(&self, py: Python, f: impl FnOnce(&mut RsClient) -> kvclient::Result<T> + Send) -> PyResult<T> {
        py.allow_threads(|| {
            let mut g = self.inner.lock().unwrap();
            if g.c.is_none() {
                self.try_reconnect(&mut g)?;
            }
            let r = f(g.c.as_mut().unwrap());
            match &r {
                Err(Error::Io(_) | Error::Protocol(_)) => {
                    g.c = None; // dead connection (or a write that timed out mid-frame): reconnect on a later call
                    g.last_attempt = Instant::now();
                    g.live = Default::default();
                }
                Err(Error::Timeout) => {
                    let silent = g.c.as_ref().map_or(Duration::ZERO, |c| c.silent_for());
                    if g.live.timed_out(silent) {
                        // hung broker (TIMEOUTS_DROP timeouts in a row and nothing received for SILENT_DROP): stop
                        // queueing requests on it, and never come back to it: closing the connection
                        // releases this engine's leases while the broker is alive, so slots it is still loading from
                        // may be handed to other entries. The tier stays off (BrokerPoolChanged) until the engine
                        // restarts; correctness over the RAM tier. A slow broker that still answers (late replies)
                        // is never dropped.
                        g.c = None;
                        g.live = Default::default();
                        g.pool_changed = Some(format!("{} (connection dropped after {TIMEOUTS_DROP} timeouts and {} s \
                                                       without any reply: leases may be gone)", self.pool_path,
                                                      SILENT_DROP.as_secs()));
                        self.drops.fetch_add(1, Relaxed);
                    }
                }
                _ => g.live.ok(),
            }
            r.map_err(CallErr::E)
        })
        .map_err(|e| match e {
            CallErr::E(e) => err(e),
            CallErr::Gone => BrokerGone::new_err("not connected to the broker (reconnecting)"),
            CallErr::Reset(epoch) => {
                BrokerReset::new_err(format!("reconnected to the broker (epoch {epoch:016x}); local transfer state is void"))
            }
            CallErr::PoolChanged(p) => BrokerPoolChanged::new_err(format!(
                "broker now serves pool {p}, not {}: the tier is unusable until this engine restarts",
                self.pool_path
            )),
        })
    }

    /// Called with the connection lock held and no connection. Rate-limited; never blocks long (connecting to a
    /// Unix socket with no listener fails at once; HELLO has its own timeout).
    fn try_reconnect(&self, g: &mut Conn) -> Result<(), CallErr> {
        if let Some(p) = &g.pool_changed {
            return Err(CallErr::PoolChanged(p.clone()));
        }
        if g.last_attempt.elapsed() < g.backoff.max(self.reconnect) {
            return Err(CallErr::Gone);
        }
        g.last_attempt = Instant::now();
        let t = *self.timeout.lock().unwrap();
        let c = match RsClient::connect_hello(&self.socket, self.engine_id, &self.name, t, RECONNECT_HELLO.max(t)) {
            Ok(c) => c,
            Err(_) => {
                g.backoff = (g.backoff.max(self.reconnect) * 2).min(RECONNECT_MAX);
                return Err(CallErr::Gone);
            }
        };
        g.backoff = Duration::ZERO;
        let h = &c.hello;
        if h.pool_path != self.pool_path || h.pool_bytes != self.pool_bytes || h.slot_bytes != self.slot_bytes {
            g.pool_changed = Some(h.pool_path.clone());
            return Err(CallErr::PoolChanged(h.pool_path.clone()));
        }
        if h.epoch == self.epoch.load(Relaxed) {
            // the same broker instance: our connection closed while it was alive, so it released our
            // leases; loads this engine still has in flight or planned may read reused slots. Tier off for good.
            g.pool_changed = Some(format!("{} (reconnected to the same broker: leases were released)", h.pool_path));
            return Err(CallErr::PoolChanged(h.pool_path.clone()));
        }
        self.epoch.store(h.epoch, Relaxed);
        self.reconnects.fetch_add(1, Relaxed);
        g.c = Some(c);
        Err(CallErr::Reset(self.epoch.load(Relaxed)))
    }
}

#[pymethods]
impl Client {
    #[new]
    /// `reconnect_ms`: minimum time between reconnect attempts after the connection is lost.
    #[pyo3(signature = (socket="/run/llmt-kv/broker.sock", engine_id=0, name="engine", timeout_ms=2.0, reconnect_ms=1000.0))]
    fn new(py: Python, socket: &str, engine_id: u16, name: &str, timeout_ms: f64, reconnect_ms: f64) -> PyResult<Self> {
        let path = PathBuf::from(socket);
        let name = name.to_string();
        let t = Duration::from_secs_f64(timeout_ms / 1000.0);
        let c = py.allow_threads(|| RsClient::connect(&path, engine_id, &name, t)).map_err(err)?;
        let h = c.hello.clone();
        Ok(Client {
            inner: Mutex::new(Conn { c: Some(c), last_attempt: Instant::now(), pool_changed: None, live: Default::default(),
                                     backoff: Duration::ZERO }),
            pool: Mutex::new(None),
            socket: path,
            engine_id,
            name,
            timeout: Mutex::new(t),
            reconnect: Duration::from_secs_f64(reconnect_ms / 1000.0),
            epoch: AtomicU64::new(h.epoch),
            reconnects: AtomicU64::new(0),
            drops: AtomicU64::new(0),
            pool_path: h.pool_path,
            pool_bytes: h.pool_bytes,
            slot_bytes: h.slot_bytes,
            n_slots: h.n_slots,
            broker_version: h.broker_version,
        })
    }

    /// Per-request timeout in milliseconds.
    fn set_timeout_ms(&self, timeout_ms: f64) {
        let t = Duration::from_secs_f64(timeout_ms / 1000.0);
        *self.timeout.lock().unwrap() = t;
        if let Some(c) = self.inner.lock().unwrap().c.as_mut() {
            c.set_timeout(t);
        }
    }

    /// Epoch of the broker this client is (or was last) connected to; changes when the broker restarts.
    #[getter]
    fn epoch(&self) -> u64 {
        self.epoch.load(Relaxed)
    }

    /// Successful reconnects so far.
    #[getter]
    fn reconnects(&self) -> u64 {
        self.reconnects.load(Relaxed)
    }

    /// Connections dropped as hung (16 consecutive timeouts and 5 s without any reply).
    #[getter]
    fn timeout_drops(&self) -> u64 {
        self.drops.load(Relaxed)
    }

    /// True while a connection is open (False between a lost connection and the next successful reconnect).
    #[getter]
    fn connected(&self) -> bool {
        self.inner.lock().unwrap().c.is_some()
    }

    /// Map the pool (once) and return its address, for cudaHostRegister / zexDriverImportExternalPointer.
    /// The mapping lives as long as this Client.
    fn pool_address(&self) -> PyResult<usize> {
        let mut p = self.pool.lock().unwrap();
        if p.is_none() {
            let h = kvclient::HelloReply {
                protocol_version: kvclient::PROTOCOL_VERSION,
                broker_version: self.broker_version.clone(),
                epoch: self.epoch.load(Relaxed),
                pool_path: self.pool_path.clone(),
                pool_bytes: self.pool_bytes,
                slot_bytes: self.slot_bytes,
                n_slots: self.n_slots,
            };
            *p = Some(PoolMap::open(&h).map_err(|e| BrokerGone::new_err(format!("cannot map pool: {e}")))?);
        }
        Ok(p.as_ref().unwrap().address())
    }

    /// Longest committed prefix of `keys` (32-byte keys, root first). Returns (n_hit, lease_id, entries).
    /// With acquire=False only n_hit is filled (no lease, no entries): for repeated scheduler checks.
    /// With acquire=True the hit is leased until `release(lease_id)` (or the connection closes / lease timeout).
    #[pyo3(signature = (namespace, keys, acquire=false))]
    fn lookup(&self, py: Python, namespace: u32, keys: Vec<Vec<u8>>, acquire: bool) -> PyResult<(u32, u64, Vec<Entry>)> {
        let ks = self::keys(keys)?;
        let l = self.with(py, |c| c.lookup(namespace, &ks, acquire))?;
        let slot = self.slot_bytes;
        Ok((l.n_hit, l.lease_id, l.entries.into_iter().map(|e| entry(py, e, slot)).collect()))
    }

    /// Reserve slots for new entries. `items`: dicts with key (32 bytes), nbytes, and optionally parent (32 bytes
    /// or None), kind (0 attention, 1 state, 2 attention+state, 3 other, 4 partial tail, 5 weights), format, cost_hint, ephemeral (bool),
    /// ttl_ms, meta (bytes, <= 1024), low / lowest (bool: the block's value class). Returns [(status, [(offset, length)])] per item, byte ranges into the pool.
    /// Write the entry's bytes into its ranges in order, then `commit`. status: OK, EXISTS, BUSY, FULL, NO_PARENT...
    fn reserve(&self, py: Python, namespace: u32, items: Vec<Bound<'_, PyDict>>) -> PyResult<Vec<(u16, Vec<(u64, u64)>)>> {
        let mut v = Vec::with_capacity(items.len());
        for d in &items {
            let get = |k: &str| d.get_item(k);
            let k: Vec<u8> = get("key")?.ok_or_else(|| PyValueError::new_err("item needs 'key'"))?.extract()?;
            let nbytes: u64 = get("nbytes")?.ok_or_else(|| PyValueError::new_err("item needs 'nbytes'"))?.extract()?;
            let parent: Option<Vec<u8>> = match get("parent")? {
                Some(p) if !p.is_none() => Some(p.extract()?),
                _ => None,
            };
            let ephemeral: bool = get("ephemeral")?.map(|x| x.extract()).transpose()?.unwrap_or(false);
            // low: the block of a low-priority request (ENTRY_LOW: evicted first under room pressure, hot set last)
            let low: bool = get("low")?.map(|x| x.extract()).transpose()?.unwrap_or(false);
            let lowest: bool = get("lowest")?.map(|x| x.extract()).transpose()?.unwrap_or(false);
            v.push(ReserveItem {
                key: key(&k)?,
                parent: parent.as_deref().map(key).transpose()?,
                bytes: nbytes,
                kind: get("kind")?.map(|x| x.extract()).transpose()?.unwrap_or(0),
                format: get("format")?.map(|x| x.extract()).transpose()?.unwrap_or(0),
                cost_hint: get("cost_hint")?.map(|x| x.extract()).transpose()?.unwrap_or(0),
                flags: if ephemeral { 1 } else { 0 } | if low { kvclient::ENTRY_LOW } else { 0 }
                    | if lowest { kvclient::ENTRY_LOWEST } else { 0 },
                ttl_ms: get("ttl_ms")?.map(|x| x.extract()).transpose()?.unwrap_or(0),
                meta: get("meta")?.map(|x| x.extract()).transpose()?.unwrap_or_default(),
            });
        }
        let r = self.with(py, |c| c.reserve(namespace, v))?;
        let slot = self.slot_bytes;
        Ok(r
            .into_iter()
            .map(|(s, rs)| (s as u16, rs.iter().map(|x| (x.start as u64 * slot, x.len as u64 * slot)).collect()))
            .collect())
    }

    fn commit(&self, py: Python, namespace: u32, keys: Vec<Vec<u8>>) -> PyResult<Vec<u16>> {
        let ks = self::keys(keys)?;
        Ok(self.with(py, |c| c.commit(namespace, &ks))?.into_iter().map(|s| s as u16).collect())
    }

    fn abort(&self, py: Python, namespace: u32, keys: Vec<Vec<u8>>) -> PyResult<Vec<u16>> {
        let ks = self::keys(keys)?;
        Ok(self.with(py, |c| c.abort(namespace, &ks))?.into_iter().map(|s| s as u16).collect())
    }

    fn touch(&self, py: Python, namespace: u32, keys: Vec<Vec<u8>>) -> PyResult<Vec<u16>> {
        let ks = self::keys(keys)?;
        Ok(self.with(py, |c| c.touch(namespace, &ks))?.into_iter().map(|s| s as u16).collect())
    }

    /// One request's TOUCH with its KV groups tagged by kind (0 attention, 1 GDN state, other unknown), in group
    /// order: `groups` = [(kind, [key, ...]), ...]. The broker links the chain from the tags; an older broker gets
    /// plain TOUCH batches.
    fn touch_request(&self, py: Python, namespace: u32, groups: Vec<(u8, Vec<Vec<u8>>)>) -> PyResult<()> {
        let gs = groups.into_iter().map(|(k, ks)| Ok((k, self::keys(ks)?))).collect::<PyResult<Vec<_>>>()?;
        self.with(py, |c| c.touch_request(namespace, &gs))
    }

    /// TOUCH_REQUEST with the request's tags: `session` / `parent` = opaque 64-bit session ids (0 = none), `cls` =
    /// 0 unknown / 1 low / 2 normal / 3 agent / 4 lowest, `agent` = an agent letter byte (0 = none), `flags` bit 0 = the
    /// session's terminal request. A broker without FEAT_REQUEST_TAGS gets a plain TOUCH_REQUEST (or TOUCH batches).
    #[pyo3(signature = (namespace, groups, session=0, parent=0, cls=0, agent=0, flags=0))]
    #[allow(clippy::too_many_arguments)]
    fn touch_tagged(&self, py: Python, namespace: u32, groups: Vec<(u8, Vec<Vec<u8>>)>, session: u64, parent: u64, cls: u8,
                    agent: u8, flags: u8) -> PyResult<()> {
        let gs = groups.into_iter().map(|(k, ks)| Ok((k, self::keys(ks)?))).collect::<PyResult<Vec<_>>>()?;
        self.with(py, |c| c.touch_tagged(namespace, &gs, session, parent, cls, agent, flags))
    }

    /// Prefix-affinity hint (docs/prefix-hints.md): the router anchor `anchor` (16 bytes) covers these keys of this
    /// engine, `groups` = [(kind, [key, ...]), ...] in group order, keys in token order, `chunk_tokens` tokens per key;
    /// flags bit 0 = the prefix is in this engine's VRAM now. Raises on a broker without FEAT_PREFIX_HINTS.
    #[pyo3(signature = (namespace, anchor, chunk_tokens, groups, flags=0))]
    fn hint(&self, py: Python, namespace: u32, anchor: Vec<u8>, chunk_tokens: u32, groups: Vec<(u8, Vec<Vec<u8>>)>, flags: u8)
            -> PyResult<()> {
        let a: kvclient::Anchor = anchor.as_slice().try_into().map_err(|_| PyValueError::new_err("anchor must be 16 bytes"))?;
        let gs = groups.into_iter().map(|(k, ks)| Ok((k, self::keys(ks)?))).collect::<PyResult<Vec<_>>>()?;
        self.with(py, |c| c.hint(namespace, &a, chunk_tokens, flags, &gs))
    }

    /// WHERE: [(anchor_idx, engine_id, name, namespace, age_ms, chunk_tokens, n_chunks, n_resident, flags)] for
    /// `anchors` (16 bytes each, longest prefix first). Read-only on the broker.
    fn where_(&self, py: Python, anchors: Vec<Vec<u8>>) -> PyResult<Vec<(u32, u16, String, u32, u64, u32, u32, u32, u8)>> {
        let v = anchors.iter().map(|a| a.as_slice().try_into().map_err(|_| PyValueError::new_err("anchor must be 16 bytes")))
            .collect::<PyResult<Vec<kvclient::Anchor>>>()?;
        Ok(self.with(py, |c| c.where_(&v))?.into_iter()
            .map(|p| (p.anchor_idx, p.engine_id, p.name, p.namespace, p.age_ms, p.chunk_tokens, p.n_chunks, p.n_resident, p.flags))
            .collect())
    }

    /// Feature bits of the broker this client is connected to (FEATURES; 0 = an older broker or not connected).
    #[getter]
    fn features(&self) -> u64 {
        self.inner.lock().unwrap().c.as_ref().map_or(0, |c| c.features)
    }

    /// Drop known-bad entries and the chain after them, in RAM and on the SSD. Per key 0 OK / NOT_FOUND;
    /// raises on a broker without INVALIDATE.
    fn invalidate(&self, py: Python, namespace: u32, keys: Vec<Vec<u8>>) -> PyResult<Vec<u16>> {
        let ks = self::keys(keys)?;
        Ok(self.with(py, |c| c.invalidate(namespace, &ks))?.into_iter().map(|s| s as u16).collect())
    }

    /// Retention hint without session bookkeeping: hot = refresh, cold = evict first.
    fn heat(&self, py: Python, namespace: u32, keys: Vec<Vec<u8>>, hot: bool) -> PyResult<Vec<u16>> {
        let ks = self::keys(keys)?;
        Ok(self.with(py, |c| c.heat(namespace, &ks, hot))?.into_iter().map(|s| s as u16).collect())
    }

    fn release(&self, py: Python, lease_ids: Vec<u64>) -> PyResult<Vec<u16>> {
        Ok(self.with(py, |c| c.release(&lease_ids))?.into_iter().map(|s| s as u16).collect())
    }

    /// Block until `key` is committed (returns OK) or `timeout_ms` passes (returns TIMEOUT). GIL released.
    fn wait(&self, py: Python, namespace: u32, key: Vec<u8>, timeout_ms: u32) -> PyResult<u16> {
        let k = self::key(&key)?;
        Ok(self.with(py, |c| c.wait(namespace, k, timeout_ms))? as u16)
    }

    /// Dynamic pool: (generation, [(start_slot, n_slots), ...] parked runs). A client that pins the pool
    /// (cudaHostRegister) must unregister the parked slots and call pool_ack(generation).
    fn pool_state(&self, py: Python) -> PyResult<(u64, Vec<(u32, u32)>)> {
        let (g, rs) = self.with(py, |c| c.pool_state())?;
        Ok((g, rs.iter().map(|r| (r.start, r.len)).collect()))
    }

    fn pool_ack(&self, py: Python, gen: u64) -> PyResult<u16> {
        Ok(self.with(py, |c| c.pool_ack(gen))? as u16)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let v = self.with(py, |c| c.stats())?;
        let d = PyDict::new(py);
        for (k, x) in v {
            d.set_item(k, x)?;
        }
        Ok(d)
    }

    /// Copy `data` into a reserved entry's byte ranges (helper for tests and CPU-side writers).
    fn write(&self, ranges: Vec<(u64, u64)>, data: &[u8]) -> PyResult<()> {
        let p = self.pool.lock().unwrap();
        let p = p.as_ref().ok_or_else(|| BrokerError::new_err("call pool_address() first"))?;
        let mut off = 0usize;
        for (o, l) in ranges {
            if off >= data.len() {
                break;
            }
            let n = (l as usize).min(data.len() - off);
            if o as usize + n > p.len {
                return Err(PyValueError::new_err("range outside the pool"));
            }
            unsafe { p.span(o as usize, n) }.write(0, &data[off..off + n]);
            off += n;
        }
        if off < data.len() {
            return Err(PyValueError::new_err("data longer than the ranges"));
        }
        Ok(())
    }

    /// Read `nbytes` from an entry's byte ranges (helper for tests and CPU-side readers).
    fn read<'py>(&self, py: Python<'py>, ranges: Vec<(u64, u64)>, nbytes: usize) -> PyResult<Bound<'py, PyBytes>> {
        let p = self.pool.lock().unwrap();
        let p = p.as_ref().ok_or_else(|| BrokerError::new_err("call pool_address() first"))?;
        let mut out = Vec::with_capacity(nbytes);
        for (o, l) in ranges {
            let n = (l as usize).min(nbytes - out.len());
            if o as usize + n > p.len {
                return Err(PyValueError::new_err("range outside the pool"));
            }
            out.extend_from_slice(&unsafe { p.span(o as usize, n) }.to_vec(n));
            if out.len() == nbytes {
                break;
            }
        }
        Ok(PyBytes::new(py, &out))
    }
}

#[pyfunction]
fn status_name(s: u16) -> &'static str {
    match Status::from_u16(s) {
        Status::Ok => "OK",
        Status::Exists => "EXISTS",
        Status::Busy => "BUSY",
        Status::Full => "FULL",
        Status::NotFound => "NOT_FOUND",
        Status::Timeout => "TIMEOUT",
        Status::BadRequest => "BAD_REQUEST",
        Status::NotOwner => "NOT_OWNER",
        Status::NoParent => "NO_PARENT",
        Status::TooBig => "TOO_BIG",
        Status::NoHello => "NO_HELLO",
        Status::Version => "VERSION",
    }
}

/// Staging-ring copies out of the broker pool (in-place slot compression): ops = [(src, dst, size)] with src inside the
/// pool [base, base + len) of `slot`-sized slots; slots the broker compressed in place are decompressed, raw ones
/// memcpy'd; parallel, without the GIL. False if a compressed slot was damaged (the caller then fails the transfer).
#[pyfunction]
fn copy_ops(py: Python, base: u64, len: u64, slot: u64, ops: Vec<(u64, u64, u64)>) -> bool {
    py.allow_threads(|| unsafe { kvslotz::copy_ops(base, len, slot, &ops) })
}

mod chains;

#[pymodule]
fn _llmt_kv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<chains::ChainsClient>()?;
    m.add_class::<Entry>()?;
    m.add_function(wrap_pyfunction!(status_name, m)?)?;
    m.add_function(wrap_pyfunction!(copy_ops, m)?)?;
    m.add("BrokerError", m.py().get_type::<BrokerError>())?;
    m.add("BrokerTimeout", m.py().get_type::<BrokerTimeout>())?;
    m.add("BrokerGone", m.py().get_type::<BrokerGone>())?;
    m.add("BrokerReset", m.py().get_type::<BrokerReset>())?;
    m.add("BrokerPoolChanged", m.py().get_type::<BrokerPoolChanged>())?;
    for (n, v) in [
        ("OK", 0u16),
        ("EXISTS", 1),
        ("BUSY", 2),
        ("FULL", 3),
        ("NOT_FOUND", 4),
        ("TIMEOUT", 5),
        ("BAD_REQUEST", 6),
        ("NOT_OWNER", 7),
        ("NO_PARENT", 8),
        ("TOO_BIG", 9),
        ("NO_HELLO", 10),
        ("VERSION", 11),
    ] {
        m.add(n, v)?;
    }
    m.add("PROTOCOL_VERSION", kvclient::PROTOCOL_VERSION)?;
    m.add("FEAT_TOUCH_REQUEST", kvclient::FEAT_TOUCH_REQUEST)?;
    m.add("FEAT_REQUEST_TAGS", kvclient::FEAT_REQUEST_TAGS)?;
    m.add("FEAT_PREFIX_HINTS", kvclient::FEAT_PREFIX_HINTS)?;
    m.add("HINT_VRAM", kvclient::HINT_VRAM)?;
    Ok(())
}
