//! Engine-side chains client. Never blocks the engine: calls only update local state and queue frames; a sender thread
//! writes them, a receiver thread keeps the latest hint per request. The client keeps every open request's tokens, so
//! after a dropped frame (queue full) or a reconnect it re-opens the request with its full token list.
use crate::engine::{self, ArmCfg, PhaseState, Tags};
use crate::proto::{self, Msg};
use std::collections::{HashMap, VecDeque};
use std::io::{BufReader, BufWriter, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Config {
    pub socket: PathBuf,
    pub engine_id: u16,
    pub model_tag: String,
    pub gate: u16,
    pub max_chain: u16,
    /// queued frames before drops (a dropped Append re-opens its request later)
    pub queue_frames: usize,
    pub reconnect_ms: u64,
}

impl Config {
    pub fn new(socket: impl Into<PathBuf>, engine_id: u16) -> Config {
        Config { socket: socket.into(), engine_id, model_tag: String::new(), gate: 16, max_chain: 64, queue_frames: 4096,
                 reconnect_ms: 1000 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hint {
    pub known_len: u32,
    pub match_len: u16,
    pub source: u8,
    pub chain_ref: u64,
    pub tokens: Vec<u32>,
}

struct ReqState {
    flags: u8,
    tokens: Vec<u32>,
    /// the last Append (its token count and phase), replayed after a re-Open so the broker can hint at once
    last_n: usize,
    last_phase: u8,
    /// the broker's copy is stale (a frame was dropped): send a full Open before the next Append
    reopen: bool,
    /// engine-side phase tracker for `step` (initialized from the prompt tail on the first step)
    ph: Option<PhaseState>,
}

/// An armed row from `Client::step`: the batch row, the hint's chain_ref and match length, the chain to draft.
#[derive(Clone, Debug, PartialEq)]
pub struct Armed {
    pub row: u32,
    pub chain_ref: u64,
    pub match_len: u16,
    pub tokens: Vec<u32>,
}

#[derive(Default)]
struct State {
    q: VecDeque<Vec<u8>>,
    reqs: HashMap<u64, ReqState>,
    hints: HashMap<u64, Hint>,
    stats: Option<String>,
}

#[derive(Default)]
pub struct Counters {
    pub frames_sent: AtomicU64,
    pub frames_dropped: AtomicU64,
    pub reopens: AtomicU64,
    pub hints_rx: AtomicU64,
    pub no_hints_rx: AtomicU64,
    pub reconnects: AtomicU64,
    pub connected: AtomicU64,
}

struct Inner {
    cfg: Config,
    st: Mutex<State>,
    cv: Condvar,
    stats_cv: Condvar,
    stop: AtomicBool,
    c: Counters,
    /// bumps on every (re)connect so a stale receiver thread exits
    epoch: AtomicU64,
}

pub struct Client {
    inner: Arc<Inner>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Open with the tokens before the last Append, then that Append (the broker hints as in normal flow).
fn reopen_frames(req: u64, r: &ReqState) -> Vec<u8> {
    let cut = r.tokens.len() - r.last_n;
    let mut v = enc(&Msg::Open { req, flags: r.flags, tokens: r.tokens[..cut].to_vec() });
    if r.last_n > 0 {
        proto::encode(&Msg::Append { req, known_len: r.tokens.len() as u32, phase: r.last_phase,
                                     tokens: r.tokens[cut..].to_vec() }, &mut v);
    }
    v
}

fn enc(m: &Msg) -> Vec<u8> {
    let mut v = Vec::new();
    proto::encode(m, &mut v);
    v
}

impl Client {
    pub fn new(cfg: Config) -> Client {
        let inner = Arc::new(Inner { cfg, st: Mutex::new(State::default()), cv: Condvar::new(), stats_cv: Condvar::new(),
                                     stop: AtomicBool::new(false), c: Counters::default(), epoch: AtomicU64::new(0) });
        let i2 = inner.clone();
        let thread = std::thread::Builder::new().name("chains-send".into()).spawn(move || sender(&i2)).unwrap();
        Client { inner, thread: Some(thread) }
    }

    fn push(&self, st: &mut State, frame: Vec<u8>) -> bool {
        if st.q.len() >= self.inner.cfg.queue_frames {
            self.inner.c.frames_dropped.fetch_add(1, Relaxed);
            return false;
        }
        st.q.push_back(frame);
        self.inner.cv.notify_one();
        true
    }

    /// A new request with its prompt (flags bit0: thinking-capable).
    pub fn open(&self, req: u64, flags: u8, tokens: &[u32]) {
        let mut st = self.inner.st.lock().unwrap();
        st.hints.remove(&req);
        let ok = self.push(&mut st, enc(&Msg::Open { req, flags, tokens: tokens.to_vec() }));
        st.reqs.insert(req, ReqState { flags, tokens: tokens.to_vec(), last_n: 0, last_phase: proto::PHASE_ANSWER, reopen: !ok, ph: None });
    }

    /// Tokens appended to a request (known_len = its length after them).
    pub fn append(&self, req: u64, known_len: u32, phase: u8, tokens: &[u32]) {
        let mut st = self.inner.st.lock().unwrap();
        self.append_locked(&mut st, req, known_len, phase, tokens);
    }

    /// One engine step for a whole batch (chains v2, `engine.rs`): for row i, request `reqs[i]` gained
    /// `new[offsets..+counts[i]]` and now has `lens[i]` tokens. Tracks each request's phase from the tag ids, appends
    /// to the broker (Append frames are queued, never sent inline), aligns the latest hint and returns the armed rows
    /// only. One lock hold for the batch; no Python involved.
    pub fn step(&self, tags: &Tags, cfg: &ArmCfg, reqs: &[u64], lens: &[u32], new: &[u32], counts: &[u32]) -> Vec<Armed> {
        let mut out = Vec::new();
        let mut st = self.inner.st.lock().unwrap();
        let mut off = 0usize;
        for (i, &req) in reqs.iter().enumerate() {
            let n = counts.get(i).copied().unwrap_or(0) as usize;
            let toks = &new[off.min(new.len())..(off + n).min(new.len())];
            off += n;
            let Some(r) = st.reqs.get_mut(&req) else { continue };
            let ph = r.ph.get_or_insert_with(|| {
                let mut p = PhaseState::default();
                let a = r.tokens.len().saturating_sub(16);
                p.feed(tags, &r.tokens[a..]);
                p
            });
            ph.feed(tags, toks);
            let phase = ph.phase();
            if !toks.is_empty() {
                self.append_locked(&mut st, req, lens[i], phase, toks);
            }
            let (Some(r), Some(h)) = (st.reqs.get(&req), st.hints.get(&req)) else { continue };
            if let Some(chain) = engine::align(cfg, phase, &r.tokens, h.known_len, h.match_len, &h.tokens) {
                out.push(Armed { row: i as u32, chain_ref: h.chain_ref, match_len: h.match_len, tokens: chain.to_vec() });
            }
        }
        out
    }

    fn append_locked(&self, st: &mut State, req: u64, known_len: u32, phase: u8, tokens: &[u32]) {
        let Some(r) = st.reqs.get_mut(&req) else { return };
        r.tokens.extend_from_slice(tokens);
        r.last_n = tokens.len();
        r.last_phase = phase;
        if r.tokens.len() != known_len as usize {
            r.reopen = true; // our copy and the caller differ (a dropped call): resync with a full Open
        }
        let frame = if r.reopen {
            self.inner.c.reopens.fetch_add(1, Relaxed);
            reopen_frames(req, r)
        } else {
            enc(&Msg::Append { req, known_len, phase, tokens: tokens.to_vec() })
        };
        let ok = self.push(st, frame);
        if let Some(r) = st.reqs.get_mut(&req) {
            r.reopen = !ok;
        }
    }

    pub fn close(&self, req: u64, aborted: bool) {
        let mut st = self.inner.st.lock().unwrap();
        st.hints.remove(&req);
        if st.reqs.remove(&req).is_some() {
            self.push(&mut st, enc(&Msg::Close { req, status: aborted as u8 }));
        }
    }

    pub fn feedback(&self, items: Vec<(u64, u16, u16, u64)>) {
        if items.is_empty() {
            return;
        }
        let mut st = self.inner.st.lock().unwrap();
        for chunk in items.chunks(4096) {
            self.push(&mut st, enc(&Msg::Feedback { items: chunk.to_vec() }));
        }
    }

    /// Corpus / template tokens for the global store (`permanent`: never evicted by recency).
    pub fn seed(&self, tokens: &[u32], permanent: bool) {
        let mut st = self.inner.st.lock().unwrap();
        self.push(&mut st, enc(&Msg::Seed { flags: permanent as u8, tokens: tokens.to_vec() }));
    }

    /// The latest hint for a request (aligned to its `known_len`; the engine checks it against its own length).
    pub fn hint(&self, req: u64) -> Option<Hint> {
        self.inner.st.lock().unwrap().hints.get(&req).cloned()
    }

    /// Broker-side stats (JSON), waiting up to `timeout`.
    pub fn stats(&self, timeout: Duration) -> Option<String> {
        let mut st = self.inner.st.lock().unwrap();
        st.stats = None;
        if !self.push(&mut st, enc(&Msg::StatsReq)) {
            return None;
        }
        let t = Instant::now();
        while st.stats.is_none() && t.elapsed() < timeout {
            st = self.inner.stats_cv.wait_timeout(st, timeout.saturating_sub(t.elapsed())).unwrap().0;
        }
        st.stats.take()
    }

    pub fn counters(&self) -> Vec<(&'static str, u64)> {
        let c = &self.inner.c;
        let st = self.inner.st.lock().unwrap();
        vec![
            ("open_requests", st.reqs.len() as u64),
            ("hints_held", st.hints.len() as u64),
            ("queued", st.q.len() as u64),
            ("frames_sent", c.frames_sent.load(Relaxed)),
            ("frames_dropped", c.frames_dropped.load(Relaxed)),
            ("reopens", c.reopens.load(Relaxed)),
            ("hints_rx", c.hints_rx.load(Relaxed)),
            ("no_hints_rx", c.no_hints_rx.load(Relaxed)),
            ("reconnects", c.reconnects.load(Relaxed)),
            ("connected", c.connected.load(Relaxed)),
        ]
    }

    /// Wait until the send queue is empty (tests).
    pub fn flush(&self, timeout: Duration) -> bool {
        let t = Instant::now();
        while t.elapsed() < timeout {
            if self.inner.st.lock().unwrap().q.is_empty() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.inner.stop.store(true, Relaxed);
        self.inner.epoch.fetch_add(1, Relaxed);
        self.inner.cv.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn connect(inner: &Inner) -> std::io::Result<UnixStream> {
    let cfg = &inner.cfg;
    let s = UnixStream::connect(&cfg.socket)?;
    s.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut w = BufWriter::new(s.try_clone()?);
    proto::send(&mut w, &Msg::Hello { proto: proto::PROTOCOL, engine_id: cfg.engine_id, model_tag: cfg.model_tag.clone(),
                                      gate: cfg.gate, max_chain: cfg.max_chain })?;
    w.flush()?;
    let mut r = BufReader::new(s.try_clone()?);
    let mut buf = Vec::new();
    match proto::read_frame(&mut r, &mut buf)? {
        Some(proto::T_HELLO_ACK) => match proto::decode(proto::T_HELLO_ACK, &buf)? {
            Msg::HelloAck { status: 0, .. } => {}
            m => return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("hello refused: {m:?}"))),
        },
        _ => return Err(std::io::Error::new(std::io::ErrorKind::Other, "no hello ack")),
    }
    s.set_read_timeout(None)?;
    Ok(s)
}

fn receiver(inner: Arc<Inner>, s: UnixStream, epoch: u64) {
    let mut r = BufReader::with_capacity(1 << 16, s);
    let mut buf = Vec::new();
    while inner.epoch.load(Relaxed) == epoch {
        let ty = match proto::read_frame(&mut r, &mut buf) {
            Ok(Some(t)) => t,
            _ => break,
        };
        let Ok(m) = proto::decode(ty, &buf) else { break };
        let mut st = inner.st.lock().unwrap();
        match m {
            Msg::Hint { req, known_len, match_len, source, chain_ref, tokens } => {
                inner.c.hints_rx.fetch_add(1, Relaxed);
                if st.reqs.contains_key(&req) {
                    st.hints.insert(req, Hint { known_len, match_len, source, chain_ref, tokens });
                }
            }
            Msg::NoHint { req, known_len } => {
                inner.c.no_hints_rx.fetch_add(1, Relaxed);
                if st.hints.get(&req).is_some_and(|h| h.known_len <= known_len) {
                    st.hints.remove(&req);
                }
            }
            Msg::Stats(j) => {
                st.stats = Some(j);
                inner.stats_cv.notify_all();
            }
            _ => {}
        }
    }
}

/// Frames that only make sense on the connection they were queued for (the re-open on a new connection covers them).
fn request_frame(f: &[u8]) -> bool {
    f.get(4).is_some_and(|&t| t == proto::T_OPEN || t == proto::T_APPEND)
}

fn sender(inner: &Arc<Inner>) {
    let mut w: Option<BufWriter<UnixStream>> = None;
    let mut next_try = Instant::now();
    let mut backoff = Duration::from_millis(inner.cfg.reconnect_ms.max(10));
    // the broker lost our state (a broken connection): the next connect re-opens the open requests; until then the
    // queue is the complete history since the client started and is sent as is
    let mut lost = false;
    loop {
        if inner.stop.load(Relaxed) {
            return;
        }
        // (re)connect eagerly, not only when frames are waiting
        if w.is_none() && Instant::now() >= next_try {
            match connect(inner) {
                Ok(s) => {
                    let epoch = inner.epoch.fetch_add(1, Relaxed) + 1;
                    let rs = s.try_clone().unwrap();
                    let i3 = inner.clone();
                    let _ = std::thread::Builder::new().name("chains-recv".into()).spawn(move || receiver(i3, rs, epoch));
                    backoff = Duration::from_millis(inner.cfg.reconnect_ms.max(10));
                    let mut st = inner.st.lock().unwrap();
                    st.hints.clear();
                    let queued: Vec<Vec<u8>> = st.q.drain(..).collect();
                    let mut out: Vec<Vec<u8>> = Vec::new();
                    if lost {
                        // the new broker connection knows none of our requests: re-open the open ones; of the frames
                        // queued since keep Close / Feedback / Seed / StatsReq (the re-open covers Open / Append)
                        for (&req, r) in st.reqs.iter_mut() {
                            r.reopen = false;
                            out.push(reopen_frames(req, r));
                        }
                        let (keep, dropped): (Vec<_>, Vec<_>) = queued.into_iter().partition(|f| !request_frame(f));
                        inner.c.frames_dropped.fetch_add(dropped.len() as u64, Relaxed);
                        out.extend(keep);
                    } else {
                        out = queued;
                    }
                    drop(st);
                    let mut bw = BufWriter::with_capacity(1 << 20, s);
                    if out.iter().try_for_each(|f| bw.write_all(f)).and_then(|_| bw.flush()).is_ok() {
                        inner.c.frames_sent.fetch_add(out.len() as u64, Relaxed);
                        inner.c.connected.store(1, Relaxed);
                        w = Some(bw);
                        lost = false;
                    } else {
                        inner.epoch.fetch_add(1, Relaxed);
                        next_try = Instant::now() + backoff;
                        lost = true;
                    }
                }
                Err(_) => {
                    // frames stay queued (push() caps the queue and marks requests whose frames it drops)
                    next_try = Instant::now() + backoff;
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                }
            }
            continue;
        }
        let batch: Vec<Vec<u8>> = {
            let mut st = inner.st.lock().unwrap();
            while st.q.is_empty() && !inner.stop.load(Relaxed) {
                let wait = if w.is_none() { next_try.saturating_duration_since(Instant::now()).min(Duration::from_millis(100)) }
                           else { Duration::from_millis(100) };
                st = inner.cv.wait_timeout(st, wait.max(Duration::from_millis(1))).unwrap().0;
                if w.is_none() && Instant::now() >= next_try {
                    break;
                }
            }
            if w.is_none() {
                continue; // not connected: frames stay queued until the next connect
            }
            st.q.drain(..).collect()
        };
        if batch.is_empty() {
            continue;
        }
        let bw = w.as_mut().unwrap();
        let n = batch.len() as u64;
        let ok = batch.iter().try_for_each(|f| bw.write_all(f)).and_then(|_| bw.flush());
        if ok.is_ok() {
            inner.c.frames_sent.fetch_add(n, Relaxed);
        } else {
            inner.c.frames_dropped.fetch_add(n, Relaxed);
            inner.c.reconnects.fetch_add(1, Relaxed);
            inner.c.connected.store(0, Relaxed);
            inner.epoch.fetch_add(1, Relaxed);
            w = None;
            lost = true;
            next_try = Instant::now() + Duration::from_millis(inner.cfg.reconnect_ms);
            inner.st.lock().unwrap().hints.clear();
        }
    }
}
