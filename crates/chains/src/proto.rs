//! Lookup-chains wire format v1. Unix stream socket, little-endian.
//! Frame: `u32 len` (bytes after this field: type byte + payload) + `u8 type` + payload. Strings: `u16 len` + UTF-8.
//!
//! Engine → broker: 1 Hello, 2 Open, 3 Append, 4 Close, 5 Feedback, 6 StatsReq, 7 Seed (all but Hello and StatsReq
//! are fire-and-forget). Broker → engine: 9 HelloAck, 10 Hint, 11 NoHint, 12 Stats.
use std::io::{self, Read, Write};

pub const PROTOCOL: u16 = 1;
pub const MAX_FRAME: usize = 64 << 20;

pub const T_HELLO: u8 = 1;
pub const T_OPEN: u8 = 2;
pub const T_APPEND: u8 = 3;
pub const T_CLOSE: u8 = 4;
pub const T_FEEDBACK: u8 = 5;
pub const T_STATS_REQ: u8 = 6;
pub const T_SEED: u8 = 7;
pub const T_HELLO_ACK: u8 = 9;
pub const T_HINT: u8 = 10;
pub const T_NO_HINT: u8 = 11;
pub const T_STATS: u8 = 12;

pub const PHASE_THINK: u8 = 0;
pub const PHASE_ANSWER: u8 = 1;
pub const PHASE_TOOL: u8 = 2;

pub const SRC_OWN: u8 = 0;
pub const SRC_GLOBAL: u8 = 1;

#[derive(Clone, Debug, PartialEq)]
pub enum Msg {
    /// gate = minimum suffix-match length for a hint (default 16); max_chain = longest hint (≤ 64)
    Hello { proto: u16, engine_id: u16, model_tag: String, gate: u16, max_chain: u16 },
    /// flags bit0: thinking-capable request
    Open { req: u64, flags: u8, tokens: Vec<u32> },
    /// known_len = the request's length after these tokens
    Append { req: u64, known_len: u32, phase: u8, tokens: Vec<u32> },
    /// status 0 finished, 1 aborted
    Close { req: u64, status: u8 },
    /// (req, offered, accepted, chain_ref)
    Feedback { items: Vec<(u64, u16, u16, u64)> },
    StatsReq,
    /// corpus / template tokens for the global store (flags bit0: permanent)
    Seed { flags: u8, tokens: Vec<u32> },
    /// status 0 ok, 1 protocol version, 2 model_tag mismatch
    HelloAck { proto: u16, status: u8 },
    /// continuation `tokens` for the request at context length `known_len`; `match_len` = suffix-match length
    Hint { req: u64, known_len: u32, match_len: u16, source: u8, chain_ref: u64, tokens: Vec<u32> },
    NoHint { req: u64, known_len: u32 },
    Stats(String),
}

struct W(Vec<u8>);
impl W {
    fn u8(&mut self, v: u8) { self.0.push(v) }
    fn u16(&mut self, v: u16) { self.0.extend_from_slice(&v.to_le_bytes()) }
    fn u32(&mut self, v: u32) { self.0.extend_from_slice(&v.to_le_bytes()) }
    fn u64(&mut self, v: u64) { self.0.extend_from_slice(&v.to_le_bytes()) }
    fn str(&mut self, s: &str) { self.u16(s.len() as u16); self.0.extend_from_slice(s.as_bytes()) }
    fn toks(&mut self, t: &[u32]) {
        for &x in t {
            self.u32(x)
        }
    }
}

struct R<'a>(&'a [u8], usize);
impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.1 + n > self.0.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "short payload"));
        }
        let s = &self.0[self.1..self.1 + n];
        self.1 += n;
        Ok(s)
    }
    fn u8(&mut self) -> io::Result<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> io::Result<u16> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    fn u32(&mut self) -> io::Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> io::Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn str(&mut self) -> io::Result<String> {
        let n = self.u16()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
    fn toks(&mut self, n: usize) -> io::Result<Vec<u32>> {
        Ok(self.take(4 * n)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// Encode one message as a complete frame (appended to `out`).
pub fn encode(m: &Msg, out: &mut Vec<u8>) {
    let mut w = W(Vec::new());
    let ty = match m {
        Msg::Hello { proto, engine_id, model_tag, gate, max_chain } => {
            w.u16(*proto); w.u16(*engine_id); w.str(model_tag); w.u16(*gate); w.u16(*max_chain);
            T_HELLO
        }
        Msg::Open { req, flags, tokens } => { w.u64(*req); w.u8(*flags); w.u32(tokens.len() as u32); w.toks(tokens); T_OPEN }
        Msg::Append { req, known_len, phase, tokens } => {
            w.u64(*req); w.u32(*known_len); w.u8(*phase); w.u16(tokens.len() as u16); w.toks(tokens);
            T_APPEND
        }
        Msg::Close { req, status } => { w.u64(*req); w.u8(*status); T_CLOSE }
        Msg::Feedback { items } => {
            w.u16(items.len() as u16);
            for &(r, o, a, c) in items {
                w.u64(r); w.u16(o); w.u16(a); w.u64(c);
            }
            T_FEEDBACK
        }
        Msg::StatsReq => T_STATS_REQ,
        Msg::Seed { flags, tokens } => { w.u8(*flags); w.u32(tokens.len() as u32); w.toks(tokens); T_SEED }
        Msg::HelloAck { proto, status } => { w.u16(*proto); w.u8(*status); T_HELLO_ACK }
        Msg::Hint { req, known_len, match_len, source, chain_ref, tokens } => {
            w.u64(*req); w.u32(*known_len); w.u16(*match_len); w.u8(*source); w.u64(*chain_ref);
            w.u16(tokens.len() as u16); w.toks(tokens);
            T_HINT
        }
        Msg::NoHint { req, known_len } => { w.u64(*req); w.u32(*known_len); T_NO_HINT }
        Msg::Stats(j) => { w.u32(j.len() as u32); w.0.extend_from_slice(j.as_bytes()); T_STATS }
    };
    out.extend_from_slice(&((w.0.len() + 1) as u32).to_le_bytes());
    out.push(ty);
    out.extend_from_slice(&w.0);
}

pub fn decode(ty: u8, p: &[u8]) -> io::Result<Msg> {
    let mut r = R(p, 0);
    let m = match ty {
        T_HELLO => Msg::Hello { proto: r.u16()?, engine_id: r.u16()?, model_tag: r.str()?, gate: r.u16()?, max_chain: r.u16()? },
        T_OPEN => {
            let (req, flags, n) = (r.u64()?, r.u8()?, r.u32()? as usize);
            Msg::Open { req, flags, tokens: r.toks(n)? }
        }
        T_APPEND => {
            let (req, known_len, phase, n) = (r.u64()?, r.u32()?, r.u8()?, r.u16()? as usize);
            Msg::Append { req, known_len, phase, tokens: r.toks(n)? }
        }
        T_CLOSE => Msg::Close { req: r.u64()?, status: r.u8()? },
        T_FEEDBACK => {
            let n = r.u16()? as usize;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                items.push((r.u64()?, r.u16()?, r.u16()?, r.u64()?));
            }
            Msg::Feedback { items }
        }
        T_STATS_REQ => Msg::StatsReq,
        T_SEED => {
            let (flags, n) = (r.u8()?, r.u32()? as usize);
            Msg::Seed { flags, tokens: r.toks(n)? }
        }
        T_HELLO_ACK => Msg::HelloAck { proto: r.u16()?, status: r.u8()? },
        T_HINT => {
            let (req, known_len, match_len, source, chain_ref, n) = (r.u64()?, r.u32()?, r.u16()?, r.u8()?, r.u64()?, r.u16()? as usize);
            Msg::Hint { req, known_len, match_len, source, chain_ref, tokens: r.toks(n)? }
        }
        T_NO_HINT => Msg::NoHint { req: r.u64()?, known_len: r.u32()? },
        T_STATS => {
            let n = r.u32()? as usize;
            Msg::Stats(String::from_utf8_lossy(r.take(n)?).into_owned())
        }
        t => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown message type {t}"))),
    };
    if r.1 != p.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "trailing bytes"));
    }
    Ok(m)
}

/// Read one frame: Ok(None) on a clean EOF.
pub fn read_frame(r: &mut impl Read, buf: &mut Vec<u8>) -> io::Result<Option<u8>> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len) as usize;
    if len == 0 || len > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("bad frame length {len}")));
    }
    buf.resize(len, 0);
    r.read_exact(buf)?;
    let ty = buf[0];
    buf.remove(0);
    Ok(Some(ty))
}

pub fn send(w: &mut impl Write, m: &Msg) -> io::Result<()> {
    let mut out = Vec::new();
    encode(m, &mut out);
    w.write_all(&out)
}
