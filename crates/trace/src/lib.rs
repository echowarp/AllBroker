//! Broker operation trace: the requests every connection sent and what the broker answered, kept in a
//! bounded in-memory ring by the broker and written to a file only on an anomaly or on request (TRACE_DUMP), for
//! replay against a test broker (`kvbench replay`). Metadata only: ops, connections, times, counts, statuses, lease
//! ids, 16-byte key prefixes and entry metadata (opaque hashes); never prompt or response content.
//!
//! File: magic `LKVT` + version u16 | header | records, each `u32 len | record`. Little-endian. Readers accept the
//! versions they know (VERSION) and refuse any other; within a version the encoding never changes (tests pin it).
//! A record: t_us u64 (since the broker's start), conn u32, op u16 (kvproto op, or OP_CLOSE), status u16,
//! lat_us u32, then the op's body (see `Body`).

pub const MAGIC: &[u8; 4] = b"LKVT";
pub const VERSION: u16 = 1;
/// Pseudo-op: the connection closed.
pub const OP_CLOSE: u16 = 0xffff;

pub type P16 = [u8; 16];

pub fn p16(k: &[u8; 32]) -> P16 {
    let mut p = [0u8; 16];
    p.copy_from_slice(&k[..16]);
    p
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub broker_version: String,
    pub epoch: u64,
    pub unix_ms: u64,
    pub reason: String,
    /// pool classes: (name, slot bytes)
    pub classes: Vec<(String, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub key: P16,
    pub parent: Option<P16>,
    pub bytes: u64,
    pub kind: u8,
    pub format: u16,
    pub flags: u8,
    pub ttl_ms: u32,
    pub meta: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Body {
    /// HELLO: engine id, name (picks the pool class)
    Hello { engine_id: u16, name: String },
    /// LOOKUP and its answer
    Lookup { ns: u32, acquire: bool, keys: Vec<P16>, n_hit: u32, lease_id: u64 },
    /// RESERVE with a status per item
    Reserve { ns: u32, items: Vec<Item>, statuses: Vec<u16> },
    /// COMMIT / ABORT / TOUCH / PROBE (by op)
    Keys { ns: u32, keys: Vec<P16> },
    Heat { ns: u32, hot: bool, keys: Vec<P16> },
    Release { lease_ids: Vec<u64> },
    TouchRequest { ns: u32, groups: Vec<(u8, Vec<P16>)> },
    Wait { ns: u32, key: P16, timeout_ms: u32 },
    /// DOMAIN descriptor (model / engine / layout lines: scopes the connection's namespaces)
    Domain { descriptor: String },
    /// any other op (STATS, FEATURES, POOL_*, NGRAM_*, TRACE_DUMP) and OP_CLOSE: nothing to replay
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub t_us: u64,
    pub conn: u32,
    pub op: u16,
    pub status: u16,
    pub lat_us: u32,
    pub body: Body,
}

// body tags (part of the format: only ever added)
const B_NONE: u8 = 0;
const B_HELLO: u8 = 1;
const B_LOOKUP: u8 = 2;
const B_RESERVE: u8 = 3;
const B_KEYS: u8 = 4;
const B_HEAT: u8 = 5;
const B_RELEASE: u8 = 6;
const B_TOUCH_REQUEST: u8 = 7;
const B_WAIT: u8 = 8;
const B_DOMAIN: u8 = 9;

#[derive(Default)]
pub struct W(pub Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) -> &mut Self {
        self.0.push(v);
        self
    }
    fn u16(&mut self, v: u16) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn p(&mut self, k: &P16) -> &mut Self {
        self.0.extend_from_slice(k);
        self
    }
    fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.u32(b.len() as u32);
        self.0.extend_from_slice(b);
        self
    }
    fn ps(&mut self, ks: &[P16]) -> &mut Self {
        self.u32(ks.len() as u32);
        for k in ks {
            self.p(k);
        }
        self
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Truncated,
    BadMagic,
    /// a file of a trace format version this build does not read
    Version(u16),
    BadTag(u8),
}

pub type R<T> = Result<T, Error>;

pub struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Rd { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        let s = self.b.get(self.p..self.p.checked_add(n).ok_or(Error::Truncated)?).ok_or(Error::Truncated)?;
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> R<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> R<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> R<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn p(&mut self) -> R<P16> {
        Ok(self.take(16)?.try_into().unwrap())
    }
    fn bytes(&mut self) -> R<Vec<u8>> {
        let n = self.u32()? as usize;
        Ok(self.take(n)?.to_vec())
    }
    fn string(&mut self) -> R<String> {
        Ok(String::from_utf8_lossy(&self.bytes()?).into_owned())
    }
    /// A count, sanity-checked against the bytes left (each element at least `min` bytes).
    fn count(&mut self, min: usize) -> R<usize> {
        let n = self.u32()? as usize;
        if n.saturating_mul(min) > self.b.len() - self.p {
            return Err(Error::Truncated);
        }
        Ok(n)
    }
    fn ps(&mut self) -> R<Vec<P16>> {
        let n = self.count(16)?;
        (0..n).map(|_| self.p()).collect()
    }
    pub fn done(&self) -> bool {
        self.p == self.b.len()
    }
}

impl Header {
    pub fn encode(&self, w: &mut W) {
        w.0.extend_from_slice(MAGIC);
        w.u16(VERSION);
        w.bytes(self.broker_version.as_bytes()).u64(self.epoch).u64(self.unix_ms).bytes(self.reason.as_bytes());
        w.u32(self.classes.len() as u32);
        for (n, s) in &self.classes {
            w.bytes(n.as_bytes()).u64(*s);
        }
    }
    pub fn decode(r: &mut Rd) -> R<Header> {
        if r.take(4)? != MAGIC {
            return Err(Error::BadMagic);
        }
        let v = r.u16()?;
        if v != VERSION {
            return Err(Error::Version(v));
        }
        let broker_version = r.string()?;
        let epoch = r.u64()?;
        let unix_ms = r.u64()?;
        let reason = r.string()?;
        let n = r.count(4 + 8)?;
        let classes = (0..n).map(|_| Ok((r.string()?, r.u64()?))).collect::<R<_>>()?;
        Ok(Header { broker_version, epoch, unix_ms, reason, classes })
    }
}

impl Record {
    pub fn encode(&self, w: &mut W) {
        w.u64(self.t_us).u32(self.conn).u16(self.op).u16(self.status).u32(self.lat_us);
        match &self.body {
            Body::None => {
                w.u8(B_NONE);
            }
            Body::Hello { engine_id, name } => {
                w.u8(B_HELLO).u16(*engine_id).bytes(name.as_bytes());
            }
            Body::Lookup { ns, acquire, keys, n_hit, lease_id } => {
                w.u8(B_LOOKUP).u32(*ns).u8(*acquire as u8).ps(keys).u32(*n_hit).u64(*lease_id);
            }
            Body::Reserve { ns, items, statuses } => {
                w.u8(B_RESERVE).u32(*ns).u32(items.len() as u32);
                for it in items {
                    w.p(&it.key);
                    match &it.parent {
                        Some(p) => w.u8(1).p(p),
                        None => w.u8(0),
                    };
                    w.u64(it.bytes).u8(it.kind).u16(it.format).u8(it.flags).u32(it.ttl_ms).bytes(&it.meta);
                }
                w.u32(statuses.len() as u32);
                for s in statuses {
                    w.u16(*s);
                }
            }
            Body::Keys { ns, keys } => {
                w.u8(B_KEYS).u32(*ns).ps(keys);
            }
            Body::Heat { ns, hot, keys } => {
                w.u8(B_HEAT).u32(*ns).u8(*hot as u8).ps(keys);
            }
            Body::Release { lease_ids } => {
                w.u8(B_RELEASE).u32(lease_ids.len() as u32);
                for l in lease_ids {
                    w.u64(*l);
                }
            }
            Body::TouchRequest { ns, groups } => {
                w.u8(B_TOUCH_REQUEST).u32(*ns).u32(groups.len() as u32);
                for (k, ks) in groups {
                    w.u8(*k).ps(ks);
                }
            }
            Body::Wait { ns, key, timeout_ms } => {
                w.u8(B_WAIT).u32(*ns).p(key).u32(*timeout_ms);
            }
            Body::Domain { descriptor } => {
                w.u8(B_DOMAIN).bytes(descriptor.as_bytes());
            }
        }
    }

    pub fn decode(r: &mut Rd) -> R<Record> {
        let (t_us, conn, op, status, lat_us) = (r.u64()?, r.u32()?, r.u16()?, r.u16()?, r.u32()?);
        let body = match r.u8()? {
            B_NONE => Body::None,
            B_HELLO => Body::Hello { engine_id: r.u16()?, name: r.string()? },
            B_LOOKUP => Body::Lookup { ns: r.u32()?, acquire: r.u8()? != 0, keys: r.ps()?, n_hit: r.u32()?, lease_id: r.u64()? },
            B_RESERVE => {
                let ns = r.u32()?;
                let n = r.count(16 + 1 + 8 + 1 + 2 + 1 + 4 + 4)?;
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    let key = r.p()?;
                    let parent = if r.u8()? != 0 { Some(r.p()?) } else { None };
                    items.push(Item { key, parent, bytes: r.u64()?, kind: r.u8()?, format: r.u16()?, flags: r.u8()?,
                                      ttl_ms: r.u32()?, meta: r.bytes()? });
                }
                let m = r.count(2)?;
                let statuses = (0..m).map(|_| r.u16()).collect::<R<_>>()?;
                Body::Reserve { ns, items, statuses }
            }
            B_KEYS => Body::Keys { ns: r.u32()?, keys: r.ps()? },
            B_HEAT => Body::Heat { ns: r.u32()?, hot: r.u8()? != 0, keys: r.ps()? },
            B_RELEASE => {
                let n = r.count(8)?;
                Body::Release { lease_ids: (0..n).map(|_| r.u64()).collect::<R<_>>()? }
            }
            B_TOUCH_REQUEST => {
                let ns = r.u32()?;
                let n = r.count(1 + 4)?;
                let groups = (0..n).map(|_| Ok((r.u8()?, r.ps()?))).collect::<R<_>>()?;
                Body::TouchRequest { ns, groups }
            }
            B_WAIT => Body::Wait { ns: r.u32()?, key: r.p()?, timeout_ms: r.u32()? },
            B_DOMAIN => Body::Domain { descriptor: r.string()? },
            t => return Err(Error::BadTag(t)),
        };
        Ok(Record { t_us, conn, op, status, lat_us, body })
    }
}

/// A whole trace file: header + length-prefixed records.
pub fn encode_file(h: &Header, records: impl IntoIterator<Item = Vec<u8>>) -> Vec<u8> {
    let mut w = W::default();
    h.encode(&mut w);
    for r in records {
        w.u32(r.len() as u32);
        w.0.extend_from_slice(&r);
    }
    w.0
}

/// Parse a trace file; a record that does not decode ends the trace with an error (never a guess).
pub fn decode_file(b: &[u8]) -> R<(Header, Vec<Record>)> {
    let mut r = Rd::new(b);
    let h = Header::decode(&mut r)?;
    let mut out = Vec::new();
    while !r.done() {
        let n = r.u32()? as usize;
        let body = r.take(n)?;
        let mut rr = Rd::new(body);
        let rec = Record::decode(&mut rr)?;
        if !rr.done() {
            return Err(Error::Truncated);
        }
        out.push(rec);
    }
    Ok((h, out))
}

/// One encoded record.
pub fn encode_record(rec: &Record) -> Vec<u8> {
    let mut w = W::default();
    rec.encode(&mut w);
    w.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> Vec<Record> {
        let k = |n: u8| [n; 16];
        let r = |op: u16, body: Body| Record { t_us: 1_000 + op as u64, conn: 3, op, status: 0, lat_us: 7, body };
        vec![
            r(1, Body::Hello { engine_id: 9, name: "arc-27b".into() }),
            r(2, Body::Lookup { ns: 1, acquire: true, keys: vec![k(1), k(2)], n_hit: 1, lease_id: 44 }),
            r(3, Body::Reserve { ns: 1, items: vec![Item { key: k(3), parent: Some(k(2)), bytes: 5, kind: 1, format: 2,
                                                         flags: 1, ttl_ms: 6, meta: vec![7, 8] }], statuses: vec![0] }),
            r(4, Body::Keys { ns: 1, keys: vec![k(3)] }),
            r(17, Body::Heat { ns: 1, hot: true, keys: vec![k(4)] }),
            r(6, Body::Release { lease_ids: vec![44, 45] }),
            r(19, Body::TouchRequest { ns: 1, groups: vec![(1, vec![k(5)]), (0, vec![k(1), k(5)])] }),
            r(9, Body::Wait { ns: 1, key: k(6), timeout_ms: 8 }),
            r(15, Body::Domain { descriptor: "model=x".into() }),
            r(OP_CLOSE, Body::None),
        ]
    }

    #[test]
    fn roundtrip() {
        let h = Header { broker_version: "0.1.0".into(), epoch: 5, unix_ms: 6, reason: "request".into(),
                         classes: vec![("arc-27b".into(), 58_839_040)] };
        let recs = samples();
        let f = encode_file(&h, recs.iter().map(encode_record));
        let (h2, r2) = decode_file(&f).unwrap();
        assert_eq!((h2, r2), (h, recs));
    }

    #[test]
    fn refuses_other_versions_and_garbage() {
        let h = Header { broker_version: "x".into(), epoch: 1, unix_ms: 2, reason: "r".into(), classes: vec![] };
        let mut f = encode_file(&h, samples().iter().map(encode_record));
        f[4] = 2; // version 2
        assert_eq!(decode_file(&f).err(), Some(Error::Version(2)));
        f[0] = b'X';
        assert_eq!(decode_file(&f).err(), Some(Error::BadMagic));
        let f = encode_file(&h, samples().iter().map(encode_record));
        assert!(decode_file(&f[..f.len() - 3]).is_err(), "a cut file is an error, not a shorter trace");
    }

    /// Format pinning: the bytes of version 1 never change (per body kind; lines only ever added for new kinds).
    #[test]
    fn format_v1_pinned() {
        fn fnv(b: &[u8]) -> u64 {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for x in b {
                h = (h ^ *x as u64).wrapping_mul(0x100_0000_01b3);
            }
            h
        }
        let got: Vec<String> = samples().iter().map(|r| format!("{} {:016x}", r.op, fnv(&encode_record(r)))).collect();
        let pinned: Vec<&str> = PINNED_V1.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        for p in &pinned {
            assert!(got.iter().any(|g| g == p), "trace v1 encoding of `{p}` changed: bump VERSION instead\nnow:\n{}",
                    got.join("\n"));
        }
        for g in &got {
            assert!(pinned.contains(&g.as_str()), "`{g}` not pinned (new body kinds only)\nnow:\n{}", got.join("\n"));
        }
    }

    const PINNED_V1: &str = "
        1 da420ab35dffc725
        2 27df1bc02b565c21
        3 813c4e6b57397a43
        4 6b98572831c98db4
        17 0bcb2d8c40223874
        6 36c643c17b16e085
        19 e7a36dc680b28ed2
        9 34cd8d1f2e8050f9
        15 a92e11aff59a2010
        65535 f6705a8ed49030dc
    ";
}
