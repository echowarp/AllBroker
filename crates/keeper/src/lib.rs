//! Weight keeper: holds the file descriptors of GPU memory an engine exported (Level Zero dma-buf on Intel, CUDA
//! POSIX-fd shareable handles on NVIDIA), so the weights outlive the engine process and the next engine start attaches
//! to the same VRAM instead of reading the checkpoint from disk again. The keeper never maps the memory and has no GPU
//! context: holding the fd is what keeps the allocation alive.
//!
//! Wire protocol (unchanged from the original Python keeper, so existing engine loaders work as they are): a Unix
//! stream socket, one request per connection. Each direction sends an 8-byte little-endian body length (the fds ride
//! on this message as SCM_RIGHTS), then a JSON body.
//!   {"op":"put","key":K,"chunks":[bytes,...],...}  + one fd per chunk -> {"ok":true}
//!   {"op":"get","key":K}                           -> {"found":true, <everything put sent but op/key>} + fds,
//!                                                     or {"found":false}
//!   {"op":"list"}                                  -> {K: {"chunks":n,"bytes":b,"gib":g}, ...}
//!   {"op":"drop","key":K}                          -> {"dropped":bool}
//!
//! Held weights are not a cache: nothing is evicted under memory pressure. The only automatic drop is `max_bytes`
//! (oldest entry first), meant for genuinely different weights piling up during experiments.
//!
//! With `fdstore`, every held fd is also parked in systemd's file descriptor store (sd_notify FDSTORE=1) and the
//! metadata is written to a state file, so a keeper restart (upgrade, crash) gets its entries back instead of freeing
//! the VRAM. The unit needs `FileDescriptorStoreMax=` (and `FileDescriptorStorePreserve=yes` to survive a stop).
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;

/// Kernel limit on fds per SCM_RIGHTS message.
pub const MAX_FDS: usize = 253;
/// Largest accepted request body (manifests of large models are a few MiB).
pub const MAX_BODY: u64 = 256 << 20;
const FD_PREFIX: &str = "whx-";

pub struct Entry {
    pub id: u64,
    pub key: String,
    pub meta: Map<String, Value>,
    pub fds: Vec<OwnedFd>,
}

impl Entry {
    pub fn bytes(&self) -> u64 {
        chunk_bytes(&self.meta)
    }
}

fn chunk_bytes(meta: &Map<String, Value>) -> u64 {
    meta.get("chunks").and_then(|c| c.as_array()).map(|a| a.iter().filter_map(|v| v.as_u64()).sum()).unwrap_or(0)
}

pub struct Keeper {
    /// insertion order = age (oldest first)
    pub entries: Vec<Entry>,
    pub max_bytes: u64,
    next_id: u64,
    fdstore: Option<FdStore>,
}

struct FdStore {
    notify: PathBuf,
    state: PathBuf,
}

impl Keeper {
    pub fn new(max_bytes: u64) -> Keeper {
        Keeper { entries: Vec::new(), max_bytes, next_id: 1, fdstore: None }
    }

    /// Enables the systemd fd store and restores the entries it holds (`listen` = fds passed by systemd with their
    /// names). Returns the number of restored entries.
    pub fn with_fdstore(&mut self, notify: PathBuf, state: PathBuf, listen: Vec<(String, OwnedFd)>) -> usize {
        self.fdstore = Some(FdStore { notify, state });
        let mut byname: HashMap<String, OwnedFd> = listen.into_iter().filter(|(n, _)| n.starts_with(FD_PREFIX)).collect();
        let saved: Value = std::fs::read(&self.fdstore.as_ref().unwrap().state)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(Value::Null);
        self.next_id = saved.get("next_id").and_then(|v| v.as_u64()).unwrap_or(1);
        for e in saved.get("entries").and_then(|v| v.as_array()).into_iter().flatten() {
            let (Some(id), Some(key), Some(meta)) =
                (e.get("id").and_then(|v| v.as_u64()), e.get("key").and_then(|v| v.as_str()), e.get("meta").and_then(|v| v.as_object()))
            else {
                continue;
            };
            let n = meta.get("chunks").and_then(|c| c.as_array()).map(|a| a.len()).unwrap_or(0);
            let fds: Vec<OwnedFd> = (0..n).filter_map(|i| byname.remove(&fd_name(id, i))).collect();
            if n > 0 && fds.len() == n {
                self.entries.push(Entry { id, key: key.to_string(), meta: meta.clone(), fds });
                self.next_id = self.next_id.max(id + 1);
            } else {
                eprintln!("keeper: entry {key} incomplete after restart ({} of {n} fds); dropped", fds.len());
                for i in 0..n {
                    self.notify_remove(&fd_name(id, i));
                }
            }
        }
        for name in byname.keys() {
            eprintln!("keeper: stored fd {name} has no entry; released");
            self.notify_remove(name);
        }
        self.save_state();
        self.entries.len()
    }

    pub fn held_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes()).sum()
    }

    pub fn put(&mut self, key: String, meta: Map<String, Value>, fds: Vec<OwnedFd>) -> Result<(), String> {
        let n = meta.get("chunks").and_then(|c| c.as_array()).map(|a| a.len()).ok_or("put needs a chunks list")?;
        if n != fds.len() {
            return Err(format!("put: {} chunks but {} fds", n, fds.len()));
        }
        if n == 0 || n > MAX_FDS {
            return Err(format!("put: {n} chunks (1..={MAX_FDS})"));
        }
        self.remove(&key);
        let need = chunk_bytes(&meta);
        while self.max_bytes > 0 && !self.entries.is_empty() && self.held_bytes() + need > self.max_bytes {
            let old = self.entries.remove(0);
            eprintln!("keeper: dropped {} ({:.2} GiB) for the {:.2} GiB cap", old.key, gib(old.bytes()), gib(self.max_bytes));
            self.forget(&old);
        }
        let id = self.next_id;
        self.next_id += 1;
        if self.fdstore.is_some() {
            for (i, fd) in fds.iter().enumerate() {
                if let Err(e) = self.notify_store(&fd_name(id, i), fd.as_raw_fd()) {
                    eprintln!("keeper: fd store failed ({e}); entry {key} will not survive a keeper restart");
                }
            }
        }
        eprintln!("keeper: put {key}: {n} chunks, {:.2} GiB, {} tensors", gib(need),
                  meta.get("manifest").and_then(|m| m.as_array()).map(|a| a.len()).unwrap_or(0));
        self.entries.push(Entry { id, key, meta, fds });
        self.save_state();
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.key == key)
    }

    /// Drops an entry; its memory is freed once no engine maps it any more.
    pub fn remove(&mut self, key: &str) -> bool {
        match self.entries.iter().position(|e| e.key == key) {
            Some(i) => {
                let e = self.entries.remove(i);
                self.forget(&e);
                self.save_state();
                true
            }
            None => false,
        }
    }

    pub fn list(&self) -> Value {
        let mut m = Map::new();
        for e in &self.entries {
            let b = e.bytes();
            m.insert(e.key.clone(), json!({"chunks": e.fds.len(), "bytes": b, "gib": (gib(b) * 100.0).round() / 100.0}));
        }
        Value::Object(m)
    }

    fn forget(&self, e: &Entry) {
        for i in 0..e.fds.len() {
            self.notify_remove(&fd_name(e.id, i));
        }
    }

    fn save_state(&self) {
        let Some(fs) = &self.fdstore else { return };
        let entries: Vec<Value> = self.entries.iter().map(|e| json!({"id": e.id, "key": e.key, "meta": e.meta})).collect();
        let body = json!({"version": 1, "next_id": self.next_id, "entries": entries});
        let tmp = fs.state.with_extension("tmp");
        let ok = std::fs::write(&tmp, serde_json::to_vec(&body).unwrap()).and_then(|_| std::fs::rename(&tmp, &fs.state));
        if let Err(e) = ok {
            eprintln!("keeper: state file {} not written ({e})", fs.state.display());
        }
    }

    fn notify_store(&self, name: &str, fd: RawFd) -> io::Result<()> {
        let fs = self.fdstore.as_ref().unwrap();
        sd_notify(&fs.notify, &format!("FDSTORE=1\nFDNAME={name}"), &[fd])
    }

    fn notify_remove(&self, name: &str) {
        if let Some(fs) = &self.fdstore {
            if let Err(e) = sd_notify(&fs.notify, &format!("FDSTOREREMOVE=1\nFDNAME={name}"), &[]) {
                eprintln!("keeper: FDSTOREREMOVE {name} failed ({e}); systemd may still hold that memory");
            }
        }
    }

    /// Serves one connection (one request, one reply).
    pub fn serve(&mut self, c: &mut UnixStream) -> io::Result<()> {
        let (req, fds) = recv_msg(c)?;
        let mut req = match req {
            Value::Object(m) => m,
            _ => return send_msg(c, &json!({"error": "request is not an object"}), &[]),
        };
        let op = req.remove("op").and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
        let key = req.remove("key").and_then(|v| v.as_str().map(String::from));
        match (op.as_str(), key) {
            ("put", Some(k)) => match self.put(k, req, fds) {
                Ok(()) => send_msg(c, &json!({"ok": true}), &[]),
                Err(e) => send_msg(c, &json!({"ok": false, "error": e}), &[]),
            },
            ("get", Some(k)) => match self.get(&k) {
                Some(e) => {
                    let mut r = e.meta.clone();
                    r.insert("found".into(), Value::Bool(true));
                    let raw: Vec<RawFd> = e.fds.iter().map(|f| f.as_raw_fd()).collect();
                    send_msg(c, &Value::Object(r), &raw)
                }
                None => send_msg(c, &json!({"found": false}), &[]),
            },
            ("list", _) => send_msg(c, &self.list(), &[]),
            ("drop", Some(k)) => {
                let d = self.remove(&k);
                if d {
                    eprintln!("keeper: drop {k}");
                }
                send_msg(c, &json!({"dropped": d}), &[])
            }
            _ => send_msg(c, &json!({"error": "unknown op"}), &[]),
        }
    }
}

pub fn fd_name(id: u64, chunk: usize) -> String {
    format!("{FD_PREFIX}{id}-{chunk}")
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

// ------------------------------------------------------------------ SCM_RIGHTS framing

/// Receives one framed message: 8-byte length (+ fds), then the JSON body.
pub fn recv_msg(c: &mut UnixStream) -> io::Result<(Value, Vec<OwnedFd>)> {
    let mut hdr = [0u8; 8];
    let (got, fds) = recvmsg_fds(c.as_raw_fd(), &mut hdr)?;
    if got == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "empty request"));
    }
    c.read_exact(&mut hdr[got..])?;
    let n = u64::from_le_bytes(hdr);
    if n > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("body of {n} B")));
    }
    let mut body = vec![0u8; n as usize];
    c.read_exact(&mut body)?;
    let v = serde_json::from_slice(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok((v, fds))
}

/// Sends one framed message; the fds are duplicated into the receiver, the caller keeps its own.
pub fn send_msg(c: &mut UnixStream, v: &Value, fds: &[RawFd]) -> io::Result<()> {
    let body = serde_json::to_vec(v).unwrap();
    sendmsg_fds(c.as_raw_fd(), &(body.len() as u64).to_le_bytes(), fds, None)?;
    c.write_all(&body)
}

fn cmsg_space(n: usize) -> usize {
    unsafe { libc::CMSG_SPACE((n * std::mem::size_of::<RawFd>()) as u32) as usize }
}

fn recvmsg_fds(sock: RawFd, buf: &mut [u8]) -> io::Result<(usize, Vec<OwnedFd>)> {
    let mut ctl = vec![0u8; cmsg_space(MAX_FDS)];
    let mut iov = libc::iovec { iov_base: buf.as_mut_ptr().cast(), iov_len: buf.len() };
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctl.as_mut_ptr().cast();
    msg.msg_controllen = ctl.len() as _;
    let n = loop {
        let r = unsafe { libc::recvmsg(sock, &mut msg, libc::MSG_CMSG_CLOEXEC) };
        if r >= 0 {
            break r as usize;
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    };
    let mut fds = Vec::new();
    unsafe {
        let mut cm = libc::CMSG_FIRSTHDR(&msg);
        while !cm.is_null() {
            if (*cm).cmsg_level == libc::SOL_SOCKET && (*cm).cmsg_type == libc::SCM_RIGHTS {
                let data = libc::CMSG_DATA(cm) as *const RawFd;
                let k = ((*cm).cmsg_len as usize - libc::CMSG_LEN(0) as usize) / std::mem::size_of::<RawFd>();
                for i in 0..k {
                    fds.push(OwnedFd::from_raw_fd(std::ptr::read_unaligned(data.add(i))));
                }
            }
            cm = libc::CMSG_NXTHDR(&msg, cm);
        }
    }
    if msg.msg_flags & libc::MSG_CTRUNC != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("more than {MAX_FDS} fds")));
    }
    Ok((n, fds))
}

/// sendmsg with fds; `to` = datagram destination (sd_notify), None for a connected stream.
fn sendmsg_fds(sock: RawFd, data: &[u8], fds: &[RawFd], to: Option<(&libc::sockaddr_un, libc::socklen_t)>) -> io::Result<()> {
    let mut ctl = vec![0u8; if fds.is_empty() { 0 } else { cmsg_space(fds.len()) }];
    let mut iov = libc::iovec { iov_base: data.as_ptr() as *mut _, iov_len: data.len() };
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    if let Some((addr, len)) = to {
        msg.msg_name = addr as *const _ as *mut _;
        msg.msg_namelen = len;
    }
    if !fds.is_empty() {
        msg.msg_control = ctl.as_mut_ptr().cast();
        msg.msg_controllen = ctl.len() as _;
        unsafe {
            let cm = libc::CMSG_FIRSTHDR(&msg);
            (*cm).cmsg_level = libc::SOL_SOCKET;
            (*cm).cmsg_type = libc::SCM_RIGHTS;
            (*cm).cmsg_len = libc::CMSG_LEN((fds.len() * std::mem::size_of::<RawFd>()) as u32) as _;
            std::ptr::copy_nonoverlapping(fds.as_ptr(), libc::CMSG_DATA(cm) as *mut RawFd, fds.len());
        }
    }
    let mut sent = loop {
        let r = unsafe { libc::sendmsg(sock, &msg, libc::MSG_NOSIGNAL) };
        if r >= 0 {
            break r as usize;
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    };
    while sent < data.len() {
        // stream only (a datagram goes whole): the fds went with the first part
        let r = unsafe { libc::send(sock, data[sent..].as_ptr().cast(), data.len() - sent, libc::MSG_NOSIGNAL) };
        if r < 0 {
            return Err(io::Error::last_os_error());
        }
        sent += r as usize;
    }
    Ok(())
}

// ------------------------------------------------------------------ systemd

/// sd_notify with fds (path or abstract `@` socket).
pub fn sd_notify(path: &std::path::Path, state: &str, fds: &[RawFd]) -> io::Result<()> {
    let p = path.as_os_str().as_encoded_bytes();
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as _;
    if p.is_empty() || p.len() >= addr.sun_path.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "bad NOTIFY_SOCKET"));
    }
    for (i, b) in p.iter().enumerate() {
        addr.sun_path[i] = if i == 0 && *b == b'@' { 0 } else { *b as _ };
    }
    let len = (std::mem::size_of::<libc::sa_family_t>() + p.len()) as libc::socklen_t;
    let s = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, 0) };
    if s < 0 {
        return Err(io::Error::last_os_error());
    }
    let s = unsafe { OwnedFd::from_raw_fd(s) };
    sendmsg_fds(s.as_raw_fd(), state.as_bytes(), fds, Some((&addr, len)))
}

/// fds passed by systemd (LISTEN_PID / LISTEN_FDS / LISTEN_FDNAMES), with their names.
pub fn listen_fds() -> Vec<(String, OwnedFd)> {
    let pid_ok = std::env::var("LISTEN_PID").ok().and_then(|v| v.parse::<u32>().ok()) == Some(std::process::id());
    let n = std::env::var("LISTEN_FDS").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0);
    if !pid_ok || n <= 0 {
        return Vec::new();
    }
    let names: Vec<String> = std::env::var("LISTEN_FDNAMES").unwrap_or_default().split(':').map(String::from).collect();
    for v in ["LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES"] {
        std::env::remove_var(v);
    }
    (0..n)
        .map(|i| {
            let fd = 3 + i;
            unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
            (names.get(i as usize).cloned().unwrap_or_default(), unsafe { OwnedFd::from_raw_fd(fd) })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memfd() -> OwnedFd {
        unsafe { OwnedFd::from_raw_fd(libc::memfd_create(c"t".as_ptr(), libc::MFD_CLOEXEC)) }
    }

    fn meta(sizes: &[u64]) -> Map<String, Value> {
        json!({"chunks": sizes}).as_object().unwrap().clone()
    }

    #[test]
    fn cap_drops_oldest_and_put_replaces() {
        let mut k = Keeper::new(10);
        k.put("a".into(), meta(&[4]), vec![memfd()]).unwrap();
        k.put("b".into(), meta(&[4]), vec![memfd()]).unwrap();
        k.put("a".into(), meta(&[3]), vec![memfd()]).unwrap(); // replace: a is now the newest
        assert_eq!(k.held_bytes(), 7);
        k.put("c".into(), meta(&[5]), vec![memfd()]).unwrap(); // over the cap: b (oldest) goes
        let keys: Vec<_> = k.entries.iter().map(|e| e.key.as_str()).collect();
        assert_eq!(keys, ["a", "c"]);
    }

    #[test]
    fn no_cap_means_no_drop() {
        let mut k = Keeper::new(0);
        for i in 0..5 {
            k.put(format!("m{i}"), meta(&[1 << 40]), vec![memfd()]).unwrap();
        }
        assert_eq!(k.entries.len(), 5);
    }

    #[test]
    fn put_checks_fd_count() {
        let mut k = Keeper::new(0);
        assert!(k.put("a".into(), meta(&[1, 2]), vec![memfd()]).is_err());
        assert!(k.put("a".into(), json!({}).as_object().unwrap().clone(), vec![]).is_err());
        assert!(k.entries.is_empty());
    }
}
