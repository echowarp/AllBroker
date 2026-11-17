//! Fault injection against a real broker process. Needs the failpoints build:
//!
//!   cargo test -p llmt-kvbroker --features failpoints --test failpoints
//!
//! Each test starts its own llmt-kvbroker (socket under /tmp/claude-1000, on-demand pool in /dev/shm/llmt-kv-fp-*,
//! SSD tier on tmpfs), switches failpoints on and off at runtime through LLMT_KV_FAILPOINTS_FILE (src/fp.rs) and
//! checks recovery: no wrong bytes are ever served (every entry carries its key id, a generation and a pattern
//! derived from both, and every read is verified), refused / aborted reservations leave nothing behind, SSD failures
//! end as misses, and the lock watchdog exits 70 on a held or poisoned lock. The broker and its directories are
//! removed when a test ends (also on failure).
#![cfg(feature = "failpoints")]

use kvclient::{Client, Error, Key, PoolMap, ReserveItem, Status};
use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

const NS: u32 = 5;
const SLOT: usize = 64 << 10;

static SEQ: AtomicU32 = AtomicU32::new(0);

struct Broker {
    child: Option<Child>,
    sock: PathBuf,
    run_dir: PathBuf,
    pool_dir: PathBuf,
    fp_file: PathBuf,
    log: PathBuf,
}

impl Broker {
    /// A broker with a pool of `slots` 64 KiB slots and these extra arguments.
    fn start(slots: u32, extra: &[&str]) -> Broker {
        Self::start_disk(slots, extra, None)
    }

    /// ... with an SSD tier (on tmpfs, inside the run directory) using this codec.
    fn start_disk(slots: u32, extra: &[&str], disk_codec: Option<&str>) -> Broker {
        Self::start_bin(env!("CARGO_BIN_EXE_llmt-kvbroker"), slots, extra, disk_codec)
    }

    /// ... with another broker binary (format compatibility with an older build).
    fn start_bin(bin: &str, slots: u32, extra: &[&str], disk_codec: Option<&str>) -> Broker {
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let id = format!("{}-{n}", std::process::id());
        let run_dir = PathBuf::from(format!("/tmp/claude-1000/fp{id}"));
        let pool_dir = PathBuf::from(format!("/dev/shm/llmt-kv-fp-{id}"));
        fs::create_dir_all(&run_dir).unwrap();
        let sock = run_dir.join("s");
        let fp_file = run_dir.join("fp");
        let log = run_dir.join("log");
        fs::write(&fp_file, "").unwrap();
        let gib = format!("{}", (slots as u64 * SLOT as u64) as f64 / (1u64 << 30) as f64);
        let mut args: Vec<String> = [
            "--pool-gib", &gib, "--slot-kib", "64", "--pool-dir", pool_dir.to_str().unwrap(),
            "--socket", sock.to_str().unwrap(), "--metrics", "off", "--events-dir", "off", "--hotset-dir", "off",
            "--scrub", "off", "--new-pool", "--no-mlock", "--stats-log-s", "0", "--mem-reserve-gib", "0",
            "--low-water-pct", "0", "--reuse-grace-ms", "0", "--pool-lazy", "on", "--sweep-s", "1",
            "--compact-mib", "0", "--lease-timeout-ms", "60000",
        ].iter().map(|s| s.to_string()).collect();
        // operation trace dumps stay in the run directory (never the default ~/.cache/llmt-kv-trace); an older build
        // (OLD_KVBROKER) has no tracing and refuses the flag
        if bin == env!("CARGO_BIN_EXE_llmt-kvbroker") {
            args.extend(["--trace-dir".to_string(), run_dir.join("trace").display().to_string()]);
        }
        args.extend(extra.iter().map(|s| s.to_string()));
        if let Some(codec) = disk_codec {
            let d = run_dir.join("d");
            args.extend(["--disk-dir", d.to_str().unwrap(), "--disk-min-hits", "0", "--disk-state-min-hits", "0",
                         "--disk-flush-s", "1", "--disk-gib", "1", "--disk-codec", codec].iter().map(|s| s.to_string()));
        }
        let child = Command::new(bin)
            .args(&args)
            .env("LLMT_KV_FAILPOINTS_FILE", &fp_file)
            .env_remove("FAILPOINTS")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(fs::File::create(&log).unwrap())
            .spawn()
            .expect("start llmt-kvbroker");
        let b = Broker { child: Some(child), sock, run_dir, pool_dir, fp_file, log };
        let t = Instant::now();
        while !b.sock.exists() {
            assert!(t.elapsed() < Duration::from_secs(10), "broker did not start:\n{}", b.log_text());
            std::thread::sleep(Duration::from_millis(20));
        }
        b
    }

    fn log_text(&self) -> String {
        fs::read_to_string(&self.log).unwrap_or_default()
    }

    /// Set the active failpoints (`name=action` lines; "" = all off). Waits for the broker to apply them.
    fn fp(&self, spec: &str) {
        let tmp = self.run_dir.join("fp.tmp");
        fs::write(&tmp, spec.replace(';', "\n")).unwrap();
        fs::rename(&tmp, &self.fp_file).unwrap();
        std::thread::sleep(Duration::from_millis(200));
    }

    /// Wait until the log shows `pat` (at least `count` times).
    fn wait_log(&self, pat: &str, count: usize, limit: Duration) -> bool {
        let t = Instant::now();
        while t.elapsed() < limit {
            if self.log_text().matches(pat).count() >= count {
                return true;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        false
    }

    fn connect(&self, tag: u16, timeout_ms: u64) -> (Client, PoolMap) {
        let c = Client::connect(&self.sock, tag, &format!("fp-{tag}"), Duration::from_millis(timeout_ms)).expect("connect");
        let p = PoolMap::open(&c.hello).expect("map pool");
        (c, p)
    }

    /// Wait for the broker to exit by itself; its exit code (None: still running after `limit`).
    fn wait_exit(&mut self, limit: Duration) -> Option<i32> {
        let t = Instant::now();
        let ch = self.child.as_mut().unwrap();
        while t.elapsed() < limit {
            if let Some(st) = ch.try_wait().unwrap() {
                self.child = None;
                return Some(st.code().unwrap_or(-1));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        None
    }
}

impl Drop for Broker {
    fn drop(&mut self) {
        if let Some(mut ch) = self.child.take() {
            let _ = ch.kill();
            let _ = ch.wait();
        }
        if std::thread::panicking() {
            eprintln!("---- broker log ----\n{}", self.log_text());
        }
        let _ = fs::remove_dir_all(&self.pool_dir);
        let _ = fs::remove_dir_all(&self.run_dir);
    }
}

// ------------------------------------------------------------------------------------ self-describing payloads

fn key(id: u32) -> Key {
    let mut k = [0u8; 32];
    k[..4].copy_from_slice(&id.to_le_bytes());
    k[4..12].copy_from_slice(b"kvfp0001");
    k
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Compressible (one value per 4 KiB block with a ripple), and every block differs between generations.
fn body(id: u32, gen: u64, i: usize) -> u8 {
    (mix((id as u64) << 40 ^ gen ^ (i / 4096) as u64) as u8) ^ ((i / 61) % 4) as u8
}

fn payload(id: u32, gen: u64, n: usize) -> Vec<u8> {
    let mut b = vec![0u8; n];
    b[..4].copy_from_slice(&id.to_le_bytes());
    b[4..12].copy_from_slice(&gen.to_le_bytes());
    b[12..20].copy_from_slice(&(n as u64).to_le_bytes());
    for (i, x) in b.iter_mut().enumerate().skip(20) {
        *x = body(id, gen, i);
    }
    b
}

/// The generation the bytes carry; panics on foreign or torn bytes.
fn check(buf: &[u8], id: u32) -> u64 {
    let hid = u32::from_le_bytes(buf[..4].try_into().unwrap());
    let len = u64::from_le_bytes(buf[12..20].try_into().unwrap()) as usize;
    assert!(hid == id && len == buf.len(), "foreign bytes for key {id}: header id {hid}, len {len}/{}", buf.len());
    let gen = u64::from_le_bytes(buf[4..12].try_into().unwrap());
    let bad = buf.iter().enumerate().skip(20).filter(|(i, b)| **b != body(id, gen, *i)).count();
    assert_eq!(bad, 0, "torn bytes for key {id} gen {gen}");
    gen
}

fn item(id: u32, bytes: usize) -> ReserveItem {
    ReserveItem { key: key(id), parent: None, bytes: bytes as u64, kind: 0, format: 1, cost_hint: 0, flags: 0, ttl_ms: 0,
                  meta: vec![] }
}

/// RESERVE, write, COMMIT. The reservation's status (Ok = committed).
fn store(c: &mut Client, p: &PoolMap, id: u32, gen: u64, bytes: usize) -> Status {
    store_item(c, p, item(id, bytes), id, gen, bytes)
}

fn store_item(c: &mut Client, p: &PoolMap, it: ReserveItem, id: u32, gen: u64, bytes: usize) -> Status {
    let r = c.reserve(NS, vec![it]).expect("reserve");
    let (st, ranges) = &r[0];
    if *st != Status::Ok {
        assert!(ranges.is_empty(), "a refused reservation came with slots");
        return *st;
    }
    let data = payload(id, gen, bytes);
    let mut off = 0;
    for rg in ranges {
        let s = unsafe { p.range(*rg) };
        let n = s.len().min(bytes - off);
        s.write(0, &data[off..off + n]);
        off += n;
    }
    assert_eq!(off, bytes);
    let cs = c.commit(NS, &[key(id)]).expect("commit");
    assert_eq!(cs[0], Status::Ok, "commit of key {id}");
    Status::Ok
}

/// Leased read of one key, verified; the generation it holds (None = miss).
fn read(c: &mut Client, p: &PoolMap, id: u32) -> Option<u64> {
    let l = c.lookup(NS, &[key(id)], true).expect("lookup");
    if l.n_hit == 0 {
        return None;
    }
    let e = &l.entries[0];
    let mut out = vec![0u8; e.bytes as usize];
    let first = unsafe { p.range(e.ranges[0]) };
    let img = first.to_vec(first.len());
    if e.ranges.len() == 1 && kvslotz::is_compressed(&img) {
        assert!(kvslotz::read_range(&img, 0, &mut out), "compressed slot of key {id} does not decode");
    } else {
        let mut off = 0;
        for r in &e.ranges {
            let s = unsafe { p.range(*r) };
            let n = s.len().min(out.len() - off);
            s.read(0, &mut out[off..off + n]);
            off += n;
        }
    }
    let g = check(&out, id);
    c.release(&[l.lease_id]).expect("release");
    Some(g)
}

fn probe(c: &mut Client, id: u32) -> u8 {
    c.probe(NS, &[key(id)]).expect("probe")[0]
}

fn stat(c: &mut Client, name: &str) -> u64 {
    c.stats().expect("stats").into_iter().find(|(n, _)| n == name).map(|x| x.1).unwrap_or(0)
}

// ------------------------------------------------------------------------------------------------------- tests

/// The pool's file system is full: RESERVE answers FULL (no slots, no entry left behind); once it has room again
/// stores work and read back intact.
#[test]
fn reserve_refused_when_pool_fs_full() {
    let b = Broker::start(16, &[]);
    let (mut c, p) = b.connect(1, 2000);
    b.fp("lazy-room-for=return");
    for id in 0..4 {
        assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Full);
        assert_eq!(read(&mut c, &p, id), None);
    }
    // a refused key can be reserved again at once (nothing reserved / quarantined under it)
    b.fp("");
    for id in 0..4 {
        assert_eq!(store(&mut c, &p, id, 2, 40_000), Status::Ok);
    }
    for id in 0..4 {
        assert_eq!(read(&mut c, &p, id), Some(2));
    }
    assert!(b.log_text().contains("failpoint lazy-room-for: reached"));
}

/// Fill the pool with other keys so the first ones leave RAM.
fn evict_by_filling(c: &mut Client, p: &PoolMap, from: u32, n: u32) {
    for id in from..from + n {
        assert_eq!(store(c, p, id, 9, 40_000), Status::Ok);
    }
}

/// Wait until key `id` is on the SSD only (probe 2) or `limit`.
fn wait_probe(c: &mut Client, id: u32, want: u8, limit: Duration) -> bool {
    let t = Instant::now();
    while t.elapsed() < limit {
        if probe(c, id) == want {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// SSD writes fail half way (EIO): no file, no partial file, no index entry; the evicted keys are plain misses.
#[test]
fn ssd_write_error_leaves_nothing() {
    let b = Broker::start_disk(8, &[], Some("none"));
    let w = Worker::start(&b);                              // SSD writes are the background worker's
    let (mut c, p) = b.connect(1, 2000);
    w.fp("disk-write-io=return");
    for id in 0..4 {
        assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Ok);
    }
    assert!(w.wait_log(&b, "failpoint disk-write-io: reached", 4, Duration::from_secs(10)), "no SSD write attempted");
    w.fp("");
    evict_by_filling(&mut c, &p, 100, 8);
    for id in 0..4 {
        assert_eq!(read(&mut c, &p, id), None, "key {id} should have left RAM");
        assert_eq!(probe(&mut c, id), 0, "key {id}: a failed write must not be on the SSD tier");
    }
    let left: Vec<String> = fs::read_dir(b.run_dir.join("d")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp")).collect();
    assert!(left.is_empty(), "partial files left: {left:?}");
}

/// SSD data read back corrupted (both codecs): the checksum catches it, the file is dropped, the lookup misses;
/// never wrong bytes.
#[test]
fn ssd_read_corruption_is_a_miss() {
    for codec in ["none", "planes-zstd"] {
        let b = Broker::start_disk(8, &[], Some(codec));
        let _w = Worker::start(&b);                         // writes the SSD files
        let (mut c, p) = b.connect(1, 2000);
        for id in 0..4 {
            assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Ok);
        }
        std::thread::sleep(Duration::from_millis(2500)); // two flushes: on the SSD tier
        b.fp("disk-read-corrupt=return");
        evict_by_filling(&mut c, &p, 100, 8);
        for id in 0..4 {
            // probe 2 = on the SSD only; it also queues the promotion, which reads the corrupted file
            let pr = probe(&mut c, id);
            assert!(pr == 2 || pr == 0, "key {id} probe {pr}");
        }
        for id in 0..4 {
            assert!(wait_probe(&mut c, id, 0, Duration::from_secs(5)), "{codec}: damaged file of key {id} still listed");
            assert_eq!(read(&mut c, &p, id), None);
        }
        assert!(b.log_text().contains("failpoint disk-read-corrupt: reached"));
    }
}

/// Promotion cannot allocate pool memory (ENOSPC): the entry is not inserted (a miss, the file kept); once memory
/// is back the promotion succeeds with the right bytes.
#[test]
fn ssd_promotion_without_pool_memory() {
    let b = Broker::start_disk(8, &[], Some("planes-zstd"));
    let _w = Worker::start(&b);                             // writes the SSD files
    let (mut c, p) = b.connect(1, 2000);
    for id in 0..2 {
        assert_eq!(store(&mut c, &p, id, 7, 40_000), Status::Ok);
    }
    std::thread::sleep(Duration::from_millis(2500));
    evict_by_filling(&mut c, &p, 100, 8);
    b.fp("lazy-ensure=return");
    for id in 0..2 {
        assert_eq!(probe(&mut c, id), 2, "key {id} should be on the SSD only");
    }
    assert!(b.wait_log("failpoint lazy-ensure: reached", 1, Duration::from_secs(5)));
    std::thread::sleep(Duration::from_millis(300));
    for id in 0..2 {
        assert_eq!(read(&mut c, &p, id), None, "key {id}: promoted without memory");
        assert_eq!(probe(&mut c, id), 2, "key {id}: the file must stay");
    }
    b.fp("");
    for id in 0..2 {
        let t = Instant::now();
        let mut got = None;
        while got.is_none() && t.elapsed() < Duration::from_secs(5) {
            probe(&mut c, id);
            std::thread::sleep(Duration::from_millis(100));
            got = read(&mut c, &p, id);
        }
        assert_eq!(got, Some(7), "key {id} not promoted after memory came back");
    }
}

/// Warm-margin allocation failing does not break stores (the client's writes allocate the pages).
#[test]
fn warm_allocation_failure_is_harmless() {
    let b = Broker::start(16, &["--warm-mib", "1"]);
    let (mut c, p) = b.connect(1, 2000);
    b.fp("warm-fallocate=return");
    for id in 0..8 {
        assert_eq!(store(&mut c, &p, id, 3, 40_000), Status::Ok);
    }
    assert!(b.wait_log("failpoint warm-fallocate: reached", 1, Duration::from_secs(5)), "warm thread never ran");
    for id in 8..16 {
        assert_eq!(store(&mut c, &p, id, 3, 40_000), Status::Ok);
    }
    for id in 0..16 {
        assert_eq!(read(&mut c, &p, id), Some(3));
    }
}

/// The core lock is held for 1.5 s inside a request: the callers time out (bounded), the timed-out writer drops its
/// connection, its reservation is gone (WriterGone) and the key can be stored again with intact bytes; a client that
/// timed out keeps working (late replies are discarded by request id). The watchdog (5 s) does not fire.
#[test]
fn lock_held_long_times_out_and_recovers() {
    let b = Broker::start(16, &["--watchdog-s", "5"]);
    let (mut a, _pa) = b.connect(1, 200);
    let (mut r, pr) = b.connect(2, 200);
    assert_eq!(store(&mut r, &pr, 50, 1, 40_000), Status::Ok);
    b.fp("handle-locked=1*sleep(1500)");
    let t = Instant::now();
    let res = a.reserve(NS, vec![item(1, 40_000)]);
    assert!(matches!(res, Err(Error::Timeout)), "reserve under a held lock: {:?}", res.map(|_| ()));
    assert!(t.elapsed() < Duration::from_millis(400));
    // a second client stuck behind the lock times out too
    let t = Instant::now();
    assert!(matches!(r.lookup(NS, &[key(50)], false), Err(Error::Timeout)));
    assert!(t.elapsed() < Duration::from_millis(400));
    drop(a); // the engine drops a connection after timeouts: its reservation's writer is gone
    std::thread::sleep(Duration::from_millis(1600));
    r.set_timeout(Duration::from_millis(2000));
    // the timed-out client works again; the late lookup reply is not taken for the next request
    assert_eq!(read(&mut r, &pr, 50), Some(1));
    assert_eq!(read(&mut r, &pr, 1), None, "reservation of the dropped writer became visible");
    assert_eq!(store(&mut r, &pr, 1, 2, 40_000), Status::Ok);
    assert_eq!(read(&mut r, &pr, 1), Some(2));
    assert!(b.log_text().contains("failpoint handle-locked: reached"));
    assert!(!b.log_text().contains("watchdog: an internal lock"), "watchdog fired on a 1.5 s hold");
}

/// A reply delayed past the client's timeout: the reservation was made; the client drops the connection, the
/// reservation is cleaned up, and a fresh connection stores the key with intact bytes.
#[test]
fn delayed_reply_reservation_cleaned_up() {
    let b = Broker::start(16, &[]);
    let (mut a, _pa) = b.connect(1, 200);
    b.fp("reply-delay=1*sleep(800)");
    assert!(matches!(a.reserve(NS, vec![item(3, 40_000)]), Err(Error::Timeout)));
    drop(a);
    std::thread::sleep(Duration::from_millis(900));
    let (mut c, p) = b.connect(2, 2000);
    assert_eq!(read(&mut c, &p, 3), None);
    assert_eq!(store(&mut c, &p, 3, 5, 40_000), Status::Ok);
    assert_eq!(read(&mut c, &p, 3), Some(5));
    assert_eq!(stat(&mut c, "slots_reserved"), 0);
    assert_eq!(stat(&mut c, "slots_quarantined"), 1, "the dropped writer's slot is quarantined (it may still write)");
}

/// A lock held for good (6 s > --watchdog-s 2): the broker exits 70 so systemd restarts it.
#[test]
fn watchdog_exits_on_held_lock() {
    let mut b = Broker::start(16, &["--watchdog-s", "2"]);
    let (mut a, _p) = b.connect(1, 200);
    b.fp("handle-locked=1*sleep(6000)");
    assert!(a.lookup(NS, &[key(1)], false).is_err());
    assert_eq!(b.wait_exit(Duration::from_secs(8)), Some(70), "watchdog did not exit 70");
}

/// A connection thread panics inside the core lock (poisoned lock): that client's call fails (a timeout: the
/// connection's writer half stays registered, so the socket only closes when the process exits), the broker exits 70.
#[test]
fn watchdog_exits_on_poisoned_lock() {
    let mut b = Broker::start(16, &["--watchdog-s", "2"]);
    let (mut a, _p) = b.connect(1, 1000);
    b.fp("handle-locked=1*panic");
    let r = a.lookup(NS, &[key(1)], false);
    assert!(r.is_err());
    assert_eq!(b.wait_exit(Duration::from_secs(6)), Some(70), "watchdog did not exit 70");
}

fn compress_args() -> Vec<&'static str> {
    // the broker's own compressor (--bg internal); the background worker's path: worker_* tests
    vec!["--class", "t=64,rcodec=1", "--ram-compress-after-s", "0", "--ram-compress-quiet-ms", "0", "--bg", "internal"]
}

/// The RAM codec fails: entries stay raw and read back intact; with the codec back, entries get compressed and
/// still read back intact.
#[test]
fn compression_failure_keeps_raw() {
    let b = Broker::start(8, &compress_args());
    let (mut c, p) = b.connect(1, 2000);
    b.fp("rcompress-fail=return");
    for id in 0..3 {
        assert_eq!(store(&mut c, &p, id, 4, 60_000), Status::Ok);
    }
    assert!(b.wait_log("failpoint rcompress-fail: reached", 3, Duration::from_secs(5)));
    b.fp("");
    for id in 3..6 {
        assert_eq!(store(&mut c, &p, id, 4, 60_000), Status::Ok);
    }
    let t = Instant::now();
    while stat(&mut c, "compressed_in_place") < 3 && t.elapsed() < Duration::from_secs(6) {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(stat(&mut c, "compressed_in_place") >= 3, "the codec never ran after the failure");
    for id in 0..6 {
        assert_eq!(read(&mut c, &p, id), Some(4));
    }
}

/// The in-place compressor copies an entry's slot out, then (here: paused) the entry is evicted and the same key is
/// stored again with new bytes in the same slot. The compressor must not write its image of the old bytes over the
/// new entry (it would serve generation 1 for a key whose committed bytes are generation 2).
#[test]
fn compression_does_not_resurrect_old_bytes() {
    let b = Broker::start(4, &compress_args());
    let (mut c, p) = b.connect(1, 3000);
    b.fp("rcompress-read=1*pause");
    assert_eq!(store(&mut c, &p, 1, 1, 60_000), Status::Ok);
    for id in 2..5 {
        assert_eq!(store(&mut c, &p, id, 1, 60_000), Status::Ok);
    }
    // the compressor took key 1 (the least recently used) and holds a copy of generation 1
    assert!(b.wait_log("failpoint rcompress-read: reached", 1, Duration::from_secs(5)), "compressor never ran");
    let slot1 = c.lookup(NS, &[key(1)], false).unwrap();
    assert_eq!(slot1.n_hit, 1);
    // evict key 1 (made the coldest), park its slot with a reservation, give it back, store key 1 again: same slot
    c.heat(NS, &[key(1)], false).unwrap();
    let r = c.reserve(NS, vec![item(9, 60_000)]).unwrap();
    assert_eq!(r[0].0, Status::Ok);
    c.abort(NS, &[key(9)]).unwrap();
    assert_eq!(store(&mut c, &p, 1, 2, 60_000), Status::Ok);
    let l = c.lookup(NS, &[key(1)], true).unwrap();
    let same_slot = l.entries[0].ranges == r[0].1;
    c.release(&[l.lease_id]).unwrap();
    assert!(same_slot, "test setup: key 1 was not stored again in its old slot");
    b.fp(""); // the compressor goes on with its copy
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(read(&mut c, &p, 1), Some(2), "key 1 serves the bytes of its evicted generation");
    for id in 2..5 {
        assert_eq!(read(&mut c, &p, id), Some(1));
    }
}

/// A bit flips in RAM after the broker checksummed the entry (no ECC): the SSD writer's copy does not match
/// the RAM checksum, so the entry is never written to the SSD; it is evicted (a miss, never wrong bytes) and the
/// other entries are written and read back intact. The SSD writer is held (disk-flush-hold) until the flip.
#[test]
fn flipped_bit_never_reaches_the_ssd() {
    let b = Broker::start_disk(8, &["--ram-compress-quiet-ms", "0", "--cksum-reverify-s", "3600"], Some("none"));
    let _w = Worker::start(&b);                            // the background worker takes the checksums
    b.fp("disk-flush-hold=return");
    let (mut c, p) = b.connect(1, 2000);
    for id in 0..4 {
        assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Ok);
    }
    std::thread::sleep(Duration::from_millis(1800));      // checksummed by the idle thread (no requests meanwhile)
    assert_eq!(stat(&mut c, "cksum_stored"), 4, "the background worker hashed every entry");
    // the fault: one byte of key 2 (the lease only finds its slot)
    let l = c.lookup(NS, &[key(2)], true).expect("lookup");
    let r = l.entries[0].ranges[0];
    c.release(&[l.lease_id]).expect("release");
    let s = unsafe { p.range(r) };
    let mut x = [0u8; 1];
    s.read(20_000, &mut x);
    x[0] ^= 0x04;
    s.write(20_000, &x);
    b.fp("");
    assert!(b.wait_log("failed its RAM checksum while being copied", 1, Duration::from_secs(8)), "copy not refused");
    std::thread::sleep(Duration::from_millis(2500));      // the other three written meanwhile
    assert_eq!(probe(&mut c, 2), 0, "the flipped entry: neither in RAM nor on the SSD");
    assert_eq!(stat(&mut c, "cksum_corrupt"), 1);
    evict_by_filling(&mut c, &p, 100, 8);                // the others leave RAM: on the SSD only now
    for id in [0, 1, 3] {
        assert!(wait_probe(&mut c, id, 2, Duration::from_secs(5)), "key {id} on the SSD");
    }
    assert_eq!(read(&mut c, &p, 2), None);
    // back from the SSD (XXH3 file checksums verified on the way): intact
    for id in [0, 1, 3] {
        assert!(wait_probe(&mut c, id, 1, Duration::from_secs(5)), "key {id} promoted");
        assert_eq!(read(&mut c, &p, id), Some(1));
    }
}

/// Files written by an older broker (OLD_KVBROKER=path; skipped without it) are still read after the checksum change
/// (SSD entry files carry a checksum algorithm byte, hot-set files LKVH2; old files use the old 64-bit
/// mix): SSD entries (both codecs) and hot-set files written by the old build load and read back intact.
#[test]
fn old_files_still_read() {
    let Ok(old) = std::env::var("OLD_KVBROKER") else {
        eprintln!("OLD_KVBROKER not set: skipped");
        return;
    };
    let base = PathBuf::from(format!("/tmp/claude-1000/fpcompat-{}", std::process::id()));
    for codec in ["none", "planes-zstd"] {
        let d = base.join(format!("ssd-{codec}"));
        fs::create_dir_all(&d).unwrap();
        let disk = ["--disk-dir", d.to_str().unwrap(), "--disk-min-hits", "0", "--disk-state-min-hits", "0", "--disk-flush-s",
                    "1", "--disk-gib", "1", "--disk-codec", codec];
        {
            let b = Broker::start_bin(&old, 8, &disk, None);
            let (mut c, p) = b.connect(1, 2000);
            for id in 0..4 {
                assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Ok);
            }
            std::thread::sleep(Duration::from_millis(2500));
        }
        let b = Broker::start(8, &disk);
        let (mut c, p) = b.connect(1, 2000);
        for id in 0..4 {
            let pr = probe(&mut c, id);
            assert!(pr == 1 || pr == 2, "{codec}: key {id} written by the old build not found (probe {pr})");
            assert!(wait_probe(&mut c, id, 1, Duration::from_secs(5)), "{codec}: key {id} promoted");
            assert_eq!(read(&mut c, &p, id), Some(1), "{codec}: key {id}");
        }
        assert!(!b.log_text().contains("is damaged"), "{codec}: an old file was judged damaged");
    }
    let h = base.join("hot");
    fs::create_dir_all(&h).unwrap();
    let hot = ["--hotset-dir", h.to_str().unwrap(), "--hotset-save-s", "1", "--hotset-min-hits", "0",
               "--hotset-stable-saves", "1"];
    {
        let b = Broker::start_bin(&old, 8, &hot, None);
        let (mut c, p) = b.connect(1, 2000);
        for id in 0..4 {
            assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Ok);
            assert_eq!(read(&mut c, &p, id), Some(1));
        }
        assert!(b.wait_log("hot set: 4 entries", 1, Duration::from_secs(6)), "old build saved no hot set");
        std::thread::sleep(Duration::from_millis(300));
    }
    let b = Broker::start(8, &hot);
    let (mut c, p) = b.connect(1, 2000);
    for id in 0..4 {
        assert_eq!(read(&mut c, &p, id), Some(1), "hot set: key {id} restored from the old build's file");
    }
    assert!(b.log_text().contains("4 entries restored") && !b.log_text().contains("is damaged"), "hot set: an old file was judged damaged");
    let _ = fs::remove_dir_all(&base);
}

/// A handler panics outside the core lock (mutation / failpoint review): the panicking connection is
/// closed at once (its client sees the socket close, not a timeout), its leases and reservations are released as on
/// a normal disconnect, and the broker keeps serving everyone else.
#[test]
fn handler_panic_closes_that_connection_only() {
    let b = Broker::start(16, &["--watchdog-s", "2"]);
    let (mut a, pa) = b.connect(1, 5000);
    let (mut c2, p2) = b.connect(2, 5000);
    assert_eq!(store(&mut a, &pa, 1, 1, 40_000), Status::Ok);
    let l = a.lookup(NS, &[key(1)], true).expect("lookup");
    assert_eq!(l.n_hit, 1);
    assert_eq!(stat(&mut c2, "leases_active"), 1, "A's lease");
    b.fp("handle-unlocked=1*panic");
    let t = Instant::now();
    let r = a.lookup(NS, &[key(1)], false);
    assert!(r.is_err(), "the panicking request cannot get a reply");
    assert!(t.elapsed() < Duration::from_millis(2500), "the client waited {:?}: the socket stayed open", t.elapsed());
    b.fp("");
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(stat(&mut c2, "leases_active"), 0, "the panicking connection's lease was not released");
    assert_eq!(read(&mut c2, &p2, 1), Some(1), "the broker keeps serving others");
    std::thread::sleep(Duration::from_millis(2500));
    assert_eq!(read(&mut c2, &p2, 1), Some(1), "no watchdog exit: nothing was poisoned");
}

fn kinds_request(c: &mut Client, ids: &[u32]) {
    // a hybrid request: one GDN state group (the last block) and the attention group (the chain)
    let groups = vec![(1u8, vec![key(ids[ids.len() - 1] + 1000)]), (0u8, ids.iter().map(|i| key(*i)).collect())];
    c.touch_request(NS, &groups).expect("touch_request");
}

/// FEATURES + TOUCH_REQUEST: a new client and broker negotiate the feature on connect, and tagged
/// touches link at once (no warm-up, requests back to back); against an older broker (OLD_KVBROKER) the same client
/// connects, sees no features and its touch_request goes out as plain TOUCH batches, which that broker still links
/// by inference.
#[test]
fn features_negotiated_and_touch_request_links() {
    let b = Broker::start(16, &[]);
    let (mut c, p) = b.connect(1, 2000);
    assert_ne!(c.features & kvclient::FEAT_TOUCH_REQUEST, 0, "feature negotiated");
    assert!(c.formats.iter().any(|(n, v)| n == "slot_codec" && *v == 1));
    for id in [1, 2, 3, 1003] {
        let mut it = item(id, 1000);
        it.kind = if id > 1000 { 1 } else { 0 };
        assert_eq!(c.reserve(NS, vec![it]).unwrap()[0].0, Status::Ok);
        c.commit(NS, &[key(id)]).unwrap();
    }
    for _ in 0..3 {
        kinds_request(&mut c, &[1, 2, 3]);
        kinds_request(&mut c, &[1, 2]);
    }
    assert_eq!(stat(&mut c, "seg_tagged_linked"), 6);
    assert_eq!(stat(&mut c, "seg_states_anchored"), 1);
    assert_eq!(stat(&mut c, "seg_runs_skipped"), 0);
    let _ = p;
    let Ok(old) = std::env::var("OLD_KVBROKER") else {
        eprintln!("OLD_KVBROKER not set: older-broker half skipped");
        return;
    };
    let b = Broker::start_bin(&old, 16, &[], None);
    let (mut c, _p) = b.connect(1, 2000);
    assert_eq!(c.features, 0, "an older broker: no features");
    for id in [1, 2, 3, 1003] {
        let mut it = item(id, 1000);
        it.kind = if id > 1000 { 1 } else { 0 };
        assert_eq!(c.reserve(NS, vec![it]).unwrap()[0].0, Status::Ok);
        c.commit(NS, &[key(id)]).unwrap();
    }
    for _ in 0..10 {
        kinds_request(&mut c, &[1, 2, 3]);
        c.release(&[]).unwrap();                       // another request ends the TOUCH run
    }
    assert!(stat(&mut c, "seg_runs_linked") >= 3, "the older broker links the plain TOUCH batches");
    assert!(!b.log_text().contains("protocol error"), "FEATURES to an older broker is not a protocol error");
}

/// Format gate: the broker writes compressed slot images (LKVZ1) only while every client of the pool
/// class reads them; a client that declares it does not keeps the class uncompressed until it leaves.
#[test]
fn compression_waits_for_every_reader() {
    let b = Broker::start(8, &compress_args());
    let (mut a, pa) = b.connect(1, 2000);
    let (mut l, _pl) = b.connect(2, 2000);
    l.negotiate(0, vec![]).unwrap();                      // reads no compressed slots
    assert_eq!(store(&mut a, &pa, 1, 1, 40_000), Status::Ok);
    std::thread::sleep(Duration::from_millis(2500));
    let slot_img = |c: &mut Client, p: &PoolMap| {
        let lk = c.lookup(NS, &[key(1)], true).unwrap();
        let s = unsafe { p.range(lk.entries[0].ranges[0]) };
        let img = s.to_vec(s.len());
        c.release(&[lk.lease_id]).unwrap();
        kvslotz::is_compressed(&img)
    };
    assert!(!slot_img(&mut a, &pa), "compressed while a client cannot read it");
    drop(l);
    std::thread::sleep(Duration::from_millis(2500));
    assert!(slot_img(&mut a, &pa), "compressed once every client reads LKVZ1");
    assert_eq!(read(&mut a, &pa, 1), Some(1));
}

/// A handler panic dumps the operation trace: the file holds the panicking connection's traffic up to the
/// panic, its HELLO, and the close; and the connection bookkeeping survives the panic: the broker still exits when
/// idle (the accept loop's open-connection count used to stay up after a panic).
#[test]
fn handler_panic_dumps_trace_and_idle_exit_still_comes() {
    let mut b = Broker::start(16, &["--idle-exit-s", "12", "--idle-exit-max-s", "12"]);
    let (mut a, pa) = b.connect(1, 5000);
    assert_ne!(a.features & kvproto::FEAT_TRACE_DUMP, 0, "trace dump advertised");
    assert_eq!(store(&mut a, &pa, 1, 1, 40_000), Status::Ok);
    b.fp("handle-unlocked=1*panic");
    assert!(a.lookup(NS, &[key(1)], false).is_err());
    b.fp("");
    drop(a);
    let dir = b.run_dir.join("trace");
    let t = Instant::now();
    let file = loop {
        let f = fs::read_dir(&dir).ok().and_then(|rd| rd.flatten().map(|e| e.path())
            .find(|p| p.to_string_lossy().contains("handler-panic") && p.extension().is_some_and(|x| x == "lkvt")));
        if let Some(f) = f {
            break f;
        }
        assert!(t.elapsed() < Duration::from_secs(15), "no trace dump after the panic:\n{}", b.log_text());
        std::thread::sleep(Duration::from_millis(200));
    };
    let (h, recs) = kvtrace::decode_file(&fs::read(&file).unwrap()).expect("a trace this build reads");
    assert_eq!(h.reason, "handler-panic");
    let ops: Vec<u16> = recs.iter().filter(|r| r.conn == 1).map(|r| r.op).collect();
    assert!(ops.contains(&1) && ops.contains(&3) && ops.contains(&4), "HELLO, RESERVE, COMMIT of the connection: {ops:?}");
    assert!(ops.contains(&kvtrace::OP_CLOSE), "its close");
    assert_eq!(b.wait_exit(Duration::from_secs(30)), Some(0), "no idle exit after a panicked connection");
}

/// Calls with the connector's 2 ms timeout for `dur`; the connection's liveness verdict (kvclient::Liveness, as the
/// PyO3 client applies it): (calls, timeouts, longest run of timeouts, dropped after).
fn hammer(c: &mut Client, dur: Duration) -> (u32, u32, u32, Option<Duration>) {
    c.set_timeout(Duration::from_millis(2));
    let mut l = kvclient::Liveness::default();
    let (mut n, mut to, mut run, mut longest) = (0, 0, 0, 0);
    let t = Instant::now();
    while t.elapsed() < dur {
        n += 1;
        match c.probe(NS, &[key(1)]) {
            Err(Error::Timeout) => {
                to += 1;
                run += 1;
                longest = longest.max(run);
                if l.timed_out(c.silent_for()) {
                    return (n, to, longest, Some(t.elapsed()));
                }
            }
            Err(e) => {
                eprintln!("hammer: call {n} at {:?}: {e:?}", t.elapsed());
                return (n, to, longest, Some(t.elapsed()));   // the client drops the connection (Io / protocol error)
            }
            Ok(_) => {
                run = 0;
                l.ok();
            }
        }
        std::thread::sleep(Duration::from_micros(500));
    }
    (n, to, longest, None)
}

/// A slow but alive broker is never dropped (in production, 16 consecutive 2 ms timeouts under load
/// dropped the engine's connection and the tier stayed off until the engine restarted). Every reply 5 ms late: the
/// old rule (16 timeouts in a row) fires, the new one (and nothing received for 5 s) does not; a broker that really
/// hangs (core lock held 6.5 s) is still dropped, after 5 s without a reply.
#[test]
fn slow_broker_kept_hung_broker_dropped() {
    let b = Broker::start(16, &["--watchdog-s", "30"]);
    let (mut c, _p) = b.connect(1, 2000);
    b.fp("reply-delay=sleep(5)");
    let (n, to, longest, dropped) = hammer(&mut c, Duration::from_secs(4));
    b.fp("");
    assert!(longest >= kvclient::TIMEOUTS_DROP, "the old rule would have dropped it: {longest} timeouts in a row of {n}");
    assert_eq!(dropped, None, "slow broker dropped ({to} of {n} calls timed out)");
    // and the connection is still good: a call with a normal timeout gets its own answer
    c.set_timeout(Duration::from_secs(2));
    std::thread::sleep(Duration::from_millis(300));
    assert!(c.stats().is_ok(), "the connection survived the slow phase");
    let b = Broker::start(16, &["--watchdog-s", "30"]);   // its own broker: no backlog of the slow phase's requests
    let (mut c, _p) = b.connect(2, 2000);
    b.fp("handle-locked=1*sleep(6500)");
    let (n2, to2, longest2, dropped) = hammer(&mut c, Duration::from_secs(7));
    b.fp("");
    let d = dropped.unwrap_or_else(|| panic!("a hung broker is dropped: {to2} of {n2} timed out, {longest2} in a row, \
                                               silent {:?}", c.silent_for()));
    // silence counts from the last reply (connect, ~0.2 s before the calls start): 5 s of silence, before the 6.5 s hang ends
    assert!(d >= Duration::from_millis(4500) && d < Duration::from_millis(6500), "dropped after {d:?}");
}

/// A LOOKUP that timed out still takes its lease when the broker gets to it; nobody will load it, and with 30-minute
/// leases it would pin the entry that long. The client releases such late leases by itself.
#[test]
fn late_lookup_lease_released_by_the_client() {
    let b = Broker::start(16, &[]);
    let (mut c, p) = b.connect(1, 2000);
    assert_eq!(store(&mut c, &p, 1, 1, 40_000), Status::Ok);
    b.fp("reply-delay=1*sleep(100)");
    c.set_timeout(Duration::from_millis(2));
    assert!(matches!(c.lookup(NS, &[key(1)], true), Err(Error::Timeout)));
    b.fp("");
    c.set_timeout(Duration::from_secs(2));
    std::thread::sleep(Duration::from_millis(300));
    let leased = |c: &mut Client| stat(c, "leases_active");
    let first = leased(&mut c);          // reads the late reply first: its lease is noted
    assert_eq!(first, 1, "the broker granted the late lease");
    assert_eq!(leased(&mut c), 0, "released by the client before its next request");
    assert_eq!(read(&mut c, &p, 1), Some(1));
}

/// A caller descheduled past its timeout still gets a reply that arrived meanwhile (the client declared a
/// timeout without looking at the socket, so CPU load on the engine side alone produced "broker timeouts").
#[test]
fn reply_already_arrived_is_not_a_timeout() {
    let b = Broker::start(16, &[]);
    let (mut c, _p) = b.connect(1, 2000);
    c.set_timeout(Duration::from_millis(2));
    c.test_pause = Duration::from_millis(20);
    for _ in 0..20 {
        assert!(c.probe(NS, &[key(1)]).is_ok(), "the reply was in the socket");
    }
}

/// The background worker process (same binary, `--bg-worker`), with its own failpoints file; killed when dropped.
struct Worker {
    child: Child,
    fp_file: PathBuf,
}

impl Worker {
    fn start(b: &Broker) -> Worker {
        let fp_file = b.run_dir.join("fpw");
        fs::write(&fp_file, "").unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_llmt-kvbroker"))
            .args(["--bg-worker", b.sock.to_str().unwrap()])
            .env("LLMT_KV_FAILPOINTS_FILE", &fp_file)
            .stdin(Stdio::null()).stdout(Stdio::null())
            .stderr(fs::File::create(b.run_dir.join("worker.log")).unwrap())
            .spawn().expect("start the worker");
        Worker { child, fp_file }
    }
    fn wait_log(&self, b: &Broker, pat: &str, count: usize, limit: Duration) -> bool {
        let t = Instant::now();
        while t.elapsed() < limit {
            if fs::read_to_string(b.run_dir.join("worker.log")).unwrap_or_default().matches(pat).count() >= count {
                return true;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        false
    }

    fn fp(&self, spec: &str) {
        let tmp = self.fp_file.with_extension("tmp");
        fs::write(&tmp, spec.replace(';', "\n")).unwrap();
        fs::rename(&tmp, &self.fp_file).unwrap();
        std::thread::sleep(Duration::from_millis(200));
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn worker_args() -> Vec<&'static str> {
    vec!["--class", "t=64,rcodec=1", "--ram-compress-after-s", "0", "--ram-compress-quiet-ms", "300"]
}

fn compressed(c: &mut Client, p: &PoolMap, id: u32) -> Option<bool> {
    let lk = c.lookup(NS, &[key(id)], true).unwrap();
    if lk.n_hit == 0 {
        return None;
    }
    let s = unsafe { p.range(lk.entries[0].ranges[0]) };
    let img = s.to_vec(s.len());
    c.release(&[lk.lease_id]).unwrap();
    Some(kvslotz::is_compressed(&img))
}

/// Background worker: it compresses and checksums while the broker is quiet; without a worker nothing
/// is compressed and serving is unaffected; an ordinary connection cannot use the worker's ops.
#[test]
fn worker_compresses_only_when_present_and_quiet() {
    let b = Broker::start(8, &worker_args());
    let (mut c, p) = b.connect(1, 2000);
    assert_ne!(c.features & kvclient::FEAT_BG_WORKER, 0);
    for id in 1..=3 {
        assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Ok);
    }
    assert!(matches!(c.bg_jobs(kvclient::BG_COMPRESS, 4), Err(Error::Status(Status::BadRequest))), "not a worker");
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(compressed(&mut c, &p, 1), Some(false), "no worker: nothing compressed");
    // steady traffic (a request every 100 ms, quiet needs 300 ms): the worker begins no rewrite
    let w = Worker::start(&b);
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(2) {
        let _ = probe(&mut c, 9);
        std::thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(compressed(&mut c, &p, 2), Some(false), "compressed although the broker was busy");
    // quiet: compressed and checksummed by the worker; reads intact
    std::thread::sleep(Duration::from_millis(2000));
    for id in 1..=3 {
        assert_eq!(compressed(&mut c, &p, id), Some(true), "key {id} not compressed by the worker");
        assert_eq!(read(&mut c, &p, id), Some(1));
    }
    assert!(stat(&mut c, "cksum_stored") >= 3);
    drop(w);
}

/// The worker dies in the middle of a rewrite (image written, end not sent): the broker evicts that entry (its slot
/// may hold part of an image), never serves it, and everything else stays intact.
#[test]
fn worker_killed_mid_rewrite_evicts_the_torn_entry() {
    let b = Broker::start(8, &worker_args());
    let (mut c, p) = b.connect(1, 2000);
    for id in 1..=2 {
        assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Ok);
    }
    let mut w = Worker::start(&b);
    w.fp("bg-rewrite-written=1*sleep(5000)");
    std::thread::sleep(Duration::from_millis(1500));        // quiet: the worker began a rewrite and is stuck in it
    let _ = w.child.kill();
    let _ = w.child.wait();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(stat(&mut c, "torn_rewrites"), 1, "the torn rewrite was not cleaned up");
    assert!(stat(&mut c, "slots_quarantined") >= 1, "the torn slot is quarantined (its writer may still write)");
    let left: Vec<u32> = (1..=2).filter(|id| probe(&mut c, *id) == 1).collect();
    assert_eq!(left.len(), 1, "exactly the torn entry left RAM");
    for id in left {
        assert_eq!(read(&mut c, &p, id), Some(1));
    }
}

fn tmp_files(b: &Broker) -> Vec<String> {
    fs::read_dir(b.run_dir.join("d")).map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp")).collect()).unwrap_or_default()
}

/// The worker dies in the middle of an SSD write (its temporary file open): the broker drops the job and the
/// temporary file; a new worker writes the entry later, and it comes back from the SSD intact.
#[test]
fn worker_killed_mid_ssd_write_is_redone() {
    let b = Broker::start_disk(8, &[], Some("none"));
    let mut w = Worker::start(&b);
    w.fp("disk-write-io=1*sleep(5000)");
    let (mut c, p) = b.connect(1, 2000);
    assert_eq!(store(&mut c, &p, 1, 1, 40_000), Status::Ok);
    assert!(w.wait_log(&b, "failpoint disk-write-io: reached", 1, Duration::from_secs(10)), "no SSD write started");
    assert_eq!(tmp_files(&b).len(), 1, "the write is in progress");
    let _ = w.child.kill();
    let _ = w.child.wait();
    std::thread::sleep(Duration::from_millis(300));
    assert!(tmp_files(&b).is_empty(), "temporary file left behind: {:?}", tmp_files(&b));
    let _w2 = Worker::start(&b);
    std::thread::sleep(Duration::from_millis(3000));       // the next flush plans it again; the new worker writes it
    assert!(tmp_files(&b).is_empty());
    evict_by_filling(&mut c, &p, 100, 8);
    assert!(wait_probe(&mut c, 1, 2, Duration::from_secs(10)) || wait_probe(&mut c, 1, 1, Duration::from_secs(1)),
            "key 1 was never written to the SSD");
    assert!(wait_probe(&mut c, 1, 1, Duration::from_secs(5)), "key 1 promoted");
    assert_eq!(read(&mut c, &p, 1), Some(1));
}

/// Rollback path (--bg internal): the broker writes the SSD tier itself, without a worker.
#[test]
fn ssd_writes_without_worker_in_internal_mode() {
    let b = Broker::start_disk(8, &["--bg", "internal"], Some("planes-zstd"));
    let (mut c, p) = b.connect(1, 2000);
    for id in 0..2 {
        assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Ok);
    }
    std::thread::sleep(Duration::from_millis(2500));
    evict_by_filling(&mut c, &p, 100, 8);
    for id in 0..2 {
        assert!(wait_probe(&mut c, id, 2, Duration::from_secs(5)) || probe(&mut c, id) == 1, "key {id} not on the SSD");
        assert!(wait_probe(&mut c, id, 1, Duration::from_secs(5)));
        assert_eq!(read(&mut c, &p, id), Some(1));
    }
}

/// The worker dies after copying an entry for compaction but before reporting it: the entry stays at its old slots
/// (intact), the target slots go to quarantine (the process might still be writing them), nothing is lost; a new
/// worker compacts later.
#[test]
fn worker_killed_mid_compaction_keeps_the_entry() {
    let b = Broker::start(16, &["--compact-mib", "64"]);
    let (mut c, p) = b.connect(1, 2000);
    // six short-lived entries in the low slots, two lasting ones above them: their expiry leaves holes below
    for id in 1..=6 {
        let mut it = item(id, 40_000);
        it.flags = kvproto::ENTRY_EPHEMERAL;
        it.ttl_ms = 300;
        let r = c.reserve(NS, vec![it]).unwrap();
        assert_eq!(r[0].0, Status::Ok);
        unsafe { p.range(r[0].1[0]) }.write(0, &payload(id, 1, 40_000));
        c.commit(NS, &[key(id)]).unwrap();
    }
    for id in 7..=8 {
        assert_eq!(store(&mut c, &p, id, 1, 40_000), Status::Ok);
    }
    let mut w = Worker::start(&b);
    w.fp("bg-compact-copied=1*sleep(8000)");
    // quiet: the sweep expires 1..=6 and gives the worker a budget; the worker copies one entry and stalls
    assert!(w.wait_log(&b, "failpoint bg-compact-copied: reached", 1, Duration::from_secs(12)), "no compaction move");
    let _ = w.child.kill();
    let _ = w.child.wait();
    std::thread::sleep(Duration::from_millis(300));
    assert!(stat(&mut c, "slots_quarantined") >= 1, "the abandoned target slots are quarantined");
    assert_eq!(stat(&mut c, "relocations"), 0, "nothing moved");
    for id in 7..=8 {
        assert_eq!(read(&mut c, &p, id), Some(1), "key {id} intact at its old slots");
    }
    // a new worker compacts (once quiet again)
    let _w2 = Worker::start(&b);
    std::thread::sleep(Duration::from_millis(6000));
    assert!(stat(&mut c, "relocations") >= 1, "the new worker compacted");
    for id in 7..=8 {
        assert_eq!(read(&mut c, &p, id), Some(1), "key {id} intact after the move");
    }
}

/// A connected background worker is not a client (found at deploy): the broker still exits when idle.
#[test]
fn worker_does_not_keep_the_broker_alive() {
    let mut b = Broker::start(8, &["--idle-exit-s", "2", "--idle-exit-max-s", "2"]);
    let _w = Worker::start(&b);
    assert!(b.wait_log("background worker (checksums", 1, Duration::from_secs(5)), "the worker connected");
    assert_eq!(b.wait_exit(Duration::from_secs(20)), Some(0), "the broker stayed up for its worker");
    assert!(!b.log_text().contains("warmed"), "no SSD warm-up for the worker");
}

// ------------------------------------------------------------------------------- weights entries (kind 5)

const KIND_WEIGHTS: u8 = 5;

fn weights_item(id: u32, bytes: usize) -> ReserveItem {
    ReserveItem { kind: KIND_WEIGHTS, cost_hint: 800, ..item(id, bytes) }
}

fn file_name(id: u32, ext: &str) -> String {
    let h: String = key(id).iter().map(|b| format!("{b:02x}")).collect();
    format!("{NS:08x}-{h}.{ext}")
}

fn files(dir: &std::path::Path, ext: &str) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir).map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(ext)).collect()).unwrap_or_default();
    v.sort();
    v
}

/// Set the kind byte of a persisted entry file (header offset `at`): a weights entry's file as an older build or an
/// edit could leave it.
fn set_kind(path: &std::path::Path, at: usize, kind: u8) {
    let mut b = fs::read(path).unwrap();
    b[at] = kind;
    fs::write(path, b).unwrap();
}

/// Weights entries (kind 5, a vision tower: the engine reloads it from the model file on a miss) never reach the SSD
/// tier or the hot set, however often they are hit, and a file of one is deleted and never restored; KV next to them
/// is persisted and restored as before. Shared reads by two clients (engine ids) hit the one entry.
#[test]
fn weights_never_persisted() {
    let base = PathBuf::from(format!("/tmp/claude-1000/fpweights-{}", std::process::id()));
    let wbytes = 4 * SLOT - 100;
    // SSD tier
    let d = base.join("ssd");
    fs::create_dir_all(&d).unwrap();
    let disk = ["--disk-dir", d.to_str().unwrap(), "--disk-min-hits", "0", "--disk-state-min-hits", "0", "--disk-flush-s",
                "1", "--disk-gib", "1", "--disk-codec", "none", "--bg", "internal"];
    {
        let b = Broker::start(16, &disk);
        let (mut c, p) = b.connect(1, 2000);
        let (mut c2, p2) = b.connect(2, 2000);
        assert_eq!(store(&mut c, &p, 1, 1, 40_000), Status::Ok);
        assert_eq!(store(&mut c, &p, 3, 1, 40_000), Status::Ok);
        assert_eq!(store_item(&mut c, &p, weights_item(2, wbytes), 2, 1, wbytes), Status::Ok);
        for _ in 0..3 {
            assert_eq!(read(&mut c, &p, 2), Some(1));
            assert_eq!(read(&mut c2, &p2, 2), Some(1), "the other engine hits the same weights entry");
            assert_eq!(read(&mut c, &p, 1), Some(1));
        }
        std::thread::sleep(Duration::from_millis(2500)); // two flushes
        let mut want = vec![file_name(1, "kvd"), file_name(3, "kvd")];
        want.sort();
        assert_eq!(files(&d, ".kvd"), want, "only the KV entries are on the SSD");
        // (the weights' 6 hits make them hotter than fresh fill entries, so they may still be in RAM: never SSD-only)
        evict_by_filling(&mut c, &p, 100, 16);
        assert_ne!(probe(&mut c, 2), 2, "weights are never on the SSD");
        assert!(wait_probe(&mut c, 1, 2, Duration::from_secs(5)) || probe(&mut c, 1) == 1);
    }
    // KV 3's file as a weights entry's: dropped at the scan, never restored; KV 1 still comes back
    set_kind(&d.join(file_name(3, "kvd")), 44, KIND_WEIGHTS);
    {
        let b = Broker::start(16, &disk);
        let (mut c, p) = b.connect(1, 2000);
        let t = Instant::now();
        while d.join(file_name(3, "kvd")).exists() && t.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!d.join(file_name(3, "kvd")).exists(), "the weights file is deleted at the scan");
        assert_eq!(probe(&mut c, 3), 0, "a weights file is never restored");
        assert_eq!(probe(&mut c, 2), 0, "after a restart the weights entry is gone (RAM only)");
        let pr = probe(&mut c, 1);
        assert!(pr == 1 || pr == 2, "KV 1 not found (probe {pr})");
        assert!(wait_probe(&mut c, 1, 1, Duration::from_secs(5)));
        assert_eq!(read(&mut c, &p, 1), Some(1));
    }
    // hot set (ranked by hits here: every entry qualifies; by use: hotset_by_use_survives_restart)
    let h = base.join("hot");
    fs::create_dir_all(&h).unwrap();
    let hot = ["--hotset-dir", h.to_str().unwrap(), "--hotset-save-s", "1", "--hotset-min-hits", "0",
               "--hotset-stable-saves", "1", "--hotset-rank", "hits"];
    {
        let b = Broker::start(16, &hot);
        let (mut c, p) = b.connect(1, 2000);
        assert_eq!(store(&mut c, &p, 1, 1, 40_000), Status::Ok);
        assert_eq!(store(&mut c, &p, 3, 1, 40_000), Status::Ok);
        assert_eq!(store_item(&mut c, &p, weights_item(2, wbytes), 2, 1, wbytes), Status::Ok);
        for id in 1..=3 {
            assert_eq!(read(&mut c, &p, id), Some(1));
        }
        assert!(b.wait_log("hot set: 2 entries", 2, Duration::from_secs(6)), "hot set not saved (or with the weights):\n{}", b.log_text());
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(files(&h, ".kv"), { let mut v = vec![file_name(1, "kv"), file_name(3, "kv")]; v.sort(); v });
    }
    set_kind(&h.join(file_name(3, "kv")), 6 + 4 + 32, KIND_WEIGHTS);
    {
        let b = Broker::start(16, &hot);
        let (mut c, p) = b.connect(1, 2000);
        assert_eq!(read(&mut c, &p, 1), Some(1), "KV restored from the hot set");
        assert_eq!(read(&mut c, &p, 3), None, "a weights file is never restored");
        assert!(b.log_text().contains("is a weights entry (never restored), removed"));
        assert!(!h.join(file_name(3, "kv")).exists());
    }
    let _ = fs::remove_dir_all(&base);
}

/// Persistent hot set ranked by use (--hotset-rank use, the default): a shared prompt that many requests touch but
/// that never gets a RAM-tier hit (the engines serve it from VRAM) is saved as a loadable unit (its attention chain +
/// the GDN state at its end), a weights entry is never saved however often it is used, and after a restart the
/// prefix is restored, loadable by key, intact, and keeps ranking (its use is in the file) without new touches.
#[test]
fn hotset_by_use_survives_restart() {
    let base = PathBuf::from(format!("/tmp/claude-1000/fp-hot-{}", std::process::id()));
    let h = base.join("hot");
    fs::create_dir_all(&h).unwrap();
    let hot = ["--hotset-dir", h.to_str().unwrap(), "--hotset-save-s", "1", "--hotset-min-uses", "3",
               "--hotset-stable-saves", "1"];
    {
        let b = Broker::start(32, &hot);
        let (mut c, p) = b.connect(1, 2000);
        for id in [1, 2, 3, 4, 1004] {
            let mut it = item(id, 40_000);
            it.kind = if id > 1000 { 1 } else { 0 };
            assert_eq!(store_item(&mut c, &p, it, id, 1, 40_000), Status::Ok);
        }
        let mut w = item(50, 40_000);
        w.kind = 5;
        w.cost_hint = 1000;
        assert_eq!(store_item(&mut c, &p, w, 50, 1, 40_000), Status::Ok);
        // the request that stored the prompt (anchors the state), then 6 conversations: their GDN touches start past
        // the prompt (hit point in VRAM), the attention touch from the root; each touched twice (waiting, storing)
        c.touch_request(NS, &[(0, (1..=4).map(key).collect()), (1, vec![key(1004)])]).unwrap();
        for conv in 0..6u32 {
            for _ in 0..2 {
                let mut att: Vec<Key> = (1..=4).map(key).collect();
                att.push(key(100 + conv));
                c.touch_request(NS, &[(0, att), (1, vec![key(1100 + conv)])]).unwrap();
            }
            c.touch(NS, &[key(50), key(200 + conv)]).unwrap();      // weights: used by many chains too
        }
        assert_eq!(stat(&mut c, "hit_entries"), 0, "no RAM-tier hit at all");
        assert!(stat(&mut c, "touch_missing_keys") >= 12, "the conversations' own blocks are not in RAM");
        assert!(b.wait_log("hot set: 5 entries", 1, Duration::from_secs(8)), "not saved by use:\n{}", b.log_text());
        assert!(b.log_text().contains("hot set by use: 1 units (5 entries)"), "{}", b.log_text());
        assert!(stat(&mut c, "hotset_ranked_bytes") >= 5 * 40_000);
        assert!(stat(&mut c, "hotset_touched_missing_keys") == 0, "each conversation's blocks: one chain");
        let files: Vec<String> = fs::read_dir(&h).unwrap().flatten().map(|f| f.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".kv")).collect();
        assert_eq!(files.len(), 5);
        assert!(!files.iter().any(|n| n.contains("-32000000")), "a weights entry (key 50) was saved");
    }
    let b = Broker::start(32, &hot);
    assert!(b.wait_log("5 entries restored", 1, Duration::from_secs(5)), "{}", b.log_text());
    let (mut c, p) = b.connect(2, 2000);
    let l = c.lookup(NS, &(1..=4).map(key).collect::<Vec<_>>(), false).unwrap();
    assert_eq!(l.n_hit, 4, "the attention chain is loadable by key");
    for id in [1, 2, 3, 4, 1004] {
        assert_eq!(read(&mut c, &p, id), Some(1), "key {id} restored intact");
    }
    assert_eq!(read(&mut c, &p, 50), None, "weights are never restored");
    // the next save passes rank the restored prefix from its saved use (no touch since the restart): files kept
    assert!(b.wait_log("hot set: 5 entries", 2, Duration::from_secs(8)), "{}", b.log_text());
    assert!(b.log_text().contains("0 removed"), "{}", b.log_text());
    let _ = fs::remove_dir_all(&base);
}

/// Request tags (FEAT_REQUEST_TAGS): a new client and broker negotiate it; a low-priority request's blocks
/// (RESERVE flag ENTRY_LOW) are stored like any entry and taken first under room pressure; a tagged session's first
/// request links at once; an agent's touch raises a low block it reuses. Against an older broker (OLD_KVBROKER) the
/// same client sees no FEAT_REQUEST_TAGS, touch_tagged goes out as TOUCH batches and the flag is ignored (ordinary
/// entries), without a protocol error.
#[test]
fn request_tags_low_class_end_to_end() {
    let low_item = |id: u32| {
        let mut it = item(id, 1000);
        it.flags = kvclient::ENTRY_LOW;
        it
    };
    let b = Broker::start(8, &[]);
    let (mut c, p) = b.connect(1, 2000);
    assert_ne!(c.features & kvclient::FEAT_REQUEST_TAGS, 0, "feature negotiated");
    for id in [1, 2, 3, 1003] {                       // an agent's turn: chain 1..3 + its state
        let mut it = item(id, 1000);
        it.kind = if id > 1000 { 1 } else { 0 };
        assert_eq!(store_item(&mut c, &p, it, id, 1, 1000), Status::Ok);
    }
    c.touch_tagged(NS, &[(0, vec![key(1), key(2), key(3)]), (1, vec![key(1003)])], 0x51, 0, kvclient::CLASS_AGENT, b'm', 0)
        .expect("touch_tagged");
    assert_eq!(stat(&mut c, "seg_tagged_linked"), 1, "the session's first request linked");
    assert_eq!(stat(&mut c, "tagged_sessions"), 1);
    for id in [11, 12, 13] {                          // bookkeeping one-offs
        assert_eq!(store_item(&mut c, &p, low_item(id), id, 1, 1000), Status::Ok);
        c.touch_tagged(NS, &[(3, vec![key(id)])], 0, 0, kvclient::CLASS_LOW, 0, 0).unwrap();
    }
    assert_eq!(stat(&mut c, "low_entries"), 3);
    assert_eq!(stat(&mut c, "low_committed"), 3);
    // the agent reuses bookkeeping block 13: raised
    c.touch_tagged(NS, &[(3, vec![key(13)])], 0x51, 0, kvclient::CLASS_AGENT, b'm', 0).unwrap();
    assert_eq!(stat(&mut c, "low_promoted"), 1);
    // pool of 8 holds 7: two more stores evict two entries; the low ones go, the agent's chain and 13 stay
    assert_eq!(store(&mut c, &p, 20, 1, 1000), Status::Ok);
    assert_eq!(store(&mut c, &p, 21, 1, 1000), Status::Ok);
    assert_eq!(store(&mut c, &p, 22, 1, 1000), Status::Ok);
    for id in [1, 2, 3, 1003, 13] {
        assert_eq!(read(&mut c, &p, id), Some(1), "entry {id} stayed");
    }
    assert_eq!(c.lookup(NS, &[key(11)], false).unwrap().n_hit + c.lookup(NS, &[key(12)], false).unwrap().n_hit, 0);
    assert_eq!(stat(&mut c, "low_evicted"), 2);
    assert_eq!(stat(&mut c, "normal_evicted"), 0);
    let Ok(old) = std::env::var("OLD_KVBROKER") else {
        eprintln!("OLD_KVBROKER not set: older-broker half skipped");
        return;
    };
    let b = Broker::start_bin(&old, 8, &[], None);
    let (mut c, p) = b.connect(1, 2000);
    assert_eq!(c.features & kvclient::FEAT_REQUEST_TAGS, 0, "an older broker: no request tags");
    assert_eq!(store_item(&mut c, &p, low_item(11), 11, 1, 1000), Status::Ok);
    c.touch_tagged(NS, &[(3, vec![key(11)])], 0x51, 0, kvclient::CLASS_LOW, 0, 0).expect("falls back");
    assert_eq!(read(&mut c, &p, 11), Some(1), "the flag is ignored: an ordinary entry");
    assert!(stat(&mut c, "touch_keys") >= 1 || stat(&mut c, "committed") == 1);
    assert!(!b.log_text().contains("protocol error"));
}
