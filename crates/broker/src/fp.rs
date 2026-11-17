//! Fault injection: named failpoints (the `fail` crate) at recovery-relevant spots. Built only with
//! `--features failpoints`; otherwise `hit` is a constant false and `setup` does nothing, so production builds carry
//! no trace of them.
//!
//! Configuration: the FAILPOINTS environment variable at start (`name=action;name=action`, fail crate syntax), and
//! at runtime the file named by LLMT_KV_FAILPOINTS_FILE, polled every 50 ms: one `name=action` per line; a failpoint
//! that was in the file and no longer is gets removed. Integration tests (tests/failpoints.rs) drive a broker
//! process this way.
//!
//! Failpoints (action `return` unless noted; `sleep(ms)` / `panic` / `N*`, `P%` prefixes work everywhere):
//!   lazy-room-for       on-demand pool: the pool's file system has no room (RESERVE answers FULL)
//!   lazy-ensure         on-demand pool: allocating slots fails with ENOSPC (SSD promotion / hot-set restore)
//!   warm-fallocate      warm margin: allocation ahead fails
//!   disk-write-io       SSD tier: a write fails with EIO after half the file was written
//!   disk-read-corrupt   SSD tier: the data read back has one flipped byte (checksum failure)
//!   handle-locked       every request, inside the core lock (sleep(ms): a lock held long; panic: a poisoned lock)
//!   reply-delay         every reply, before it is written (outside the core lock; sleep(ms): a slow broker)
//!   rcompress-fail      in-place RAM compression: the codec fails (entry stays raw)
//!   rcompress-read      in-place RAM compression: after the slot was copied out, before the rewrite starts
//!                       (sleep(ms): the entry may change meanwhile)
//!   handle-unlocked     every request, before the core lock (panic: a handler panic without a poisoned lock)
//!   disk-flush-hold     SSD tier: the writer skips its pass (test ordering)
//!   background worker process (--bg-worker; its own LLMT_KV_FAILPOINTS_FILE):
//!   bg-rewrite-written  the image is in the slot, the rewrite not ended yet (sleep + kill: a torn rewrite)
//!   bg-checksum-read    after an entry's bytes were read for its checksum
//!   bg-checksum-flip    the computed checksum is wrong (a corrupt entry)
//!   bg-compact-copied   a compaction copy is done, not yet reported (sleep + kill: a move left open)

/// Failpoint `name`: runs its action (sleep / panic / ...) and says whether it fired with `return`.
#[cfg(feature = "failpoints")]
#[inline]
pub fn hit(name: &str) -> bool {
    // a configured failpoint logs each time it is reached (tests wait for "failpoint NAME: reached")
    if fail::list().iter().any(|(n, a)| n == name && !a.is_empty()) {
        eprintln!("failpoint {name}: reached");
    }
    fail::eval(name, |_| ()).is_some()
}

#[cfg(not(feature = "failpoints"))]
#[inline(always)]
pub fn hit(_name: &str) -> bool {
    false
}

/// Read FAILPOINTS and start the runtime control thread (LLMT_KV_FAILPOINTS_FILE).
#[cfg(feature = "failpoints")]
pub fn setup() {
    let sc = fail::FailScenario::setup();
    std::mem::forget(sc); // for the life of the process
    eprintln!("FAILPOINTS build: fault injection enabled");
    let Ok(path) = std::env::var("LLMT_KV_FAILPOINTS_FILE") else { return };
    std::thread::Builder::new().name("failpoints".into()).spawn(move || {
        let mut last = String::new();
        let mut on: Vec<String> = Vec::new();
        loop {
            let s = std::fs::read_to_string(&path).unwrap_or_default();
            if s != last {
                let mut now = Vec::new();
                for l in s.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
                    let Some((name, act)) = l.split_once('=') else { continue };
                    match fail::cfg(name.trim(), act.trim()) {
                        Ok(()) => eprintln!("failpoint {}: {}", name.trim(), act.trim()),
                        Err(e) => eprintln!("failpoint {}: bad action {}: {e}", name.trim(), act.trim()),
                    }
                    now.push(name.trim().to_string());
                }
                for n in on.iter().filter(|n| !now.contains(n)) {
                    fail::remove(n);
                    eprintln!("failpoint {n}: off");
                }
                on = now;
                last = s;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }).unwrap();
}

#[cfg(not(feature = "failpoints"))]
#[inline(always)]
pub fn setup() {}
