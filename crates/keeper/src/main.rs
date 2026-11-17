//! llmt-keeper: holds exported GPU weight memory across engine restarts (see the library docs for the protocol).
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::time::Duration;

const USAGE: &str = "llmt-keeper --socket PATH [--max-gib N (0 = no cap; default 0)] [--fdstore STATE_FILE]
  --fdstore: park every held fd in systemd's fd store (needs NOTIFY_SOCKET, FileDescriptorStoreMax=) and keep the
             metadata in STATE_FILE, so a keeper restart keeps the weights in VRAM";

fn main() {
    let mut socket = None;
    let mut max_gib = 0.0f64;
    let mut state = None;
    let mut a = std::env::args().skip(1);
    while let Some(k) = a.next() {
        match k.as_str() {
            "--socket" => socket = a.next().map(PathBuf::from),
            "--max-gib" => max_gib = a.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| die("--max-gib needs a number")),
            "--fdstore" => state = a.next().map(PathBuf::from),
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            _ => die(&format!("unknown argument {k}")),
        }
    }
    let socket = socket.unwrap_or_else(|| die("--socket is required"));
    let mut k = llmt_keeper::Keeper::new((max_gib * (1u64 << 30) as f64) as u64);
    if let Some(state) = state {
        let notify = std::env::var_os("NOTIFY_SOCKET").map(PathBuf::from).unwrap_or_else(|| die("--fdstore needs NOTIFY_SOCKET (run under systemd)"));
        let n = k.with_fdstore(notify, state, llmt_keeper::listen_fds());
        eprintln!("keeper: fd store on, {n} entries restored ({:.2} GiB)", k.held_bytes() as f64 / (1u64 << 30) as f64);
    }
    let _ = std::fs::remove_file(&socket);
    let l = UnixListener::bind(&socket).unwrap_or_else(|e| die(&format!("bind {}: {e}", socket.display())));
    let _ = std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600));
    eprintln!("keeper on {}", socket.display());
    if let Some(n) = std::env::var_os("NOTIFY_SOCKET") {
        let _ = llmt_keeper::sd_notify(std::path::Path::new(&n), "READY=1", &[]);
    }
    for c in l.incoming() {
        let Ok(mut c) = c else { continue };
        // one client at a time, as before; a stuck client must not block the next engine start for long
        let _ = c.set_read_timeout(Some(Duration::from_secs(30)));
        let _ = c.set_write_timeout(Some(Duration::from_secs(30)));
        if let Err(e) = k.serve(&mut c) {
            eprintln!("keeper: request failed: {e}");
        }
    }
}

fn die(m: &str) -> ! {
    eprintln!("llmt-keeper: {m}\n{USAGE}");
    std::process::exit(2)
}
