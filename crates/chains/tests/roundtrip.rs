use kvchains::{client, proto, server};
use std::io::{BufReader, BufWriter, Write};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

fn sock(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("kvchains-test-{}-{name}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d.join("chains.sock")
}

fn wait_hint(c: &client::Client, req: u64, known: u32) -> Option<client::Hint> {
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(2) {
        if let Some(h) = c.hint(req).filter(|h| h.known_len == known) {
            return Some(h);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    None
}

fn prompt_with_span(span: &[u32]) -> Vec<u32> {
    let mut p: Vec<u32> = (0..500u32).map(|i| 100_000 + i).collect();
    p.extend_from_slice(span);
    p.extend((0..200u32).map(|i| 200_000 + i));
    p
}

#[test]
fn client_gets_aligned_hints_and_none_in_think() {
    let path = sock("rt");
    server::serve(server::Config::new(&path)).unwrap();
    let c = client::Client::new(client::Config::new(&path, 2));
    let span: Vec<u32> = (1..=60).collect();
    let p = prompt_with_span(&span);
    c.open(7, 0, &p);
    let mut n = p.len() as u32;
    // the output repeats the span: thinking phase first (no hints), then answer phase
    for &t in &span[..10] {
        n += 1;
        c.append(7, n, proto::PHASE_THINK, &[t]);
    }
    assert!(c.flush(Duration::from_secs(2)));
    std::thread::sleep(Duration::from_millis(50));
    assert!(c.hint(7).is_none(), "no hints in the thinking phase");
    for &t in &span[10..20] {
        n += 1;
        c.append(7, n, proto::PHASE_ANSWER, &[t]);
    }
    let h = wait_hint(&c, 7, n).expect("hint after 20 repeated tokens");
    assert_eq!(h.match_len, 20);
    assert_eq!(h.source, proto::SRC_OWN);
    assert_eq!(h.tokens[..40], span[20..60]);
    // the repeat breaks: the hint is cleared
    n += 1;
    c.append(7, n, proto::PHASE_ANSWER, &[999_999]);
    let t = Instant::now();
    while c.hint(7).is_some() && t.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(c.hint(7).is_none(), "NoHint clears the hint");
    c.close(7, false);
    let st = c.stats(Duration::from_secs(2)).expect("stats");
    assert!(st.contains("\"hints\":"), "{st}");
    eprintln!("{st}");
}

#[test]
fn client_started_before_broker_reopens_and_hints() {
    let path = sock("late");
    let mut cfg = client::Config::new(&path, 3);
    cfg.reconnect_ms = 50;
    let c = client::Client::new(cfg);
    let span: Vec<u32> = (500..540).collect();
    let p = prompt_with_span(&span);
    c.open(1, 0, &p);
    let mut n = p.len() as u32;
    for &t in &span[..18] {
        n += 1;
        c.append(1, n, proto::PHASE_TOOL, &[t]);
    }
    std::thread::sleep(Duration::from_millis(200)); // broker not up: frames dropped, request marked for re-open
    server::serve(server::Config::new(&path)).unwrap();
    let t = Instant::now();
    let mut got = None;
    while got.is_none() && t.elapsed() < Duration::from_secs(10) {
        n += 1;
        let next = span[(n as usize - p.len() - 1) % span.len()]; // continue the repeated span
        c.append(1, n, proto::PHASE_TOOL, &[next]);
        std::thread::sleep(Duration::from_millis(30));
        got = c.hint(1).filter(|h| h.known_len == n);
    }
    let h = got.expect("hint after the broker came up");
    assert!(h.match_len >= 16);
    let cn: std::collections::HashMap<_, _> = c.counters().into_iter().collect();
    // frames queued while the broker was down are delivered on the first connect (complete history, no re-open)
    assert_eq!((cn["connected"], cn["frames_dropped"]), (1, 0), "{cn:?}");
}

#[test]
fn append_to_hint_latency_16_requests() {
    // raw protocol (no client threads in the measurement): 16 open requests, each appends 1 token per step at 60 steps/s
    let path = sock("lat");
    server::serve(server::Config::new(&path)).unwrap();
    let s = UnixStream::connect(&path).unwrap();
    let mut w = BufWriter::new(s.try_clone().unwrap());
    let mut r = BufReader::new(s);
    let mut buf = Vec::new();
    proto::send(&mut w, &proto::Msg::Hello { proto: 1, engine_id: 1, model_tag: String::new(), gate: 16, max_chain: 64 }).unwrap();
    w.flush().unwrap();
    proto::read_frame(&mut r, &mut buf).unwrap();
    let span: Vec<u32> = (1..=2000).collect(); // long repeat: every append from token 16 on yields a hint
    let mut lens = vec![0u32; 16];
    for req in 0..16u64 {
        let mut p: Vec<u32> = (0..40_000u32).map(|i| 1_000_000 + i + req as u32 * 50_000).collect();
        p.extend_from_slice(&span);
        lens[req as usize] = p.len() as u32;
        proto::send(&mut w, &proto::Msg::Open { req, flags: 0, tokens: p }).unwrap();
    }
    w.flush().unwrap();
    let mut lat = Vec::new();
    for step in 0..120 {
        let t0 = Instant::now();
        for req in 0..16u64 {
            let i = req as usize;
            lens[i] += 1;
            proto::send(&mut w, &proto::Msg::Append { req, known_len: lens[i], phase: 1, tokens: vec![span[step % 2000]] }).unwrap();
        }
        w.flush().unwrap();
        let mut got = 0;
        while got < 16 && step >= 16 {
            let ty = proto::read_frame(&mut r, &mut buf).unwrap().unwrap();
            if ty == proto::T_HINT {
                got += 1;
            }
        }
        if step >= 16 {
            lat.push(t0.elapsed());
        }
        std::thread::sleep(Duration::from_micros(16_600));
    }
    lat.sort();
    let p50 = lat[lat.len() / 2];
    let p99 = lat[lat.len() * 99 / 100];
    eprintln!("16 requests, Append burst -> 16 Hints: p50 {:?}, p99 {:?}", p50, p99);
    assert!(cfg!(debug_assertions) || p99 < Duration::from_micros(1000), "p99 {p99:?}");
}

#[test]
fn global_store_hints_across_requests_and_snapshot_restores() {
    let path = sock("glob");
    let snap = path.with_file_name("chains.bin");
    let mut cfg = server::Config::new(&path);
    cfg.snapshot = Some(snap.clone());
    cfg.snapshot_s = 0;
    cfg.idle_ms = 50;
    server::serve(cfg).unwrap();
    let c = client::Client::new(client::Config::new(&path, 2));
    // request A writes a 120-token answer, then closes (finished): its output goes to the global store
    let ans: Vec<u32> = (0..120u32).map(|i| 300_000 + i * 7).collect();
    c.open(1, 0, &(0..100u32).collect::<Vec<_>>());
    let mut n = 100u32;
    for &t in &ans {
        n += 1;
        c.append(1, n, proto::PHASE_ANSWER, &[t]);
    }
    c.close(1, false);
    assert!(c.flush(Duration::from_secs(2)));
    std::thread::sleep(Duration::from_millis(600)); // idle GC inserts the document
    // request B (a different prompt) starts writing the same answer: the hint comes from the global store
    let pb: Vec<u32> = (0..80u32).map(|i| 400_000 + i).collect();
    c.open(2, 0, &pb);
    let mut m = pb.len() as u32;
    let mut got = None;
    for &t in &ans[..40] {
        m += 1;
        c.append(2, m, proto::PHASE_ANSWER, &[t]);
        std::thread::sleep(Duration::from_millis(3));
        if let Some(h) = c.hint(2).filter(|h| h.known_len == m && h.source == proto::SRC_GLOBAL) {
            got = Some((m, h));
            break;
        }
    }
    let (at, h) = got.expect("global hint");
    let k = (at - pb.len() as u32) as usize;
    assert!(h.match_len >= 16);
    assert_eq!(h.tokens[..], ans[k..(k + 64).min(ans.len())], "continuation from request A's answer");
    c.feedback(vec![(2, h.tokens.len() as u16, 10, h.chain_ref)]);
    c.close(2, true); // aborted: not stored
    let st = c.stats(Duration::from_secs(2)).unwrap();
    assert!(st.contains("\"global_hints\":") && st.contains("\"store\":{"), "{st}");
    eprintln!("{st}");
    // snapshot written while idle; a new server restores it
    let t = Instant::now();
    while !snap.exists() && t.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(snap.exists(), "snapshot written");
    let path2 = sock("glob2");
    let mut cfg2 = server::Config::new(&path2);
    cfg2.snapshot = Some(snap.clone());
    server::serve(cfg2).unwrap();
    let c2 = client::Client::new(client::Config::new(&path2, 1));
    c2.open(9, 0, &(500_000..500_050).collect::<Vec<u32>>());
    let mut z = 50u32;
    let mut ok = false;
    for &t in &ans[..40] {
        z += 1;
        c2.append(9, z, proto::PHASE_TOOL, &[t]);
        std::thread::sleep(Duration::from_millis(3));
        if c2.hint(9).is_some_and(|h| h.known_len == z && h.source == proto::SRC_GLOBAL) {
            ok = true;
            break;
        }
    }
    assert!(ok, "restored store gives global hints");
}

#[test]
fn own_context_beats_global_and_seed_is_used() {
    let path = sock("seed");
    let mut cfg = server::Config::new(&path);
    cfg.idle_ms = 20;
    server::serve(cfg).unwrap();
    let c = client::Client::new(client::Config::new(&path, 2));
    let corpus: Vec<u32> = (0..200u32).map(|i| 700_000 + i * 3).collect();
    c.seed(&corpus, true);
    assert!(c.flush(Duration::from_secs(2)));
    std::thread::sleep(Duration::from_millis(400));
    c.open(5, 0, &(0..64u32).collect::<Vec<_>>());
    let mut n = 64u32;
    let mut src = None;
    for &t in &corpus[..30] {
        n += 1;
        c.append(5, n, proto::PHASE_ANSWER, &[t]);
        std::thread::sleep(Duration::from_millis(3));
        if let Some(h) = c.hint(5).filter(|h| h.known_len == n) {
            src = Some(h.source);
            break;
        }
    }
    assert_eq!(src, Some(proto::SRC_GLOBAL), "seeded corpus hint");
    // a request whose own prompt contains the span gets the own-context hint, not the global one
    let mut p: Vec<u32> = (0..10u32).collect();
    p.extend_from_slice(&corpus[..100]);
    p.extend(900_000..900_050);
    c.open(6, 0, &p);
    let mut m = p.len() as u32;
    let mut s6 = None;
    for &t in &corpus[..30] {
        m += 1;
        c.append(6, m, proto::PHASE_ANSWER, &[t]);
        std::thread::sleep(Duration::from_millis(3));
        if let Some(h) = c.hint(6).filter(|h| h.known_len == m && h.match_len >= 16) {
            s6 = Some(h.source);
        }
    }
    assert_eq!(s6, Some(proto::SRC_OWN), "own context first");
}

fn global_hint_for(c: &client::Client, req: u64, span: &[u32]) -> bool {
    c.open(req, 0, &(600_000..600_050).collect::<Vec<u32>>());
    let mut n = 50u32;
    for &t in span {
        n += 1;
        c.append(req, n, proto::PHASE_ANSWER, &[t]);
        std::thread::sleep(Duration::from_millis(3));
        if c.hint(req).is_some_and(|h| h.known_len == n && h.source == proto::SRC_GLOBAL) {
            return true;
        }
    }
    false
}

#[test]
fn shutdown_snapshot_and_model_tag_on_restore() {
    let path = sock("tag");
    let snap = path.with_file_name("chains.bin");
    let _ = std::fs::remove_file(&snap);
    let mut cfg = server::Config::new(&path);
    cfg.snapshot = Some(snap.clone());
    cfg.snapshot_s = 1_000_000; // no periodic snapshot: only the shutdown one
    cfg.idle_ms = 20;
    let h = server::serve(cfg.clone()).unwrap();
    let mut cc = client::Config::new(&path, 2);
    cc.model_tag = "model-a".into();
    let c = client::Client::new(cc);
    let corpus: Vec<u32> = (0..200u32).map(|i| 800_000 + i * 5).collect();
    c.seed(&corpus, false); // hot tier
    assert!(c.flush(Duration::from_secs(2)));
    std::thread::sleep(Duration::from_millis(500));
    assert!(!snap.exists());
    let d = h.save_now().expect("snapshot configured").expect("saved");
    assert!(snap.exists());
    eprintln!("shutdown snapshot {:?}; {}", d, h.summary());
    assert!(h.metrics().contains("llmt_kv_chains_occurrences 192"), "{}", h.metrics());
    // restored with the same tag: hints at once
    let p2 = sock("tag2");
    let mut c2 = cfg.clone();
    c2.socket = p2.clone();
    let h2 = server::serve(c2).unwrap();
    assert_eq!(h2.view().occ, 192);
    let mut k = client::Config::new(&p2, 1);
    k.model_tag = "model-a".into();
    assert!(global_hint_for(&client::Client::new(k), 1, &corpus[..40]), "restored chains used");
    // restored, but the first engine runs another model: the restored store is discarded, not the engine
    let p3 = sock("tag3");
    let mut c3 = cfg.clone();
    c3.socket = p3.clone();
    let h3 = server::serve(c3).unwrap();
    let mut k = client::Config::new(&p3, 1);
    k.model_tag = "model-b".into();
    assert!(!global_hint_for(&client::Client::new(k), 1, &corpus[..40]), "another model's chains are not used");
    assert_eq!(h3.stats.restore_discarded.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(h3.global.store.lock().unwrap().model_tag, "model-b");
    assert_eq!(h3.global.store.lock().unwrap().index.is_empty(), true);
}
