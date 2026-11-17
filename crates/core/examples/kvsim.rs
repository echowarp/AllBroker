//! Offline simulation of the shared broker pool with the real `kvcore::Core`, to try an
//! eviction algorithm offline before it serves. Input: a step file (an
//! engine cache trace split into two engines by conversation). For each eviction setting, prints the RAM-tier hit rate per
//! engine and how many evicted entries were never hit.
//!
//!   cargo run --release --example kvsim -- STEPS.txt [--slots 437]
//!
//! Model: VRAM hit as measured; RAM hit = deepest block b past the VRAM hit whose
//! attention blocks are all stored and whose GDN state at b is stored (at least 2 blocks, like production); only
//! what is loaded is looked up (counts as hits); missing attention blocks are stored (1 slot) and the prompt-end state
//! (3 slots: 3 GDN groups); output blocks at fin. Engines: 1 = arc, 2 = rtx5090. Numbers only.
use kvcore::policy::PolicyKind;
use kvcore::{Config, Core};
use kvproto::{Key, ReserveItem};
use std::collections::HashMap;

struct Step {
    fin: bool,
    t: u64,
    ns: u32,
    bt: u64,
    b0: usize,
    state: usize,
    np: u64,
    keys: Vec<u64>,
}

fn key(h: u64, kind: u8) -> Key {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&h.to_le_bytes());
    k[8] = kind;
    k
}

fn item(k: Key, slots: u64, kind: u8) -> ReserveItem {
    ReserveItem { key: k, parent: None, bytes: slots * 100, kind, format: 1, cost_hint: 0, flags: 0, ttl_ms: 0, meta: vec![] }
}

#[derive(Default, Debug)]
struct Res {
    prompt: [u64; 3],
    vram: [u64; 3],
    ram: [u64; 3],
    evicted: [u64; 3],
    never: [u64; 3],
}

fn store(c: &mut Core, ns: u32, k: Key, slots: u64, kind: u8, t: u64) {
    if c.contains(ns, &k) {
        return;
    }
    let r = c.reserve(ns as kvcore::ConnId, ns, &[item(k, slots, kind)], t);
    if r[0].0 == kvproto::Status::Ok {
        c.commit(ns as kvcore::ConnId, ns, &[k], t);
    }
}

fn run(steps: &[Step], cfg: Config, split: bool, shared: bool) -> Res {
    // split = two private pools (half each); else one shared pool
    let mut cores: Vec<Core> = if split {
        let half = Config { n_slots: cfg.n_slots / 2, low_water_slots: cfg.low_water_slots / 2, ..cfg.clone() };
        vec![Core::new(half.clone()), Core::new(half)]
    } else {
        vec![Core::new(cfg)]
    };
    for c in cores.iter_mut() {
        for ns in 1..=2u32 {
            c.connect(ns as kvcore::ConnId);
            c.hello(ns as kvcore::ConnId, ns as u16);
        }
    }
    let mut r = Res::default();
    for s in steps {
        let c = &mut cores[if split { s.ns as usize - 1 } else { 0 }];
        c.tick(s.t);
        let i = s.ns as usize;
        let ns = if shared { 1 } else { s.ns };
        if s.fin {
            for &h in &s.keys {
                store(c, ns, key(h, 0), 1, 0, s.t);
            }
            continue;
        }
        r.prompt[i] += s.np;
        r.vram[i] += (s.b0 as u64 * s.bt).min(s.np);
        let n = s.keys.len();
        // deepest RAM hit past the VRAM hit
        let (mut b, mut best) = (s.b0, s.b0);
        while b < n && c.contains(ns, &key(s.keys[b], 0)) {
            b += 1;
            if c.contains(ns, &key(s.keys[b - 1], 1)) {
                best = b;
            }
        }
        if best - s.b0 >= 2 {
            let ks: Vec<Key> = (s.b0..best).map(|j| key(s.keys[j], 0)).collect();
            c.lookup(ns as kvcore::ConnId, ns, &ks, false, s.t);
            c.lookup(ns as kvcore::ConnId, ns, &[key(s.keys[best - 1], 1)], false, s.t);
            r.ram[i] += ((best - s.b0) as u64 * s.bt).min(s.np.saturating_sub(s.b0 as u64 * s.bt));
        }
        for j in 0..n {
            store(c, ns, key(s.keys[j], 0), 1, 0, s.t);
        }
        if s.state > 0 {
            store(c, ns, key(s.keys[s.state - 1], 1), 3, 1, s.t);
        }
        for rec in c.drain_records() {
            if rec.reason == "evicted" || rec.reason == "low_water" {
                r.evicted[rec.ns as usize] += 1;
                if rec.hits == 0 {
                    r.never[rec.ns as usize] += 1;
                }
            }
        }
    }
    r
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: kvsim STEPS.txt [--slots 437]");
    let slots: u32 = args.iter().position(|a| a == "--slots").map(|i| args[i + 1].parse().unwrap()).unwrap_or(437);
    let mut steps = Vec::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let p = |i: usize| f[i].parse::<u64>().unwrap();
        if f[0] == "A" {
            steps.push(Step { fin: false, t: p(1), ns: p(2) as u32, bt: p(3), b0: p(4) as usize, state: p(5) as usize,
                              np: p(6), keys: f[7..].iter().map(|x| x.parse().unwrap()).collect() });
        } else {
            steps.push(Step { fin: true, t: p(1), ns: p(2) as u32, bt: 0, b0: 0, state: 0, np: 0,
                              keys: f[3..].iter().map(|x| x.parse().unwrap()).collect() });
        }
    }
    let base = Config {
        n_slots: slots, slot_bytes: 100, reserve_timeout_ms: 300_000, lease_timeout_ms: 60_000,
        low_water_slots: (slots as u64) / 100, policy: PolicyKind::Arc, stale_unhit_ms: 0, ns_floor_slots: 0, ns_cap_slots: 0, tail_age_div: 4, heat_hit_ms: 0, shared_slots: 0, shared_age_mult: 24, shared_min_cont: 3, shared_max_depth: 19, seg_evict: false, seg_grace_ms: 300_000, weights_kv_slot_ms: 0,
    };
    let pct = |p: f64| (slots as f64 * p / 100.0) as u64;
    let variants: Vec<(&str, Config, bool)> = vec![
        ("shared, ARC (today)", base.clone(), false),
        ("shared, stale-unhit 15 min", Config { stale_unhit_ms: 900_000, ..base.clone() }, false),
        ("shared, floor 20% + cap 80%", Config { ns_floor_slots: pct(20.0), ns_cap_slots: pct(80.0), ..base.clone() }, false),
        ("shared, floor/cap + stale 15 min", Config { ns_floor_slots: pct(20.0), ns_cap_slots: pct(80.0), stale_unhit_ms: 900_000, ..base.clone() }, false),
        ("shared, floor/cap + stale 5 min", Config { ns_floor_slots: pct(20.0), ns_cap_slots: pct(80.0), stale_unhit_ms: 300_000, ..base.clone() }, false),
        ("shared, floor/cap + stale 60 min", Config { ns_floor_slots: pct(20.0), ns_cap_slots: pct(80.0), stale_unhit_ms: 3_600_000, ..base.clone() }, false),
        ("private halves (reference)", base.clone(), true),
    ];
    println!("{} steps, pool {} slots (attention 1, state 3 per position)", steps.len(), slots);
    println!("{:34} {:>9} {:>9} {:>9}   {:>18} {:>18}", "variant", "arc RAM", "5090 RAM", "total", "arc evict/never", "5090 evict/never");
    let mut names: HashMap<&str, ()> = HashMap::new();
    for (name, cfg, split) in variants {
        names.insert(name, ());
        let r = run(&steps, cfg.clone(), split, false);
        let pc = |i: usize| r.ram[i] as f64 / r.prompt[i].max(1) as f64 * 100.0;
        let tot = (r.ram[1] + r.ram[2]) as f64 / (r.prompt[1] + r.prompt[2]).max(1) as f64 * 100.0;
        println!("{:34} {:>8.1}% {:>8.1}% {:>8.1}%   {:>10}/{:<7} {:>10}/{:<7}", name, pc(1), pc(2), tot,
                 r.evicted[1], r.never[1], r.evicted[2], r.never[2]);
        if name.starts_with("shared, ARC") {
            let r = run(&steps, cfg, false, true);
            let pc = |i: usize| r.ram[i] as f64 / r.prompt[i].max(1) as f64 * 100.0;
            let tot = (r.ram[1] + r.ram[2]) as f64 / (r.prompt[1] + r.prompt[2]).max(1) as f64 * 100.0;
            println!("{:34} {:>8.1}% {:>8.1}% {:>8.1}%", "  ... one shared namespace", pc(1), pc(2), tot);
        }
    }
}
