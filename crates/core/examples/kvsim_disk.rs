//! Disk tier under the shared RAM pool, simulated with the real `kvcore::Core`: entries hit
//! several times but not recently are candidates to move from RAM to disk entirely.
//!
//!   cargo run --release --example kvsim_disk -- STEPS.txt [--slots 437]
//!
//! RAM: one shared pool, ARC (today's broker). An entry evicted from RAM with at least `min_hits` hits is written to
//! a disk tier (LRU, fixed capacity in slots). A request's lookup walks the chain past the VRAM hit through RAM and
//! disk; the hit is the deepest block whose attention blocks and GDN state are in RAM or on disk (>= 2 blocks).
//! Blocks found only on disk are read back (and stored in RAM again). Reports extra prompt tokens served from disk,
//! disk bytes written / read, and the net time saved per engine for a USB stick (~0.2 GB/s read) and an NVMe SSD
//! (~3 GB/s), against prefill at 2,600 tok/s (Arc) and 9,000 tok/s (5090). Slot = 58.8 MB. Numbers only.
use kvcore::policy::PolicyKind;
use kvcore::{Config, Core};
use kvproto::{Key, ReserveItem};
use std::collections::{BTreeMap, HashMap};

const SLOT_GB: f64 = 0.058839;
const PREFILL: [f64; 3] = [0.0, 2600.0, 9000.0]; // tok/s: engine 1 Arc, engine 2 5090

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

/// LRU disk tier: (ns, hash, kind) -> (last use, slots).
struct Disk {
    cap: u64,
    used: u64,
    by_key: HashMap<(u32, u64, u8), (u64, u64)>,
    by_age: BTreeMap<(u64, u32, u64, u8), ()>,
    clock: u64,
    written: u64,
}

impl Disk {
    fn has(&self, ns: u32, h: u64, kind: u8) -> bool {
        self.by_key.contains_key(&(ns, h, kind))
    }
    fn touch(&mut self, ns: u32, h: u64, kind: u8) {
        if let Some((t, s)) = self.by_key.get(&(ns, h, kind)).copied() {
            self.by_age.remove(&(t, ns, h, kind));
            self.clock += 1;
            self.by_key.insert((ns, h, kind), (self.clock, s));
            self.by_age.insert((self.clock, ns, h, kind), ());
        }
    }
    fn put(&mut self, ns: u32, h: u64, kind: u8, slots: u64) {
        if self.cap == 0 || self.has(ns, h, kind) {
            return;
        }
        while self.used + slots > self.cap {
            let Some((&(t, n, hh, k), _)) = self.by_age.iter().next() else { return };
            self.by_age.remove(&(t, n, hh, k));
            self.used -= self.by_key.remove(&(n, hh, k)).unwrap().1;
        }
        self.clock += 1;
        self.by_key.insert((ns, h, kind), (self.clock, slots));
        self.by_age.insert((self.clock, ns, h, kind), ());
        self.used += slots;
        self.written += slots;
    }
}

fn item(k: Key, slots: u64, kind: u8) -> ReserveItem {
    ReserveItem { key: k, parent: None, bytes: slots * 100, kind, format: 1, cost_hint: 0, flags: 0, ttl_ms: 0, meta: vec![] }
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

#[derive(Default)]
struct Res {
    prompt: [u64; 3],
    ram_tok: [u64; 3],
    disk_tok: [u64; 3],      // tokens of hits that needed at least one disk block
    disk_read_slots: [u64; 3],
    written: u64,
}

fn run(steps: &[Step], slots: u32, disk_cap: u64, min_hits: u64) -> Res {
    let cfg = Config {
        n_slots: slots, slot_bytes: 100, reserve_timeout_ms: 300_000, lease_timeout_ms: 60_000,
        low_water_slots: (slots as u64) / 100, policy: PolicyKind::Arc, stale_unhit_ms: 0, ns_floor_slots: 0, ns_cap_slots: 0, tail_age_div: 4, heat_hit_ms: 0, shared_slots: 0, shared_age_mult: 24, shared_min_cont: 3, shared_max_depth: 19, seg_evict: false, seg_grace_ms: 300_000, weights_kv_slot_ms: 0,
    };
    let mut c = Core::new(cfg);
    for ns in 1..=2u32 {
        c.connect(ns as kvcore::ConnId);
        c.hello(ns as kvcore::ConnId, ns as u16);
    }
    let mut d = Disk { cap: disk_cap, used: 0, by_key: HashMap::new(), by_age: BTreeMap::new(), clock: 0, written: 0 };
    let mut r = Res::default();
    for s in steps {
        c.tick(s.t);
        let ns = s.ns;
        let i = ns as usize;
        if !s.fin {
            r.prompt[i] += s.np;
            let n = s.keys.len();
            let inram = |c: &Core, h: u64, k: u8| c.contains(ns, &key(h, k));
            let (mut b, mut best) = (s.b0, s.b0);
            while b < n && (inram(&c, s.keys[b], 0) || d.has(ns, s.keys[b], 0)) {
                b += 1;
                if inram(&c, s.keys[b - 1], 1) || d.has(ns, s.keys[b - 1], 1) {
                    best = b;
                }
            }
            if best - s.b0 >= 2 {
                let mut from_disk = 0u64;
                for j in s.b0..best {
                    if !inram(&c, s.keys[j], 0) {
                        from_disk += 1;
                        d.touch(ns, s.keys[j], 0);
                    }
                }
                if !inram(&c, s.keys[best - 1], 1) {
                    from_disk += 3;
                    d.touch(ns, s.keys[best - 1], 1);
                }
                let tok = ((best - s.b0) as u64 * s.bt).min(s.np.saturating_sub(s.b0 as u64 * s.bt));
                // what RAM alone held is looked up (counts hits); disk blocks are read back and stored in RAM below
                let ram_keys: Vec<Key> = (s.b0..best).map(|j| key(s.keys[j], 0)).filter(|k| c.contains(ns, k)).collect();
                if !ram_keys.is_empty() {
                    c.lookup(ns as kvcore::ConnId, ns, &ram_keys, false, s.t);
                }
                if c.contains(ns, &key(s.keys[best - 1], 1)) {
                    c.lookup(ns as kvcore::ConnId, ns, &[key(s.keys[best - 1], 1)], false, s.t);
                }
                if from_disk > 0 {
                    r.disk_tok[i] += tok;
                    r.disk_read_slots[i] += from_disk;
                } else {
                    r.ram_tok[i] += tok;
                }
            }
            for j in 0..n {
                store(&mut c, ns, key(s.keys[j], 0), 1, 0, s.t);
            }
            if s.state > 0 {
                store(&mut c, ns, key(s.keys[s.state - 1], 1), 3, 1, s.t);
            }
        } else {
            for &h in &s.keys {
                store(&mut c, ns, key(h, 0), 1, 0, s.t);
            }
        }
        for rec in c.drain_records() {
            if (rec.reason == "evicted" || rec.reason == "low_water") && rec.hits >= min_hits {
                let h = u64::from_le_bytes(rec.key_prefix);
                d.put(rec.ns, h, rec.kind, if rec.kind == 1 { 3 } else { 1 });
            }
        }
    }
    r.written = d.written;
    r
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: kvsim_disk STEPS.txt [--slots 437]");
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
    let days = steps.last().map(|s| s.t).unwrap_or(1) as f64 / 86_400_000.0;
    println!("{} steps over {:.1} days, RAM pool {} slots ({:.0} GiB)", steps.len(), days, slots, slots as f64 * SLOT_GB / 1.0737);
    let net_mode = args.iter().any(|a| a == "--net");
    if net_mode {
        // second tier over a network: GB/s per link (payload): 1, 2.5, 10, 25 GbE
        println!("{:28} {:>7} {:>8} {:>9} | net s saved per day at link: {:>8} {:>8} {:>8} {:>8}", "remote tier", "tier %",
                 "total %", "xfer GB/d", "1GbE", "2.5GbE", "10GbE", "25GbE");
    } else {
    println!("{:28} {:>7} {:>7} {:>8} {:>9} {:>9} {:>10} {:>10}", "disk tier", "RAM %", "disk %", "total %",
             "write GB/d", "read GB/d", "net s USB", "net s NVMe");
    }
    if let Some(i) = args.iter().position(|a| a == "--remote") {
        // one remote RAM tier: --remote GIB GBPS (payload GB/s), every eviction; GPU time = prefill avoided (the
        // transfer does not use GPU compute), latency = prefill avoided - transfer; decode collateral: Arc streams
        // run at ~60% during a prefill, so each prefill second avoided frees ~0.4 s per running stream there.
        let gib: f64 = args[i + 1].parse().unwrap();
        let bw: f64 = args[i + 2].parse().unwrap();
        let cap = (gib * 1.0737 / SLOT_GB) as u64;
        let base = run(&steps, slots, 0, 0);
        let r = run(&steps, slots, cap, 0);
        let p = (r.prompt[1] + r.prompt[2]) as f64;
        let tot = |x: &Res| (x.ram_tok[1] + x.ram_tok[2] + x.disk_tok[1] + x.disk_tok[2]) as f64 / p * 100.0;
        println!("remote tier {gib:.0} GiB at {bw} GB/s, every eviction: hit {:.1}% -> {:.1}%, transfer {:.0} GB/day",
                 tot(&base), tot(&r), (r.written + r.disk_read_slots[1] + r.disk_read_slots[2]) as f64 * SLOT_GB / days);
        for (e, name) in [(1usize, "arc"), (2, "rtx5090")] {
            let gpu = r.disk_tok[e] as f64 / PREFILL[e];
            let xfer = r.disk_read_slots[e] as f64 * SLOT_GB / bw;
            println!("  {name:8} remote-hit tokens/day {:>9.0}  GPU prefill s/day saved {:>6.0}  latency s/day saved {:>6.0}  \
                      transfer s/day {:>5.0}  mean transfer per hit {:.2} s",
                     r.disk_tok[e] as f64 / days, gpu / days, (gpu - xfer) / days, xfer / days,
                     xfer / (r.disk_read_slots[e] as f64 / 4.0).max(1.0));
        }
        return;
    }
    for (name, cap_gib, min_hits) in [("none", 0u64, 0u64), ("50 GiB, >=1 hit", 50, 1), ("50 GiB, >=2 hits", 50, 2),
                                      ("200 GiB, >=1 hit", 200, 1), ("200 GiB, >=2 hits", 200, 2),
                                      ("200 GiB, every eviction", 200, 0)] {
        let cap = (cap_gib as f64 * 1.0737 / SLOT_GB) as u64;
        let r = run(&steps, slots, cap, min_hits);
        let p = (r.prompt[1] + r.prompt[2]) as f64;
        let ram = (r.ram_tok[1] + r.ram_tok[2]) as f64 / p * 100.0;
        let dsk = (r.disk_tok[1] + r.disk_tok[2]) as f64 / p * 100.0;
        // net time saved by disk hits = recompute avoided - disk read time, per engine
        let net = |bw: f64| -> f64 {
            (1..=2).map(|i| r.disk_tok[i] as f64 / PREFILL[i] - r.disk_read_slots[i] as f64 * SLOT_GB / bw).sum()
        };
        let rd = (r.disk_read_slots[1] + r.disk_read_slots[2]) as f64 * SLOT_GB;
        if net_mode {
            let per_engine = |bw: f64, i: usize| (r.disk_tok[i] as f64 / PREFILL[i] - r.disk_read_slots[i] as f64 * SLOT_GB / bw) / days;
            let cols: Vec<String> = [0.117, 0.29, 1.17, 2.9].iter()
                .map(|&bw| format!("{:>4.0}/{:<4.0}", per_engine(bw, 1), per_engine(bw, 2))).collect();
            println!("{:28} {:>6.1}% {:>7.1}% {:>9.0} | (arc/5090)                {}", name, dsk, ram + dsk,
                     (r.written as f64 * SLOT_GB + rd) / days, cols.join(" "));
        } else {
        println!("{:28} {:>6.1}% {:>6.1}% {:>7.1}% {:>9.0} {:>9.1} {:>10.0} {:>10.0}", name, ram, dsk, ram + dsk,
                 r.written as f64 * SLOT_GB / days, rd / days, net(0.2), net(3.0));
        }
    }
}
