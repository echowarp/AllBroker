//! Segment eviction A/B with the real `kvcore::Core`: synthetic agent sessions on a hybrid model (3 GDN
//! groups + attention, one slot per entry), connector-shaped traffic (lookup, TOUCH run per request: the GDN groups'
//! suffix batches then the attention batch, RESERVE / COMMIT of new attention blocks and the 3 prompt-end states,
//! output blocks stored at the end of the turn: attention only). GPU: the last `--gpu` served sessions hit there.
//! A RAM hit = deepest block b whose attention 0..b is in RAM and whose 3 states at b are; "state_limited" = blocks of
//! the attention match past b (attention in RAM, no state to use it). Numbers only.
//!
//!   cargo run --release --example kvsim_seg -- [--turns 6000] [--sessions 16] [--seeds 3]
use kvcore::policy::PolicyKind;
use kvcore::{Config, ConnId, Core};
use kvproto::{Key, ReserveItem};

const NS: u32 = 1;
const C: ConnId = 1;
const G: u32 = 3;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// kind 0 = attention, 1..=3 = GDN group; block id = the block's content incl. its prefix (vLLM chained hash)
fn key(kind: u32, id: u32) -> Key {
    let mut k = [0u8; 32];
    k[..4].copy_from_slice(&kind.to_le_bytes());
    k[4..8].copy_from_slice(&id.to_le_bytes());
    k[16] = 0xa5;
    k
}

fn item(k: Key, kind: u8) -> ReserveItem {
    ReserveItem { key: k, parent: None, bytes: 100, kind, format: 1, cost_hint: 0, flags: 0, ttl_ms: 0, meta: vec![] }
}

#[derive(Default)]
struct Res {
    prompt: u64,
    gpu: u64,
    ram: u64,
    limited: u64,
    turns_limited: u64,
    orphan_attn: u64,
    orphan_state: u64,
    attn_slots: u64,
    state_slots: u64,
    samples: u64,
}

struct Sess {
    chain: Vec<u32>,
    ends: Vec<u32>,
}

fn touch(c: &mut Core, ch: &[u32], from: u32, t: u64) {
    for g in 1..=G {
        let v: Vec<Key> = ch[from as usize..].iter().map(|b| key(g, *b)).collect();
        c.touch_conn(C, NS, &v, t);
    }
    let v: Vec<Key> = ch.iter().map(|b| key(0, *b)).collect();
    c.touch_conn(C, NS, &v, t);
}

fn store(c: &mut Core, ks: &[(Key, u8)], t: u64) {
    let items: Vec<ReserveItem> = ks.iter().map(|(k, kd)| item(*k, *kd)).collect();
    let r = c.reserve(C, NS, &items, t);
    let ok: Vec<Key> = ks.iter().zip(r).filter(|(_, (st, _))| *st == kvproto::Status::Ok).map(|((k, _), _)| *k).collect();
    if !ok.is_empty() {
        c.commit(C, NS, &ok, t);
    }
}

fn run(seg: bool, lazy: bool, slots: u32, turns: u32, nsess: u32, gpu: usize, retry_pct: u64, seed: u64) -> Res {
    let cfg = Config {
        n_slots: slots, slot_bytes: 100, reserve_timeout_ms: 300_000, lease_timeout_ms: 60_000,
        low_water_slots: (slots as u64) / 100, policy: PolicyKind::Arc, stale_unhit_ms: 0, ns_floor_slots: 0,
        ns_cap_slots: 0, tail_age_div: 4, heat_hit_ms: 0, shared_slots: 0, shared_age_mult: 24, shared_min_cont: 3, shared_max_depth: 19,
        seg_evict: seg, seg_grace_ms: 300_000, weights_kv_slot_ms: 0,
    };
    let mut c = Core::new(cfg);
    c.connect(C);
    c.hello(C, 1);
    let mut rng = Rng(seed * 0x9e3779b97f4a7c15 + 1);
    let mut ss: Vec<Sess> = (0..nsess).map(|_| Sess { chain: Vec::new(), ends: Vec::new() }).collect();
    let mut next_id = 1u32;
    let mut recent: Vec<u32> = Vec::new();
    let mut r = Res::default();
    let mut t = 0u64;
    for _ in 0..turns {
        t += 2_000 + rng.below(20_000);
        // skewed activity: session i with weight ~ 1 / (i + 1)
        let s = loop {
            let x = rng.below(nsess as u64) as u32;
            if rng.below(x as u64 + 1) == 0 {
                break x;
            }
        };
        let on_gpu = recent.contains(&s);
        let st = &mut ss[s as usize];
        if st.chain.is_empty() || st.chain.len() > 60 {
            st.chain.clear();
            st.ends.clear();
        }
        // a retry / edit / branch: back to one of the last 3 prompt ends, then different blocks
        let mut base = st.chain.len() as u32;
        if st.ends.len() > 1 && rng.below(100) < retry_pct {
            let k = st.ends.len() - 1 - rng.below(st.ends.len().min(3) as u64) as usize;
            base = st.ends[k];
            st.chain.truncate(base as usize);
            st.ends.truncate(k + 1);
        }
        for _ in 0..1 + rng.below(3) {
            st.chain.push(next_id);
            next_id += 1;
        }
        let ch = st.chain.clone();
        let n = ch.len() as u32;
        st.ends.push(n);
        r.prompt += n as u64;
        // GPU: the last `gpu` sessions hold their chain (resume at the branch point / previous prompt end)
        let gpu_hit = if on_gpu { base } else { 0 };
        r.gpu += gpu_hit as u64;
        let (mut b, mut best) = (gpu_hit, gpu_hit);
        while b < n && c.contains(NS, &key(0, ch[b as usize])) {
            b += 1;
            if (1..=G).all(|g| c.contains(NS, &key(g, ch[b as usize - 1]))) {
                best = b;
            }
        }
        if best > gpu_hit {
            let ks: Vec<Key> = (gpu_hit..best).map(|j| key(0, ch[j as usize])).collect();
            c.lookup(C, NS, &ks, false, t);
            let sk: Vec<Key> = (1..=G).map(|g| key(g, ch[best as usize - 1])).collect();
            c.lookup(C, NS, &sk, false, t);
            r.ram += (best - gpu_hit) as u64;
        }
        if b > best {
            r.limited += (b - best) as u64;
            r.turns_limited += 1;
        }
        touch(&mut c, &ch, best.saturating_sub(1), t);
        // inclusive stores: new attention blocks + the prompt-end states (3 groups); lazy: the states only, attention
        // when the session leaves the GPU
        let na = if lazy { 0 } else { n };
        let mut ks: Vec<(Key, u8)> = (0..na).map(|j| (key(0, ch[j as usize]), 0)).filter(|(k, _)| !c.contains(NS, k)).collect();
        ks.extend((1..=G).map(|g| (key(g, ch[n as usize - 1]), 1)).filter(|(k, _)| !c.contains(NS, k)));
        store(&mut c, &ks, t);
        touch(&mut c, &ch, best.saturating_sub(1), t);
        // output: 0-1 blocks, attention only (no state past the prompt end)
        if rng.below(2) == 1 {
            let id = next_id;
            next_id += 1;
            ss[s as usize].chain.push(id);
            if !lazy {
                store(&mut c, &[(key(0, id), 0)], t + 1000);
                let ch = ss[s as usize].chain.clone();
                touch(&mut c, &ch, n - 1, t + 1000);
            }
        }
        c.release(C, &[]);
        c.tick(t + 1500);
        recent.retain(|x| *x != s);
        recent.push(s);
        if recent.len() > gpu {
            let x = recent.remove(0);
            if lazy {
                // the GPU evicts the session: its blocks are copied to RAM (tail end first)
                let ks: Vec<(Key, u8)> = ss[x as usize].chain.iter().rev().map(|id| (key(0, *id), 0))
                    .filter(|(k, _)| !c.contains(NS, k)).collect();
                store(&mut c, &ks, t + 1600);
            }
        }
        for (k, n) in c.ns_stats().into_iter().map(|(k, g, _)| (k, g)) {
            if k.0 == NS {
                let sl = n.iter().find(|x| x.0 == "slots").unwrap().1;
                if k.1 == 0 {
                    r.attn_slots += sl;
                } else if k.1 == 1 {
                    r.state_slots += sl;
                }
            }
        }
        r.samples += 1;
    }
    for (n, v) in c.counters() {
        match n.as_str() {
            "evict_orphan_attn" => r.orphan_attn = v,
            "evict_orphan_state" => r.orphan_state = v,
            _ => {}
        }
    }
    r
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, d: u64| args.iter().position(|a| a == name).map(|i| args[i + 1].parse().unwrap()).unwrap_or(d);
    let (turns, nsess, seeds, gpu) = (arg("--turns", 6000) as u32, arg("--sessions", 16) as u32, arg("--seeds", 3), arg("--gpu", 2));
    let retry = arg("--retry-pct", 10);
    println!("{turns} turns, {nsess} sessions, GPU = last {gpu} sessions, {retry}% retries / edits, {seeds} seeds; blocks summed over seeds");
    println!("{:>4} {:>6} {:>4} {:>10} {:>9} {:>9} {:>7} {:>11} {:>8} {:>13} {:>12}", "mode", "slots", "seg", "prompt", "RAM hit",
             "state_lim", "lim_turn", "RAM % prompt", "attn %", "orphan a / s", "time");
    for (lazy, slots) in [(false, 150u32), (false, 300), (false, 600), (true, 150), (true, 300), (true, 600)] {
        for seg in [false, true] {
            let t0 = std::time::Instant::now();
            let mut a = Res::default();
            for sd in 0..seeds {
                let r = run(seg, lazy, slots, turns, nsess, gpu as usize, retry, sd + 1);
                a.prompt += r.prompt;
                a.gpu += r.gpu;
                a.ram += r.ram;
                a.limited += r.limited;
                a.turns_limited += r.turns_limited;
                a.orphan_attn += r.orphan_attn;
                a.orphan_state += r.orphan_state;
                a.attn_slots += r.attn_slots;
                a.state_slots += r.state_slots;
                a.samples += r.samples;
            }
            println!("{:>4} {:>6} {:>4} {:>10} {:>9} {:>9} {:>7} {:>10.1}% {:>7.0}% {:>6}/{:<6} {:>10.1}s", if lazy { "lazy" } else { "incl" }, slots,
                     if seg { "on" } else { "off" }, a.prompt, a.ram, a.limited, a.turns_limited,
                     a.ram as f64 / a.prompt as f64 * 100.0,
                     a.attn_slots as f64 / (a.attn_slots + a.state_slots).max(1) as f64 * 100.0,
                     a.orphan_attn, a.orphan_state, t0.elapsed().as_secs_f64());
        }
    }
}
