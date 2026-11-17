//! Model-based property test of the core's data-ownership invariants.
//! proptest drives random sequences of RESERVE / COMMIT / ABORT / acquiring LOOKUP / RELEASE / TOUCH / HEAT, clock
//! jumps (reservation and lease timeouts, quarantine, aging, segment cleanup), disconnects, compaction moves and
//! in-place rewrites against one Core, and a model of what bytes every slot holds. After every step:
//!   1. a LOOKUP hit's slots hold exactly that key's last committed bytes;
//!   2. a RESERVE never gets a slot that a live lease covers, or that a dead writer may still be writing
//!      (reservation timed out / socket closed less than QUARANTINE_MS ago);
//!   3. a compaction move or a rewrite never touches a leased entry;
//!   4. Core::check() (slot accounting, index consistency);
//!   5. an integrity checksum mismatch never evicts a leased entry (it waits: Busy), and the evicted entry's key misses.
//! A failure shrinks to a minimal operation sequence.
use super::*;
use proptest::prelude::*;
use std::collections::HashMap;

const NSM: u32 = 3;
const SLOTS: u32 = 16;
const LEASE_MS: u64 = 60_000;
const RES_MS: u64 = 30_000;

#[derive(Clone, Debug)]
enum Op {
    Reserve { conn: u64, key: u8, slots: u8 },
    Commit { conn: u64, key: u8 },
    Abort { conn: u64, key: u8 },
    Lookup { conn: u64, key: u8 },
    Release { pick: u8 },
    Touch { conn: u64, key: u8 },
    Heat { key: u8, hot: bool },
    Tick { ms: u32 },
    Disconnect { conn: u64 },
    Compact,
    Rewrite { key: u8 },
    Checksum { key: u8, bad: bool },
}

fn op() -> impl Strategy<Value = Op> {
    let conn = 1u64..=3;
    let key = 0u8..12;
    prop_oneof![
        5 => (conn.clone(), key.clone(), 1u8..=3).prop_map(|(conn, key, slots)| Op::Reserve { conn, key, slots }),
        5 => (conn.clone(), key.clone()).prop_map(|(conn, key)| Op::Commit { conn, key }),
        1 => (conn.clone(), key.clone()).prop_map(|(conn, key)| Op::Abort { conn, key }),
        4 => (conn.clone(), key.clone()).prop_map(|(conn, key)| Op::Lookup { conn, key }),
        3 => (0u8..8).prop_map(|pick| Op::Release { pick }),
        1 => (conn.clone(), key.clone()).prop_map(|(conn, key)| Op::Touch { conn, key }),
        1 => (key.clone(), any::<bool>()).prop_map(|(key, hot)| Op::Heat { key, hot }),
        3 => prop_oneof![1u32..2_000, 25_000u32..70_000].prop_map(|ms| Op::Tick { ms }),
        1 => conn.prop_map(|conn| Op::Disconnect { conn }),
        2 => Just(Op::Compact),
        1 => key.clone().prop_map(|key| Op::Rewrite { key }),
        2 => (key, prop_oneof![9 => Just(false), 1 => Just(true)]).prop_map(|(key, bad)| Op::Checksum { key, bad }),
    ]
}

fn mk(k: u8) -> Key {
    let mut x = [0u8; 32];
    x[0] = k;
    x[1] = 0x6d;
    x
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Owner {
    Free,
    Writing(u8, u64),   // a reservation's writer: key, generation
    Data(u8, u64),      // committed bytes
}

struct Model {
    slot: Vec<Owner>,
    dirty_until: Vec<u64>,                         // a dead writer may still write this slot until then
    res: HashMap<(u64, u8), (Vec<Range>, u64, u64)>,  // (conn, key) -> ranges, gen, deadline
    gen: HashMap<u8, u64>,                         // key -> generation of its live committed entry
    leases: Vec<(u64, u64, Vec<Range>, u64)>,      // conn, lease id, ranges, deadline
    next_gen: u64,
}

impl Model {
    fn leased(&self, s: u32, now: u64) -> bool {
        self.leases.iter().any(|(_, _, rs, dl)| *dl > now && rs.iter().any(|r| s >= r.start && s < r.start + r.len))
    }
    fn writer_dies(&mut self, conn: u64, key: u8, now: u64) {
        if let Some((rs, _, _)) = self.res.remove(&(conn, key)) {
            for r in rs {
                for s in r.start..r.start + r.len {
                    self.slot[s as usize] = Owner::Free;
                    self.dirty_until[s as usize] = now + QUARANTINE_MS;
                }
            }
        }
    }
}

fn slots_of(rs: &[Range]) -> impl Iterator<Item = u32> + '_ {
    rs.iter().flat_map(|r| r.start..r.start + r.len)
}

/// `seg`: segment-wise or per-entry eviction order; `fair`: the shared-pool fairness rules on (stale never-hit
/// first, namespace cap and floor), so every victim path is checked against the ownership invariants.
fn run(ops: Vec<Op>, seg: bool, fair: bool) -> Result<(), TestCaseError> {
    let mut cf = Config {
        n_slots: SLOTS, slot_bytes: 100, reserve_timeout_ms: RES_MS, lease_timeout_ms: LEASE_MS, low_water_slots: 1,
        policy: PolicyKind::Arc, stale_unhit_ms: 0, ns_floor_slots: 0, ns_cap_slots: 0, tail_age_div: 4, heat_hit_ms: 0, shared_slots: 0,
        shared_age_mult: 24, shared_min_cont: 3, shared_max_depth: 19, seg_evict: seg, seg_grace_ms: 300_000, weights_kv_slot_ms: 0,
    };
    if fair {
        (cf.stale_unhit_ms, cf.ns_cap_slots, cf.ns_floor_slots) = (20_000, 10, 2);
    }
    cf.low_water_slots = 1;
    let mut c = Core::new(cf);
    for conn in 1..=3 {
        c.connect(conn);
        c.hello(conn, conn as u16);
    }
    let mut m = Model { slot: vec![Owner::Free; SLOTS as usize], dirty_until: vec![0; SLOTS as usize], res: HashMap::new(),
                        gen: HashMap::new(), leases: Vec::new(), next_gen: 1 };
    let mut now = 1u64;
    for o in ops {
        match o {
            Op::Reserve { conn, key, slots } => {
                let it = ReserveItem { key: mk(key), parent: None, bytes: slots as u64 * 100, kind: 0, format: 1,
                                       cost_hint: 0, flags: 0, ttl_ms: 0, meta: vec![] };
                let r = c.reserve(conn, NSM, &[it], now);
                if r[0].0 == Status::Ok {
                    for s in slots_of(&r[0].1) {
                        prop_assert!(!m.leased(s, now), "slot {s} handed out while leased");
                        prop_assert!(m.dirty_until[s as usize] <= now,
                                     "slot {s} handed out while a dead writer may still write it (until {}, now {now})",
                                     m.dirty_until[s as usize]);
                    }
                    let g = m.next_gen;
                    m.next_gen += 1;
                    for s in slots_of(&r[0].1) {
                        m.slot[s as usize] = Owner::Writing(key, g);
                    }
                    m.res.insert((conn, key), (r[0].1.clone(), g, now + RES_MS));
                }
            }
            Op::Commit { conn, key } => {
                let (st, _) = c.commit(conn, NSM, &[mk(key)], now);
                if st[0] == Status::Ok {
                    let (rs, g, _) = m.res.remove(&(conn, key)).expect("model: committed without a reservation");
                    for s in slots_of(&rs) {
                        prop_assert_eq!(m.slot[s as usize], Owner::Writing(key, g));
                        m.slot[s as usize] = Owner::Data(key, g);
                    }
                    m.gen.insert(key, g);
                }
            }
            Op::Abort { conn, key } => {
                let st = c.abort(conn, NSM, &[mk(key)]);
                if st[0] == Status::Ok {
                    if let Some((rs, _, _)) = m.res.remove(&(conn, key)) {
                        for s in slots_of(&rs) {
                            m.slot[s as usize] = Owner::Free;     // the writer is done: no late writes
                        }
                    }
                }
            }
            Op::Lookup { conn, key } => {
                let (n, lease, es) = c.lookup(conn, NSM, &[mk(key)], true, now);
                if n > 0 {
                    let g = *m.gen.get(&key).expect("model: hit on a key never committed");
                    for s in slots_of(&es[0].ranges) {
                        prop_assert_eq!(m.slot[s as usize], Owner::Data(key, g), "slot {} of key {} holds other bytes", s, key);
                    }
                    m.leases.push((conn, lease, es[0].ranges.clone(), now + LEASE_MS));
                }
            }
            Op::Release { pick } => {
                if !m.leases.is_empty() {
                    let (conn, l, _, _) = m.leases.remove(pick as usize % m.leases.len());
                    c.release(conn, &[l]);
                }
            }
            Op::Touch { conn, key } => {
                c.touch_conn(conn, NSM, &[mk(key)], now);
            }
            Op::Heat { key, hot } => {
                c.heat_keys(NSM, &[mk(key)], hot, now);
            }
            Op::Tick { ms } => {
                now += ms as u64;
                c.tick(now);
                // reservations past their deadline are gone (ResTimeout): their writers may still write
                let dead: Vec<(u64, u8)> = m.res.iter().filter(|(_, v)| v.2 <= now).map(|(k, _)| *k).collect();
                for (conn, key) in dead {
                    let dl = m.res[&(conn, key)].2;
                    m.writer_dies(conn, key, dl);
                }
                m.leases.retain(|l| l.3 > now);
            }
            Op::Disconnect { conn } => {
                c.disconnect(conn);
                let mine: Vec<u8> = m.res.keys().filter(|(cc, _)| *cc == conn).map(|(_, k)| *k).collect();
                for key in mine {
                    m.writer_dies(conn, key, now);
                }
                m.leases.retain(|l| l.0 != conn);
                c.connect(conn);
                c.hello(conn, conn as u16);
            }
            Op::Compact => {
                if let Some((ns, key, old, new)) = c.relocation_candidate() {
                    for s in slots_of(&old) {
                        prop_assert!(!m.leased(s, now), "compaction moves leased slot {s}");
                    }
                    for s in slots_of(&new) {
                        prop_assert!(m.dirty_until[s as usize] <= now, "compaction into slot {s} a dead writer may write");
                    }
                    let src: Vec<Owner> = slots_of(&old).map(|s| m.slot[s as usize]).collect();
                    for (s, o) in slots_of(&new).zip(src) {
                        m.slot[s as usize] = o;
                    }
                    c.finish_relocation(ns, &key, &old, new);
                }
            }
            Op::Checksum { key, bad } => {
                if let Some(g) = c.copy_guard(NSM, &mk(key)) {
                    let leased = slots_of(&g.1).any(|s| m.leased(s, now));
                    let stored = c.checksum_of(NSM, &mk(key));
                    let sum = if bad { stored.map(|s| s ^ 1) } else { Some(stored.unwrap_or(key as u64)) };
                    match c.checksum_result(NSM, &mk(key), &g, sum, now) {
                        ChecksumResult::Corrupt => {
                            prop_assert!(!leased, "a leased entry evicted as corrupt");
                            prop_assert!(!c.contains(NSM, &mk(key)));
                            m.gen.remove(&key);
                        }
                        ChecksumResult::Busy => prop_assert!(bad),
                        _ => {}
                    }
                }
            }
            Op::Rewrite { key } => {
                if let Some((_, rs, _, _)) = c.copy_guard(NSM, &mk(key)) {
                    if c.begin_rewrite(NSM, &mk(key), &rs) {
                        for s in slots_of(&rs) {
                            prop_assert!(!m.leased(s, now), "in-place rewrite of leased slot {s}");
                        }
                        c.end_rewrite(NSM, &mk(key), true);
                    }
                }
            }
        }
        c.check();
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2000, max_shrink_iters: 20_000, ..ProptestConfig::default() })]
    #[test]
    fn data_ownership_model(ops in proptest::collection::vec(op(), 1..200), seg in any::<bool>(), fair in any::<bool>()) {
        run(ops, seg, fair)?;
    }
}
