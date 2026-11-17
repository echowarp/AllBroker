use super::*;
use kvproto::{CLASS_AGENT, CLASS_LOWEST, TAG_TERMINAL};

const NS: u32 = 1;

fn cfg(n_slots: u32, policy: PolicyKind) -> Config {
    Config { n_slots, slot_bytes: 100, reserve_timeout_ms: 30_000, lease_timeout_ms: 60_000, low_water_slots: 0, policy,
             stale_unhit_ms: 0, ns_floor_slots: 0, ns_cap_slots: 0, tail_age_div: 4, heat_hit_ms: 0, shared_slots: 0, shared_age_mult: 24, shared_min_cont: 3, shared_max_depth: 19,
             seg_evict: true, seg_grace_ms: 300_000, weights_kv_slot_ms: 1000 }
}

fn k(n: u32) -> Key {
    let mut x = [0u8; 32];
    x[..4].copy_from_slice(&n.to_le_bytes());
    x
}

fn item(key: u32, parent: Option<u32>, bytes: u64) -> ReserveItem {
    ReserveItem {
        key: k(key),
        parent: parent.map(k),
        bytes,
        kind: 0,
        format: 1,
        cost_hint: 0,
        flags: 0,
        ttl_ms: 0,
        meta: vec![],
    }
}

fn core(n: u32) -> Core {
    let mut c = Core::new(cfg(n, PolicyKind::Arc));
    for conn in 1..=3 {
        c.connect(conn);
        c.hello(conn, conn as u16 * 10);
    }
    c
}

fn store(c: &mut Core, conn: ConnId, key: u32, parent: Option<u32>, slots: u64, now: u64) -> Status {
    let r = c.reserve(conn, NS, &[item(key, parent, slots * 100)], now);
    if r[0].0 != Status::Ok {
        return r[0].0;
    }
    c.commit(conn, NS, &[k(key)], now).0[0]
}

#[test]
fn two_writers_same_key() {
    let mut c = core(10);
    let a = c.reserve(1, NS, &[item(1, None, 250)], 0);
    assert_eq!(a[0].0, Status::Ok);
    assert_eq!(a[0].1.iter().map(|r| r.len).sum::<u32>(), 3);
    let b = c.reserve(2, NS, &[item(1, None, 250)], 0);
    assert_eq!(b[0].0, Status::Busy);
    // readers don't see the reservation; the other writer can't commit or abort it
    assert_eq!(c.lookup(3, NS, &[k(1)], true, 0).0, 0);
    assert_eq!(c.commit(2, NS, &[k(1)], 0).0, vec![Status::NotOwner]);
    assert_eq!(c.abort(2, NS, &[k(1)]), vec![Status::NotOwner]);
    assert_eq!(c.commit(1, NS, &[k(1)], 0).0, vec![Status::Ok]);
    assert_eq!(c.reserve(2, NS, &[item(1, None, 250)], 0)[0].0, Status::Exists);
    let (n, lease, d) = c.lookup(3, NS, &[k(1), k(2)], true, 1);
    assert_eq!((n, d.len(), d[0].producer), (1, 1, 10));
    assert!(lease > 0);
    assert_eq!(c.c.cross_engine_hit_entries, 1);
    c.check();
}

#[test]
fn lookup_without_acquire_takes_no_lease() {
    let mut c = core(10);
    store(&mut c, 1, 1, None, 1, 0);
    let (n, lease, d) = c.lookup(2, NS, &[k(1)], false, 0);
    assert_eq!((n, lease, d.len()), (1, 0, 0));
    assert_eq!(c.gauges().iter().find(|g| g.0 == "leases_active").unwrap().1, 0);
}

#[test]
fn eviction_skips_leased() {
    let mut c = core(4);
    for key in 1..=4 {
        assert_eq!(store(&mut c, 1, key, None, 1, key as u64), Status::Ok);
    }
    let (_, lease, _) = c.lookup(2, NS, &[k(1)], true, 10); // oldest, but leased
    assert_eq!(store(&mut c, 1, 5, None, 1, 11), Status::Ok);
    assert_eq!(c.lookup(2, NS, &[k(1)], false, 12).0, 1, "leased entry survived");
    assert_eq!(c.lookup(2, NS, &[k(2)], false, 12).0, 0, "next-oldest evicted");
    // everything leased -> FULL
    let all: Vec<Key> = [1, 3, 4, 5].map(k).to_vec();
    for key in &all {
        c.lookup(2, NS, std::slice::from_ref(key), true, 13);
    }
    assert_eq!(store(&mut c, 1, 6, None, 1, 14), Status::Full);
    c.release(2, &[lease]);
    c.check();
}

#[test]
fn chain_evicts_from_the_tail() {
    let mut c = core(4);
    // root <- a <- b, plus a lone entry x
    store(&mut c, 1, 1, None, 1, 0);
    store(&mut c, 1, 2, Some(1), 1, 1);
    store(&mut c, 1, 3, Some(2), 1, 2);
    store(&mut c, 1, 9, None, 1, 3);
    c.lookup(2, NS, &[k(9)], false, 4); // x is the most recent
    // need 2 slots: the root is least recent but has children; the tail (3) then its parent (2) go
    assert_eq!(c.reserve(1, NS, &[item(7, None, 200)], 5)[0].0, Status::Ok);
    assert_eq!(c.lookup(2, NS, &[k(1)], false, 6).0, 1, "root kept");
    assert_eq!(c.lookup(2, NS, &[k(1), k(2)], false, 6).0, 1);
    assert_eq!(c.lookup(2, NS, &[k(9)], false, 6).0, 1);
    c.check();
}

#[test]
fn parent_required_and_protected_during_reserve() {
    let mut c = core(2);
    assert_eq!(c.reserve(1, NS, &[item(2, Some(1), 100)], 0)[0].0, Status::NoParent);
    store(&mut c, 1, 1, None, 1, 0);
    // pool: 1 used, 1 free. Child needs 2: the only candidate is its own parent -> FULL, parent intact
    assert_eq!(c.reserve(1, NS, &[item(2, Some(1), 200)], 1)[0].0, Status::Full);
    assert_eq!(c.lookup(2, NS, &[k(1)], false, 2).0, 1);
    c.check();
}

#[test]
fn crash_with_live_leases_and_reservations() {
    let mut c = core(10);
    store(&mut c, 1, 1, None, 2, 0);
    let (_, _lease, _) = c.lookup(2, NS, &[k(1)], true, 1);
    c.reserve(2, NS, &[item(2, Some(1), 300), item(3, None, 100)], 1);
    c.reserve(2, NS, &[item(4, Some(2), 100)], 1);
    assert_eq!(c.alloc.free(), 3);
    c.disconnect(2);
    c.check();
    // the writer's reservations are gone, but their slots stay quarantined for
    // QUARANTINE_MS (its copy may still be running), then return to the allocator
    assert_eq!(c.alloc.free(), 3, "reservations quarantined");
    assert_eq!(c.c.leases_closed_by_disconnect, 1);
    c.tick(1 + QUARANTINE_MS);
    assert_eq!(c.alloc.free(), 8, "reservations freed after the quarantine");
    // entry 1 is unleased again: it can be evicted to fit 10 slots
    assert_eq!(store(&mut c, 1, 5, None, 10, 2 + QUARANTINE_MS), Status::Ok);
    c.check();
}

#[test]
fn reservation_and_lease_timeouts() {
    let mut c = core(10);
    c.reserve(1, NS, &[item(1, None, 100)], 0);
    store(&mut c, 1, 2, None, 1, 0);
    c.reserve(1, NS, &[item(3, Some(2), 100)], 0);
    let (_, lease, _) = c.lookup(2, NS, &[k(2)], true, 0);
    c.tick(29_999);
    assert_eq!(c.alloc.free(), 7);
    c.tick(30_000);
    assert_eq!(c.alloc.free(), 7, "timed-out reservations quarantined");
    assert_eq!(c.c.reservation_timeouts, 2);
    c.check();
    assert_eq!(c.commit(1, NS, &[k(1)], 30_001).0, vec![Status::NotFound]);
    c.tick(60_000);
    assert_eq!(c.c.lease_timeouts, 1);
    assert_eq!(c.release(2, &[lease]), vec![Status::NotFound]);
    c.tick(30_000 + QUARANTINE_MS);
    assert_eq!(c.alloc.free(), 9, "quarantine over");
    c.check();
}

#[test]
fn explicit_abort_frees_at_once() {
    // an ABORT comes after the engine's copy (failed store / NaN scan): no quarantine
    let mut c = core(10);
    c.reserve(1, NS, &[item(1, None, 300)], 0);
    assert_eq!(c.alloc.free(), 7);
    assert_eq!(c.abort(1, NS, &[k(1)]), vec![Status::Ok]);
    assert_eq!(c.alloc.free(), 10);
    c.check();
}

#[test]
fn abort_cascades_to_reserved_children() {
    let mut c = core(10);
    c.reserve(1, NS, &[item(1, None, 100), item(2, Some(1), 100), item(3, Some(2), 100)], 0);
    c.commit(1, NS, &[k(3)], 0); // child committed before its parent (pipelined stores)
    assert_eq!(c.abort(1, NS, &[k(1)]), vec![Status::Ok]);
    // 2 (reserved) went with it; 3 (committed) is detached and stays reachable on its own
    assert_eq!(c.lookup(2, NS, &[k(2)], false, 1).0, 0);
    assert_eq!(c.lookup(2, NS, &[k(3)], false, 1).0, 1);
    c.check();
}

#[test]
fn wait_for_commit() {
    let mut c = core(10);
    assert_eq!(c.wait(2, 77, NS, k(1), 1000, 0), None);
    assert_eq!(c.wait(3, 78, NS, k(1), 10, 0), None);
    c.reserve(1, NS, &[item(1, None, 100)], 0);
    assert_eq!(c.tick(10), vec![Notify { conn: 3, req_id: 78, status: Status::Timeout }]);
    let (_, notes) = c.commit(1, NS, &[k(1)], 20);
    assert_eq!(notes, vec![Notify { conn: 2, req_id: 77, status: Status::Ok }]);
    assert_eq!(c.wait(2, 79, NS, k(1), 1000, 21), Some(Status::Ok));
}

#[test]
fn ephemeral_goes_first() {
    let mut c = core(3);
    store(&mut c, 1, 1, None, 1, 0);
    store(&mut c, 1, 2, None, 1, 1);
    let mut e = item(3, None, 100);
    e.flags = ENTRY_EPHEMERAL;
    e.ttl_ms = 5_000;
    c.reserve(1, NS, &[e], 2);
    c.commit(1, NS, &[k(3)], 2);
    // read once by the decoding engine
    let (_, lease, _) = c.lookup(2, NS, &[k(3)], true, 3);
    c.release(2, &[lease]);
    // the ephemeral (newest) goes before the LRU entries
    assert_eq!(store(&mut c, 1, 4, None, 1, 4), Status::Ok);
    assert_eq!(c.lookup(2, NS, &[k(3)], false, 5).0, 0);
    assert_eq!(c.lookup(2, NS, &[k(1)], false, 5).0, 1);
    // TTL expiry without a read
    let mut e = item(5, None, 100);
    e.flags = ENTRY_EPHEMERAL;
    e.ttl_ms = 5_000;
    c.reserve(1, NS, &[e], 10);
    c.commit(1, NS, &[k(5)], 10);
    assert_eq!(store(&mut c, 1, 6, None, 1, 11), Status::Ok); // evicts by policy
    c.tick(5_010);
    assert_eq!(c.c.evict_expired, 2);
    c.check();
}

#[test]
fn low_water_mark() {
    let mut c = Core::new(Config { low_water_slots: 3, ..cfg(10, PolicyKind::Lru) });
    c.connect(1);
    c.hello(1, 1);
    for key in 1..=10 {
        store(&mut c, 1, key, None, 1, key as u64);
    }
    c.tick(100);
    assert_eq!(c.alloc.free(), 3);
    assert_eq!(c.lookup(1, NS, &[k(1)], false, 101).0, 0);
    assert_eq!(c.lookup(1, NS, &[k(4)], false, 101).0, 1);
}

#[test]
fn too_big_and_bad_meta() {
    let mut c = core(4);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 401)], 0)[0].0, Status::TooBig);
    let mut it = item(2, None, 1);
    it.meta = vec![0; MAX_META + 1];
    assert_eq!(c.reserve(1, NS, &[it], 0)[0].0, Status::BadRequest);
}

/// Random operations from three connections; invariants checked after every step.
#[test]
fn randomized_invariants() {
    for policy in [PolicyKind::Arc, PolicyKind::Lru] {
        let mut rng = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut c = Core::new(Config { low_water_slots: 2, ..cfg(64, policy) });
        let mut leases: Vec<(ConnId, u64)> = Vec::new();
        for conn in 1..=3 {
            c.connect(conn);
            c.hello(conn, conn as u16);
        }
        let mut now = 0u64;
        for step in 0..20_000 {
            now += next() % 50;
            let conn = 1 + next() % 3;
            let key = (next() % 40) as u32;
            match next() % 10 {
                0..=2 => {
                    let parent = if key > 0 && next() % 2 == 0 { Some(key - 1) } else { None };
                    let mut it = item(key, parent, 1 + next() % 900);
                    if next() % 8 == 0 {
                        it.flags = ENTRY_EPHEMERAL;
                        it.ttl_ms = 500;
                    }
                    c.reserve(conn, NS, &[it], now);
                }
                3..=4 => {
                    c.commit(conn, NS, &[k(key)], now);
                }
                5 => {
                    c.abort(conn, NS, &[k(key)]);
                }
                6 => {
                    let keys: Vec<Key> = (key..key + 4).map(k).collect();
                    let (_, l, _) = c.lookup(conn, NS, &keys, true, now);
                    if l > 0 {
                        leases.push((conn, l));
                    }
                }
                7 => {
                    if !leases.is_empty() {
                        let (conn, l) = leases.swap_remove((next() % leases.len() as u64) as usize);
                        c.release(conn, &[l]);
                    }
                }
                8 => {
                    c.tick(now);
                }
                _ => {
                    if next() % 20 == 0 {
                        c.disconnect(conn);
                        c.connect(conn);
                        c.hello(conn, conn as u16);
                        leases.retain(|x| x.0 != conn);
                    } else {
                        c.wait(conn, step, NS, k(key), 100, now);
                    }
                }
            }
            c.check();
        }
        // a disconnect every ~5 s here quarantines that writer's open reservations for 60 s, so fewer slots circulate than before (300 / 100); the point is that the pool keeps working
        assert!(c.c.committed > 120 && c.c.evict_room > 60, "{:?}", c.c);
    }
}

#[test]
fn per_namespace_counters() {
    // two namespaces in a 3-slot pool: hits, EXISTS and evictions (used / never hit) are counted per namespace
    let mut c = core(3);
    const N2: u32 = 2;
    store(&mut c, 1, 1, None, 1, 0);
    store(&mut c, 1, 2, None, 1, 1);
    assert_eq!(c.lookup(2, NS, &[k(1), k(9)], false, 2).0, 1);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 100)], 3)[0].0, Status::Exists);
    for (i, key) in [11u32, 12].iter().enumerate() {
        let r = c.reserve(3, N2, &[item(*key, None, 100)], 4 + i as u64);
        assert_eq!(r[0].0, Status::Ok);
        assert_eq!(c.commit(3, N2, &[k(*key)], 4 + i as u64).0[0], Status::Ok);
    }
    let s: HashMap<(u32, u8), (Vec<(&str, u64)>, Vec<(&str, u64)>)> =
        c.ns_stats().into_iter().map(|(ns, g, cs)| (ns, (g, cs))).collect();
    let get = |v: &Vec<(&str, u64)>, n: &str| v.iter().find(|(k, _)| *k == n).unwrap().1;
    // test items are kind 0 (attention); lookups count under KIND_ANY
    let (g1, c1) = &s[&(NS, 0)];
    let (g2, c2) = &s[&(N2, 0)];
    assert_eq!(get(&s[&(NS, KIND_ANY)].1, "lookup_keys"), 2);
    assert_eq!((get(c1, "hit_entries"), get(c1, "committed"), get(c1, "exists")), (1, 2, 1));
    // 3 slots, 4 stores: one NS entry evicted; ARC keeps the hit one (key 1) and evicts key 2 (never hit)
    assert_eq!((get(c1, "evicted"), get(c1, "evicted_unused"), get(g1, "entries")), (1, 1, 1));
    assert_eq!((get(c2, "committed"), get(g2, "entries"), get(g2, "slots"), get(c2, "evicted")), (2, 2, 2, 0));
    c.check();
}

#[test]
fn entry_lifecycle_records() {
    // committed at 1 s, hit by engine 20 at 6 s and 9 s, evicted at 20 s: one record, histograms per kind
    let mut c = core(1);
    store(&mut c, 1, 1, None, 1, 1_000);
    assert_eq!(c.lookup(2, NS, &[k(1)], false, 6_000).0, 1);
    assert_eq!(c.lookup(2, NS, &[k(1)], false, 9_000).0, 1);
    c.tick(20_000);
    assert_eq!(c.reserve(3, NS, &[item(2, None, 100)], 20_000)[0].0, Status::Ok); // evicts key 1
    assert_eq!(c.records.len(), 1);
    let r = &c.records[0];
    assert_eq!((r.born, r.first_hit, r.last_hit, r.left, r.hits, r.reason), (1_000, 6_000, 9_000, 20_000, 2, "evicted"));
    assert_eq!(r.hit_engines, 1 << 20);
    assert_eq!(r.recent, vec![(6_000, 20), (9_000, 20)]);
    assert_eq!(r.key_prefix, k(1)[..8]);
    let fh = &c.life[&("first_hit", 0)];
    assert_eq!((fh.count, fh.sum_ms), (1, 5_000));
    assert_eq!(fh.buckets[0], 0); // > 1 s
    assert_eq!(fh.buckets[1], 1); // <= 5 s
    assert_eq!(c.life[&("rehit", 0)].sum_ms, 3_000);
    assert_eq!(c.life[&("age", 0)].sum_ms, 19_000);
    c.check();
}

#[test]
fn recent_hits_keep_the_last_ten() {
    let mut c = core(1);
    store(&mut c, 1, 1, None, 1, 0);
    for t in 1..=15u64 {
        c.lookup(1 + (t % 2) as ConnId, NS, &[k(1)], false, t * 1000);
    }
    c.tick(20_000);
    store(&mut c, 3, 2, None, 1, 20_000); // 1-slot pool: evicts key 1
    let r = c.records.iter().find(|r| r.key_prefix == k(1)[..8]).unwrap();
    assert_eq!(r.hits, 15);
    assert_eq!(r.recent.len(), RECENT_HITS);
    assert_eq!(r.recent[0], (6_000, 10));      // hits 6..15 kept; engine of conn 1 = 10, conn 2 = 20
    assert_eq!(r.recent[9], (15_000, 20));
    assert_eq!(r.hit_engines, (1 << 10) | (1 << 20));
}

#[test]
fn event_journal_in_order() {
    let mut c = core(1);
    c.log_events = true;
    store(&mut c, 1, 1, None, 1, 100);
    c.lookup(2, NS, &[k(1), k(9)], true, 200);
    store(&mut c, 3, 2, None, 1, 300); // the lease from conn 2 pins key 1: FULL, nothing evicted
    c.disconnect(2);                   // lease ends
    store(&mut c, 3, 2, None, 1, 400); // evicts key 1
    let kinds: Vec<&str> = c.events.iter().map(|e| match e {
        Event::Store { .. } => "store",
        Event::Lookup { .. } => "lookup",
        Event::Remove(_) => "remove",
        Event::Content { .. } => "content",
    }).collect();
    assert_eq!(kinds, vec!["store", "lookup", "remove", "store"]);
    match &c.events[1] {
        Event::Lookup { t, engine, n_keys, n_hit, acquire, .. } => assert_eq!((*t, *engine, *n_keys, *n_hit, *acquire), (200, 20, 2, 1, true)),
        _ => unreachable!(),
    }
    match &c.events[2] {
        Event::Remove(r) => assert_eq!((r.hits, r.recent.clone(), r.left), (1, vec![(200, 20)], 400)),
        _ => unreachable!(),
    }
}

#[test]
fn hot_set_select_and_restore() {
    let mut c = core(4);
    store(&mut c, 1, 1, None, 1, 0);
    store(&mut c, 1, 2, None, 1, 0);
    for t in 1..=3 {
        c.lookup(2, NS, &[k(1)], false, t);
    }
    c.lookup(2, NS, &[k(2)], false, 5);
    let hot = c.hot_entries(2);
    assert_eq!(hot.len(), 1);
    assert_eq!((hot[0].key, hot[0].hits), (k(1), 3));
    // restore into a fresh core: committed at once, hits kept, visible to lookups
    let mut d = core(4);
    let r = d.insert_committed(NS, k(1), 0, 1, 10, vec![1, 2], 100, 3, 0).unwrap();
    assert_eq!(r.iter().map(|x| x.len).sum::<u32>(), 1);
    assert!(d.insert_committed(NS, k(1), 0, 1, 10, vec![], 100, 3, 0).is_none());
    assert_eq!(d.lookup(3, NS, &[k(1)], false, 1).0, 1);
    assert_eq!(d.hot_entries(1)[0].hits, 4);
    d.check();
}

fn store_kind(c: &mut Core, conn: ConnId, key: u32, kind: u8, slots: u64, now: u64) -> Status {
    let mut it = item(key, None, slots * 100);
    it.kind = kind;
    let r = c.reserve(conn, NS, &[it], now);
    if r[0].0 != Status::Ok {
        return r[0].0;
    }
    c.commit(conn, NS, &[k(key)], now).0[0]
}

#[test]
fn tail_entries_evict_first() {
    // Arc tail-hold spill: RESERVE evicts partial tails (KIND_TAIL) before any other entry, even a colder one
    let mut c = core(10);
    assert_eq!(store(&mut c, 1, 1, None, 4, 0), Status::Ok); // old prefix entry, never hit
    assert_eq!(store_kind(&mut c, 1, 2, KIND_TAIL, 4, 100), Status::Ok);
    assert_eq!(store_kind(&mut c, 1, 3, KIND_TAIL, 2, 200), Status::Ok);
    c.lookup(1, NS, &[k(2)], false, 300); // tail 2 used more recently than tail 3
    // needs 2 slots: tail 3 (least recently used tail) goes, not the older prefix entry 1
    assert_eq!(store(&mut c, 1, 4, None, 2, 400), Status::Ok);
    assert_eq!(c.lookup(1, NS, &[k(3)], false, 500).0, 0);
    assert_eq!(c.lookup(1, NS, &[k(1)], false, 500).0, 1);
    assert_eq!(c.lookup(1, NS, &[k(2)], false, 500).0, 1);
    // a leased tail is never taken: the next victim is a normal entry
    let (n, lease, _) = c.lookup(2, NS, &[k(2)], true, 600);
    assert_eq!((n, lease > 0), (1, true));
    assert_eq!(store(&mut c, 1, 5, None, 4, 700), Status::Ok);
    assert_eq!(c.lookup(1, NS, &[k(2)], false, 800).0, 1);
    c.check();
}

#[test]
fn tail_entries_age_faster() {
    // gradual sweep with age A: a tail unused for A / tail_age_div (4) goes; no TTL otherwise
    let mut c = core(20);
    assert_eq!(store(&mut c, 1, 1, None, 1, 1000), Status::Ok);
    assert_eq!(store_kind(&mut c, 1, 2, KIND_TAIL, 4, 1000), Status::Ok);
    // now 1200, age 1000: prefix last use 1000 is not before 200; tail heat 1200 - 4*200 = 400, not before 200 either
    c.tick(1200);
    assert_eq!(c.evict_aged_by(&|_| 200, 0), 0);
    // now 1300: tail heat 1300 - 4*300 = 100 < 300 -> the tail goes (unused 300 ms > 1000 / 4); the prefix stays
    c.tick(1300);
    assert_eq!(c.evict_aged_by(&|_| 300, 0), 4);
    assert_eq!(c.lookup(1, NS, &[k(1)], false, 1300).0, 1);
    // heat order across pools (RAM ceiling / pressure)
    let mut c = core(20);
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    assert_eq!(store_kind(&mut c, 1, 2, KIND_TAIL, 1, 900), Status::Ok);
    c.tick(1000);
    // prefix 0; tail 1000 - 4*100 = 600: the prefix entry is still the oldest
    assert_eq!(c.oldest_evictable(), Some(0));
    assert_eq!(store_kind(&mut c, 1, 3, KIND_TAIL, 1, 1000), Status::Ok);
    c.tick(1500);
    // tail 2: 1500 - 4*600 -> 0 (saturates) ties the prefix entry at 0; tail 3: 1500 - 2000 -> 0 too; all three go
    // before anything used later, and the prefix entry is not protected by being older
    assert_eq!(c.evict_oldest() + c.evict_oldest() + c.evict_oldest(), 3);
    c.check();
}

#[test]
fn tail_age_div_one_is_plain() {
    let mut cf = cfg(10, PolicyKind::Arc);
    cf.tail_age_div = 1;
    let mut c = Core::new(cf);
    c.connect(1);
    c.hello(1, 10);
    assert_eq!(store_kind(&mut c, 1, 2, KIND_TAIL, 1, 500), Status::Ok);
    c.tick(1000);
    assert_eq!(c.oldest_evictable(), Some(500));
}

fn cfg_shared(n_slots: u32, shared_slots: u64) -> Config {
    let mut c = cfg(n_slots, PolicyKind::Arc);
    c.shared_slots = shared_slots;
    c
}

/// Reserve + commit one slot with entry metadata "L1" + the parent's 16-byte key prefix (the clients' META_V1).
fn store_meta(c: &mut Core, key: u32, parent: Option<u32>, flags: u8, now: u64) -> Status {
    let mut meta = b"L1".to_vec();
    match parent {
        Some(p) => meta.extend_from_slice(&k(p)[..16]),
        None => meta.extend_from_slice(&[0u8; 16]),
    }
    meta.extend_from_slice(&[0u8; 14]);
    let mut it = item(key, None, 100);
    it.meta = meta;
    it.flags = flags;
    let r = c.reserve(1, NS, &[it], now);
    if r[0].0 != Status::Ok {
        return r[0].0;
    }
    c.commit(1, NS, &[k(key)], now).0[0]
}

fn gauge(c: &Core, name: &str) -> u64 {
    c.gauges().into_iter().find(|(n, _)| n == name).map(|x| x.1).unwrap()
}

#[test]
fn shared_prefix_detected_by_three_continuations() {
    // system prompt chunks 1 -> 2, then sessions continue from 2 with 3, 4, 5: a second continuation is only a
    // regenerate / retry (not shared); the third makes 2 and its ancestor 1 shared
    let mut c = Core::new(cfg_shared(8, 4));
    for conn in 1..=2 {
        c.connect(conn);
        c.hello(conn, 10);
    }
    assert_eq!(store_meta(&mut c, 1, None, 0, 0), Status::Ok);
    assert_eq!(store_meta(&mut c, 2, Some(1), 0, 1), Status::Ok);
    assert_eq!(store_meta(&mut c, 3, Some(2), 0, 2), Status::Ok);
    assert_eq!(gauge(&c, "shared_entries"), 0, "one continuation is not a shared prefix");
    assert_eq!(store_meta(&mut c, 4, Some(2), 0, 3), Status::Ok);
    assert_eq!(gauge(&c, "shared_entries"), 0, "two continuations (one regenerate) are not a shared prefix");
    assert_eq!(store_meta(&mut c, 5, Some(2), 0, 4), Status::Ok);
    assert_eq!(gauge(&c, "shared_entries"), 2);
    assert_eq!(gauge(&c, "shared_slots"), 2);
    // fill the pool: the oldest entries (1, 2: the shared prefix) are evicted last
    for key in 10..13 {
        assert_eq!(store_meta(&mut c, key, None, 0, 10 + key as u64), Status::Ok);
    }
    assert_eq!(gauge(&c, "slots_free"), 0);
    for key in 20..24 {
        assert_eq!(store_meta(&mut c, key, None, 0, 100 + key as u64), Status::Ok);
    }
    assert_eq!(c.lookup(1, NS, &[k(1), k(2)], false, 200).0, 2, "the shared prefix stays");
    assert_eq!(c.lookup(1, NS, &[k(3)], false, 200).0, 0, "an unshared older entry went");
    c.check();
}

#[test]
fn shared_prefix_ages_slower_and_hint_flag() {
    let mut c = Core::new(cfg_shared(8, 4));
    c.connect(1);
    c.hello(1, 10);
    assert_eq!(store_meta(&mut c, 1, None, ENTRY_SHARED_HINT, 1000), Status::Ok);   // client hint: shared at once
    assert_eq!(store_meta(&mut c, 2, None, 0, 1000), Status::Ok);
    assert_eq!(gauge(&c, "shared_entries"), 1);
    // age 1000 ms: at now 3000 the plain entry (idle 2000) goes, the shared one looks idle 2000/24 = 83 ms
    c.tick(3000);
    assert_eq!(c.evict_aged_by(&|_| 2000, 0), 1);
    assert_eq!(c.lookup(1, NS, &[k(1)], false, 3000).0, 1);
    assert_eq!(c.lookup(1, NS, &[k(2)], false, 3000).0, 0);
    c.check();
}

#[test]
fn shared_class_budget_demotes_least_recent() {
    let mut c = Core::new(cfg_shared(8, 2));
    c.connect(1);
    c.hello(1, 10);
    for (key, t) in [(1u32, 10u64), (2, 20), (3, 30)] {
        assert_eq!(store_meta(&mut c, key, None, ENTRY_SHARED_HINT, t), Status::Ok);
    }
    assert_eq!(gauge(&c, "shared_slots"), 2, "budget 2 slots");
    let demoted = c.counters().into_iter().find(|(n, _)| n == "shared_demoted").unwrap().1;
    assert_eq!(demoted, 1);
    // the demoted one (1, least recently used) is now an ordinary entry: evicted before 2 and 3
    for key in 10..16 {
        assert_eq!(store_meta(&mut c, key, None, 0, 100 + key as u64), Status::Ok);
    }
    assert_eq!(store_meta(&mut c, 30, None, 0, 200), Status::Ok);
    assert_eq!(c.lookup(1, NS, &[k(1)], false, 300).0, 0);
    assert_eq!(c.lookup(1, NS, &[k(2)], false, 300).0 + c.lookup(1, NS, &[k(3)], false, 300).0, 2);
    c.check();
}

#[test]
fn shared_class_off_by_default() {
    let mut c = core(8);
    assert_eq!(store_meta(&mut c, 1, None, ENTRY_SHARED_HINT, 0), Status::Ok);
    assert_eq!(store_meta(&mut c, 2, Some(1), 0, 1), Status::Ok);
    assert_eq!(store_meta(&mut c, 3, Some(1), 0, 2), Status::Ok);
    assert_eq!(gauge(&c, "shared_entries"), 0);
}

#[test]
fn shared_hint_in_meta_and_on_exists() {
    // the Arc client's hint: "L1" metadata byte 32 bit 0; a reserve of an already committed entry applies it
    let mut c = Core::new(cfg_shared(8, 4));
    c.connect(1);
    c.hello(1, 10);
    assert_eq!(store_meta(&mut c, 1, None, 0, 0), Status::Ok);
    assert_eq!(gauge(&c, "shared_entries"), 0);
    let mut meta = b"L1".to_vec();
    meta.extend_from_slice(&[0u8; 30]);
    meta.push(1);
    let mut it = item(1, None, 100);
    it.meta = meta.clone();
    assert_eq!(c.reserve(1, NS, &[it], 1)[0].0, Status::Exists);
    assert_eq!(gauge(&c, "shared_entries"), 1);
    let mut it = item(2, None, 100);
    it.meta = meta;
    assert_eq!(c.reserve(1, NS, &[it], 2)[0].0, Status::Ok);
    c.commit(1, NS, &[k(2)], 2);
    assert_eq!(gauge(&c, "shared_entries"), 2);
}

#[test]
fn shared_prefix_depth_cap() {
    // a chain 1 -> 2 -> 3 -> 4 continued three ways from 4 (depth 4): with a cap of 3 it is not a chat beginning
    let mut cf = cfg_shared(16, 8);
    cf.shared_max_depth = 3;
    let mut c = Core::new(cf);
    c.connect(1);
    c.hello(1, 10);
    assert_eq!(store_meta(&mut c, 1, None, 0, 0), Status::Ok);
    for key in 2..5 {
        assert_eq!(store_meta(&mut c, key, Some(key - 1), 0, key as u64), Status::Ok);
    }
    for key in 10..13 {
        assert_eq!(store_meta(&mut c, key, Some(4), 0, key as u64), Status::Ok);
    }
    assert_eq!(gauge(&c, "shared_entries"), 0, "deeper than the cap");
    for key in 20..23 {
        assert_eq!(store_meta(&mut c, key, Some(3), 0, key as u64), Status::Ok);
    }
    assert_eq!(gauge(&c, "shared_entries"), 3, "depth 3 junction: 1, 2, 3");
}

// ------------------------------------------------------------ sessions, strict SSD candidates, pressure protection

const MIN: u64 = 60_000;

/// a request's chain: 3 attention entries (1 <- 2 <- 3) and a GDN state at its end (kind 1, key 103)
fn session_chain(c: &mut Core, base: u32, now: u64) {
    assert_eq!(store(c, 1, base + 1, None, 1, now), Status::Ok);
    assert_eq!(store(c, 1, base + 2, Some(base + 1), 1, now), Status::Ok);
    assert_eq!(store(c, 1, base + 3, Some(base + 2), 1, now), Status::Ok);
    assert_eq!(store_kind(c, 1, base + 103, 1, 1, now), Status::Ok);
}

fn touch_chain(c: &mut Core, keys: &[u32], now: u64) {
    let ks: Vec<Key> = keys.iter().map(|&x| k(x)).collect();
    c.touch(NS, &ks, now);
}

#[test]
fn sessions_count_resumes_after_real_gaps_only() {
    let mut c = core(40);
    touch_chain(&mut c, &[1, 2, 3], 0);
    session_chain(&mut c, 0, 0);
    let sid = *c.sessions.keys().next().unwrap();
    // the same request touching again, and a continuation after 1 minute: no resume
    touch_chain(&mut c, &[1, 2, 3], 1_000);
    touch_chain(&mut c, &[1, 2, 3, 4], MIN);
    assert_eq!(c.sessions[&sid].resumes, 0);
    // continuations after >= 10 min idle: resumes; the chain end moves along (4, then 5, ...)
    touch_chain(&mut c, &[1, 2, 3, 4, 5], 12 * MIN);
    touch_chain(&mut c, &[1, 2, 3, 4, 5, 6], 30 * MIN);
    assert_eq!((c.sessions.len(), c.sessions[&sid].resumes), (1, 2));
    // a branch from an older end (3) carries the session's counts
    touch_chain(&mut c, &[1, 2, 3, 9], 45 * MIN);
    assert_eq!((c.sessions.len(), c.sessions[&sid].resumes), (1, 3));
    // an unrelated chain is a new session
    touch_chain(&mut c, &[50, 51], 46 * MIN);
    assert_eq!(c.sessions.len(), 2);
}

#[test]
fn strict_candidates_short_lived_session_never() {
    // a one-off / short-lived session that goes idle: nothing to write
    let mut c = core(40);
    touch_chain(&mut c, &[1, 2, 3, 103], 0);
    session_chain(&mut c, 0, 0);
    touch_chain(&mut c, &[1, 2, 3, 103], 15 * MIN); // one resume only, span 15 min
    let all = |_: u32| u64::MAX;
    assert!(c.strict_age_candidates(&all, 0, 3, 60 * MIN).is_empty());
}

#[test]
fn strict_candidates_long_term_session_newest_end_once() {
    for seg in [true, false] {
        strict_long_term(seg);
    }
}

fn strict_long_term(seg: bool) {
    let mut cf = cfg(60, PolicyKind::Arc);
    cf.seg_evict = seg;
    let mut c = core_with(cf);
    // a long-term session: stored at t=0, resumed 3 times after >= 10 min gaps over 70 min; a newer state at the end
    touch_chain(&mut c, &[1, 2, 3, 103], 0);
    session_chain(&mut c, 0, 0);
    for (i, t) in [20 * MIN, 45 * MIN, 70 * MIN].iter().enumerate() {
        let mut keys = vec![1, 2, 3, 103];
        keys.extend((0..=i as u32).map(|j| 4 + j));
        keys.push(200 + i as u32);
        touch_chain(&mut c, &keys, *t);
    }
    assert_eq!(store(&mut c, 1, 4, Some(3), 1, 70 * MIN), Status::Ok);
    assert_eq!(store_kind(&mut c, 1, 202, 1, 1, 70 * MIN), Status::Ok); // the newest state (touched before stored)
    let sid = *c.sessions.keys().next().unwrap();
    assert_eq!(c.sessions[&sid].resumes, 3);
    let all = |_: u32| u64::MAX;
    let v: HashSet<Key> = c.strict_age_candidates(&all, 0, 3, 60 * MIN).into_iter().map(|x| x.1).collect();
    // leaves about to age out: the attention chain end (4) and the newest state (202); the superseded state (103) never
    assert!(v.contains(&k(4)) && v.contains(&k(202)) && !v.contains(&k(103)), "{v:?}");
    // the same session under a stricter rule (4 resumes, or a 71 min span) qualifies for nothing
    assert!(c.strict_age_candidates(&all, 0, 4, 60 * MIN).is_empty());
    assert!(c.strict_age_candidates(&all, 0, 3, 71 * MIN).is_empty());
    // only entries about to age out: nothing is older than 0; a slot budget stops the walk
    assert!(c.strict_age_candidates(&|_| 0, 0, 3, 60 * MIN).is_empty());
    assert!(c.strict_age_candidates(&all, 1, 3, 60 * MIN).len() <= 1, "seg {seg}: one slot's worth");
    c.check();
}

#[test]
fn pressure_protection_skips_recent_and_shared() {
    let mut c = core(40);
    assert_eq!(store(&mut c, 1, 1, None, 2, 0), Status::Ok); // cold (last used at 0)
    assert_eq!(store(&mut c, 1, 2, None, 2, 50 * MIN), Status::Ok); // active session (within 15 min of now)
    assert_eq!(store(&mut c, 1, 3, None, 2, 10 * MIN), Status::Ok); // cold
    let now = 55 * MIN;
    let prot: HashSet<EntryId> = c.protect_candidates(now - 15 * MIN).into_iter().map(|x| x.2).collect();
    assert_eq!(prot.len(), 1);
    // pressure: the cold ones go (oldest first), then nothing: the pool shrinks to the protected set
    assert_eq!(c.oldest_evictable_except(&prot), Some(0));
    assert_eq!(c.evict_oldest_except(&prot), 2);
    assert_eq!(c.evict_oldest_except(&prot), 2);
    assert_eq!(c.evict_oldest_except(&prot), 0);
    assert!(c.contains(NS, &k(2)) && !c.contains(NS, &k(1)) && !c.contains(NS, &k(3)));
    // extreme pressure (no protection): the protected entry can go too
    assert_eq!(c.evict_oldest(), 2);
    c.check();
}

// ------------------------------------------------------------ segment eviction

const GRACE: u64 = 300_000;

fn keys(v: &[u32]) -> Vec<Key> {
    v.iter().map(|x| k(*x)).collect()
}

fn hs(c: &mut Core, key: u32, kind: u8, now: u64) {
    let mut it = item(key, None, 100);
    it.kind = kind;
    assert_eq!(c.reserve(1, NS, &[it], now)[0].0, Status::Ok);
    assert_eq!(c.commit(1, NS, &[k(key)], now).0[0], Status::Ok);
}

/// One request's TOUCH run as the connector sends it: the GDN groups' batches (suffixes), then the attention batch
/// (from the root); the next request of the connection ends the run.
fn treq(c: &mut Core, states: &[Vec<u32>], attn: &[u32], now: u64) {
    for s in states {
        c.touch_conn(1, NS, &keys(s), now);
    }
    c.touch_conn(1, NS, &keys(attn), now);
    c.release(1, &[]);
}

/// A core that learned the connection's request shape (`groups` GDN groups + attention), warm-up entries removed.
fn hybrid(n_slots: u32, groups: u32, cf: impl Fn(&mut Config)) -> Core {
    let mut x = cfg(n_slots, PolicyKind::Arc);
    cf(&mut x);
    let mut c = Core::new(x);
    for conn in 1..=3 {
        c.connect(conn);
        c.hello(conn, conn as u16 * 10);
    }
    for g in 0..groups {
        hs(&mut c, 9000 + g, 1, 0);
    }
    hs(&mut c, 9100, 0, 0);
    let st: Vec<Vec<u32>> = (0..groups).map(|g| vec![9000 + g]).collect();
    for _ in 0..8 {
        treq(&mut c, &st, &[9100], 0);
    }
    assert!(!c.config().seg_evict || c.c.seg_runs_linked >= 3, "learned");
    for g in 0..groups {
        assert!(c.remove_key(NS, &k(9000 + g)));
    }
    assert!(c.remove_key(NS, &k(9100)));
    c
}

fn has(c: &Core, key: u32) -> bool {
    c.contains(NS, &k(key))
}

/// Chain attention 101..=100+n, GDN state (one group) stored at the given blocks (key 200 + block), touched once.
fn chain(c: &mut Core, n: u32, states: &[u32], now: u64) {
    for b in 1..=n {
        hs(c, 100 + b, 0, now);
    }
    for b in states {
        hs(c, 200 + b, 1, now);
    }
    treq(c, &[(1..=n).map(|b| 200 + b).collect()], &(1..=n).map(|b| 100 + b).collect::<Vec<_>>(), now);
}

#[test]
fn seg_orphan_attention_after_grace() {
    let mut c = hybrid(32, 1, |_| {});
    chain(&mut c, 6, &[3], 1000);
    c.tick(1000 + GRACE - 1);
    assert!((101..=106).all(|b| has(&c, b)), "young: kept");
    c.tick(1000 + GRACE + 2000);
    // attention past the deepest state is useless; the chain up to the state stays
    assert!((101..=103).all(|b| has(&c, b)) && has(&c, 203));
    assert!((104..=106).all(|b| !has(&c, b)));
    assert_eq!(c.c.evict_orphan_attn + c.c.cascade_evicted, 3, "an orphan takes the deeper ones with it");
    assert_eq!(c.c.evict_orphan_state, 0);
    c.check();
}

#[test]
fn seg_off_keeps_orphans() {
    let mut c = hybrid(32, 1, |x| x.seg_evict = false);
    chain(&mut c, 6, &[3], 1000);
    c.tick(1000 + 10 * GRACE);
    assert!((101..=106).all(|b| has(&c, b)) && has(&c, 203));
    assert_eq!(c.c.evict_orphan_attn, 0);
}

#[test]
fn seg_orphan_state_when_chain_gone() {
    let mut c = hybrid(32, 1, |_| {});
    chain(&mut c, 3, &[3], 1000);
    assert!(c.remove_key(NS, &k(102)));   // a block below the state leaves by another path
    // the rest of its chain goes with it at once: block 103 and the state anchored at it
    assert!(!has(&c, 103) && !has(&c, 203) && has(&c, 101));
    assert_eq!(c.c.cascade_evicted, 2);
    c.tick(1000 + GRACE + 1000);
    c.tick(1000 + GRACE + 3000);
    assert!(!has(&c, 101), "then the attention no state uses");
    assert_eq!(c.c.evict_orphan_state, 0);
    assert_eq!(c.c.evict_orphan_attn, 1);
    c.check();
}

#[test]
fn seg_state_kept_when_block_on_ssd_or_touched_again() {
    // on the SSD: still a usable chain
    let mut c = hybrid(32, 1, |_| {});
    let on_ssd = k(102);
    c.set_elsewhere(Box::new(move |_, key| *key == on_ssd));
    chain(&mut c, 3, &[3], 1000);
    assert!(c.remove_key(NS, &k(102)));
    c.tick(1000 + GRACE + 1000);
    assert!(has(&c, 203) && has(&c, 101) && has(&c, 103));
    assert_eq!(c.c.cascade_evicted, 0);
    // thrown out (not on the SSD): the chain after it goes at once, even if a request still has it on the GPU
    // (no lookup can reach past the gap; the GPU stores the blocks again if it still has them)
    let mut c = hybrid(32, 1, |_| {});
    chain(&mut c, 3, &[3], 1000);
    assert!(c.remove_key(NS, &k(102)));
    assert!(!has(&c, 203) && !has(&c, 103) && has(&c, 101));
}

#[test]
fn seg_partial_block_states_are_orphans() {
    // three GDN groups: one group's state of the block leaves, the other two are useless
    let mut c = hybrid(32, 3, |_| {});
    for b in 1..=3 {
        hs(&mut c, 100 + b, 0, 1000);
    }
    for g in 0..3 {
        hs(&mut c, 300 + g * 10 + 3, 1, 1000);
    }
    let st: Vec<Vec<u32>> = (0..3).map(|g| (1..=3).map(|b| 300 + g * 10 + b).collect()).collect();
    treq(&mut c, &st, &[101, 102, 103], 1000);
    c.tick(1000 + GRACE + 1000);
    assert!((0..3).all(|g| has(&c, 300 + g * 10 + 3)) && has(&c, 103), "whole block: a segment, kept");
    assert!(c.remove_key(NS, &k(323)));
    c.tick(1000 + GRACE + 3000);
    assert!(!has(&c, 303) && !has(&c, 313));
    assert_eq!(c.c.evict_orphan_state, 2);
    c.check();
}

#[test]
fn seg_segments_leave_together() {
    // turn 1 stored blocks 1..4 + state 4, turn 2 blocks 5..8 + state 8 (LRU: state 4 is the coldest entry). Pressure
    // takes the superseded state 4 alone (state 8 still uses blocks 1..4), then state 8 with every block only it makes
    // usable, at once: never attention without a state
    let mut c = hybrid(16, 1, |x| x.policy = PolicyKind::Lru);
    chain(&mut c, 4, &[4], 1000);
    for b in 5..=8 {
        hs(&mut c, 100 + b, 0, 2000);
    }
    hs(&mut c, 208, 1, 2000);
    treq(&mut c, &[(1..=8).map(|b| 200 + b).collect()], &(101..=108).collect::<Vec<_>>(), 2000);
    for x in 500..506 {
        hs(&mut c, x, 3, 3000);                // fills the pool
    }
    hs(&mut c, 506, 3, 3000);
    assert!(!has(&c, 204) && (101..=108).all(|b| has(&c, b)) && has(&c, 208));
    hs(&mut c, 507, 3, 3000);
    assert!(!has(&c, 208) && (101..=108).all(|b| !has(&c, b)), "the segment left whole");
    assert_eq!(c.c.seg_units, 1);
    assert_eq!(c.c.seg_unit_entries, 9);
    c.check();
}

#[test]
fn seg_young_uncovered_attention_held_under_pressure() {
    // a prompt being stored (attention ahead of its prompt-end state) is not taken while older entries can go
    let mut c = hybrid(8, 1, |_| {});
    hs(&mut c, 600, 3, 0);     // an unrelated old entry (unknown kind)
    for b in 1..=5 {
        hs(&mut c, 100 + b, 0, 1000);
    }
    treq(&mut c, &[(1..=5).map(|b| 200 + b).collect()], &(101..=105).collect::<Vec<_>>(), 1000);
    hs(&mut c, 601, 3, 2000);
    hs(&mut c, 602, 3, 2000);
    hs(&mut c, 603, 3, 2000);  // full: 600 goes, not the young chain
    assert!(!has(&c, 600) && (101..=105).all(|b| has(&c, b)));
    hs(&mut c, 205, 1, 3000);  // the prompt-end state arrives: all covered
    treq(&mut c, &[(1..=5).map(|b| 200 + b).collect()], &(101..=105).collect::<Vec<_>>(), 3000);
    c.tick(3000 + GRACE + 1000);
    assert!((101..=105).all(|b| has(&c, b)) && has(&c, 205));
    assert_eq!(c.c.evict_orphan_attn, 0);
}

#[test]
fn seg_lazy_attention_after_state_is_covered() {
    // lazy stores: the state is stored at once, the attention blocks only later when the GPU evicts them
    let mut c = hybrid(32, 1, |_| {});
    hs(&mut c, 204, 1, 1000);
    treq(&mut c, &[(1..=4).map(|b| 200 + b).collect()], &(101..=104).collect::<Vec<_>>(), 1000);
    for b in [104, 103, 102, 101] {
        hs(&mut c, 100 + (b - 100), 0, 5000);
    }
    c.tick(5000 + GRACE + 1000);
    assert!((101..=104).all(|b| has(&c, b)) && has(&c, 204));
    assert_eq!(c.c.evict_orphan_attn, 0);
}

#[test]
fn seg_tails_still_first_then_orphans() {
    let mut c = hybrid(12, 1, |_| {});
    chain(&mut c, 4, &[2], 1000);              // 103, 104: orphans after the grace
    hs(&mut c, 700, KIND_TAIL, 1000 + GRACE);  // a younger partial tail
    c.tick(1000 + GRACE + 10);
    // eager cleanup skips namespaces with live tails that carry no anchor (a tail may need that attention)
    assert!(has(&c, 103) && has(&c, 104) && has(&c, 700));
    for x in 801..807 {
        hs(&mut c, x, 3, 1000 + GRACE + 20);   // 6 free -> 0 free: no eviction yet
    }
    hs(&mut c, 807, 3, 1000 + GRACE + 30);     // needs one slot: the tail goes first
    assert!(!has(&c, 700) && has(&c, 103) && has(&c, 104));
    hs(&mut c, 808, 3, 1000 + GRACE + 40);     // then an orphan, before the segment {202 + 101, 102}
    assert!(!has(&c, 103) && !has(&c, 104), "an orphan takes the rest of its chain");
    assert!((101..=102).all(|b| has(&c, b)) && has(&c, 202));
    assert_eq!(c.c.evict_orphan_attn + c.c.cascade_evicted, 2);
    c.check();
}

#[test]
fn seg_age_sweep_by_units() {
    // a sweep drops a segment whole; a trunk shared by two branches stays until both branches are gone
    let mut c = hybrid(32, 1, |_| {});
    for b in [101, 102, 103, 104, 114] {
        hs(&mut c, b, 0, 1000);
    }
    hs(&mut c, 204, 1, 1000);
    hs(&mut c, 214, 1, 1000);
    treq(&mut c, &[vec![201, 202, 203, 204]], &[101, 102, 103, 104], 1000);
    treq(&mut c, &[vec![201, 202, 203, 214]], &[101, 102, 103, 114], 1000);
    // branch 114/214 used again later
    c.lookup(2, NS, &keys(&[101, 102, 103, 114]), false, 50_000);
    c.lookup(2, NS, &keys(&[214]), false, 50_000);
    c.tick(60_000);
    let n = c.evict_aged_by(&|_| 10_000, 0);
    assert_eq!(n, 2, "segment 104 + 204");
    assert!(!has(&c, 104) && !has(&c, 204) && (101..=103).all(|b| has(&c, b)) && has(&c, 214));
    let n = c.evict_aged_by(&|_| 60_000, 0);
    assert_eq!(n, 5, "then the branch with the trunk, now its own segment");
    assert_eq!(c.gauges().iter().find(|g| g.0 == "entries").unwrap().1, 0);
    c.check();
}

#[test]
fn seg_attention_only_namespace_untouched() {
    // an engine without GDN states (one KV group): never linked, no orphan rule
    let mut c = core(16);
    for b in 1..=4 {
        hs(&mut c, 100 + b, 0, 1000);
    }
    for _ in 0..8 {
        c.touch_conn(1, NS, &keys(&[101, 102, 103, 104]), 1000);
        c.release(1, &[]);
    }
    c.tick(1000 + 10 * GRACE);
    assert!((101..=104).all(|b| has(&c, b)));
    assert_eq!(c.c.evict_orphan_attn, 0);
}

#[test]
fn seg_lazy_copy_without_state_gets_no_grace() {
    // the chain's state left RAM; the GPU then evicts the blocks and the lazy store copies them: useless. The eager
    // scan leaves them until they are older than the grace (it must not race a long prefill whose state
    // comes last), but a RESERVE that needs room takes them first, young or not.
    let mut c = hybrid(8, 1, |_| {});
    hs(&mut c, 204, 1, 1000);
    treq(&mut c, &[(1..=4).map(|b| 200 + b).collect()], &(101..=104).collect::<Vec<_>>(), 1000);
    assert!(c.remove_key(NS, &k(204)));
    for b in 101..=104 {
        hs(&mut c, b, 0, 5000);
    }
    hs(&mut c, 300, 0, 5000);
    hs(&mut c, 301, 0, 5000);
    hs(&mut c, 302, 0, 5000);
    hs(&mut c, 303, 0, 5000);
    c.tick(7000);
    assert!((101..=104).all(|b| has(&c, b)), "eager scan: young orphans stay");
    hs(&mut c, 400, 0, 8000);                      // pool full: the RESERVE takes an orphan, not 300..303
    assert_eq!((101..=104).filter(|b| has(&c, *b)).count(), 0, "with the rest of its chain");
    assert!((300..=303).all(|b| has(&c, b)) && has(&c, 400));
    c.tick(5000 + GRACE + 2000);
    assert!((101..=104).all(|b| !has(&c, b)), "old orphans: cleaned");
    c.check();
}

#[test]
fn seg_long_prefill_attention_survives_until_its_state() {
    // as seen in production: one touch at the request's start, then attention stored block by block for minutes,
    // the prompt-end state last. Nothing of it may be cleaned in between, with room to spare.
    let mut c = hybrid(64, 1, |_| {});
    treq(&mut c, &[vec![240]], &(101..=140).collect::<Vec<_>>(), 1000);
    for b in 1..=40u32 {
        hs(&mut c, 100 + b, 0, 1000 + b as u64 * 6000);   // 6 s per block: 240 s of prefill
        c.tick(1000 + b as u64 * 6000 + 1000);
    }
    assert!((101..=140).all(|b| has(&c, b)), "attention kept during the prefill");
    hs(&mut c, 240, 1, 245_000);                        // the prompt-end state
    treq(&mut c, &[vec![240]], &(101..=140).collect::<Vec<_>>(), 246_000);
    c.tick(246_000 + 3 * GRACE);
    assert!((101..=140).all(|b| has(&c, b)) && has(&c, 240), "covered by its state: kept for good");
    assert_eq!(c.c.evict_orphan_attn, 0);
    c.check();
}

#[test]
fn heat_cold_goes_first_hot_refreshes() {
    for pk in [PolicyKind::Arc, PolicyKind::Lru] {
        let mut c = Core::new(cfg(4, pk));
        c.connect(1);
        c.hello(1, 10);
        for key in 1..=4 {
            assert_eq!(store(&mut c, 1, key, None, 1, key as u64), Status::Ok);
        }
        // 4 is the newest: demoted, it is the next victim instead of 1
        assert_eq!(c.heat_keys(NS, &[k(4), k(99)], false, 10), vec![Status::Ok, Status::NotFound]);
        assert_eq!(store(&mut c, 1, 5, None, 1, 11), Status::Ok);
        assert_eq!(c.lookup(1, NS, &[k(4)], false, 12).0, 0, "{pk:?}: demoted entry evicted");
        assert_eq!(c.lookup(1, NS, &[k(1)], false, 12).0, 1, "{pk:?}: oldest kept");
        // hot refresh: 1 survives the next store, 2 goes
        c.heat_keys(NS, &[k(1)], true, 13);
        assert_eq!(store(&mut c, 1, 6, None, 1, 14), Status::Ok);
        assert_eq!(c.lookup(1, NS, &[k(1)], false, 15).0, 1, "{pk:?}: refreshed entry kept");
        assert_eq!(c.lookup(1, NS, &[k(2)], false, 15).0, 0, "{pk:?}: next-oldest evicted");
        let h = c.heat_hist();
        assert_eq!(h.iter().map(|x| x.2).sum::<u64>(), 4, "{pk:?}: heat histogram covers every slot");
        assert_eq!(c.c.heat_cold, 1);
        assert_eq!(c.c.heat_hot, 1);
        c.check();
    }
}

#[test]
fn copy_guard_sees_reuse_rewrite_and_moves() {
    // an unleased copier compares the entry's identity before and after its copy
    let mut c = core(8);
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    let g0 = c.copy_guard(NS, &k(1)).unwrap();
    assert_eq!(c.copy_guard(NS, &k(1)), Some(g0.clone()), "unchanged: same guard");
    assert!(c.remove_key(NS, &k(1)));
    assert_eq!(store(&mut c, 1, 1, None, 1, 1), Status::Ok);      // same key stored again
    assert_ne!(c.copy_guard(NS, &k(1)), Some(g0), "evicted + stored again: different guard");
    let g1 = c.copy_guard(NS, &k(1)).unwrap();
    let r = g1.1.clone();
    assert!(c.begin_rewrite(NS, &k(1), &r));
    assert!(c.copy_guard(NS, &k(1)).unwrap().3, "rewriting is visible");
    c.end_rewrite(NS, &k(1), true);
    assert_ne!(c.copy_guard(NS, &k(1)), Some(g1), "rewritten in place: different guard");
    c.check();
}

// ------------------------------------------------------------ cascade + integrity checksums

#[test]
fn cascade_leased_descendant_goes_when_released() {
    let mut c = hybrid(32, 1, |_| {});
    chain(&mut c, 3, &[3], 1000);
    let (n, lease, _) = c.lookup(2, NS, &keys(&[103]), true, 1100);
    assert_eq!(n, 1);
    assert!(c.remove_key(NS, &k(102)));
    assert!(has(&c, 103), "leased: a load may be reading it");
    assert!(!has(&c, 203) && has(&c, 101));
    c.release(2, &[lease]);
    c.tick(1300);
    assert!(!has(&c, 103), "gone once its lease ended");
    c.check();
}

#[test]
fn checksum_store_verify_corrupt_cascades() {
    let mut c = hybrid(32, 1, |_| {});
    chain(&mut c, 3, &[3], 1000);
    let cands = c.checksum_candidates(0, 100);
    assert_eq!(cands.len(), 4, "all never-checksummed entries");
    for (ns, key, ..) in &cands {
        let g = c.copy_guard(*ns, key).unwrap();
        assert_eq!(c.checksum_result(*ns, key, &g, Some(7), 2000), ChecksumResult::Stored);
    }
    assert!(c.checksum_candidates(2000, 100).is_empty(), "all checksummed, none due");
    assert_eq!(c.checksum_candidates(2001, 100).len(), 4, "due for re-verification");
    let g = c.copy_guard(NS, &k(103)).unwrap();
    assert_eq!(c.checksum_result(NS, &k(103), &g, Some(7), 3000), ChecksumResult::Verified);
    // a mismatch on a leased entry waits for the next pass
    let (_, lease, _) = c.lookup(2, NS, &keys(&[101]), true, 3000);
    let g = c.copy_guard(NS, &k(101)).unwrap();
    assert_eq!(c.checksum_result(NS, &k(101), &g, Some(8), 3000), ChecksumResult::Busy);
    c.release(2, &[lease]);
    // the entry changed while copied: the result says nothing about it
    let g = c.copy_guard(NS, &k(102)).unwrap();
    assert!(c.remove_key(NS, &k(203)));
    let g2 = (g.0 + 1_000_000, g.1.clone(), g.2, g.3);
    assert_eq!(c.checksum_result(NS, &k(102), &g2, Some(8), 3000), ChecksumResult::Raced);
    // corrupt: thrown out with the rest of its chain
    let g = c.copy_guard(NS, &k(101)).unwrap();
    assert_eq!(c.checksum_result(NS, &k(101), &g, Some(8), 3000), ChecksumResult::Corrupt);
    assert!((101..=103).all(|b| !has(&c, b)));
    assert_eq!((c.c.cksum_corrupt, c.c.cascade_evicted), (1, 2));
    c.check();
}

#[test]
fn checksum_undecodable_is_corrupt() {
    let mut c = core(8);
    hs(&mut c, 1, 0, 1000);
    let g = c.copy_guard(NS, &k(1)).unwrap();
    assert_eq!(c.checksum_result(NS, &k(1), &g, None, 1000), ChecksumResult::Corrupt);
    assert!(!has(&c, 1));
}

#[test]
fn checksum_corrupt_parent_with_leased_child() {
    // explicit parent links: the corrupt root goes, its leased child stays detached until its lease ends
    let mut c = core(8);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 100)], 1000)[0].0, Status::Ok);
    c.commit(1, NS, &[k(1)], 1000);
    assert_eq!(c.reserve(1, NS, &[item(2, Some(1), 100)], 1000)[0].0, Status::Ok);
    c.commit(1, NS, &[k(2)], 1000);
    assert_eq!(c.reserve(1, NS, &[item(3, Some(2), 100)], 1000)[0].0, Status::Ok);
    c.commit(1, NS, &[k(3)], 1000);
    let (_, lease, _) = c.lookup(2, NS, &keys(&[2]), true, 1000);
    let g = c.copy_guard(NS, &k(1)).unwrap();
    assert_eq!(c.checksum_result(NS, &k(1), &g, None, 1000), ChecksumResult::Corrupt);
    assert!(!has(&c, 1) && !has(&c, 3) && has(&c, 2));
    c.check();
    c.release(2, &[lease]);
    c.tick(1100);
    assert!(!has(&c, 2));
    c.check();
}

#[test]
fn chain_gauges_describe_the_index() {
    let mut c = hybrid(32, 1, |_| {});
    chain(&mut c, 6, &[3, 6], 1000);
    let g = |c: &Core| ["chain_roots", "chain_linked_entries", "chain_unknown_entries", "chain_dangling_entries",
                        "chain_chains", "chain_depth_max", "cksum_unchecked_entries"].map(|n| gauge(c, n));
    assert_eq!(g(&c), [1, 7, 0, 0, 1, 6, 8], "root 101, links 102..106 + 2 anchors, one chain of 6, nothing hashed");
    // a block stored after its predecessor left (an attention block's link is known from the touch run)
    assert!(c.remove_key(NS, &k(106)));
    assert!(c.remove_key(NS, &k(105)));
    hs(&mut c, 106, 0, 2000);
    assert_eq!(gauge(&c, "chain_dangling_entries"), 1);
    assert_eq!(gauge(&c, "chain_depth_max"), 4);
}

#[test]
fn seg_merged_runs_do_not_hide_the_group_count() {
    // several requests touched in one scheduler step (nothing between them) arrive as one run of k x groups batches;
    // when merged runs are the most common length, single-request runs must still link (43 % skipped)
    let mut c = hybrid(64, 1, |_| {});
    let skipped0 = c.c.seg_runs_skipped;
    for i in 0..30u32 {
        let a = 1000 + i * 10;
        let b = 5000 + i * 10;
        c.touch_conn(1, NS, &keys(&[a + 2]), 2000);           // request A: state batch, attention batch
        c.touch_conn(1, NS, &keys(&[a, a + 1, a + 2]), 2000);
        if i % 3 != 0 {
            c.touch_conn(1, NS, &keys(&[b + 2]), 2000);       // request B in the same step: the run is 4 batches
            c.touch_conn(1, NS, &keys(&[b, b + 1, b + 2]), 2000);
        }
        if i % 5 == 0 {
            for x in [7000 + i * 10] {                         // three requests at once
                c.touch_conn(1, NS, &keys(&[x + 2]), 2000);
                c.touch_conn(1, NS, &keys(&[x, x + 1, x + 2]), 2000);
            }
        }
        c.release(1, &[]);
    }
    assert_eq!(c.c.seg_runs_skipped - skipped0, 0, "runs skipped");
}

#[test]
fn seg_odd_runs_do_not_break_learning() {
    // 1 in 5 runs has an odd length (a lone batch): those are skipped, the others still link
    let mut c = hybrid(64, 1, |_| {});
    let (l0, s0) = (c.c.seg_runs_linked, c.c.seg_runs_skipped);
    for i in 0..40u32 {
        let a = 1000 + i * 10;
        c.touch_conn(1, NS, &keys(&[a + 2]), 2000);
        if i % 5 != 4 {
            c.touch_conn(1, NS, &keys(&[a, a + 1, a + 2]), 2000);
        }
        c.release(1, &[]);
    }
    assert_eq!((c.c.seg_runs_linked - l0, c.c.seg_runs_skipped - s0, c.c.seg_skip_len), (32, 8, 8));
}

// ------------------------------------------------------------ gaps found by mutation testing (cargo mutants)

/// Per-entry eviction order (--seg-evict off) on an LRU policy: tests below compare against plain LRU.
fn cfg_flat(n_slots: u32) -> Config {
    let mut c = cfg(n_slots, PolicyKind::Lru);
    c.seg_evict = false;
    c
}

fn core_with(cf: Config) -> Core {
    let mut c = Core::new(cf);
    for conn in 1..=3 {
        c.connect(conn);
        c.hello(conn, conn as u16 * 10);
    }
    c
}

fn store_ns(c: &mut Core, ns: u32, key: u32, slots: u64, now: u64) -> Status {
    let r = c.reserve(1, ns, &[item(key, None, slots * 100)], now);
    if r[0].0 != Status::Ok {
        return r[0].0;
    }
    c.commit(1, ns, &[k(key)], now).0[0]
}

fn has_ns(c: &Core, ns: u32, key: u32) -> bool {
    c.contains(ns, &k(key))
}

#[test]
fn hello_required_and_per_connection() {
    let mut c = Core::new(cfg(4, PolicyKind::Arc));
    assert!(!c.has_hello(1), "unknown connection");
    c.connect(1);
    assert!(!c.has_hello(1), "connected, no HELLO yet");
    c.hello(1, 7);
    assert!(c.has_hello(1));
    assert!(!c.has_hello(2));
    // connect() resets a reused connection id (a new socket): no HELLO, no leases
    c.connect(1);
    assert!(!c.has_hello(1));
}

#[test]
fn disconnect_drops_only_its_waiters() {
    let mut c = core(10);
    assert_eq!(c.reserve(1, NS, &[item(5, None, 100)], 0)[0].0, Status::Ok);
    assert_eq!(c.wait(2, 77, NS, k(5), 10_000, 0), None);
    assert_eq!(c.wait(3, 78, NS, k(5), 10_000, 0), None);
    c.disconnect(2);
    let (st, notes) = c.commit(1, NS, &[k(5)], 1);
    assert_eq!(st[0], Status::Ok);
    assert_eq!(notes, vec![Notify { conn: 3, req_id: 78, status: Status::Ok }], "only the live waiter is told");
}

#[test]
fn release_of_another_connections_lease_is_refused() {
    let mut c = core(4);
    assert_eq!(store(&mut c, 1, 1, None, 4, 0), Status::Ok);
    let (n, lease, _) = c.lookup(1, NS, &[k(1)], true, 1);
    assert_eq!(n, 1);
    assert_eq!(c.release(2, &[lease]), vec![Status::NotOwner]);
    assert_eq!(c.release(2, &[lease + 100]), vec![Status::NotFound]);
    // still leased: a full-pool reserve cannot take it
    assert_eq!(store(&mut c, 2, 2, None, 1, 2), Status::Full);
    assert_eq!(c.release(1, &[lease]), vec![Status::Ok]);
    assert_eq!(store(&mut c, 2, 2, None, 1, 3), Status::Ok);
    c.check();
}

#[test]
fn heat_on_a_reservation_is_not_found() {
    let mut c = core(4);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 100)], 0)[0].0, Status::Ok);
    assert_eq!(c.heat_keys(NS, &[k(1)], false, 1), vec![Status::NotFound]);
    assert_eq!(c.heat_keys(NS, &[k(1)], true, 1), vec![Status::NotFound]);
    assert_eq!(c.c.heat_cold + c.c.heat_hot, 0);
    assert_eq!(c.commit(1, NS, &[k(1)], 2).0[0], Status::Ok);
    c.check();
}

#[test]
fn whole_pool_entry_fits_exactly() {
    let mut c = core(8);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 800)], 0)[0].0, Status::Ok, "8 slots in an 8-slot pool");
    assert_eq!(c.reserve(1, NS, &[item(2, None, 801)], 0)[0].0, Status::TooBig);
    // parked slots shrink what fits
    let mut c = core(8);
    assert_eq!(c.cull(2).iter().map(|r| r.len).sum::<u32>(), 2);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 700)], 0)[0].0, Status::TooBig);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 600)], 0)[0].0, Status::Ok);
    c.check();
}

#[test]
fn slots_being_freed_are_not_handed_out_until_the_punch_is_done() {
    let mut c = core(8);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 200)], 0)[0].0, Status::Ok);   // slots 0-1
    let free = c.free_runs();
    let took = c.begin_freeing(&free);
    assert_eq!(took, free);
    assert!(c.begin_freeing(&free).is_empty(), "already taken: nothing twice");
    assert_eq!(c.live_and_high_water().0, 2, "the slots being freed are not live");
    c.check();
    // the pool looks full to a RESERVE meanwhile (the reserved entry is not evictable)
    assert_ne!(c.reserve(1, NS, &[item(2, None, 100)], 0)[0].0, Status::Ok);
    c.check();
    c.end_freeing(&took);
    c.end_freeing(&took);   // twice: ignored
    c.check();
    assert_eq!(c.reserve(1, NS, &[item(2, None, 600)], 0)[0].0, Status::Ok);
    c.check();
}

#[test]
fn shared_hint_on_exists_only_with_the_class_on() {
    // class off: a hinted RESERVE of a committed key does not mark it
    let mut c = core(8);
    assert_eq!(store_meta(&mut c, 1, None, 0, 0), Status::Ok);
    assert_eq!(store_meta(&mut c, 1, None, ENTRY_SHARED_HINT, 1), Status::Exists);
    assert_eq!(gauge(&c, "shared_entries"), 0);
    // class on: a hinted RESERVE of a key still being written does not mark it either (only committed entries)
    let mut c = core_with(cfg_shared(8, 4));
    let mut it = item(2, None, 100);
    it.flags = ENTRY_SHARED_HINT;
    assert_eq!(c.reserve(1, NS, &[it.clone()], 0)[0].0, Status::Ok);
    assert_eq!(c.reserve(2, NS, &[it], 0)[0].0, Status::Busy);
    assert_eq!(gauge(&c, "shared_entries"), 0);
    // ... and once committed, an unhinted RESERVE does not mark it
    assert_eq!(c.commit(1, NS, &[k(2)], 1).0[0], Status::Ok);
    assert_eq!(gauge(&c, "shared_entries"), 1, "hint given at reserve applies at commit");
    assert_eq!(store_meta(&mut c, 3, None, 0, 2), Status::Ok);
    assert_eq!(c.reserve(2, NS, &[item(3, None, 100)], 3)[0].0, Status::Exists);
    assert_eq!(gauge(&c, "shared_entries"), 1);
    c.check();
}

#[test]
fn shared_prefix_without_depth_cap() {
    // shared_max_depth 0 = no cap: a deep junction qualifies by its continuations alone
    let mut cf = cfg_shared(64, 32);
    cf.shared_max_depth = 0;
    let mut c = core_with(cf);
    let mut now = 0;
    for d in 1..25u32 {
        assert_eq!(store_meta(&mut c, d, if d > 1 { Some(d - 1) } else { None }, 0, now), Status::Ok);
        now += 1;
    }
    for x in 0..3u32 {
        assert_eq!(store_meta(&mut c, 100 + x, Some(24), 0, now), Status::Ok);
        now += 1;
    }
    assert!(c.entry_info(NS, &k(24)).unwrap().shared, "deep junction shared with no depth cap");
    assert!(c.entry_info(NS, &k(1)).unwrap().shared, "its root too");
    c.check();
}

#[test]
fn shared_class_slots_return_on_removal() {
    let mut c = core_with(cfg_shared(8, 4));
    assert_eq!(store_meta(&mut c, 1, None, ENTRY_SHARED_HINT, 0), Status::Ok);
    assert_eq!(gauge(&c, "shared_slots"), 1);
    assert!(c.remove_key(NS, &k(1)));
    assert_eq!(gauge(&c, "shared_slots"), 0, "a removed member leaves the class budget");
    // the budget is usable again: four new members fit without demotions
    for key in 10..14 {
        assert_eq!(store_meta(&mut c, key, None, ENTRY_SHARED_HINT, 1), Status::Ok);
    }
    assert_eq!(gauge(&c, "shared_slots"), 4);
    assert_eq!(c.c.shared_demoted, 0);
    c.check();
}

#[test]
fn remove_key_and_remove_corrupt_spare_leased_and_parents() {
    let mut c = core(8);
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    assert_eq!(store(&mut c, 1, 2, Some(1), 1, 0), Status::Ok);
    let (_, lease, _) = c.lookup(1, NS, &[k(2)], true, 1);
    assert!(!c.remove_key(NS, &k(2)), "leased");
    assert!(!c.remove_key(NS, &k(1)), "has a child");
    let id2 = c.scrub_candidates(&HashSet::new()).into_iter().find(|x| x.5 == k(2)).unwrap().0;
    assert!(c.has_entry(id2));
    assert!(!c.remove_corrupt(id2), "leased");
    assert!(c.contains(NS, &k(2)) && c.contains(NS, &k(1)));
    c.release(1, &[lease]);
    assert!(c.remove_corrupt(id2));
    assert!(!c.has_entry(id2));
    assert!(!c.remove_corrupt(id2), "gone already");
    assert_eq!(c.c.corrupt, 1);
    assert!(c.remove_key(NS, &k(1)));
    assert!(!c.remove_key(NS, &k(1)));
    // a reservation is not removable either (its writer owns it)
    assert_eq!(c.reserve(1, NS, &[item(3, None, 100)], 2)[0].0, Status::Ok);
    assert!(!c.remove_key(NS, &k(3)));
    c.check();
}

#[test]
fn scrub_candidates_committed_newest_first_skipping_done() {
    let mut c = core(8);
    for key in 1..=3 {
        assert_eq!(store_kind(&mut c, 1, key, key as u8, 1, key as u64), Status::Ok);
    }
    assert_eq!(c.reserve(1, NS, &[item(9, None, 100)], 5)[0].0, Status::Ok);
    let v = c.scrub_candidates(&HashSet::new());
    assert_eq!(v.iter().map(|x| x.5).collect::<Vec<_>>(), vec![k(3), k(2), k(1)], "committed only, newest first");
    assert!(v.iter().all(|x| x.1 == NS && x.4 == 1 && x.3 as u32 == u32::from_le_bytes(x.5[..4].try_into().unwrap())));
    assert_eq!(v[0].2, c.entry_info(NS, &k(3)).unwrap().ranges);
    let done: HashSet<_> = [v[0].0].into_iter().collect();
    assert_eq!(c.scrub_candidates(&done).len(), 2);
}

#[test]
fn compress_candidates_filter_and_order() {
    let mut c = core(16);
    for key in 1..=6 {
        assert_eq!(store(&mut c, 1, key, None, 1, 10 + key as u64), Status::Ok);
    }
    let (_, lease, _) = c.lookup(1, NS, &[k(2)], true, 12);     // leased (and used at 12)
    let r3 = c.entry_info(NS, &k(3)).unwrap().ranges;
    assert!(c.begin_rewrite(NS, &k(3), &r3));                    // being rewritten
    let r4 = c.entry_info(NS, &k(4)).unwrap().ranges;
    assert!(c.begin_rewrite(NS, &k(4), &r4));
    c.end_rewrite(NS, &k(4), true);                               // compressed already
    assert_eq!(c.reserve(1, NS, &[item(7, None, 100)], 0)[0].0, Status::Ok);   // reserved (last_used 0)
    // used before 16: 1 (11), 5 (15); 6 (16) is not before 16
    let v = c.compress_candidates(16, 10);
    assert_eq!(v.iter().map(|x| x.1).collect::<Vec<_>>(), vec![k(1), k(5)]);
    assert_eq!(v[0], (NS, k(1), c.entry_info(NS, &k(1)).unwrap().ranges, 100));
    assert_eq!(c.compress_candidates(16, 1).len(), 1);
    assert_eq!(c.compress_candidates(11, 10).len(), 0);
    assert!(c.is_compressed(NS, &k(4)) && !c.is_compressed(NS, &k(1)));
    c.release(1, &[lease]);
    c.end_rewrite(NS, &k(3), false);
    assert!(c.compress_candidates(100, 10).iter().all(|x| x.1 != k(3) && x.1 != k(4)), "done = never again");
    c.check();
}

#[test]
fn begin_rewrite_conditions_and_lookup_during_rewrite() {
    let mut c = core(8);
    for key in 1..=3 {
        assert_eq!(store(&mut c, 1, key, None, 1, 0), Status::Ok);
    }
    let r1 = c.entry_info(NS, &k(1)).unwrap().ranges;
    let r2 = c.entry_info(NS, &k(2)).unwrap().ranges;
    assert!(!c.begin_rewrite(NS, &k(1), &r2), "other ranges");
    let (_, lease, _) = c.lookup(1, NS, &[k(1)], true, 1);
    assert!(!c.begin_rewrite(NS, &k(1), &r1), "leased");
    c.release(1, &[lease]);
    assert!(c.begin_rewrite(NS, &k(1), &r1));
    assert!(!c.begin_rewrite(NS, &k(1), &r1), "already rewriting");
    assert_eq!(c.lookup(1, NS, &[k(1)], true, 2).0, 0, "a rewriting entry misses");
    assert!(!c.remove_key(NS, &k(1)), "a rewriting entry is not evictable");
    c.end_rewrite(NS, &k(1), true);
    assert_eq!(c.c.compressed, 1);
    assert!(!c.begin_rewrite(NS, &k(1), &r1), "compressed already");
    assert_eq!(c.lookup(1, NS, &[k(1)], false, 3).0, 1, "visible again after the rewrite");
    c.end_rewrite(NS, &k(2), false);
    assert_eq!(c.c.compressed, 1, "not compressed: not counted");
    assert_eq!(c.reserve(1, NS, &[item(9, None, 100)], 4)[0].0, Status::Ok);
    let r9 = c.reserve(1, NS, &[item(9, None, 100)], 4);
    assert_eq!(r9[0].0, Status::Busy);
    assert!(!c.begin_rewrite(NS, &k(9), &[]), "a reservation");
    assert!(!c.begin_rewrite(NS, &k(42), &r1), "unknown key");
    c.check();
}

#[test]
fn relocation_moves_highest_entry_down_and_checks_identity() {
    let mut c = core(8);
    for key in 1..=4 {
        assert_eq!(store(&mut c, 1, key, None, 1, key as u64), Status::Ok);
    }
    assert!(c.remove_key(NS, &k(1)));
    assert!(c.remove_key(NS, &k(2)));
    let top = c.entry_info(NS, &k(4)).unwrap().ranges;
    let (ns, key, old, new) = c.relocation_candidate().expect("a move");
    assert_eq!((ns, key, &old), (NS, k(4), &top), "the entry reaching highest moves first");
    assert!(new.iter().all(|r| r.start + r.len <= old[0].start), "to slots below it");
    assert!(c.finish_relocation(ns, &key, &old, new.clone()));
    assert_eq!(c.entry_info(NS, &k(4)).unwrap().ranges, new);
    assert_eq!(c.c.relocations, 1);
    c.check();
    // the entry changed between candidate and finish (leased, or moved): the new slots go back
    let (ns, key, old, new) = c.relocation_candidate().expect("3 moves down too");
    let (_, lease, _) = c.lookup(1, NS, &[key], true, 9);
    assert!(!c.finish_relocation(ns, &key, &old, new.clone()), "leased meanwhile");
    assert_eq!(c.entry_info(NS, &key).unwrap().ranges, old);
    c.check();
    c.release(1, &[lease]);
    let (ns, key, _, new) = c.relocation_candidate().unwrap();
    assert!(!c.finish_relocation(ns, &key, &new, new.clone()), "other ranges");
    c.check();
    let (_, _, old, new) = c.relocation_candidate().unwrap();
    assert!(!c.finish_relocation(ns, &k(77), &old, new), "unknown key");
    c.check();
    // a leased or rewriting entry is never a candidate; a packed pool has none
    let mut c = core(4);
    for key in 1..=2 {
        assert_eq!(store(&mut c, 1, key, None, 1, 0), Status::Ok);
    }
    assert!(c.relocation_candidate().is_none(), "packed");
    assert!(c.remove_key(NS, &k(1)));
    let (_, lease, _) = c.lookup(1, NS, &[k(2)], true, 1);
    assert!(c.relocation_candidate().is_none(), "leased");
    c.release(1, &[lease]);
    let r2 = c.entry_info(NS, &k(2)).unwrap().ranges;
    assert!(c.begin_rewrite(NS, &k(2), &r2));
    assert!(c.relocation_candidate().is_none(), "rewriting");
    c.check();
}

#[test]
fn insert_committed_fits_exactly_and_never_overwrites() {
    let mut c = core(4);
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    let r = c.insert_committed(NS, k(2), 0, 1, 5, vec![], 300, 1, 1).expect("3 slots free: fits exactly");
    assert_eq!(r.iter().map(|x| x.len).sum::<u32>(), 3);
    assert!(c.insert_committed(NS, k(2), 0, 1, 5, vec![], 100, 1, 1).is_none(), "exists");
    assert!(c.insert_committed(NS, k(3), 0, 1, 5, vec![], 100, 1, 1).is_none(), "full: no eviction");
    assert!(c.contains(NS, &k(1)) && c.contains(NS, &k(2)));
    assert!(c.remove_key(NS, &k(1)));
    assert!(c.insert_committed(NS, k(3), 0, 1, 5, vec![], 100, 1, 1).is_some());
    assert!(c.contains(NS, &k(2)) && c.contains(NS, &k(3)), "a new insert keeps the others");
    assert_eq!(c.lookup(1, NS, &[k(2)], false, 2).0, 1);
    c.check();
}

#[test]
fn restored_hot_entry_outlives_a_cold_one_under_arc() {
    // insert_committed with hits > 1 starts among ARC's frequent entries: a newer one-hit entry goes first
    let mut c = core(3);
    assert!(c.insert_committed(NS, k(1), 0, 1, 5, vec![], 100, 5, 1).is_some());
    assert!(c.insert_committed(NS, k(2), 0, 1, 5, vec![], 100, 1, 2).is_some());
    assert!(c.insert_committed(NS, k(3), 0, 1, 5, vec![], 100, 1, 3).is_some());
    assert!(c.make_room_for(NS, 100));
    assert!(c.contains(NS, &k(1)), "the hot restored entry stays");
    assert!(!c.contains(NS, &k(2)), "the cold one goes");
    c.check();
}

#[test]
fn make_room_for_reports_failure() {
    let mut c = core(2);
    assert!(c.make_room_for(NS, 200), "free already");
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    assert_eq!(store(&mut c, 1, 2, None, 1, 1), Status::Ok);
    let (_, l1, _) = c.lookup(1, NS, &[k(1)], true, 2);
    let (_, l2, _) = c.lookup(1, NS, &[k(2)], true, 2);
    assert!(!c.make_room_for(NS, 100), "everything leased");
    c.release(1, &[l1]);
    assert!(c.make_room_for(NS, 100));
    assert!(!c.contains(NS, &k(1)) && c.contains(NS, &k(2)));
    assert!(!c.make_room_for(NS, 200), "the other is leased");
    c.release(1, &[l2]);
    c.check();
}

#[test]
fn evict_coldest_frees_what_is_asked() {
    let mut c = core(8);
    for key in 1..=6 {
        assert_eq!(store(&mut c, 1, key, None, 1, key as u64), Status::Ok);
    }
    assert_eq!(c.evict_coldest(0), 0);
    assert_eq!(c.evict_coldest(2), 2);
    assert!(!c.contains(NS, &k(1)) && !c.contains(NS, &k(2)) && c.contains(NS, &k(3)));
    let (_, l, _) = c.lookup(1, NS, &[k(3), k(4), k(5), k(6)], true, 9);
    assert_eq!(c.evict_coldest(3), 0, "everything left is leased");
    c.release(1, &[l]);
    assert_eq!(c.evict_coldest(100), 4);
    c.check();
}

#[test]
fn cull_parks_free_slots_then_evicts() {
    let mut c = core(8);
    for key in 1..=6 {
        assert_eq!(store(&mut c, 1, key, None, 1, key as u64), Status::Ok);
    }
    assert!(c.cull(0).is_empty());
    let p = c.cull(2);
    assert_eq!(p.iter().map(|r| r.len).sum::<u32>(), 2, "the two free slots");
    assert_eq!(c.parked_slots(), 2);
    assert_eq!(c.parked_runs().iter().map(|r| r.len).sum::<u32>(), 2);
    assert_eq!(c.entries.len(), 6, "nothing evicted for them");
    assert!(c.cull(2).is_empty(), "already parked");
    let p = c.cull(4);
    assert_eq!(p.iter().map(|r| r.len).sum::<u32>(), 2);
    assert!(!c.contains(NS, &k(1)) && !c.contains(NS, &k(2)) && c.contains(NS, &k(3)), "the coldest two evicted");
    assert_eq!(c.live_and_high_water().0, 4);
    c.unpark(&c.parked_runs());
    assert_eq!(c.parked_slots(), 0);
    assert_eq!(c.free_runs().iter().map(|r| r.len).sum::<u32>(), 4);
    c.check();
}

#[test]
fn live_slots_and_high_water() {
    let mut c = core(8);
    assert_eq!(c.live_and_high_water(), (0, 0));
    assert_eq!(store(&mut c, 1, 1, None, 2, 0), Status::Ok);
    assert_eq!(c.reserve(1, NS, &[item(2, None, 300)], 0)[0].0, Status::Ok);
    let hw = c.entries.values().flat_map(|e| e.ranges.iter()).map(|r| r.start + r.len).max().unwrap();
    assert_eq!(c.live_and_high_water(), (5, hw));
    assert_eq!(c.free_runs().iter().map(|r| r.len).sum::<u32>(), 3);
    c.check();
}

#[test]
fn committed_keys_and_ancestors() {
    let mut c = core(8);
    assert_eq!(store(&mut c, 1, 1, None, 1, 5), Status::Ok);
    assert_eq!(store(&mut c, 1, 2, Some(1), 1, 6), Status::Ok);
    assert_eq!(store(&mut c, 1, 3, Some(2), 1, 7), Status::Ok);
    assert_eq!(c.reserve(1, NS, &[item(4, Some(3), 100)], 8)[0].0, Status::Ok);
    let mut ck = c.committed_keys();
    ck.sort_by_key(|x| x.2);
    assert_eq!(ck, vec![(NS, k(1), 5), (NS, k(2), 6), (NS, k(3), 7)]);
    assert_eq!(c.ancestors(NS, &k(3)), vec![k(2), k(1)], "parent first");
    assert_eq!(c.ancestors(NS, &k(4)), vec![k(3), k(2), k(1)]);
    assert!(c.ancestors(NS, &k(1)).is_empty());
    assert!(c.ancestors(NS, &k(99)).is_empty());
}

#[test]
fn flat_order_stale_unhit_first() {
    for seg in [false, true] {
        stale_unhit_first(seg);
    }
}

fn stale_unhit_first(seg: bool) {
    let mut cf = cfg_flat(2);
    cf.seg_evict = seg;
    cf.stale_unhit_ms = 1000;
    let mut c = core_with(cf);
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    assert_eq!(c.lookup(1, NS, &[k(1)], false, 1).0, 1);       // hit once, used at 1
    assert_eq!(store(&mut c, 1, 2, None, 1, 10), Status::Ok);   // never hit
    // plain LRU would take 1 (used at 1); 2 is never hit and 1990 ms old: stale, it goes first
    assert_eq!(store(&mut c, 1, 3, None, 1, 2000), Status::Ok);
    assert!(c.contains(NS, &k(1)) && !c.contains(NS, &k(2)));
    // not stale yet (younger than stale_unhit_ms): plain order again
    assert_eq!(store(&mut c, 1, 4, None, 1, 2500), Status::Ok);
    assert!(!c.contains(NS, &k(1)) && c.contains(NS, &k(3)));
    c.check();
}

#[test]
fn namespace_cap_evicts_from_itself() {
    for seg in [false, true] {
        let mut cf = cfg_flat(4);
        cf.seg_evict = seg;
        cf.ns_cap_slots = 2;
        let mut c = core_with(cf);
        assert_eq!(store_ns(&mut c, 2, 1, 1, 0), Status::Ok);
        assert_eq!(store_ns(&mut c, 2, 2, 1, 1), Status::Ok);
        assert_eq!(store_ns(&mut c, 1, 3, 1, 2), Status::Ok);
        assert_eq!(store_ns(&mut c, 1, 4, 1, 3), Status::Ok);
        // ns 1 at its cap (2): its own oldest goes, not ns 2's older entries
        assert_eq!(store_ns(&mut c, 1, 5, 1, 4), Status::Ok);
        assert!(has_ns(&c, 2, 1) && has_ns(&c, 2, 2), "seg {seg}: other namespace kept");
        assert!(!has_ns(&c, 1, 3) && has_ns(&c, 1, 4), "seg {seg}: own oldest evicted");
        // below its cap ns 2 stores by plain order (oldest: ns 2's key 1)
        c.ns_c.clear();
        assert!(c.remove_key(1, &k(4)));
        assert_eq!(store_ns(&mut c, 2, 6, 1, 5), Status::Ok);    // free slot
        assert_eq!(store_ns(&mut c, 1, 7, 1, 6), Status::Ok);    // ns 1 holds 1 < cap: plain LRU
        assert!(!has_ns(&c, 2, 1), "seg {seg}: below the cap: plain order");
        c.check();
    }
}

#[test]
fn namespace_floor_is_spared() {
    for seg in [false, true] {
        let mut cf = cfg_flat(4);
        cf.seg_evict = seg;
        cf.ns_floor_slots = 1;
        let mut c = core_with(cf);
        assert_eq!(store_ns(&mut c, 2, 1, 1, 0), Status::Ok);     // ns 2 at its floor (1 slot), the oldest
        for (i, key) in (2..=4).enumerate() {
            assert_eq!(store_ns(&mut c, 1, key, 1, 1 + i as u64), Status::Ok);
        }
        assert_eq!(store_ns(&mut c, 1, 5, 1, 9), Status::Ok);
        assert!(has_ns(&c, 2, 1), "seg {seg}: namespace at its floor spared");
        assert!(!has_ns(&c, 1, 2), "seg {seg}: oldest above-floor entry evicted");
        // above its floor ns 2 is evicted from like any other
        assert_eq!(store_ns(&mut c, 2, 6, 1, 10), Status::Ok);    // evicts ns 1's key 3 (ns 2 still at floor)
        assert!(!has_ns(&c, 1, 3));
        assert_eq!(store_ns(&mut c, 1, 7, 1, 11), Status::Ok);    // ns 2 holds 2 > 1: its oldest (key 1) goes
        assert!(!has_ns(&c, 2, 1) && has_ns(&c, 2, 6), "seg {seg}: above the floor: plain order");
        c.check();
    }
}

#[test]
fn flat_order_tails_then_shared_last() {
    for seg in [false, true] {
        tails_then_shared_last(seg);
    }
}

fn tails_then_shared_last(seg: bool) {
    let mut cf = cfg_flat(3);
    cf.seg_evict = seg;
    cf.shared_slots = 2;
    let mut c = core_with(cf);
    assert_eq!(store_meta(&mut c, 1, None, ENTRY_SHARED_HINT, 0), Status::Ok);   // shared, the oldest
    assert_eq!(store(&mut c, 1, 2, None, 1, 1), Status::Ok);
    assert_eq!(store_kind(&mut c, 1, 3, KIND_TAIL, 1, 2), Status::Ok);           // a tail, the newest
    assert_eq!(store(&mut c, 1, 4, None, 1, 3), Status::Ok);
    assert!(!c.contains(NS, &k(3)), "the tail goes first");
    assert_eq!(store(&mut c, 1, 5, None, 1, 4), Status::Ok);
    assert!(c.contains(NS, &k(1)) && !c.contains(NS, &k(2)), "the shared entry outlives an unshared newer one");
    // only shared left evictable: it goes (with_shared pass)
    let (_, l, _) = c.lookup(1, NS, &[k(4), k(5)], true, 5);
    assert_eq!(store(&mut c, 1, 6, None, 1, 6), Status::Ok);
    assert!(!c.contains(NS, &k(1)));
    c.release(1, &[l]);
    c.check();
}

#[test]
fn flat_age_sweeps_skip_protected_and_leased() {
    for seg in [false, true] {
        age_sweeps(seg);
    }
}

fn age_sweeps(seg: bool) {
    let mut cf = cfg_flat(8);
    cf.seg_evict = seg;
    let mut c = core_with(cf);
    for key in 1..=5 {
        assert_eq!(store(&mut c, 1, key, None, 1, key as u64 * 10), Status::Ok);
    }
    let (_, l, _) = c.lookup(1, NS, &[k(1)], true, 10);            // leased, used at 10
    // evict_aged: last used before 35 (1 leased; 2, 3), at most 1 slot
    assert_eq!(c.evict_aged(35, 1), 1);
    assert!(c.contains(NS, &k(1)) && !c.contains(NS, &k(2)) && c.contains(NS, &k(3)));
    assert_eq!(c.evict_aged(35, 0), 1, "no limit: everything before 35 but the leased one");
    assert!(!c.contains(NS, &k(3)) && c.contains(NS, &k(4)));
    // evict_aged_by: per-namespace threshold (strictly before: 40 is not before 40), a slot budget
    assert_eq!(c.evict_aged_by(&|ns| if ns == NS { 40 } else { u64::MAX }, 0), 0);
    assert_eq!(store(&mut c, 1, 6, None, 1, 41), Status::Ok);
    assert_eq!(c.evict_aged_by(&|ns| if ns == NS { 42 } else { u64::MAX }, 1), 1, "one slot's worth");
    assert!(!c.contains(NS, &k(4)) && c.contains(NS, &k(6)));
    assert_eq!(c.evict_aged_by(&|ns| if ns == NS { 45 } else { u64::MAX }, 0), 1);
    assert!(!c.contains(NS, &k(6)) && c.contains(NS, &k(5)));
    assert!(!c.contains(NS, &k(4)) && c.contains(NS, &k(5)));
    assert_eq!(c.evict_aged_by(&|_| 0, 0), 0);
    // oldest_evictable / evict_oldest_except skip the protected set
    let id5 = c.protect_candidates(50).into_iter().map(|x| x.2).collect::<HashSet<_>>();
    assert_eq!(id5.len(), 1);
    assert_eq!(c.oldest_evictable_except(&id5), None, "1 leased, 5 protected");
    assert_eq!(c.evict_oldest_except(&id5), 0);
    assert_eq!(c.oldest_evictable_except(&HashSet::new()), Some(50));
    c.release(1, &[l]);
    assert_eq!(c.oldest_evictable_except(&id5), Some(10));
    assert_eq!(c.evict_oldest_except(&id5), 1);
    assert!(!c.contains(NS, &k(1)) && c.contains(NS, &k(5)));
    assert_eq!(c.evict_aged(u64::MAX, 0), 1);
    c.check();
}

#[test]
fn lease_deadline_counts_from_the_lookup() {
    let mut c = core(4);
    assert_eq!(store(&mut c, 1, 1, None, 4, 1_000), Status::Ok);
    let (_, lease, _) = c.lookup(2, NS, &[k(1)], true, 5_000);
    c.tick(5_000 + 60_000 - 1);
    assert_eq!(store(&mut c, 1, 2, None, 1, 64_999), Status::Full, "still leased");
    c.tick(5_000 + 60_000);
    assert_eq!(c.c.lease_timeouts, 1);
    assert_eq!(c.release(2, &[lease]), vec![Status::NotFound]);
    assert_eq!(store(&mut c, 1, 2, None, 1, 65_001), Status::Ok);
    c.check();
}

#[test]
fn reservation_deadline_counts_from_the_reserve() {
    let mut c = core(4);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 100)], 7_000)[0].0, Status::Ok);
    c.tick(7_000 + 30_000 - 1);
    assert_eq!(c.commit(1, NS, &[k(1)], 36_999).0[0], Status::Ok, "within its deadline");
    assert_eq!(c.reserve(1, NS, &[item(2, None, 100)], 40_000)[0].0, Status::Ok);
    c.tick(70_000);
    assert_eq!(c.commit(1, NS, &[k(2)], 70_000).0[0], Status::NotFound, "timed out");
    c.check();
}

#[test]
fn touch_refreshes_committed_only() {
    let mut c = core_with(cfg_flat(3));
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    assert_eq!(store(&mut c, 1, 2, None, 1, 1), Status::Ok);
    assert_eq!(c.reserve(1, NS, &[item(3, None, 100)], 2)[0].0, Status::Ok);
    assert_eq!(c.touch(NS, &[k(1), k(3), k(9)], 5), vec![Status::Ok, Status::NotFound, Status::NotFound]);
    assert_eq!(c.commit(1, NS, &[k(3)], 6).0[0], Status::Ok);
    assert_eq!(store(&mut c, 1, 4, None, 1, 7), Status::Ok);
    assert!(c.contains(NS, &k(1)) && !c.contains(NS, &k(2)), "the touched entry outlives an untouched newer one");
    c.check();
}

#[test]
fn tail_and_shared_age_exactly() {
    // heat: a tail looks tail_age_div times longer unused, a shared entry shared_age_mult times shorter
    let mut cf = cfg_flat(8);
    cf.shared_slots = 4;
    cf.shared_age_mult = 24;
    let mut c = core_with(cf);
    assert_eq!(store_kind(&mut c, 1, 1, KIND_TAIL, 1, 1_000), Status::Ok);
    assert_eq!(store_meta(&mut c, 2, None, ENTRY_SHARED_HINT, 1_000), Status::Ok);
    assert_eq!(store(&mut c, 1, 3, None, 1, 1_000), Status::Ok);
    c.tick(1_300);
    // tail: 1000 - 3 x 300 = 100; shared: 1000 + 300 - 300 / 24 = 1288; plain: 1000
    assert_eq!(c.oldest_evictable(), Some(100));
    assert_eq!(c.evict_aged(100, 0), 0);
    assert_eq!(c.evict_aged(101, 0), 1);
    assert!(!c.contains(NS, &k(1)));
    assert_eq!(c.evict_aged(1_000, 0), 0);
    assert_eq!(c.evict_aged(1_001, 0), 1);
    assert!(!c.contains(NS, &k(3)));
    assert_eq!(c.oldest_evictable(), Some(1_288));
    assert_eq!(c.evict_aged(1_288, 0), 0, "shared ages 24x slower");
    assert_eq!(c.evict_aged(1_289, 0), 1);
    c.check();
}

#[test]
fn shared_needs_three_continuations_without_depth_cap_too() {
    let mut cf = cfg_shared(16, 8);
    cf.shared_max_depth = 0;
    let mut c = core_with(cf);
    assert_eq!(store_meta(&mut c, 1, None, 0, 0), Status::Ok);
    assert_eq!(store_meta(&mut c, 2, Some(1), 0, 1), Status::Ok);
    assert_eq!(store_meta(&mut c, 3, Some(1), 0, 2), Status::Ok);
    assert!(!c.entry_info(NS, &k(1)).unwrap().shared, "two continuations are not enough");
    assert_eq!(store_meta(&mut c, 4, Some(1), 0, 3), Status::Ok);
    assert!(c.entry_info(NS, &k(1)).unwrap().shared);
    c.check();
}

#[test]
fn shared_parent_from_minimal_metadata() {
    // an "L1" record of exactly 18 bytes (no fingerprint / model id) still names the parent
    let mut c = core_with(cfg_shared(16, 8));
    assert_eq!(store_meta(&mut c, 1, None, 0, 0), Status::Ok);
    for (i, key) in (2..=4).enumerate() {
        let mut meta = b"L1".to_vec();
        meta.extend_from_slice(&k(1)[..16]);
        let mut it = item(key, None, 100);
        it.meta = meta;
        assert_eq!(c.reserve(1, NS, &[it], 1 + i as u64)[0].0, Status::Ok);
        assert_eq!(c.commit(1, NS, &[k(key)], 1 + i as u64).0[0], Status::Ok);
    }
    assert!(c.entry_info(NS, &k(1)).unwrap().shared);
}

#[test]
fn metadata_parent_cycles_terminate() {
    // entry metadata is client input: A names B as parent and B names A; marking / depth walks must end
    let mut c = core_with(cfg_shared(16, 8));
    let meta_to = |p: u32| {
        let mut m = b"L1".to_vec();
        m.extend_from_slice(&k(p)[..16]);
        m.extend_from_slice(&[0u8; 14]);
        m
    };
    for (key, parent, flags) in [(1u32, 2u32, 0u8), (2, 1, ENTRY_SHARED_HINT), (3, 2, 0), (4, 2, 0), (5, 2, 0)] {
        let mut it = item(key, None, 100);
        it.meta = meta_to(parent);
        it.flags = flags;
        assert_eq!(c.reserve(1, NS, &[it], key as u64)[0].0, Status::Ok);
        assert_eq!(c.commit(1, NS, &[k(key)], key as u64).0[0], Status::Ok);
    }
    assert!(c.entry_info(NS, &k(1)).unwrap().shared && c.entry_info(NS, &k(2)).unwrap().shared);
    assert_eq!(gauge(&c, "shared_slots"), 2);
    c.check();
}

#[test]
fn session_span_counts_from_its_first_use() {
    // a session first used at 100 min and resumed 3 times by 145 min spans 45 min, not 145
    let mut c = core(40);
    let t0 = 100 * MIN;
    touch_chain(&mut c, &[1, 2, 3, 103], t0);
    session_chain(&mut c, 0, t0);
    touch_chain(&mut c, &[1, 2, 3, 103], t0 + 1_000);
    assert_eq!(c.sessions.values().next().unwrap().resumes, 0, "a second after its start: no resume");
    for (i, dt) in [15 * MIN, 30 * MIN, 45 * MIN].iter().enumerate() {
        let mut keys = vec![1, 2, 3, 103];
        keys.extend((0..=i as u32).map(|j| 4 + j));
        touch_chain(&mut c, &keys, t0 + dt);
    }
    let sid = *c.sessions.keys().next().unwrap();
    assert_eq!(c.sessions[&sid].resumes, 3);
    touch_chain(&mut c, &[1, 2, 3, 103, 4, 5, 6], t0 + 45 * MIN + 1_000);
    assert_eq!(c.sessions[&sid].resumes, 3, "the same request again a second later: no resume");
    let all = |_: u32| u64::MAX;
    assert!(c.strict_age_candidates(&all, 0, 3, 60 * MIN).is_empty(), "45 min span < 60");
    assert!(!c.strict_age_candidates(&all, 0, 3, 45 * MIN).is_empty());
}

#[test]
fn metadata_size_limit_inclusive() {
    let mut c = core(4);
    let mut it = item(1, None, 100);
    it.meta = vec![0u8; MAX_META];
    assert_eq!(c.reserve(1, NS, &[it.clone()], 0)[0].0, Status::Ok);
    it.key = k(2);
    it.meta.push(0);
    assert_eq!(c.reserve(1, NS, &[it], 0)[0].0, Status::BadRequest);
}

#[test]
fn self_parented_metadata_is_no_continuation() {
    // an entry whose metadata names its own key prefix as parent does not count as a continuation of itself
    let mut c = core_with(cfg_shared(16, 8));
    let mut meta = b"L1".to_vec();
    meta.extend_from_slice(&k(1)[..16]);
    meta.extend_from_slice(&[0u8; 14]);
    let mut it = item(1, None, 100);
    it.meta = meta;
    assert_eq!(c.reserve(1, NS, &[it], 0)[0].0, Status::Ok);
    assert_eq!(c.commit(1, NS, &[k(1)], 0).0[0], Status::Ok);
    assert_eq!(store_meta(&mut c, 2, Some(1), 0, 1), Status::Ok);
    assert_eq!(store_meta(&mut c, 3, Some(1), 0, 2), Status::Ok);
    assert!(!c.entry_info(NS, &k(1)).unwrap().shared, "two real continuations only");
}

#[test]
fn seg_idle_flush_takes_held_entries_too() {
    // evict_aged(u64::MAX) (no clients for --idle-shrink-s): everything evictable goes, held young attention included
    let mut c = hybrid(16, 1, |_| {});
    for b in 1..=4 {
        hs(&mut c, 100 + b, 0, 1000);
    }
    treq(&mut c, &[(1..=4).map(|b| 200 + b).collect()], &(101..=104).collect::<Vec<_>>(), 1000);
    assert_eq!(c.evict_aged(5_000, 0), 0, "young uncovered attention is held by an age sweep");
    assert_eq!(c.evict_aged(u64::MAX, 0), 4);
    assert_eq!(c.entries.len(), 0);
    c.check();
}

#[test]
fn relocation_takes_the_entry_reaching_highest() {
    // by the end of its highest range (start + len), not start x len: A ends at 9, B (4 slots) at 8
    let mut c = core(12);
    assert_eq!(store(&mut c, 1, 1, None, 4, 0), Status::Ok);   // [0, 4)
    assert_eq!(store(&mut c, 1, 2, None, 4, 0), Status::Ok);   // B [4, 8)
    assert_eq!(store(&mut c, 1, 3, None, 1, 0), Status::Ok);   // A [8, 9)
    assert_eq!(c.entry_info(NS, &k(3)).unwrap().ranges, vec![Range { start: 8, len: 1 }]);
    assert!(c.remove_key(NS, &k(1)));
    let (_, key, _, new) = c.relocation_candidate().unwrap();
    assert_eq!(key, k(3));
    c.finish_relocation(NS, &key, &[Range { start: 8, len: 1 }], new);
    c.check();
}

#[test]
fn unexpired_ephemeral_waits_its_turn_and_a_read_one_stays_while_leased() {
    let mut c = core_with(cfg_flat(3));
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    let mut e = item(2, None, 100);
    e.flags = ENTRY_EPHEMERAL;
    e.ttl_ms = 60_000;
    assert_eq!(c.reserve(1, NS, &[e], 1)[0].0, Status::Ok);
    assert_eq!(c.commit(1, NS, &[k(2)], 1).0[0], Status::Ok);
    assert_eq!(store(&mut c, 1, 3, None, 1, 2), Status::Ok);
    assert_eq!(store(&mut c, 1, 4, None, 1, 3), Status::Ok);
    assert!(!c.contains(NS, &k(1)) && c.contains(NS, &k(2)), "not read, not expired: plain order");
    // two readers; one releases (read once: expires now), the other still holds its lease
    let (_, l1, _) = c.lookup(1, NS, &[k(2)], true, 4);
    let (_, l2, _) = c.lookup(2, NS, &[k(2)], true, 4);
    c.release(1, &[l1]);
    assert_eq!(store(&mut c, 1, 5, None, 1, 5), Status::Ok);
    assert!(c.contains(NS, &k(2)), "still leased by the other reader");
    c.release(2, &[l2]);
    assert_eq!(store(&mut c, 1, 6, None, 1, 6), Status::Ok);
    assert!(!c.contains(NS, &k(2)), "read and released: first to go");
    c.check();
}

#[test]
fn low_first_allocation_switch() {
    let mut c = core(8);
    assert_eq!(store(&mut c, 1, 1, None, 3, 0), Status::Ok);         // [0,3)
    assert_eq!(store(&mut c, 1, 2, None, 4, 0), Status::Ok);         // [3,7), free [7,8)
    assert!(c.remove_key(NS, &k(1)));                                 // free [0,3) and [7,8)
    c.set_alloc_low_first(true);
    assert_eq!(store(&mut c, 1, 9, None, 1, 1), Status::Ok);
    assert_eq!(c.entry_info(NS, &k(9)).unwrap().ranges, vec![Range { start: 0, len: 1 }], "lowest first");
    c.set_alloc_low_first(false);
    assert_eq!(store(&mut c, 1, 10, None, 1, 1), Status::Ok);
    assert_eq!(c.entry_info(NS, &k(10)).unwrap().ranges, vec![Range { start: 7, len: 1 }], "best fit: the 1-slot hole");
}

#[test]
fn seg_restored_block_mends_the_chain() {
    // a block that left RAM for the SSD (no cascade: still reachable) and is stored again before the grace: its
    // state is not an orphan
    let mut c = hybrid(32, 1, |_| {});
    let on_ssd = k(102);
    c.set_elsewhere(Box::new(move |_, key| *key == on_ssd));
    chain(&mut c, 3, &[3], 1000);
    assert!(c.remove_key(NS, &k(102)));
    assert!(has(&c, 103) && has(&c, 203), "on the SSD: the chain after it stays");
    c.set_elsewhere(Box::new(|_, _| false));      // promoted back: RAM only again
    hs(&mut c, 102, 0, 2000);
    c.tick(2000 + GRACE + 1000);
    c.tick(2000 + GRACE + 3000);
    assert!(has(&c, 203) && (101..=103).all(|b| has(&c, b)));
    assert_eq!(c.c.evict_orphan_state, 0);
    c.check();
}

#[test]
fn seg_aborted_lazy_reservation_does_not_break_the_chain() {
    // lazy stores: the state is in RAM, its attention not yet; a reservation of one block aborted is no loss
    let mut c = hybrid(32, 1, |_| {});
    hs(&mut c, 204, 1, 1000);
    treq(&mut c, &[(1..=4).map(|b| 200 + b).collect()], &(101..=104).collect::<Vec<_>>(), 1000);
    assert_eq!(c.reserve(1, NS, &[item(102, None, 100)], 1500)[0].0, Status::Ok);
    assert_eq!(c.abort(1, NS, &[k(102)]), vec![Status::Ok]);
    c.tick(1000 + GRACE + 1000);
    c.tick(1000 + GRACE + 3000);
    assert!(has(&c, 204), "state kept: its chain may be on the GPU");
    assert_eq!(c.c.evict_orphan_state, 0);
}

#[test]
fn seg_state_on_ssd_still_covers_its_chain() {
    // the state left RAM but the SSD tier holds it: the attention it makes usable is not orphaned
    let mut c = hybrid(32, 1, |_| {});
    let on_ssd = k(203);
    c.set_elsewhere(Box::new(move |_, key| *key == on_ssd));
    chain(&mut c, 3, &[3], 1000);
    assert!(c.remove_key(NS, &k(203)));
    c.tick(1000 + GRACE + 1000);
    c.tick(1000 + GRACE + 3000);
    assert!((101..=103).all(|b| has(&c, b)), "covered by the SSD copy of its state");
    assert_eq!(c.c.evict_orphan_attn, 0);
    // without the SSD copy the same blocks are orphans
    let mut c = hybrid(32, 1, |_| {});
    chain(&mut c, 3, &[3], 1000);
    assert!(c.remove_key(NS, &k(203)));
    c.tick(1000 + GRACE + 1000);
    c.tick(1000 + GRACE + 3000);
    assert!((101..=103).all(|b| !has(&c, b)));
}

#[test]
fn sessions_bounded() {
    let mut c = core(4);
    for i in 0..SESSIONS_MAX as u32 {
        touch_chain(&mut c, &[1_000_000 + i], i as u64);
    }
    assert_eq!(c.sessions.len(), SESSIONS_MAX, "at the bound: nothing dropped");
    touch_chain(&mut c, &[999], SESSIONS_MAX as u64);
    assert_eq!(c.sessions.len(), SESSIONS_MAX * 3 / 4, "over it: the least recently used quarter goes");
    assert!(c.sess_by_end.len() == SESSIONS_MAX * 3 / 4 && c.key_sess.len() == SESSIONS_MAX * 3 / 4);
    assert!(c.sess_by_end.contains_key(&(NS, k(999))), "the newest stays");
    assert!(!c.sess_by_end.contains_key(&(NS, k(1_000_000))), "the oldest went");
}

#[test]
fn shared_hint_meta_byte_32() {
    // "L1" metadata longer than 32 bytes: byte 32 bit 0 is the shared hint of clients without a flags field
    let mut c = core_with(cfg_shared(16, 8));
    for (key, b32) in [(1u32, 0u8), (2, 2), (3, 1), (4, 3)] {
        let mut meta = b"L1".to_vec();
        meta.extend_from_slice(&[0u8; 30]);
        meta.push(b32);
        let mut it = item(key, None, 100);
        it.meta = meta;
        assert_eq!(c.reserve(1, NS, &[it], key as u64)[0].0, Status::Ok);
        assert_eq!(c.commit(1, NS, &[k(key)], key as u64).0[0], Status::Ok);
        assert_eq!(c.entry_info(NS, &k(key)).unwrap().shared, b32 & 1 != 0, "byte 32 = {b32}");
    }
}

#[test]
fn touch_request_links_from_tags() {
    // TOUCH_REQUEST: no warm-up, no runs; many requests back to back in one step all link
    let mut c = core(64);
    for b in 1..=3 {
        hs(&mut c, 100 + b, 0, 1000);
        hs(&mut c, 300 + b, 0, 1000);
    }
    hs(&mut c, 203, 1, 1000);
    hs(&mut c, 403, 1, 1000);
    for _ in 0..3 {
        c.touch_request(1, NS, &[(1, keys(&[201, 202, 203])), (0, keys(&[101, 102, 103]))], 1000);
        c.touch_request(1, NS, &[(1, keys(&[403])), (0, keys(&[301, 302, 303]))], 1000);
    }
    c.touch_request(1, NS, &[(3, keys(&[999])), (0, keys(&[101]))], 1000);   // not a hybrid layout: touched only
    assert_eq!((c.c.seg_tagged_linked, c.c.seg_tagged_unlinked, c.c.seg_runs_skipped), (6, 1, 0));
    assert_eq!(gauge(&c, "seg_states_anchored"), 2);
    assert_eq!(gauge(&c, "chain_depth_max"), 3);
    assert_eq!(gauge(&c, "chain_dangling_entries"), 0);
    // the links drive the cascade
    assert!(c.remove_key(NS, &k(302)));
    assert!(!has(&c, 303) && !has(&c, 403) && has(&c, 301));
    assert!((101..=103).all(|b| has(&c, b)) && has(&c, 203));
    c.check();
}

#[test]
fn torn_rewrite_evicts_with_chain() {
    let mut c = core(8);
    assert_eq!(c.reserve(1, NS, &[item(1, None, 100)], 1000)[0].0, Status::Ok);
    c.commit(1, NS, &[k(1)], 1000);
    assert_eq!(c.reserve(1, NS, &[item(2, Some(1), 100)], 1000)[0].0, Status::Ok);
    c.commit(1, NS, &[k(2)], 1000);
    let g = c.copy_guard(NS, &k(1)).unwrap();
    assert!(!c.abort_torn_rewrite(NS, &k(1)), "not being rewritten: nothing to do");
    assert!(c.begin_rewrite(NS, &k(1), &g.1));
    assert_eq!(c.lookup(2, NS, &keys(&[1]), true, 1000).0, 0, "unreadable while rewritten");
    assert!(c.abort_torn_rewrite(NS, &k(1)));
    assert!(!has(&c, 1) && !has(&c, 2), "the torn entry and its child are gone");
    assert_eq!(c.c.torn_rewrites, 1);
    assert_eq!(gauge(&c, "slots_quarantined"), 1, "the torn slot is quarantined (its writer may still write)");
    c.check();
}

#[test]
fn guarded_relocation_and_abandon() {
    let mut c = core(16);
    for x in 1..=6 {
        assert_eq!(c.reserve(1, NS, &[item(x, None, 100)], 1000)[0].0, Status::Ok);
        c.commit(1, NS, &[k(x)], 1000);
    }
    for x in 1..=4 {
        assert!(c.remove_key(NS, &k(x)));                  // holes below 5 and 6
    }
    let (ns, key, old, new) = c.relocation_candidate().expect("a candidate");
    let g = c.copy_guard(ns, &key).unwrap();
    assert_eq!(g.1, old);
    // the entry changed while copied (compressed in place): refused, the new slots freed
    assert!(c.begin_rewrite(ns, &key, &old));
    c.end_rewrite(ns, &key, true);
    assert!(!c.finish_relocation_guarded(ns, &key, &g, new));
    c.check();
    // unchanged: moved
    let (ns, key, old, new) = c.relocation_candidate().expect("a candidate");
    let g = c.copy_guard(ns, &key).unwrap();
    assert!(c.finish_relocation_guarded(ns, &key, &g, new.clone()));
    assert_eq!(c.copy_guard(ns, &key).unwrap().1, new);
    assert_ne!(old, new);
    c.check();
    // the copier went away: its target slots quarantined, not free
    let (_, _, _, new) = c.relocation_candidate().expect("a candidate");
    let q0 = gauge(&c, "slots_quarantined");
    c.abandon_relocation(new.clone());
    assert_eq!(gauge(&c, "slots_quarantined"), q0 + new.iter().map(|r| r.len as u64).sum::<u64>());
    c.check();
    c.tick(1000 + QUARANTINE_MS + 1);
    assert_eq!(gauge(&c, "slots_quarantined"), 0, "released after the quarantine");
    c.check();
}

#[test]
fn invalidate_drops_the_chain_after_and_lets_a_fresh_copy_in() {
    let mut c = hybrid(32, 1, |_| {});
    chain(&mut c, 3, &[3], 1000); // attention 101 -> 102 -> 103, GDN state 203 at block 3
    let (st, gone) = c.invalidate(NS, &keys(&[102, 999]));
    assert_eq!(st, vec![Status::Ok, Status::NotFound]);
    assert_eq!(gone.len(), 3, "the removed keys (for the SSD copies)");
    assert!(has(&c, 101), "what comes before stays");
    assert!(!has(&c, 102) && !has(&c, 103) && !has(&c, 203), "the entry and the chain after it go");
    assert_eq!(c.lookup(2, NS, &keys(&[101, 102, 103]), false, 1100).0, 1, "lookups stop before it");
    hs(&mut c, 102, 0, 1200); // the recompute's fresh copy
    assert!(has(&c, 102));
    assert_eq!(c.counters().iter().find(|(n, _)| n == "invalidated").map(|x| x.1), Some(3));
    c.check();
}

#[test]
fn invalidate_leased_leaves_lookups_now_and_keeps_a_fresh_copy() {
    let mut c = hybrid(32, 1, |_| {});
    chain(&mut c, 3, &[3], 1000);
    let (n, lease, _) = c.lookup(2, NS, &keys(&[101, 102, 103]), true, 1100);
    assert_eq!(n, 3);
    assert_eq!(c.invalidate(NS, &keys(&[103])).0, vec![Status::Ok]);
    assert!(!has(&c, 103), "no lookup finds it although it is leased");
    hs(&mut c, 103, 0, 1150); // a fresh copy committed while the old one is still leased
    c.release(2, &[lease]);
    c.tick(1300);
    assert!(has(&c, 103), "the old copy's removal leaves the fresh one's index alone");
    c.check();
}

#[test]
fn invalidate_keeps_shared_beginnings() {
    let mut c = hybrid(32, 1, |x| x.shared_slots = 8);
    chain(&mut c, 3, &[3], 1000);
    c.mark_shared_for_test(NS, &k(101));
    let (st, gone) = c.invalidate(NS, &keys(&[101]));
    assert_eq!(st, vec![Status::Exists]);
    assert!(gone.is_empty() && has(&c, 101) && has(&c, 103), "a shared beginning and its sessions stay");
    c.check();
}

#[test]
fn heat_counts_hits_so_a_reused_entry_outlives_a_newer_unused_one() {
    let mut x = cfg(16, PolicyKind::Arc);
    x.seg_evict = false;
    x.heat_hit_ms = 600_000; // a hit = 10 min of recency
    let mut c = Core::new(x);
    c.connect(1);
    c.hello(1, 10);
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok); // old, reused 3 times
    for t in [1000, 2000, 3000] {
        let (n, lease, _) = c.lookup(1, NS, &keys(&[1]), true, t);
        assert_eq!(n, 1);
        c.release(1, &[lease]);
    }
    assert_eq!(store(&mut c, 1, 2, None, 1, 900_000), Status::Ok); // 15 min newer, never reused
    c.tick(1_000_000);
    c.evict_oldest_except(&Default::default());
    assert!(has(&c, 1) && !has(&c, 2), "the reused entry is hotter (3 x 10 min > 15 min)");
    c.check();
}

#[test]
fn heat_branching_prefix_decays_slower() {
    let mut x = cfg(16, PolicyKind::Arc);
    x.seg_evict = false;
    x.shared_slots = 8;
    x.shared_min_cont = 9; // never "shared": only the branching rule applies
    let mut c = Core::new(x);
    c.connect(1);
    c.hello(1, 10);
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    c.note_continuation_for_test(NS, &k(1), 77);
    c.note_continuation_for_test(NS, &k(1), 78); // two different chains continue from entry 1
    assert_eq!(store(&mut c, 1, 2, None, 1, 400_000), Status::Ok);
    c.tick(1_000_000); // idle: entry 1 1000 s / 3 = 333 s, entry 2 600 s
    c.evict_oldest_except(&Default::default());
    assert!(has(&c, 1) && !has(&c, 2), "the branching prefix cools 3x slower");
    c.check();
}

// ------------------------------------------------------------ weights entries (KIND_WEIGHTS, mmproj)

fn store_w(c: &mut Core, conn: ConnId, ns: u32, key: u32, slots: u64, cost_hint: u32, now: u64) -> Status {
    let mut it = item(key, None, slots * 100);
    it.kind = KIND_WEIGHTS;
    it.cost_hint = cost_hint;
    let r = c.reserve(conn, ns, &[it], now);
    if r[0].0 != Status::Ok {
        return r[0].0;
    }
    c.commit(conn, ns, &[k(key)], now).0[0]
}

fn here(c: &Core, ns: u32, key: u32) -> bool {
    c.contains(ns, &k(key))
}

/// Every policy x segment-eviction setting: weights leave by per-slot value against KV, not by recency alone.
#[test]
fn weights_valued_per_slot_against_kv() {
    for (policy, seg) in [(PolicyKind::Arc, true), (PolicyKind::Arc, false), (PolicyKind::Lru, true), (PolicyKind::Lru, false)] {
        let mut cf = cfg(20, policy);
        cf.seg_evict = seg;
        // worth less per slot than KV: 16 slots, reload 1,600 ms vs 1,000 ms per KV slot -> idle counts 10 x
        let mut c = core_with(cf.clone());
        assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
        assert_eq!(store_w(&mut c, 1, NS, 2, 16, 1600, 100), Status::Ok);
        for key in 3..=5 {
            assert_eq!(store(&mut c, 1, key, None, 1, 105), Status::Ok);
        }
        assert_eq!(gauge(&c, "slots_free"), 0);
        // t 110: weights heat 110 - 10 x 10 = 100 > KV 1's 0: KV 1 goes (as plain LRU would)
        assert_eq!(store(&mut c, 1, 6, None, 1, 110), Status::Ok, "{policy:?} seg {seg}");
        assert!(!here(&c, NS, 1) && here(&c, NS, 2), "{policy:?} seg {seg}: the colder KV entry leaves");
        // t 400: weights heat 400 - 10 x 300 < KV 3's 105: 300 ms of weights idle count as 3 s of KV idle, so the
        // weights go before KV 3, which is older
        assert_eq!(store(&mut c, 1, 7, None, 1, 400), Status::Ok);
        assert!(!here(&c, NS, 2), "{policy:?} seg {seg}: weights colder per slot than the KV victim");
        assert!((3..=7).filter(|x| *x != 1 && *x != 2).all(|x| here(&c, NS, x)), "{policy:?} seg {seg}: no KV evicted");
        c.check();

        // worth more per slot than KV (reload 64 s for 16 slots: idle counts 1/4): older weights outlive newer KV
        let mut c = core_with(cf.clone());
        assert_eq!(store_w(&mut c, 1, NS, 2, 16, 64_000, 0), Status::Ok);
        for key in 3..=6 {
            assert_eq!(store(&mut c, 1, key, None, 1, 100), Status::Ok);
        }
        // t 1000: weights heat 1000 - 1000 / 4 = 750 > KV 3's 100: KV 3 goes, the (older) weights stay
        assert_eq!(store(&mut c, 1, 7, None, 1, 1000), Status::Ok);
        assert!(here(&c, NS, 2) && !here(&c, NS, 3), "{policy:?} seg {seg}: valuable weights stay");
        // t 5000: weights heat 5000 - 1250 = 3750 > every KV entry's last use: KV keeps going first
        assert_eq!(store(&mut c, 1, 8, None, 1, 5000), Status::Ok);
        assert!(here(&c, NS, 2) && !here(&c, NS, 4));
        c.check();

        // no cost_hint (or weights_kv_slot_ms 0): plain recency like KV, the older weights go first
        for (cost, kv_ms) in [(0u32, 1000u64), (64_000, 0)] {
            let mut cf0 = cf.clone();
            cf0.weights_kv_slot_ms = kv_ms;
            let mut c = core_with(cf0);
            assert_eq!(store_w(&mut c, 1, NS, 2, 16, cost, 0), Status::Ok);
            for key in 3..=6 {
                assert_eq!(store(&mut c, 1, key, None, 1, 100), Status::Ok);
            }
            assert_eq!(store(&mut c, 1, 7, None, 1, 1000), Status::Ok);
            assert!(!here(&c, NS, 2) && here(&c, NS, 3), "{policy:?} seg {seg} cost {cost} kv_ms {kv_ms}: plain LRU");
            c.check();
        }
    }
}

#[test]
fn weights_age_sweep_and_pressure_by_value() {
    for seg in [true, false] {
        let mut cf = cfg(40, PolicyKind::Arc);
        cf.seg_evict = seg;
        let mut c = core_with(cf);
        assert_eq!(store(&mut c, 1, 1, None, 1, 1000), Status::Ok);
        assert_eq!(store_w(&mut c, 1, NS, 2, 16, 1600, 1000), Status::Ok); // idle counts 10 x
        c.tick(1100);
        // age sweep "unused for 500 ms": KV idle 100 stays; weights 100 x 10 = 1000 > 500 go
        assert_eq!(c.evict_aged_by(&|_| 1100 - 500, 0), 16, "seg {seg}");
        assert!(here(&c, NS, 1) && !here(&c, NS, 2));
        // heat order across pools (RAM ceiling): a fresh weights entry is behind older KV, then overtakes it
        assert_eq!(store_w(&mut c, 1, NS, 2, 16, 1600, 1100), Status::Ok);
        c.tick(1105);
        // weights 1105 - 10 x 5 = 1055, KV 1105 - 105 = 1000: KV is the colder one
        assert_eq!(c.oldest_evictable(), Some(1000), "seg {seg}");
        c.tick(1300);
        // weights 1300 - 2000 < 0 -> 0; KV 1000
        assert_eq!(c.oldest_evictable(), Some(0));
        assert_eq!(c.evict_oldest(), 16);
        assert!(here(&c, NS, 1));
        // memory pressure (policy order + weights by heat)
        assert_eq!(store_w(&mut c, 1, NS, 2, 16, 1600, 1300), Status::Ok);
        c.tick(2000);
        assert_eq!(c.evict_coldest(1), 16, "seg {seg}: pressure takes the weights (idle 700 x 10) before KV (idle 1000)");
        c.check();
    }
}

#[test]
fn weights_low_water_applies_and_nothing_pins_them() {
    let mut c = core_with(Config { low_water_slots: 2, ..cfg(20, PolicyKind::Arc) });
    for key in 1..=4 {
        assert_eq!(store(&mut c, 1, key, None, 1, 0), Status::Ok);
    }
    assert_eq!(store_w(&mut c, 1, NS, 9, 16, 1600, 10), Status::Ok);
    assert_eq!(gauge(&c, "slots_free"), 0);
    // an image request: lease, copy, release; the entry stays evictable (no pin, no TTL, no lease while idle)
    let (n, lease, d) = c.lookup(2, NS, &[k(9)], true, 20);
    assert_eq!((n, d[0].kind, d[0].ranges.iter().map(|r| r.len).sum::<u32>()), (1, KIND_WEIGHTS, 16));
    c.tick(1000);
    // low water (2 free slots): the leased weights are colder per slot but cannot go; KV 1 and 2 do
    assert!(here(&c, NS, 9) && !here(&c, NS, 1) && !here(&c, NS, 2), "leased: never evicted");
    assert_eq!(c.release(2, &[lease]), vec![Status::Ok]);
    assert_eq!(store(&mut c, 1, 5, None, 1, 1000), Status::Ok);
    assert_eq!(store(&mut c, 1, 6, None, 1, 1000), Status::Ok);
    // released: at t 1100 the weights (1100 - 10 x 1080 < 0) are colder than KV 3 (0): they go, all KV stays
    c.tick(1100);
    assert!(!here(&c, NS, 9) && (3..=6).all(|x| here(&c, NS, x)));
    assert_eq!(gauge(&c, "leases_active"), 0);
    c.check();
}

#[test]
fn weights_shared_namespace_two_engines() {
    const SHARED: u32 = 0x5eed;
    let mut c = core(40);
    // engine 10 (conn 1) loads the tower from disk and stores it; engine 20 (conn 2) hits the same entry
    assert_eq!(store_w(&mut c, 1, SHARED, 7, 16, 1200, 0), Status::Ok);
    let (n, lease, d) = c.lookup(2, SHARED, &[k(7)], true, 10);
    assert_eq!((n, d.len(), d[0].producer, d[0].kind), (1, 1, 10, KIND_WEIGHTS));
    assert_eq!(d[0].ranges.iter().map(|r| r.len).sum::<u32>(), 16);
    assert_eq!(c.c.cross_engine_hit_entries, 1);
    // both engines at once (two image requests on two GPUs): two leases on one entry
    let (n1, lease1, _) = c.lookup(1, SHARED, &[k(7)], true, 11);
    assert_eq!(n1, 1);
    // the other engine's store of the same bytes: EXISTS, nothing written twice
    let mut it = item(7, None, 1600);
    it.kind = KIND_WEIGHTS;
    assert_eq!(c.reserve(2, SHARED, &[it], 12)[0].0, Status::Exists);
    assert_eq!(c.release(2, &[lease]), vec![Status::Ok]);
    assert_eq!(c.release(1, &[lease1]), vec![Status::Ok]);
    // the storing engine going away does not take the entry with it
    c.disconnect(1);
    assert_eq!(c.lookup(2, SHARED, &[k(7)], false, 20).0, 1);
    let st = c.ns_stats();
    assert!(st.iter().any(|((ns, kind), g, _)| *ns == SHARED && *kind == KIND_WEIGHTS
        && g.iter().any(|(x, v)| *x == "slots" && *v == 16)), "per-kind stats under kind 5");
    c.check();
}

#[test]
fn weights_reserve_in_a_fragmented_pool() {
    let mut c = core(40);
    for key in 1..=40 {
        assert_eq!(store(&mut c, 1, key, None, 1, key as u64), Status::Ok);
    }
    // every other KV entry leased: the evictable slots are 20 single slots, scattered
    let mut leases = Vec::new();
    for key in (2..=40).step_by(2) {
        leases.push(c.lookup(2, NS, &[k(key)], true, 100).1);
    }
    let mut it = item(1000, None, 16 * 100 - 50);
    it.kind = KIND_WEIGHTS;
    it.cost_hint = 1000;
    let r = c.reserve(3, NS, &[it.clone()], 200);
    assert_eq!(r[0].0, Status::Ok, "evicts enough by the normal order");
    assert_eq!(r[0].1.iter().map(|x| x.len).sum::<u32>(), 16);
    assert!(r[0].1.len() > 1, "scattered ranges, not one contiguous run: the client copies each range");
    assert_eq!((1..=31).step_by(2).filter(|x| here(&c, NS, *x)).count(), 0, "the 16 least recently used unleased went");
    assert!((33..=39).step_by(2).all(|x| here(&c, NS, x)));
    assert_eq!(c.commit(3, NS, &[k(1000)], 200).0, vec![Status::Ok]);
    // everything else leased (the weights too): FULL, nothing changes
    for key in (33..=39).step_by(2) {
        leases.push(c.lookup(2, NS, &[k(key)], true, 300).1);
    }
    leases.push(c.lookup(2, NS, &[k(1000)], true, 300).1);
    let mut it2 = it.clone();
    it2.key = k(1001);
    assert_eq!(c.reserve(3, NS, &[it2.clone()], 400)[0].0, Status::Full);
    assert!(here(&c, NS, 1000));
    // leases end: the next RESERVE finds room (the old weights entry is the coldest per slot)
    for l in leases {
        c.release(2, &[l]);
    }
    assert_eq!(c.reserve(3, NS, &[it2], 5000)[0].0, Status::Ok);
    assert!(!here(&c, NS, 1000), "the older weights entry made the room");
    c.check();
}

#[test]
fn weights_never_persisted_or_restored() {
    let mut c = core(40);
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    assert_eq!(store_w(&mut c, 1, NS, 2, 16, 1200, 0), Status::Ok);
    for t in 1..=5 {
        c.lookup(2, NS, &[k(1), k(2)], false, t);
        c.lookup(2, NS, &[k(2)], false, t);
    }
    // hot set (main.rs HotSet and the SSD tier's core pass both use hot_entries): KV only
    let hot = c.hot_entries(1);
    assert_eq!(hot.iter().map(|e| e.key).collect::<Vec<_>>(), vec![k(1)]);
    // in-place RAM compression: never (the engine copies raw slots)
    let cand = c.compress_candidates(u64::MAX, 100);
    assert_eq!(cand.iter().map(|x| x.1).collect::<Vec<_>>(), vec![k(1)]);
    // a restore (hot-set file, SSD file) of a weights entry is refused
    let mut d = core(40);
    assert!(d.insert_committed(NS, k(2), KIND_WEIGHTS, 1, 10, vec![], 1600, 9, 0).is_none());
    assert!(!here(&d, NS, 2));
    assert!(d.insert_committed(NS, k(1), 0, 1, 10, vec![], 100, 9, 0).is_some());
    d.check();
}

#[test]
fn weights_counted_as_their_own_kind() {
    let mut c = core(40);
    assert_eq!(store_w(&mut c, 1, NS, 2, 16, 1200, 0), Status::Ok);
    assert_eq!(store_kind(&mut c, 1, 3, KIND_TAIL, 1, 0), Status::Ok);
    assert_eq!(store_kind(&mut c, 1, 4, 7, 1, 0), Status::Ok); // an unknown kind: counted with 4 as before
    assert_eq!((gauge(&c, "entries_kind5"), gauge(&c, "entries_kind4")), (1, 2));
    let h = c.heat_hist();
    assert_eq!(h.iter().filter(|x| x.0 == KIND_WEIGHTS).map(|x| x.2).sum::<u64>(), 16);
    assert_eq!(h.iter().filter(|x| x.0 == KIND_TAIL).map(|x| x.3).sum::<u64>(), 2);
}

// ------------------------------------------------------------ persistent hot set by use (hot.rs)

/// One request of conversation `conv` as the Arc connector sends it (TOUCH_REQUEST): attention from the root
/// (shared prompt 101..=104, then the conversation's own blocks 1000 + 10 conv + j), the GDN group from `from` on
/// (its hit point; suffix).
fn conv_req(c: &mut Core, conv: u32, own: u32, from: usize, now: u64) {
    let mut att: Vec<u32> = (101..=104).collect();
    att.extend((1..=own).map(|j| 1000 + 10 * conv + j));
    let st: Vec<u32> = att[from.min(att.len() - 1)..].iter().map(|a| a + 100_000).collect();
    c.touch_request(1, NS, &[(0, keys(&att)), (1, keys(&st))], now);
}

/// Shared prompt in RAM: attention 101..=104, GDN states (one group) at blocks 2 and 4 (keys 100102, 100104).
fn shared_prompt(c: &mut Core, now: u64) {
    for b in 101..=104 {
        hs(c, b, 0, now);
    }
    hs(c, 100_102, 1, now);
    hs(c, 100_104, 1, now);
    conv_req(c, 0, 0, 0, now);      // the request that stored it: links the chain, anchors the states
}

fn hot_keys(v: &[HotEntry]) -> Vec<u32> {
    v.iter().map(|e| u32::from_le_bytes(e.key[..4].try_into().unwrap())).collect()
}

#[test]
fn hotset_by_use_keeps_the_shared_prompt_without_hits() {
    let mut c = core(64);
    shared_prompt(&mut c, 1000);
    // conversation 1 stores its own blocks too and runs 4 turns (+ re-touches while waiting): one chain
    for j in 1..=3 {
        hs(&mut c, 1010 + j, 0, 1000);
    }
    hs(&mut c, 101_013, 1, 1000);
    for t in 0..4 {
        for _ in 0..3 {
            conv_req(&mut c, 1, 1 + t.min(2), if t == 0 { 4 } else { 5 }, 2000 + t as u64);
        }
    }
    // 5 more conversations: their hit point is past the shared prompt (VRAM), so the GDN touch skips its state
    for conv in 2..=6 {
        conv_req(&mut c, conv, 2, 5, 3000);
        conv_req(&mut c, conv, 3, 6, 3100);
    }
    assert_eq!(c.c.hit_entries, 0, "no RAM-tier hits at all");
    assert!(c.hot_entries(3).is_empty(), "ranked by hits: nothing");
    assert!(c.use_of(NS, &k(104)) >= 5.9, "one use per conversation: {}", c.use_of(NS, &k(104)));
    assert!(c.use_of(NS, &k(1011)) < 2.0, "one conversation's turns and re-touches are one chain");
    let hot = c.hot_entries_by_use(3.0, u64::MAX);
    // the shared prompt's deepest boundary, root first; the block-2 state is superseded by block 4 (same users);
    // conversation 1's own blocks have one user
    assert_eq!(hot_keys(&hot), vec![101, 102, 103, 104, 100_104]);
    let st = c.hot_stats();
    assert_eq!((st.units, st.entries, st.bytes, st.unloadable), (1, 5, 500, 0));
    assert!(hot.iter().all(|e| e.kind != KIND_WEIGHTS));
    c.check();
}

#[test]
fn hotset_by_use_budget_and_loadability() {
    let mut c = core(64);
    c.hot_min_use = 3.0;
    for b in [101, 103, 104] {
        hs(&mut c, b, 0, 1000);                    // block 102 only in VRAM (never stored, or evicted)
    }
    hs(&mut c, 100_104, 1, 1000);
    conv_req(&mut c, 0, 0, 0, 1000);
    for conv in 1..=5 {
        conv_req(&mut c, conv, 1, 5, 2000 + conv as u64);
    }
    assert!(c.hot_entries_by_use(3.0, u64::MAX).is_empty(), "a chain with a block not in RAM is not loadable");
    assert!(c.hot_stats().unloadable >= 1);
    assert_eq!(gauge(&c, "hotset_touched_missing_keys") >= 1, true, "block 102: touched by 5 chains, not in RAM");
    assert!(c.c.touch_missing_keys > 0 && c.c.touch_keys > c.c.touch_missing_keys);
    // the engine stores 102 again: it takes the use its key gathered, and the prefix ranks
    hs(&mut c, 102, 0, 3000);
    assert!(c.use_of(NS, &k(102)) >= 4.9, "took the use its key gathered");
    let hot = c.hot_entries_by_use(3.0, u64::MAX);
    assert_eq!(hot_keys(&hot), vec![101, 102, 103, 104, 100_104]);
    // budget: a unit that does not fit is left out whole (never a chain without its state)
    assert!(c.hot_entries_by_use(3.0, 400).is_empty());
    assert_eq!(c.hot_stats().over_budget, 1);
    c.check();
}

#[test]
fn hotset_use_decays_and_restores_with_relations() {
    let mut c = core(64);
    shared_prompt(&mut c, 1000);
    for conv in 1..=6 {
        conv_req(&mut c, conv, 1, 5, 2000);
    }
    c.use_half_life_ms = 3_600_000;
    let hot = c.hot_entries_by_use(3.0, u64::MAX);
    assert_eq!(hot.len(), 5);
    // restart: entries restored from files with their use and relation
    let saved: Vec<(HotEntry, f64, HotRel)> = hot.iter().map(|e| (e.clone(), c.use_of(e.ns, &e.key), c.hot_rel(e.ns, &e.key))).collect();
    assert!(matches!(saved[4].2, HotRel::Anchor(_, 1)), "the state carries its anchor block");
    assert!(matches!(saved[0].2, HotRel::Parent(p) if p == [0u8; 16]), "the root is a root");
    let mut d = core(64);
    d.use_half_life_ms = 3_600_000;
    d.tick(5000);
    for (e, u, rel) in &saved {
        assert!(d.insert_committed(e.ns, e.key, e.kind, e.format, e.producer, e.meta.clone(), e.bytes, e.hits, 5000).is_some());
        d.restore_hot(e.ns, &e.key, *u, *rel);
    }
    assert_eq!(hot_keys(&d.hot_entries_by_use(3.0, u64::MAX)), vec![101, 102, 103, 104, 100_104],
               "restored: still a loadable, ranked prefix (no touch yet)");
    assert_eq!(d.lookup(2, NS, &keys(&[101, 102, 103, 104]), false, 5001).0, 4, "loadable by key");
    assert_eq!(d.lookup(2, NS, &keys(&[100_104]), false, 5001).0, 1);
    // two half-lives without use: 6 -> 1.5 < 3, out of the ranking
    d.tick(5000 + 2 * 3_600_000);
    assert!(d.hot_entries_by_use(3.0, u64::MAX).is_empty());
    d.check();
}

#[test]
fn hotset_boundary_state_keeps_its_anchors_recency() {
    let mut c = core(64);
    c.hot_min_use = 3.0;
    shared_prompt(&mut c, 1000);
    for conv in 1..=4 {
        conv_req(&mut c, conv, 1, 5, 2000);
    }
    let lu = |c: &Core, key: u32| c.entry_info(NS, &k(key)).unwrap().last_used;
    // later requests touch the shared prompt's attention, the GDN group only past it
    for conv in 5..=8 {
        conv_req(&mut c, conv, 1, 5, 100_000);
    }
    assert_eq!(lu(&c, 104), 100_000);
    assert!(lu(&c, 100_104) < 100_000);
    c.tick(100_000);
    assert_eq!(lu(&c, 100_104), 100_000, "the boundary state is as recent as its anchor block");
    assert!(lu(&c, 100_102) < 100_000, "a superseded state is not held");
    assert!(c.c.hot_state_refreshed >= 1);
}

#[test]
fn hotset_by_use_lookup_only_clients_and_weights() {
    // a client without TOUCH (llama.cpp-style): lookup hits count, per distinct chain; weights never rank
    let mut c = core(64);
    assert_eq!(store(&mut c, 1, 1, None, 1, 0), Status::Ok);
    assert_eq!(store(&mut c, 1, 2, Some(1), 1, 0), Status::Ok);
    for x in 3..=6 {
        assert_eq!(store(&mut c, 1, x, Some(2), 1, 0), Status::Ok);
    }
    assert_eq!(store_w(&mut c, 1, NS, 50, 2, 1200, 0), Status::Ok);
    for x in 3..=6 {
        for _ in 0..3 {
            c.lookup(2, NS, &keys(&[1, 2, x]), false, 10);       // the same chain thrice: one use
        }
        c.lookup(2, NS, &keys(&[50]), false, 10);
    }
    assert!((c.use_of(NS, &k(2)) - 4.0).abs() < 0.01 && (c.use_of(NS, &k(3)) - 1.0).abs() < 0.01);
    let hot = c.hot_entries_by_use(3.0, u64::MAX);
    assert_eq!(hot_keys(&hot), vec![1, 2], "the shared trunk with its root; weights never");
    // a touching connection's lookups are not counted again (its touches are)
    c.touch_conn(3, NS, &keys(&[1, 2, 3]), 20);
    let u = c.use_of(NS, &k(1));
    c.lookup(3, NS, &keys(&[1, 2, 7]), false, 20);
    assert_eq!(c.use_of(NS, &k(1)), u);
    c.check();
}

/// Seen on an RTX 5090 RAM tier (three windows): a cold prompt is touched at its lookup, then its
/// attention and prompt-end state are stored. Its run is linked before the state exists, so the state is not anchored
/// and the attention is uncovered; it was stored after its chain's last touch, so it is not "young" either. Once the
/// namespace is linked (any anchored state), the age sweep took it as an orphan (heat i64::MIN) seconds after commit,
/// with free slots, and its whole chain by cascade: 11 of 14 attention entries of three prompts, also with the
/// production engine stack. Production 7 d: 183 of 523 commits removed this way.
fn cold_prompt(c: &mut Core, base: u32, n: u32, t_touch: u64, t_store: u64) {
    // lookup: the request's TOUCH run (GDN group suffix + attention from the root), before anything is stored
    treq(c, &[(1..=n).map(|b| base + 100 + b).collect()], &(1..=n).map(|b| base + b).collect::<Vec<_>>(), t_touch);
    for b in 1..=n {
        hs(c, base + b, 0, t_store);
    }
    hs(c, base + 100 + n, 1, t_store);           // prompt-end state
}

#[test]
fn seg_cold_prompt_survives_age_sweep() {
    let mut c = hybrid(64, 1, |_| {});
    chain(&mut c, 3, &[3], 500);                 // an anchored state: the namespace is linked
    cold_prompt(&mut c, 1000, 5, 1000, 2000);
    c.tick(3000);
    // a sweep right after the stores on a young broker (now - age saturates to 0), and with a real age cut-off
    assert_eq!(c.evict_aged_by(&|_| 0, 0), 0, "nothing is older than the age limit");
    c.evict_aged_by(&|_| 1500, 0);              // a real age cut-off: the older warm-up chain (t 500) may go
    assert!((1001..=1005).all(|b| has(&c, b)) && has(&c, 1105), "a just-stored prompt and its state stay");
    c.check();
}

#[test]
fn seg_cold_prompt_state_anchored() {
    // the prompt-end state committed after its request's run was linked is anchored at its block: the prompt is a
    // segment (state + attention), not an orphan chain, also after the grace
    let mut c = hybrid(64, 1, |_| {});
    chain(&mut c, 3, &[3], 500);
    cold_prompt(&mut c, 1000, 5, 1000, 2000);
    c.tick(2000 + 2 * GRACE);
    assert_eq!(c.evict_aged_by(&|_| 0, 0), 0);
    assert!((1001..=1005).all(|b| has(&c, b)) && has(&c, 1105), "covered by its state: kept");
    assert_eq!(c.c.evict_orphan_attn + c.c.cascade_evicted + c.c.evict_orphan_state, 0);
    c.check();
}

#[test]
fn seg_cold_prompt_orphan_tail_still_goes_after_grace() {
    // attention past the prompt-end state (no state at blocks 4-5) is still useless: it goes once older than the grace
    let mut c = hybrid(64, 1, |_| {});
    chain(&mut c, 3, &[3], 500);
    treq(&mut c, &[(1..=5).map(|b| 1100 + b).collect()], &(1001..=1005).collect::<Vec<_>>(), 1000);
    for b in 1..=5 {
        hs(&mut c, 1000 + b, 0, 2000);
    }
    hs(&mut c, 1103, 1, 2000);                    // state at block 3 only
    c.tick(2500);
    assert_eq!(c.evict_aged_by(&|_| 0, 0), 0, "young: kept");
    c.tick(2000 + GRACE + 2000);
    c.evict_aged_by(&|_| 0, 0);
    assert!((1001..=1003).all(|b| has(&c, b)) && has(&c, 1103), "the segment up to the state stays");
    assert!((1004..=1005).all(|b| !has(&c, b)), "attention past the deepest state goes after the grace");
    c.check();
}

// ------------------------------------------------------------ request classes (ENTRY_LOW, TOUCH_TAGGED)

/// Reserve + commit one slot of `kind` with RESERVE flags `flags` (ENTRY_LOW = a low-priority request's block).
fn hsf(c: &mut Core, key: u32, kind: u8, flags: u8, now: u64) {
    let mut it = item(key, None, 100);
    it.kind = kind;
    it.flags = flags;
    assert_eq!(c.reserve(1, NS, &[it], now)[0].0, Status::Ok);
    assert_eq!(c.commit(1, NS, &[k(key)], now).0[0], Status::Ok);
}

fn is_low(c: &Core, key: u32) -> bool {
    c.entry_flags(NS, &k(key)).is_some_and(|f| f & ENTRY_LOW != 0)
}

#[test]
fn low_class_goes_first_under_room_pressure_aging_unchanged() {
    // an agent entry (older) and a bookkeeping entry (newer, comparable heat): RESERVE takes the bookkeeping one
    for margin in [1_800_000, 0] {
        let mut c = hybrid(3, 1, |x| x.policy = PolicyKind::Lru);
        c.low_margin_ms = margin;
        hsf(&mut c, 1, 3, 0, 1000);            // agent
        hsf(&mut c, 2, 3, ENTRY_LOW, 60_000);  // bookkeeping, a minute later
        hsf(&mut c, 3, 3, 0, 61_000);
        hsf(&mut c, 4, 3, 0, 62_000);          // pool of 3: one must go
        if margin > 0 {
            assert!(has(&c, 1) && !has(&c, 2), "class order: the low entry goes first");
            assert_eq!((c.c.low_evicted, c.c.low_picked.get()), (1, 1));
        } else {
            assert!(!has(&c, 1) && has(&c, 2), "margin 0: plain policy order (LRU)");
        }
        c.check();
    }
    // ageing with room is the same for every class: the age sweep takes both at the same threshold
    let mut c = hybrid(8, 1, |_| {});
    hsf(&mut c, 1, 3, 0, 1000);
    hsf(&mut c, 2, 3, ENTRY_LOW, 1000);
    c.evict_aged_by(&|_| 500, 0);
    assert!(has(&c, 1) && has(&c, 2), "neither is older than the threshold");
    c.evict_aged_by(&|_| 1500, 0);
    assert!(!has(&c, 1) && !has(&c, 2));
}

#[test]
fn hot_low_prefix_outlives_a_cold_agent_entry() {
    // a bookkeeping prefix reused all the time (recent, hit often) is worth more than an agent entry idle for hours
    let mut c = hybrid(3, 1, |x| { x.policy = PolicyKind::Lru; x.heat_hit_ms = 600_000; });
    hsf(&mut c, 1, 3, 0, 1000);                         // agent, then idle
    hsf(&mut c, 2, 3, ENTRY_LOW, 1000);                 // bookkeeping prefix ...
    for t in 0..6u64 {                                  // ... reused by six bookkeeping requests over 2 h
        let now = 1000 + (t + 1) * 1_200_000;
        c.touch_tagged1(1, NS, 0, 0, CLASS_LOW, &[(3, keys(&[2]))], now);
        assert_eq!(c.lookup(1, NS, &[k(2)], false, now).0, 1);
    }
    assert!(is_low(&c, 2), "low-class reuse keeps it in the low class");
    hsf(&mut c, 3, 3, 0, 7_300_000);
    hsf(&mut c, 4, 3, 0, 7_300_000);
    assert!(!has(&c, 1) && has(&c, 2), "the cold agent entry went, the hot bookkeeping prefix stayed");
    c.check();
}

#[test]
fn low_class_raised_by_normal_use_not_by_low_use() {
    let mut c = core(16);
    hsf(&mut c, 1, 0, ENTRY_LOW, 0);
    hsf(&mut c, 2, 0, ENTRY_LOW, 0);
    hsf(&mut c, 3, 0, ENTRY_LOW, 0);
    assert_eq!((c.c.low_committed, gauge(&c, "low_entries"), gauge(&c, "low_slots")), (3, 3, 3));
    c.touch_tagged1(1, NS, 7, 0, CLASS_LOW, &[(0, keys(&[1, 2, 3]))], 10);
    assert!(is_low(&c, 1) && is_low(&c, 2) && is_low(&c, 3), "a low request's touch keeps them low");
    c.touch_tagged1(1, NS, 8, 0, CLASS_AGENT, &[(0, keys(&[1]))], 20);
    assert!(!is_low(&c, 1), "an agent reuses block 1");
    c.touch(NS, &keys(&[2]), 30);
    assert!(!is_low(&c, 2), "an untagged (plain TOUCH) reuse counts as normal");
    // a normal request storing the same block (EXISTS) raises it; a low one does not
    let mut it = item(3, None, 100);
    it.flags = ENTRY_LOW;
    assert_eq!(c.reserve(2, NS, &[it], 40)[0].0, Status::Exists);
    assert!(is_low(&c, 3));
    assert_eq!(c.reserve(2, NS, &[item(3, None, 100)], 40)[0].0, Status::Exists);
    assert!(!is_low(&c, 3));
    assert_eq!(c.c.low_promoted, 3);
    assert_eq!(gauge(&c, "low_entries"), 0);
    c.check();
}

#[test]
fn tagged_first_request_is_linked_and_names_its_session() {
    // no warm-up runs: a tagged session's very first request links its chain and anchors its state
    let mut c = core(64);
    for b in 1..=3 {
        hs(&mut c, 100 + b, 0, 1000);
    }
    hs(&mut c, 203, 1, 1000);
    c.touch_tagged1(1, NS, 0xabc, 0, CLASS_AGENT, &[(0, keys(&[101, 102, 103])), (1, keys(&[201, 202, 203]))], 1000);
    assert_eq!((c.c.seg_tagged_linked, c.c.seg_runs_skipped, c.c.tagged_sessions), (1, 0, 1));
    assert_eq!(gauge(&c, "seg_states_anchored"), 1);
    // the same session after a compaction: a new chain (no common end), still the same broker session
    hs(&mut c, 301, 0, 2000);
    hs(&mut c, 401, 1, 2000);
    c.touch_tagged1(1, NS, 0xabc, 0, CLASS_AGENT, &[(0, keys(&[301])), (1, keys(&[401]))], 2000);
    assert_eq!(c.c.tagged_sessions, 1, "named by its tag, not by a chain end");
    assert_eq!(c.sessions.len(), 1);
    // another session tag: another session; an untagged request continuing the first chain's end joins its session
    c.touch_tagged1(1, NS, 0xdef, 0xabc, CLASS_AGENT, &[(0, keys(&[101]))], 3000);
    assert_eq!(c.sessions.len(), 2);
    c.touch_request(1, NS, &[(0, keys(&[101, 102, 103])), (1, keys(&[203]))], 4000);
    assert_eq!(c.sessions.len(), 2);
    c.check();
}

#[test]
fn hotset_ranks_low_prefixes_after_agent_prefixes() {
    let mut c = core(64);
    // agent prefix 1..=2 used by 4 requests, bookkeeping prefix 11..=12 used by 9 (more), attention-only namespace
    hs(&mut c, 1, 0, 0);
    hsf(&mut c, 11, 0, ENTRY_LOW, 0);
    store_meta(&mut c, 2, Some(1), 0, 0);
    let mut it = item(12, None, 100);
    it.flags = ENTRY_LOW;
    let mut meta = b"L1".to_vec();
    meta.extend_from_slice(&k(11)[..16]);
    meta.extend_from_slice(&[0u8; 14]);
    it.meta = meta;
    assert_eq!(c.reserve(1, NS, &[it], 0)[0].0, Status::Ok);
    c.commit(1, NS, &[k(12)], 0);
    for r in 0..4u32 {
        c.touch_tagged1(1, NS, 100 + r as u64, 0, CLASS_AGENT, &[(0, keys(&[1, 2, 900 + r]))], 10);
    }
    for r in 0..9u32 {
        c.touch_tagged1(1, NS, 0, 0, CLASS_LOW, &[(0, keys(&[11, 12, 800 + r]))], 10);
    }
    assert!(is_low(&c, 11) && is_low(&c, 12));
    let hot = hot_keys(&c.hot_entries_by_use(3.0, u64::MAX));
    assert_eq!(hot, vec![1, 2, 11, 12], "agent prefix first, the bookkeeping prefix included after it");
    // a budget for one unit: the agent prefix, though the bookkeeping prefix has more use
    assert_eq!(hot_keys(&c.hot_entries_by_use(3.0, 200)), vec![1, 2]);
    assert_eq!(c.hot_stats().over_budget, 2, "units [11] and [11, 12] left out");
}

#[test]
fn seg_low_segment_goes_before_an_agent_segment() {
    // hybrid namespace: an agent turn (blocks 1..3 + state 3) and a later bookkeeping prompt (blocks 11..13 + its
    // state, all ENTRY_LOW): room pressure takes the bookkeeping segment whole, the agent's stays loadable
    let mut c = hybrid(9, 1, |x| x.policy = PolicyKind::Lru);
    chain(&mut c, 3, &[3], 1000);
    for b in 11..=13 {
        hsf(&mut c, 100 + b, 0, ENTRY_LOW, 50_000);
    }
    hsf(&mut c, 213, 1, ENTRY_LOW, 50_000);
    c.touch_tagged1(1, NS, 0, 0, CLASS_LOW, &[(1, keys(&[211, 212, 213])), (0, keys(&[111, 112, 113]))], 50_000);
    assert_eq!(c.c.seg_tagged_linked, 1);
    hs(&mut c, 500, 3, 60_000);
    hs(&mut c, 501, 3, 60_000);   // pool of 9 full: one unit must go
    assert!((101..=103).all(|b| has(&c, b)) && has(&c, 203), "the agent segment stays");
    assert!((111..=113).all(|b| !has(&c, b)) && !has(&c, 213), "the bookkeeping segment left whole");
    c.check();
}

impl Core {
    /// touch_tagged without flags (tests)
    fn touch_tagged1(&mut self, conn: ConnId, ns: u32, session: u64, parent: u64, class: u8, groups: &[(u8, Vec<Key>)], now: u64) {
        self.touch_tagged(conn, ns, session, parent, class, 0, groups, now);
    }
}

fn rank(c: &Core, key: u32) -> u8 {
    rank_of(c.entry_flags(NS, &k(key)).unwrap())
}

#[test]
fn lowest_goes_before_low_before_normal() {
    let mut c = hybrid(4, 1, |x| x.policy = PolicyKind::Lru);
    hsf(&mut c, 1, 3, 0, 1000);                 // normal, oldest
    hsf(&mut c, 2, 3, ENTRY_LOW, 2000);
    hsf(&mut c, 3, 3, ENTRY_LOWEST, 3000);      // lowest, newest
    hsf(&mut c, 4, 3, 0, 4000);
    hsf(&mut c, 5, 3, 0, 5000);
    assert!(!has(&c, 3) && has(&c, 2) && has(&c, 1), "lowest first");
    hsf(&mut c, 6, 3, 0, 6000);
    assert!(!has(&c, 2) && has(&c, 1), "then low");
    hsf(&mut c, 7, 3, 0, 7000);
    assert!(!has(&c, 1), "then the policy order");
    assert_eq!((c.c.lowest_evicted, c.c.low_evicted, c.c.normal_evicted), (1, 1, 1));
    // raises go to the toucher's class: a low request lifts a lowest entry to low, not to normal
    hsf(&mut c, 8, 3, ENTRY_LOWEST, 8000);
    c.touch_tagged1(1, NS, 0, 0, CLASS_LOWEST, &[(3, keys(&[8]))], 8100);
    assert_eq!(rank(&c, 8), 2);
    c.touch_tagged1(1, NS, 0, 0, CLASS_LOW, &[(3, keys(&[8]))], 8200);
    assert_eq!(rank(&c, 8), 1);
    c.touch_tagged1(1, NS, 0, 0, CLASS_AGENT, &[(3, keys(&[8]))], 8300);
    assert_eq!(rank(&c, 8), 0);
    c.check();
}

#[test]
fn terminal_request_drops_its_own_chain_not_the_shared_prefix() {
    // a delegated session: shared specialist prefix 1..2 (every session of that kind uses it), its own blocks 11..13;
    // its terminal request (the write-up) ends it: its own blocks drop to lowest, the shared prefix stays normal
    let mut c = core(64);
    for b in [1, 2, 11, 12, 13] {
        hs(&mut c, b, 0, 0);
    }
    for (s, own) in [(0x10u64, 900u32), (0x20, 901), (0x30, 902)] {     // sibling sessions use the prefix
        c.touch_tagged1(1, NS, s, 0, CLASS_AGENT, &[(0, keys(&[1, 2, own]))], 10);
    }
    for t in 0..3u64 {                                                 // the session's rounds
        c.touch_tagged1(1, NS, 0x40, 0, CLASS_AGENT, &[(0, keys(&[1, 2, 11, 12, 13 + 0 * t as u32]))], 20 + t);
    }
    assert!([1, 2, 11, 12, 13].iter().all(|b| rank(&c, *b) == 0));
    c.touch_tagged(1, NS, 0x40, 0, CLASS_AGENT, TAG_TERMINAL, &[(0, keys(&[1, 2, 11, 12, 13]))], 30);
    assert_eq!((rank(&c, 1), rank(&c, 2)), (0, 0), "the shared prefix keeps its class");
    assert!([11, 12, 13].iter().all(|b| rank(&c, *b) == 2), "the session's own chain is lowest now");
    assert_eq!(c.c.terminal_demoted, 3);
    // a later readout of that session's transcript still hits it (cached, only first to go)
    assert_eq!(c.lookup(1, NS, &keys(&[1, 2, 11, 12, 13]), false, 40).0, 5);
    c.check();
}
