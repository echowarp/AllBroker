"""Logic test of llmt_kv.vllm_tier (the engines' shared broker tier) against a private broker that this script starts,
kills and restarts. Needs vllm importable (run inside an engine image); no GPU work (the manager runs in the
scheduler; the worker side is covered by the engine runs).
  python test_vllm_tier.py BROKER_BINARY SCRATCH_DIR     (with a 1 MiB tmpfs on /small: also the full-tmpfs case)
"""
import os, signal, subprocess, sys, time

import llmt_kv
from llmt_kv import vllm_tier as T
from vllm.v1.kv_offload.base import LookupResult, ReqContext, ScheduleEndContext, make_offload_key

BIN, S = sys.argv[1], sys.argv[2]
SOCK, POOL = f"{S}/broker.sock", f"{S}/pool"
SLOT = 256 << 10
N_SLOTS = 6
CTX = ReqContext(req_id="r1")
EXTRA = {"broker_socket": SOCK, "broker_wait_s": 1, "broker_reconnect_ms": 100, "broker_timeout_ms": 500}


def start(*extra):
    p = subprocess.Popen([BIN, "--pool-gib", str(N_SLOTS * SLOT / 2**30), "--slot-kib", str(SLOT >> 10), "--pool-dir",
                          POOL, "--socket", SOCK, "--metrics", "off", "--low-water-pct", "0", "--reserve-timeout-ms",
                          "300", "--reuse-grace-ms", "500", "--hotset-dir", "off", *extra], stderr=open(f"{S}/broker.log", "a"))
    for _ in range(100):
        if os.path.exists(SOCK):
            try:
                llmt_kv.Client(SOCK)
                return p
            except llmt_kv.BrokerError:
                pass
        time.sleep(0.05)
    raise SystemExit("broker did not start")


def stop(p):
    p.send_signal(signal.SIGKILL)
    p.wait()
    os.unlink(SOCK)


def key(i):
    return make_offload_key(bytes([i]) * 32, 0)


def manager(name, lease_steps=1024, lease_cap=0):
    return T.BrokerOffloadingManager(T.connect(EXTRA, 1, name), "test-ns", SLOT, lease_steps, lease_cap)


def steps(m, n, admitted=None):
    """n scheduler steps; admitted = request ids admitted in the last one (None: no admission info, older vLLM)."""
    for i in range(n):
        m.on_schedule_end(None if admitted is None else ScheduleEndContext(admitted if i == n - 1 else (), ()))


def store(m, keys):
    out = m.prepare_store(keys, CTX)
    assert out is not None and out.keys_to_store == keys, out
    m.complete_store(keys, CTX, True)
    return list(out.store_spec.block_ids)


def check(cond, what):
    assert cond, what
    print("ok -", what)


# 1. no broker at engine start: tier off, nothing raises
m0 = manager("nobroker")
check(isinstance(m0.c, T._NoBroker) and m0._off, "no broker at start -> tier OFF, engine starts")
check(m0.lookup(key(1), CTX) == LookupResult.MISS and m0.prepare_store([key(1)], CTX) is None,
      "tier OFF: lookups miss, stores skipped")

# 1b. pool that can't be fully allocated (full tmpfs; needs a 1 MiB tmpfs on /small, e.g. docker run --tmpfs
#     /small:size=1m): tier off instead of SIGBUS on the first write
if os.path.ismount("/small"):
    open("/small/pool", "wb").close()
    check(not T._pool_backed("/small/pool", 8 << 20), "pool larger than a full tmpfs -> not backed -> tier OFF")
    check(T._pool_backed("/small/pool", 256 << 10), "pool that fits -> backed")

b = start("--new-pool")                                         # a pool left in SCRATCH_DIR by an earlier run
# 2. store from one engine, load from another
a = manager("a")
check(a.lookup(key(1), CTX) == LookupResult.MISS, "cold lookup misses")
slots = store(a, [key(1), key(2)])
check(a.n["committed"] == 2 and len(set(slots)) == 2, "store: 2 reserved, 2 committed")
e = manager("e")
check(e.lookup(key(1), CTX) == LookupResult.HIT and key(1) in e._leases, "other engine: hit + lease held")
spec = e.prepare_load([key(1)], CTX)
check(list(spec.block_ids) == [slots[0]], "prepare_load returns the stored slot")
e.complete_load([key(1)], CTX)
check(not e._loading, "complete_load releases")

# 3. leases belong to requests: kept while any looking-up request needs them, released when it finishes
CTX2 = ReqContext(req_id="r2")
e.lookup(key(2), CTX)
e.lookup(key(2), CTX2)
steps(e, 10, admitted=())
check(key(2) in e._leases and e._leases[key(2)][3] == {"r1", "r2"}, "lease held for both waiting requests")
spec = e.prepare_load([key(2)], CTX)
check(list(spec.block_ids) == [slots[1]] and key(2) in e._leases, "r1 loads; lease stays for waiting r2")
e.complete_load([key(2)], CTX)
e.on_request_finished(CTX2)
check(key(2) not in e._leases, "r2 finished -> lease released")
e.on_request_finished(CTX)
spec = e.prepare_load([key(2)], CTX)
check(list(spec.block_ids) == [slots[1]] and e.n["unleased"] == 0, "load without a held lease re-acquires it")
e.complete_load([key(2)], CTX)
c1 = manager("cap", lease_cap=1)
check(c1.lookup(key(1), CTX) == LookupResult.HIT and c1.lookup(key(2), CTX) == LookupResult.MISS
      and c1.n["lease_cap"] == 1, "lease cap: lookups beyond the cap miss instead of pinning more")
c1.on_request_finished(CTX)
check(not c1._leases, "cap manager: finish releases")

# 4. no admission info (older vLLM: context None): a lease nobody loaded is released at the end of the lookup's step
d = manager("default")
check(d.lookup(key(1), CTX) == LookupResult.HIT and d.lookup(key(2), CTX) == LookupResult.HIT, "default: hits leased")
spec = d.prepare_load([key(1)], CTX)
steps(d, 1)
check(list(d._leases) == [] and d.n["lease_expired"] == 1 and key(1) in d._loading,
      "no admission info: unloaded lease released at step end; the load in flight keeps its lease")
d.complete_load([key(1)], CTX)
check(not d._loading and not d._req_keys.get("r1"), "default: nothing held after the load")
# 4b. with admission info (vLLM 0.26): a waiting request keeps its leases (re-lookups answered locally), admission
#     releases what it did not load, the safety net still covers a request that never gets admitted
w = manager("waiting")
CTX3 = ReqContext(req_id="r3")
check(w.lookup(key(1), CTX3) == LookupResult.HIT and w.lookup(key(2), CTX3) == LookupResult.HIT, "waiting: hits")
lk = w.n["hits"]
steps(w, 5, admitted=())
check(w.lookup(key(1), CTX3) == LookupResult.HIT and w.n["hits"] == lk and w.n["local"] == 1 and len(w._leases) == 2,
      "waiting request keeps its leases across steps; its re-lookup is answered locally (no broker call)")
w.prepare_load([key(1)], CTX3)
steps(w, 1, admitted=["r3"])
check(not w._leases and w.n["lease_admitted"] == 1 and key(1) in w._loading,
      "admitted: the lease it did not load is released, the load in flight keeps its own")
w.complete_load([key(1)], CTX3)
w.lookup(key(2), CTX3)
steps(w, 1100, admitted=())
check(not w._leases and w.n["lease_expired"] == 1, "never admitted, never finished: safety net releases it")
e.lookup(key(1), CTX)
steps(e, 1100, admitted=())
check(key(1) not in e._leases and e.n["lease_expired"] == 1, "lease_steps 1024: idle lease released by the safety net")
store(a, [key(10 + i) for i in range(N_SLOTS)])                  # fills the pool: evicts keys 1 and 2
try:                                     # an unleased slot is never loaded (its bytes may be another block's)
    e.prepare_load([key(1)], CTX)
    refused = False
except RuntimeError:
    refused = True
check(refused and e.n["unleased"] == 1, "evicted between lookup and load -> load refused (never an unleased slot)")

# 4c. per-request namespace (llmt_ns, allowlisted) + conversation hints (llmt_conv)
HINTS = f"{S}/hints/test.json"
x1 = T.BrokerOffloadingManager(T.connect(EXTRA, 1, "x1"), "ns-one", SLOT, 1024, 0, ns_allow=["xgpu-test"],
                               hints_path=HINTS, client_name="test")
x2 = T.BrokerOffloadingManager(T.connect(EXTRA, 2, "x2"), "ns-two", SLOT, 1024, 0, ns_allow=["xgpu-test"])
XS = ReqContext(req_id="xs", kv_transfer_params={"llmt_ns": "xgpu-test", "llmt_conv": "c1"})
XL = ReqContext(req_id="xl", kv_transfer_params={"llmt_ns": "xgpu-test"})
NOPE = ReqContext(req_id="xn", kv_transfer_params={"llmt_ns": "not-allowed"})
xk = key(20)
check(x2.lookup(xk, XL) == LookupResult.MISS, "shared test namespace starts empty")
out = x1.prepare_store([xk], XS)
check(out is not None and out.keys_to_store == [xk], "store under the requested namespace returns the vLLM key")
x1.complete_store([xk], XS, True)
check(x1.n["committed"] == 1 and x2.lookup(xk, XL) == LookupResult.HIT,
      "the other engine finds it under the same requested namespace")
check(x2.lookup(xk, ReqContext(req_id="own")) == LookupResult.MISS and x2.lookup(xk, NOPE) == LookupResult.MISS,
      "own namespace and a non-allowlisted one do not see it")
spec = x2.prepare_load([xk], XL)
x2.complete_load([xk], XL)
check(len(list(spec.block_ids)) == 1 and not x2._loading, "load from the requested namespace")
x1.on_request_finished(XS)
x1._hints_t = 0
steps(x1, 1, admitted=())
import json as _json
h = _json.load(open(HINTS))
check(h["client"] == "test" and h["convs"]["c1"]["chunks"] == 1 and h["convs"]["c1"]["ns"] == "xgpu-test",
      "conversation hint written: c1 has 1 chunk in RAM")
x2.on_request_finished(XL)

# 4d. entry kinds per KV group (0 attention, 1 GDN state) and per-kind counters
kk = T.BrokerOffloadingManager(T.connect(EXTRA, 1, "kinds"), "ns-kinds", SLOT, 1024, 0, group_kinds=(1, 0))
ks, ka = make_offload_key(bytes([60]) * 32, 0), make_offload_key(bytes([61]) * 32, 1)
check(kk._kind(ks) == 1 and kk._kind(ka) == 0 and kk._kind(make_offload_key(bytes([62]) * 32, 7)) == 3,
      "group 0 -> state, group 1 -> attention, unknown group -> 3")
out = kk.prepare_store([ks, ka], CTX)
kk.complete_store([ks, ka], CTX, True)
check(kk.n["committed_state"] == 1 and kk.n["committed_attn"] == 1, "commits counted per kind")
KC = ReqContext(req_id="kc")
kk.lookup(ks, KC)
kk.lookup(ka, KC)
kk.prepare_load([ka], KC)
steps(kk, 1, admitted=["kc"])
kk.complete_load([ka], KC)
check(kk.n["hits_state"] == 1 and kk.n["hits_attn"] == 1 and kk.n["loads_attn"] == 1 and kk.n["loads_state"] == 0
      and kk.n["admitted_state"] == 1, "hits/loads per kind; the state hit that was not loaded counted as admitted_state")
kk.on_request_finished(CTX)
kk.on_request_finished(KC)

# 5. a reservation that expired before commit is counted as lost
out = a.prepare_store([key(30)], CTX)
time.sleep(0.6)
a.complete_store([key(30)], CTX, True)
check(a.n["lost"] == 1, "expired reservation -> commit refused, counted as lost")

# 6. broker restart (pool reused): no crash, Reset, nothing held from before is loaded (the restarted
#    broker zeroed / may reuse those slots), stores again after grace
k = key(15)                                                      # newest of 10..15 (4c stored one more entry)
check(e.lookup(k, CTX) == LookupResult.HIT, "lease before the restart")
held_slot = e._leases[k][1]
stop(b)
check(e.lookup(key(12), CTX) == LookupResult.MISS, "broker down: lookup misses (no raise)")
check(e.prepare_store([key(40)], CTX) is None, "broker down: store skipped")
b = start()
time.sleep(0.15)
e.lookup(key(12), CTX)                                           # reconnects -> BrokerReset handled inside
check(e.n["resets"] == 1 and k not in e._leases and not e._hint, "reconnect: reset counted, every lease / hint forgotten")
try:
    e.prepare_load([k], CTX)
    refused = False
except RuntimeError:
    refused = True
check(refused, "the slot held before the restart is not loaded (the new index is empty: refused)")
check(e.prepare_store([key(41)], CTX) is None, "grace period: stores refused (FULL)")
CR = ReqContext(req_id="chain")
check(e.prepare_store([key(43), key(44)], CR) is None and not e._last_stored.get("chain"),
      "refused stores do not advance the request's parent chain")
time.sleep(0.6)
store(e, [key(41)])
check(e.lookup(key(41), CTX) == LookupResult.HIT, "after grace: store + hit work again")
seen = []
_call = e._call
e._call = lambda fn, *args: (seen.extend(args[1]) if fn == e.c.reserve else None, _call(fn, *args))[1]
out = e.prepare_store([key(43), key(44)], CR)
e.complete_store([key(43), key(44)], CR, True)
e._call = _call
par = {it["key"]: it["meta"][2:18] for it in seen}
b43, b44 = e._bk(key(43)), e._bk(key(44))
check(out is not None and par[b43] == bytes(16) and par[b44] == b43[:16],
      "retried store after a refusal: first block has no parent, second points to the first (no self-parent cycle)")

# 6b. llmt_nostore: the request stores nothing (it may still load)
NS = ReqContext(req_id="ns", kv_transfer_params={"llmt_nostore": True})
out = e.prepare_store([key(45)], NS)
check(out is not None and out.keys_to_store == [] and e.n["nostore"] == 1 and e.lookup(key(45), CTX) == LookupResult.MISS,
      "llmt_nostore: nothing reserved or stored")
check(e.lookup(key(41), NS) == LookupResult.HIT, "llmt_nostore request still finds stored entries")

# 7. broker restarted with a new pool: tier off for good, no raise
stop(b)
b = start("--new-pool")
e.lookup(key(50), CTX)
time.sleep(0.15)
e.lookup(key(50), CTX)
check(e._off and e.lookup(key(42), CTX) == LookupResult.MISS, "pool changed -> tier OFF until engine restart (held leases still served from the old mapping)")
stop(b)
print("PASS vllm_tier")
