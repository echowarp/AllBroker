"""Engine-side lease safety of llmt_kv/vllm_tier.py: stubbed vllm/torch and a fake
broker with 60 s lease timeouts. Every check must PASS: an engine never loads a slot whose lease may be gone.
  PYTHONDONTWRITEBYTECODE=1 python3 -I test_tier_leases.py [path/to/vllm_tier.py]   (default: the in-repo copy)"""
import enum, importlib.util, sys, types
from dataclasses import dataclass, field

import os
TIER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "llmt_kv", "vllm_tier.py")

def mod(name, **kw):
    m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m; return m

class LookupResult(enum.Enum):
    HIT = 1; MISS = 2; HIT_PENDING = 3; RETRY = 4
@dataclass
class ReqContext:
    req_id: str
    kv_transfer_params: dict | None = None
@dataclass
class ScheduleEndContext:
    new_req_ids: tuple = ()
    finished: tuple = ()
class CPULoadStoreSpec:
    def __init__(self, ids): self.block_ids = list(ids)
class _Log:
    def __getattr__(self, n): return lambda *a, **k: None
for n in ("vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.cpu"):
    mod(n)
mod("torch")
mod("vllm.logger", init_logger=lambda n: _Log())
mod("vllm.v1.kv_offload.base", LoadStoreSpec=object, LookupResult=LookupResult, OffloadingManager=object, OffloadKey=bytes,
    PrepareStoreOutput=object, ReqContext=ReqContext, RequestOffloadingContext=object)
mod("vllm.v1.kv_offload.cpu.common", CPULoadStoreSpec=CPULoadStoreSpec)
mod("vllm.v1.kv_offload.cpu.gpu_worker", CPUOffloadingWorker=object)
mod("vllm.v1.kv_offload.cpu.shared_offload_region", SharedOffloadRegion=object)
mod("vllm.v1.kv_offload.cpu.spec", CPUOffloadingSpec=object)
class BrokerError(Exception): pass
class BrokerReset(BrokerError): pass
class BrokerPoolChanged(BrokerError): pass
kv = mod("llmt_kv", BrokerError=BrokerError, BrokerReset=BrokerReset, BrokerPoolChanged=BrokerPoolChanged,
         BrokerTimeout=BrokerError, BrokerGone=BrokerError, OK=0, EXISTS=1, BUSY=2, FULL=3)
spec = importlib.util.spec_from_file_location("llmt_kv.vllm_tier", TIER)
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)

SLOT = 1000
class Ent:
    def __init__(self, s): self.byte_ranges = [(s * SLOT, SLOT)]
class FakeBroker:
    """key -> slot; leases with a deadline (broker clock `now`, ms)."""
    def __init__(self): self.now, self.idx, self.leases, self.next, self.calls, self.mode = 0, {}, {}, 1, [], None
    def lookup(self, ns, keys, acquire=False):
        self.calls.append("lookup")
        if self.mode: raise self.mode("x")
        k = keys[0]
        if k not in self.idx: return (0, 0, [])
        lid = self.next; self.next += 1; self.leases[lid] = (k, self.now + 60_000)
        return (1, lid, [Ent(self.idx[k])])
    def release(self, ids): self.calls.append("release"); return [0 if self.leases.pop(i, None) else 4 for i in ids]
    def touch(self, ns, keys): self.calls.append("touch")
    def tick(self, now):
        self.now = now
        for i, (k, dl) in list(self.leases.items()):
            if dl <= now: del self.leases[i]
    def leased(self, k): return any(v[0] == k for v in self.leases.values())
    def evict_and_reuse(self, k, newkey):
        assert not self.leased(k), "broker would never evict a leased entry"
        self.idx[newkey] = self.idx.pop(k)

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); cond or fails.append(name)

def mk():
    b = FakeBroker(); m = T.BrokerOffloadingManager(b, "ns", SLOT); return b, m


CLOCK = [0.0]
T.time.monotonic = lambda: CLOCK[0]

# 1) a request waits 90 s (lookups answered every second); leases are refreshed before the broker's 60 s timeout, so
#    the broker can never evict / reuse the slot, and the load uses a live lease
b, m = mk(); K = b"\x01" * 36; ctx = ReqContext("R")
b.idx[m._bk(K)] = 7
check("1: lookup hit", m.lookup(K, ctx) == LookupResult.HIT)
for t in range(1, 91):
    CLOCK[0] = float(t); b.tick(t * 1000); m.lookup(K, ctx); m.on_schedule_end(ScheduleEndContext(()))
check("1: the lease is still alive on the broker after 90 s", b.leased(m._bk(K)))
check("1: refreshed (not answered from a dead lease)", m.n["lease_refresh"] + m.n["renewed"] >= 2)   # lookup-side refresh or broker-side renewal (every LLMT_KV_LEASE_RENEW_S)
check("1: load gets slot 7 under a live lease", m.prepare_load([K], ctx).block_ids == [7])

# 1b) the entry left the broker while the lease was old: the next lookup is a MISS, nothing loads it
b, m = mk(); b.idx[m._bk(K)] = 4; CLOCK[0] = 0.0
m.lookup(K, ReqContext("R")); CLOCK[0] = 40.0; b.tick(70_000); del b.idx[m._bk(K)]
check("1b: stale lease, entry gone -> MISS", m.lookup(K, ReqContext("R")) == LookupResult.MISS)
check("1b: lease dropped", K not in m._leases)

# 2) tier OFF: no HIT from old leases, ever
b, m = mk(); b.idx[m._bk(K)] = 3; CLOCK[0] = 0.0
m.lookup(K, ReqContext("A")); b.mode = BrokerPoolChanged
m._call(b.lookup, 0, [b"x"])
check("2: tier off", m._off)
check("2: a new request gets MISS", m.lookup(K, ReqContext("B")) == LookupResult.MISS)
check("2: old leases forgotten", not m._leases and not m._hint)

# 3) broker reset: old leases / slots forgotten; the load asks the broker again (fresh lease, current slot)
b, m = mk(); b.idx[m._bk(K)] = 5
m.lookup(K, ReqContext("A")); b.mode = BrokerReset
m._call(b.lookup, 0, [b"x"]); b.mode = None
check("3: leases cleared at the reset", not m._leases)
b.idx[m._bk(K)] = 6                                  # the restarted broker holds the key elsewhere now
check("3: load re-acquires and uses the current slot", m.prepare_load([K], ReqContext("A")).block_ids == [6])

# 3b) after a reset the entry is gone: the load refuses instead of reading an unleased slot
b, m = mk(); b.idx[m._bk(K)] = 5
m.lookup(K, ReqContext("A")); b.mode = BrokerReset
m._call(b.lookup, 0, [b"x"]); b.mode = None; del b.idx[m._bk(K)]
try:
    m.prepare_load([K], ReqContext("A")); refused = False
except RuntimeError:
    refused = True
check("3b: unleased load refused", refused)

# 4) every load holds its own lease. A and B wait on the same block; A loads; B gives up (its
#    lease dropped): A's copy is still protected. Then two loads in flight: the first to finish releases its own.
b, m = mk(); b.idx[m._bk(K)] = 9; CLOCK[0] = 0.0
m.lookup(K, ReqContext("A")); m.lookup(K, ReqContext("B"))
m.prepare_load([K], ReqContext("A"))                # B still waits: A takes a lease of its own
m._drop(K, "B")                                     # B aborted: the waiting lease goes
check("4: A's load in flight keeps a broker lease after B gave up", b.leased(m._bk(K)))
m.lookup(K, ReqContext("C"))
m.prepare_load([K], ReqContext("C"))                # C loads the same slot while A's copy runs
m.complete_load([K], ReqContext("A"))               # A finishes first
check("4: C's load still leased after A's completion", b.leased(m._bk(K)) and m._loading.get(K) and m._loading[K][0][2] == "C")
m.complete_load([K], ReqContext("C"))
check("4: all leases released when both loads are done", not b.leased(m._bk(K)) and K not in m._loading)

print("all checks passed" if not fails else "FAILED: %s" % fails)
sys.exit(1 if fails else 0)
