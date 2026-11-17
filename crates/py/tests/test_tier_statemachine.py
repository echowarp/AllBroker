"""Stateful property test of the engine connector's lease logic (llmt_kv/vllm_tier.py).

Hypothesis drives vllm_tier.py (vllm / torch stubbed) against a fake broker that behaves like the real
one where it matters: leases end at a deadline (60 s) unless released; it evicts only unleased entries and reuses
their slots for other data at once; it can reset the connection (BrokerReset) or switch pools (BrokerPoolChanged).
Rules: requests arrive; scheduling steps look a request's chain up and maybe admit it (prepare_load in the same step,
as vLLM does), loads complete, requests finish or abort, the clock jumps, the broker evicts / stores / resets.
Invariants:
  1. prepare_load only returns slots that hold that key right now, under a live broker lease;
  2. while a load is in flight, its slot is never evictable on the broker (a lease of the engine covers it);
  3. a load promised by a lookup in the same step is never refused.
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_tier_statemachine.py     (needs hypothesis + pytest)
  or: python test_tier_statemachine.py [path/to/vllm_tier.py]
"""
import enum
import importlib.util
import os
import sys
import types
from dataclasses import dataclass

from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

TIER = os.environ.get("TIER_PY") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "python", "llmt_kv", "vllm_tier.py")


def mod(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


class LookupResult(enum.Enum):
    HIT = 1
    MISS = 2
    HIT_PENDING = 3
    RETRY = 4


@dataclass
class ReqContext:
    req_id: str
    kv_transfer_params: dict | None = None


@dataclass
class ScheduleEndContext:
    new_req_ids: tuple = ()
    finished: tuple = ()


class CPULoadStoreSpec:
    def __init__(self, ids):
        self.block_ids = list(ids)


class _Log:
    def __getattr__(self, n):
        return lambda *a, **k: None


for n in ("vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.cpu"):
    mod(n)
mod("torch")
mod("vllm.logger", init_logger=lambda n: _Log())
mod("vllm.v1.kv_offload.base", LoadStoreSpec=object, LookupResult=LookupResult, OffloadingManager=object,
    OffloadKey=bytes, PrepareStoreOutput=object, ReqContext=ReqContext, RequestOffloadingContext=object)
mod("vllm.v1.kv_offload.cpu.common", CPULoadStoreSpec=CPULoadStoreSpec)
mod("vllm.v1.kv_offload.cpu.gpu_worker", CPUOffloadingWorker=object)
mod("vllm.v1.kv_offload.cpu.shared_offload_region", SharedOffloadRegion=object)
mod("vllm.v1.kv_offload.cpu.spec", CPUOffloadingSpec=object)


class BrokerError(Exception):
    pass


class BrokerReset(BrokerError):
    pass


class BrokerPoolChanged(BrokerError):
    pass


mod("llmt_kv", BrokerError=BrokerError, BrokerReset=BrokerReset, BrokerPoolChanged=BrokerPoolChanged,
    BrokerTimeout=BrokerError, BrokerGone=BrokerError, OK=0, EXISTS=1, BUSY=2, FULL=3)
_spec = importlib.util.spec_from_file_location("llmt_kv.vllm_tier", TIER)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

CLOCK = [0.0]
T.time.monotonic = lambda: CLOCK[0]
SLOT = 1000
LEASE_S = 60.0
KEYS = [bytes([k]) * 36 for k in range(1, 9)]       # chain of 8 blocks: a request asks for a prefix of it


class Ent:
    def __init__(self, s):
        self.byte_ranges = [(s * SLOT, SLOT)]


class FakeBroker:
    """broker key -> slot; leases (id -> (bk, deadline)); evicts only unleased entries; reuses slots at once."""

    def __init__(self):
        self.idx, self.leases, self.next_lease, self.next_slot, self.free = {}, {}, 1, 0, []
        self.mode = None
        self.owner = {}                                  # slot -> broker key whose bytes it holds
        self.connected, self.reconnects = True, 0        # read by the connector's stats log

    def _slot(self):
        return self.free.pop() if self.free else self._new()

    def _new(self):
        self.next_slot += 1
        return self.next_slot

    def store(self, bk):
        if bk in self.idx:
            return
        s = self._slot()
        self.idx[bk], self.owner[s] = s, bk

    def leased(self, bk):
        return any(v[0] == bk and v[1] > CLOCK[0] for v in self.leases.values())

    def evict(self, bk):
        if bk in self.idx and not self.leased(bk):
            s = self.idx.pop(bk)
            self.owner[s] = b"other data"
            self.free.append(s)                          # the next store reuses it
            return True
        return False

    def expire(self):
        for i, (_, dl) in list(self.leases.items()):
            if dl <= CLOCK[0]:
                del self.leases[i]

    # client API used by vllm_tier
    def lookup(self, ns, keys, acquire=False):
        if self.mode:
            m, self.mode = self.mode, None
            raise m("x")
        k = keys[0]
        if k not in self.idx:
            return (0, 0, [])
        lid = self.next_lease
        self.next_lease += 1
        self.leases[lid] = (k, CLOCK[0] + LEASE_S)
        return (1, lid, [Ent(self.idx[k])])

    def release(self, ids):
        return [0 if self.leases.pop(i, None) else 4 for i in ids]

    def touch(self, ns, keys):
        return [0] * len(keys)

    def heat(self, ns, keys, hot):
        return [0] * len(keys)


class TierMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        CLOCK[0] = 0.0
        self.b = FakeBroker()
        self.m = T.BrokerOffloadingManager(self.b, "ns", SLOT)
        self.waiting = {}        # rid -> chain length wanted
        self.loading = {}        # rid -> [(key, slot)]
        self.running = set()
        self.n = 0
        self.refused = 0
        self.epoch = 0           # broker resets so far; loads started before one cannot be protected (re-zeroed pool)
        self.load_epoch = {}     # rid -> epoch its load started in

    def bk(self, key):
        return self.m._bk(key)

    # ---- requests
    @rule(length=st.integers(1, len(KEYS)))
    def new_request(self, length):
        self.n += 1
        self.waiting[f"r{self.n}"] = length

    @precondition(lambda self: self.waiting)
    @rule(data=st.data(), admit=st.booleans())
    def schedule_step(self, data, admit):
        rid = data.draw(st.sampled_from(sorted(self.waiting)))
        ctx = ReqContext(rid)
        hits = []
        for key in KEYS[:self.waiting[rid]]:              # the connector's chunk lookups, prefix until a miss
            if self.m.lookup(key, ctx) != LookupResult.HIT:
                break
            hits.append(key)
        if admit:
            del self.waiting[rid]
            if hits:
                try:
                    spec = self.m.prepare_load(hits, ctx)
                except RuntimeError:
                    self.refused += 1
                    raise AssertionError("a load the same step's lookup promised was refused")
                pairs = list(zip(hits, spec.block_ids))
                for key, slot in pairs:
                    bk = self.bk(key)
                    assert self.b.owner.get(slot) == bk, f"load of {bk[:2].hex()} got slot {slot} holding other bytes"
                    assert self.b.leased(bk), f"load of {bk[:2].hex()} from slot {slot} without a live broker lease"
                self.loading[rid] = pairs
                self.load_epoch[rid] = self.epoch
            else:
                self.running.add(rid)
            self.m.on_schedule_end(ScheduleEndContext(new_req_ids=(rid,)))
        else:
            self.m.on_schedule_end(ScheduleEndContext(new_req_ids=()))

    @precondition(lambda self: self.loading)
    @rule(data=st.data())
    def complete_load(self, data):
        rid = data.draw(st.sampled_from(sorted(self.loading)))
        pairs = self.loading.pop(rid)
        same_epoch = self.load_epoch.pop(rid, self.epoch) == self.epoch
        for key, slot in pairs:
            if same_epoch:
                assert self.b.owner.get(slot) == self.bk(key), "a slot changed owner while its load was in flight"
        self.m.complete_load([k for k, _ in pairs], ReqContext(rid))
        self.running.add(rid)

    @precondition(lambda self: self.running or self.waiting)
    @rule(data=st.data())
    def finish(self, data):
        rid = data.draw(st.sampled_from(sorted(self.running | set(self.waiting))))
        self.running.discard(rid)
        self.waiting.pop(rid, None)                       # an abort of a waiting request
        self.m.on_request_finished(ReqContext(rid))

    # ---- time and the broker
    @rule(dt=st.sampled_from([0.5, 5.0, 29.0, 31.0, 61.0, 120.0]))
    def advance(self, dt):
        # copies take ms..s: long jumps only while no load is in flight (leases outlive any real copy)
        if dt >= 29.0 and self.loading:
            dt = 0.5
        CLOCK[0] += dt
        self.b.expire()

    @rule(i=st.integers(0, len(KEYS) - 1))
    def broker_store(self, i):
        self.b.store(self.bk(KEYS[i]))

    @rule(i=st.integers(0, len(KEYS) - 1))
    def broker_evict(self, i):
        self.b.evict(self.bk(KEYS[i]))

    @rule(kind=st.sampled_from(["reset", "pool"]))
    def broker_fault(self, kind):
        self.b.mode = BrokerReset if kind == "reset" else BrokerPoolChanged
        self.m._call(self.b.lookup, 0, [b"x"])
        if kind == "reset":                                # a reset broker lost its index and leases
            self.epoch += 1
            for bk in list(self.b.idx):
                s = self.b.idx.pop(bk)
                self.b.owner[s] = b"zeros"
                self.b.free.append(s)
            self.b.leases.clear()

    @invariant()
    def in_flight_loads_protected(self):
        for rid, pairs in self.loading.items():
            if self.load_epoch.get(rid, self.epoch) != self.epoch:
                continue     # started before a broker reset (pool re-zeroed): unprotectable; production uses --new-pool
            for key, slot in pairs:
                assert self.b.owner.get(slot) == self.bk(key), "in-flight load lost its slot"


TierMachine.TestCase.settings = settings(max_examples=600, stateful_step_count=80, deadline=None)
test_tier_statemachine = TierMachine.TestCase

if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.environ["TIER_PY"] = sys.argv[1]
    import unittest
    unittest.main(argv=[sys.argv[0]], defaultTest="test_tier_statemachine", verbosity=1)
