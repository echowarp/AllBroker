"""The connector's HINT path (docs/prefix-hints.md): a request with kv_transfer_params.llmt_anchor sends one HINT when
it finishes, with its keys per KV group in token order (looked-up and stored keys, deduplicated), the group kinds and
the chunk size; requests without an anchor, brokers without FEAT_PREFIX_HINTS and broker errors send nothing / change
nothing. Stubs as in test_tier_statemachine, without hypothesis.
  python test_prefix_hint.py [path/to/vllm_tier.py]
"""
import enum
import importlib.util
import os
import sys
import types
from dataclasses import dataclass

TIER = (sys.argv[1] if len(sys.argv) > 1 else None) or os.environ.get("TIER_PY") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "python", "llmt_kv", "vllm_tier.py")


def mod(name, **kw):
    m = types.ModuleType(name)
    m.__dict__.update(kw)
    sys.modules[name] = m
    return m


class LookupResult(enum.Enum):
    HIT = 1
    MISS = 2


@dataclass
class ReqContext:
    req_id: str
    kv_transfer_params: dict | None = None


class CPULoadStoreSpec:
    def __init__(self, ids):
        self.block_ids = list(ids)


class PrepareStoreOutput:
    def __init__(self, keys_to_store, store_spec, evicted_keys):
        self.keys_to_store = keys_to_store


class _Log:
    def __getattr__(self, n):
        return lambda *a, **k: None


class BrokerError(Exception):
    pass


for n in ("vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.cpu"):
    mod(n)
mod("torch")
mod("vllm.logger", init_logger=lambda n: _Log())
mod("vllm.v1.kv_offload.base", LoadStoreSpec=object, LookupResult=LookupResult, OffloadingManager=object,
    OffloadKey=bytes, PrepareStoreOutput=PrepareStoreOutput, ReqContext=ReqContext, RequestOffloadingContext=object)
mod("vllm.v1.kv_offload.cpu.common", CPULoadStoreSpec=CPULoadStoreSpec)
mod("vllm.v1.kv_offload.cpu.gpu_worker", CPUOffloadingWorker=object)
mod("vllm.v1.kv_offload.cpu.shared_offload_region", SharedOffloadRegion=object)
mod("vllm.v1.kv_offload.cpu.spec", CPUOffloadingSpec=object)
mod("llmt_kv", BrokerError=BrokerError, BrokerReset=BrokerError, BrokerPoolChanged=type("PC", (BrokerError,), {}),
    BrokerTimeout=BrokerError, BrokerGone=BrokerError, OK=0, EXISTS=1, BUSY=2, FULL=3, FEAT_PREFIX_HINTS=1 << 5,
    status_name=str)
_spec = importlib.util.spec_from_file_location("llmt_kv.vllm_tier", TIER)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

SLOT = 1000
ANCHOR = bytes(range(16))


def key(group: int, i: int) -> bytes:
    return bytes([i]) * 32 + group.to_bytes(4, "big")


class Broker:
    """Misses every lookup; reserve -> OK; records HINTs."""

    def __init__(self, features=1 << 5, fail=False):
        self.features, self.fail, self.hints, self.slot = features, fail, [], 0
        self.connected, self.reconnects = True, 0

    def lookup(self, ns, keys, acquire=False):
        return (0, 0, [])

    def reserve(self, ns, items):
        out = []
        for _ in items:
            self.slot += 1
            out.append((0, [(self.slot * SLOT, SLOT)]))
        return out

    def commit(self, ns, keys):
        return [0] * len(keys)

    def touch(self, ns, keys):
        return [0] * len(keys)

    def hint(self, ns, anchor, chunk_tokens, groups, flags=0):
        if self.fail:
            raise BrokerError("x")
        self.hints.append((ns, anchor, chunk_tokens, groups, flags))


def manager(b):
    return T.BrokerOffloadingManager(b, "ns", SLOT, group_kinds=(0, 1), hint_chunk_tokens=3200)


def run(m, ctx):
    for i in (1, 2, 3):                                  # scheduler lookups (miss at 1 here), both groups
        m.lookup(key(0, i), ctx)
        m.lookup(key(1, i), ctx)
    m.lookup(key(0, 1), ctx)                             # a repeated check: deduplicated
    m.prepare_store([key(0, i) for i in (1, 2, 3, 4)] + [key(1, 4)], ctx)
    m.on_request_finished(ctx)


def test_hint_sent_with_groups_in_order():
    b = Broker()
    m = manager(b)
    run(m, ReqContext("r1", {"llmt_anchor": ANCHOR.hex()}))
    assert len(b.hints) == 1, b.hints
    ns, anchor, ct, groups, flags = b.hints[0]
    assert anchor == ANCHOR and ct == 3200 and flags == 1 and ns == m.ns
    assert [g[0] for g in groups] == [0, 1], "group kinds from group_kinds, in group order"
    assert groups[0][1] == [m._bk(key(0, i)) for i in (1, 2, 3, 4)], "attention keys in token order, deduplicated"
    assert groups[1][1] == [m._bk(key(1, i)) for i in (1, 2, 3, 4)]
    assert not m._anchor, "per-request state dropped at finish"


def test_no_anchor_no_hint():
    b = Broker()
    m = manager(b)
    run(m, ReqContext("r2", {"llmt_ns": "shared"}))
    run(m, ReqContext("r3", None))
    run(m, ReqContext("r4", {"llmt_anchor": "zz"}))
    assert b.hints == [] and not m._anchor


def test_old_broker_and_errors():
    b = Broker(features=0)
    m = manager(b)
    run(m, ReqContext("r5", {"llmt_anchor": ANCHOR.hex()}))
    assert b.hints == [], "broker without FEAT_PREFIX_HINTS: nothing sent"
    b = Broker(fail=True)
    m = manager(b)
    run(m, ReqContext("r6", {"llmt_anchor": ANCHOR.hex()}))
    assert b.hints == [] and not m._anchor, "a broker error is swallowed (tier logic unchanged)"


def test_shared_namespace_id():
    b = Broker()
    m = T.BrokerOffloadingManager(b, "arc", SLOT, group_kinds=(0,), ns_allow=("shared",))
    run(m, ReqContext("r7", {"llmt_anchor": ANCHOR.hex(), "llmt_ns": "shared"}))
    assert b.hints and b.hints[0][0] == m._nsinfo("shared")[0], "hint in the request's namespace"


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
