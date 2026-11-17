"""Lend fences and the one-tower guard: CPU checks.
 1 load fence: an in-flight load's destination ids are never lent (non-contiguous and contiguous lends)
 2 finished-store fence: a finished request's not-yet-built store chunks are never lent
 3 pending store jobs (_jobs / _block_id_to_pending_jobs) are never lent
 4 unreadable connector state: nothing is lent
 5 no fence hit: the plain get_new_blocks path (coldest first)
 6 two lazy vision towers in one process: vision lending refuses (one GPU copy of the tower at most)
EXL3_LORA_HOT=1."""
import os, types
import numpy as np
os.environ["EXL3_LORA_HOT"] = "1"
import torch
from llmt_kv import lend as L
from vllm.v1.core.kv_cache_utils import KVCacheBlock, FreeKVCacheBlockQueue
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as osch

NB, P, NS = 24, 2048, 2
raw = [torch.zeros(NB * P, dtype=torch.int8) for _ in range(NS)]
L._kv_layout_init({f"l{i}": raw[i].view(torch.float16).view(NB, P // 2) for i in range(NS)}, NB)
L._ST.update(layers=[], dev=torch.device("cpu"), ready=True)
L.LEND_KEEP_FRAC = 0.0


class Pool:
    enable_caching = True

    def __init__(self):
        self.blocks = [KVCacheBlock(i) for i in range(NB)]; self.blocks[0].is_null = True
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks[1:])
        self.cached_block_hashes_by_block = {}

    def get_num_free_blocks(self): return self.free_block_queue.num_free_blocks

    def _maybe_evict_cached_block(self, b): pass

    def get_new_blocks(self, n):
        out = [self.free_block_queue.popleft() for _ in range(n)]
        for b in out:
            b.ref_cnt += 1
        return out


class FakeCS:
    def __init__(self):
        self._current_batch_load_jobs = {}; self._jobs = {}; self._req_status = {}; self._block_id_to_pending_jobs = {}
        self.config = types.SimpleNamespace(blocks_per_chunk=2)

    def build_connector_meta(self, out):
        self._current_batch_load_jobs = {}
        return "meta"


real = osch.OffloadingConnectorScheduler
osch.OffloadingConnectorScheduler = FakeCS
L._patch_load_fence()


def setup(cs):
    L._TENANTS.clear(); L._ST["n"] = {}; L._LOADS.clear()
    L._ST["sched"] = types.SimpleNamespace(running=[], waiting=[], connector=types.SimpleNamespace(connector_scheduler=cs))
    return Pool()



# 2 load fence: ids 1..3 are the destination of an in-flight load handed to the worker
cs = FakeCS()
cs._current_batch_load_jobs = {7: types.SimpleNamespace(dst_spec=types.SimpleNamespace(block_ids=np.array([1, 2, 3])))}
cs._jobs = {7: object()}
pool = setup(cs)
L._LOADS.clear()
cs.build_connector_meta("so")            # captures the destinations, then the connector forgets them
assert {1, 2, 3} == L._LOADS[7]
b = L._lend(pool, "lora", 4, "t")
assert b is not None and not ({x.block_id for x in b} & {1, 2, 3}), [x.block_id for x in b]
pool = setup(cs); L._LOADS[7] = {1, 2, 3}
r = L._lend(pool, "vision", 5, "t", contiguous=True)
assert r is not None and min(x.block_id for x in r) >= 4, [x.block_id for x in r]
del cs._jobs[7]
pool = setup(cs)
b = L._lend(pool, "lora", 3, "t")
assert [x.block_id for x in b] == [1, 2, 3]
print("1 OK: load fence")

# 3 finished-store fence: a finished request remembers ids 5..8, chunk 0 (ids 5, 6) already built
cs = FakeCS()
rq = types.SimpleNamespace(is_finished=lambda: True)
cs._req_status = {"r": types.SimpleNamespace(req=rq, group_states=[types.SimpleNamespace(next_stored_chunk_idx=1, block_ids=[5, 6, 7, 8])])}
pool = setup(cs)
for b in pool.blocks[1:9]:
    pool.free_block_queue.remove(b); b.ref_cnt = 1       # 1..8 busy except what we free below
for i in (5, 6, 7, 8):
    pool.blocks[i].ref_cnt = 0; pool.free_block_queue.append(pool.blocks[i])
blocked, ok = L._lend_fence()
assert ok and {7, 8} <= blocked and not ({5, 6} & blocked), blocked
b = L._lend(pool, "lora", 2, "t")
assert not ({x.block_id for x in b} & {7, 8}), [x.block_id for x in b]
rq.is_finished = lambda: False           # running: its blocks have ref_cnt > 0 anyway; not fenced
blocked, ok = L._lend_fence()
assert not ({7, 8} & blocked)
print("2 OK: finished-store fence")

# 4 pending store jobs
cs = FakeCS()
cs._block_id_to_pending_jobs = {2: {1}}
cs._jobs = {1: types.SimpleNamespace(non_sliding_window_block_ids=[3, 4], sliding_window_block_ids=None)}
pool = setup(cs)
b = L._lend(pool, "lora", 3, "t")
assert not ({x.block_id for x in b} & {2, 3, 4}), [x.block_id for x in b]
assert L._ST["n"].get("lends_fenced") == 1
print("3 OK: pending store jobs fenced")


# 5 unreadable state
class Bad:
    @property
    def _block_id_to_pending_jobs(self): raise RuntimeError("x")
pool = setup(Bad())
assert L._lend(pool, "lora", 2, "t") is None and L._lend(pool, "vision", 2, "t", contiguous=True) is None
assert L._ST["n"]["fence_errors"] >= 1
print("4 OK: unreadable connector state lends nothing")

# 6 plain path
pool = setup(FakeCS())
b = L._lend(pool, "lora", 3, "t")
assert [x.block_id for x in b] == [1, 2, 3] and "lends_fenced" not in L._ST["n"]
print("5 OK: plain coldest-first path")
# 7 one tower: two lazy-vision towers in the process -> no tower for lending (never a second lent run)
import sys
t1, t2 = object(), object()
sys.modules["llmt_lazyvis"] = types.SimpleNamespace(_STATE={"towers": [t1]})
assert L._vis_tower() is t1
sys.modules["exl3xpu.lazyvis"] = types.SimpleNamespace(_STATE={"towers": [t2]})
assert L._vis_tower() is None
sys.modules["exl3xpu.lazyvis"] = types.SimpleNamespace(_STATE={"towers": [t1]})     # same object twice: one tower
assert L._vis_tower() is t1
del sys.modules["llmt_lazyvis"], sys.modules["exl3xpu.lazyvis"]
print("6 OK: one tower only")
osch.OffloadingConnectorScheduler = real
print("ALL OK")
