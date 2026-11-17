"""CPU checks for the move guard and the value model:
 1 no worker in this process: in-use moves refused, free / drop-only regions still usable
 2 pending-store / remote-load blocks are never moved, dropped, used as a destination or lent
 3 production default: EXL3_LEND_MOVE_INUSE unset -> in-use moves off even with the worker present"""
import os, random, time, types
os.environ["EXL3_LORA_HOT"] = "1"; os.environ["EXL3_LEND_COMPACT"] = "1"; os.environ["EXL3_LEND_MOVE_INUSE"] = "1"
import torch
from llmt_kv import lend as L
from vllm.v1.core.kv_cache_utils import KVCacheBlock, FreeKVCacheBlockQueue

NB, P, NS, n = 24, 2048, 3, 4
raw = [torch.zeros(NB * P, dtype=torch.int8) for _ in range(NS)]
L._kv_layout_init({f"l{i}": raw[i].view(torch.float16).view(NB, P // 2) for i in range(NS)}, NB)
L._ST.update(layers=[], dev=torch.device("cpu"), ready=True, max_slots=1, slot_bytes=1)

class HM:
    def __init__(self): self._cache = {}
class Pool:
    enable_caching = True
    metrics_collector = None
    def __init__(self):
        self.blocks = [KVCacheBlock(i) for i in range(NB)]; self.blocks[0].is_null = True
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks[1:])
        self.cached_block_hashes_by_block = {}; self.cached_block_hash_to_block = HM()
    def get_num_free_blocks(self): return self.free_block_queue.num_free_blocks
    def _maybe_evict_cached_block(self, b):
        if b.block_hash is not None:
            self.cached_block_hash_to_block._cache.pop(b.block_hash, None); b.reset_hash()
    def free_blocks(self, bl):
        for b in bl:
            b.ref_cnt -= 1
            if b.ref_cnt == 0: self.free_block_queue.append(b)

def fresh(layout, chains, sess=None):
    L._CH.clear(); L._TENANTS.clear(); L._ST["n"] = {}; L._ST.pop("runner", None); L._SESS.clear(); L._REGRET.clear()
    L._ST["to_return"] = []
    pool = Pool(); reqs = {}
    for i, what in layout.items():
        b = pool.blocks[i]
        if what == "free": continue
        if what[0] in ("use", "hold"):
            pool.free_block_queue.remove(b); b.ref_cnt = 1; reqs.setdefault(what[1], []).append(b)
        else:
            b.set_block_hash(("%s%d" % (what[1], what[2]), 0)); pool.cached_block_hash_to_block._cache[b.block_hash] = b
    for name, cls in chains.items():
        hs = sorted({b.block_hash[0] for b in pool.blocks if b.block_hash and b.block_hash[0].startswith(name)},
                    key=lambda h: int(h[len(name):]))
        kv = {"llmt_cls": cls}
        if sess and name in sess: kv["llmt_sess"] = sess[name]
        L._chain_note(pool, types.SimpleNamespace(block_hashes=hs, kv_transfer_params=kv), 0, len(hs))
    for s_, (u8, PP) in enumerate(L._KVL["stores"]):
        for b in range(NB): u8[b * PP:(b + 1) * PP] = (b * 7 + s_) % 251
    return pool, reqs
def content(bid): return [int(u8[bid * PP]) for u8, PP in L._KVL["stores"]]
def cached(pool): return {b.block_hash[0]: b for b in pool.blocks if b.block_hash}


from vllm.v1.request import RequestStatus
RUNNER = types.SimpleNamespace(input_batch=types.SimpleNamespace(block_table=types.SimpleNamespace(block_tables=[])), requests={})

# 1 no worker in this process -> in-use moves refused, drop/free-only regions still claimable
lay = {i: "free" for i in range(1, NB)}; lay[NB - 2] = ("use", "dec")
pool, reqs = fresh(lay, {})
L._ST.pop("runner", None); L._ST["sched"] = types.SimpleNamespace(running=[], waiting=[], connector=None)
assert L._compact_for_run(pool, n, at="top") is None, "in-use move without the worker refused"
st = L._compact_for_run(pool, n)
assert st is not None and not (st <= NB - 2 < st + n), "a region without in-use blocks is used instead"
print("1 NO-WORKER OK: in-use moves refused, a free region claimed instead (start %d)" % st)

# 2 blocked ids: pending store of a running request, pending store of a finished request (freed, ref 0), a request
#   waiting for remote KV -- never moved, dropped, used as a destination or lent
lay = {i: "free" for i in range(1, NB)}
for i in (NB - 1, NB - 2): lay[i] = ("use", "run")            # running request, a store job reads these
lay[NB - 3] = ("chain", "f", 0)                               # finished request's cached block, store pending (ref 0)
lay[5] = ("use", "remote")                                    # loading from the RAM tier
pool, reqs = fresh(lay, {"f": "lowest"})
L._ST["runner"] = RUNNER
cs = types.SimpleNamespace(_jobs={7: types.SimpleNamespace(non_sliding_window_block_ids=[NB - 1, NB - 2], sliding_window_block_ids=None)},
                           _block_id_to_pending_jobs={NB - 3: {8}}, _current_batch_load_jobs={})
remote = types.SimpleNamespace(request_id="remote", status=RequestStatus.WAITING_FOR_REMOTE_KVS)
km = types.SimpleNamespace(get_block_ids=lambda rid: ([5],) if rid == "remote" else ([],))
L._ST["sched"] = types.SimpleNamespace(running=[], waiting=[remote], kv_cache_manager=km,
                                       connector=types.SimpleNamespace(connector_scheduler=cs))
allow, blocked = L._move_ctx()
assert allow and blocked == {NB - 1, NB - 2, NB - 3, 5}, blocked
assert L._compact_for_run(pool, n, at="top") is None, "top region holds blocked blocks"
before = {i: content(i) for i in (NB - 1, NB - 2, NB - 3, 5)}
for k in range(6):                                            # churn: lends anywhere never touch the blocked ids
    got = L._lend(pool, "t%d" % k, 2, "x", contiguous=(k % 2 == 0))
    if got:
        assert not {b.block_id for b in got} & blocked, ("blocked id lent", [b.block_id for b in got])
assert all(content(i) == before[i] for i in before) and pool.blocks[NB - 3].block_hash is not None
print("2 BLOCKED OK: pending-store / remote-load blocks never moved, dropped, used as destination or lent")

# 3 production default: EXL3_LEND_MOVE_INUSE unset -> in-use moves off even with the worker present
L.MOVE_INUSE = False; L._ST["runner"] = RUNNER
assert L._move_ctx()[0] is False
lay = {i: "free" for i in range(1, NB)}; lay[NB - 2] = ("use", "dec")
pool, reqs = fresh(lay, {}); L._ST["runner"] = RUNNER; L._ST["sched"] = types.SimpleNamespace(running=[], waiting=[], connector=None)
assert L._compact_for_run(pool, n, at="top") is None and reqs["dec"][0].block_id == NB - 2
print("3 DEFAULT-OFF OK: no in-use moves unless EXL3_LEND_MOVE_INUSE=1")
print("ALL OK")
