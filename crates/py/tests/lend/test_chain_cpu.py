"""CPU check of chain-aware compaction: (A) cached-unreferenced blocks in the chosen window are moved, not dropped,
while free blocks exist outside; (B) a forced drop takes a low-value chain suffix leaf-first, never orphans
descendants, and the high-value chain survives intact; score_region: free run 0, lent block inf."""
import os, types
os.environ["EXL3_LORA_HOT"] = "1"; os.environ["EXL3_LEND_COMPACT"] = "1"; os.environ["EXL3_LEND_MOVE_INUSE"] = "1"
import numpy as np, torch
from llmt_kv import lend as L

NB, P, NS, n = 16, 2048, 3, 4
raw = [torch.zeros(NB * P, dtype=torch.int8) for _ in range(NS)]
L._kv_layout_init({f"l{i}": raw[i].view(torch.float16).view(NB, P // 2) for i in range(NS)}, NB)
L._ST.update(layers=[], dev=torch.device("cpu"), ready=True, max_slots=1, slot_bytes=1)

class Blk:
    def __init__(self, i): self.block_id, self.ref_cnt, self.is_null, self.block_hash = i, 0, i == 0, None
class FQ:
    def __init__(self, bl): self.l = [b for b in bl if not b.is_null]
    def remove(self, b): self.l.remove(b)
    def prepend_n(self, bs): self.l[:0] = bs
    @property
    def num_free_blocks(self): return len(self.l)
class HM:
    def __init__(self): self._cache = {}
class Pool:
    enable_caching = True
    def __init__(self):
        self.blocks = [Blk(i) for i in range(NB)]; self.free_block_queue = FQ(self.blocks)
        self.cached_block_hashes_by_block = {}; self.cached_block_hash_to_block = HM()
    def get_num_free_blocks(self): return self.free_block_queue.num_free_blocks
    def _maybe_evict_cached_block(self, b):
        if b.block_hash is not None: self.cached_block_hash_to_block._cache.pop(b.block_hash, None); b.block_hash = None

def setup(layout, chains):
    """layout: id -> ('use', req) | ('chain', name, pos) | 'free'; chains: name -> llmt_cls."""
    L._CH.clear(); L._TENANTS.clear(); L._ST["n"] = {}; L._ST["runner"] = types.SimpleNamespace(input_batch=None, requests={}); L._ST.pop("sched", None)
    pool = Pool(); reqs = {}
    for i, what in layout.items():
        b = pool.blocks[i]
        if what == "free":
            continue
        if what[0] == "use":
            b.ref_cnt = 1; pool.free_block_queue.remove(b); reqs.setdefault(what[1], []).append(b)
        else:
            b.block_hash = ("%s%d" % (what[1], what[2]), 0); pool.cached_block_hash_to_block._cache[b.block_hash] = b
    for name, cls in chains.items():
        hs = sorted({b.block_hash[0] for b in pool.blocks if b.block_hash and b.block_hash[0].startswith(name)},
                    key=lambda h: int(h[len(name):]))
        L._chain_note(pool, types.SimpleNamespace(block_hashes=hs, kv_transfer_params={"llmt_cls": cls}), 0, len(hs))
    for s_, (u8, PP) in enumerate(L._KVL["stores"]):
        for b in range(NB): u8[b * PP:(b + 1) * PP] = (b * 7 + s_) % 251
    return pool, reqs

def content(bid): return [int(u8[bid * PP]) for u8, PP in L._KVL["stores"]]
def cached(pool): return {b.block_hash[0]: b for b in pool.blocks if b.block_hash}

# (A) free uncached runs of 2 only; cached chain x0..x3 (mid-chain blocks inside every window) + in-use blocks
lay = {i: "free" for i in range(1, NB)}
for i, p in zip((2, 6, 10, 14), range(4)): lay[i] = ("chain", "x", p)
for i in (1, 5, 9, 13): lay[i] = ("use", "r")
pool, reqs = setup(lay, {"x": "normal"})
assert L._take_run(pool, n) is None or True
cb = {h: content(b.block_id) for h, b in cached(pool).items()}
rb = [content(b.block_id) for b in reqs["r"]]
assert L.score_region(pool, 3, 2) == 0.0
st = L._compact_for_run(pool, n)
assert st is not None and all(pool.blocks[i].ref_cnt == 0 and pool.blocks[i].block_hash is None for i in range(st, st + n))
assert L._ST["n"].get("compact_dropped_blocks", 0) == 0, "nothing dropped while free blocks exist outside"
c2 = cached(pool)
assert set(c2) == set(cb) and all(content(b.block_id) == cb[h] for h, b in c2.items()), "chain moved intact"
assert all(pool.cached_block_hash_to_block._cache[b.block_hash] is b for b in c2.values())
assert [content(b.block_id) for b in reqs["r"]] == rb
assert L._ST["n"]["compacted_blocks"] >= 1 and any(h in ("x1", "x2") for h in cb), "mid-chain block moved"
print("A MOVE-NOT-DROP OK: window %d-%d, moved %d, dropped 0" % (st, st + n - 1, L._ST["n"]["compacted_blocks"]))

# (B) full pool: in-use request R, low chain L (lowest), high chain H (agent, hit 10x); 3 free blocks
lay = {}
for k in range(4):
    lay[1 + 4 * k] = ("use", "R"); lay[2 + 4 * k] = ("chain", "l", k); lay[3 + 4 * k] = ("chain", "h", k)
    if k < 3: lay[4 + 4 * k] = "free"
pool, reqs = setup(lay, {"l": "lowest", "h": "agent"})
for h in ("h0", "h1", "h2", "h3"): L._CH[h][3] = 10
hb = {h: content(b.block_id) for h, b in cached(pool).items() if h.startswith("h")}
rb = [content(b.block_id) for b in reqs["R"]]
t = pool.blocks[4]; t.ref_cnt = 1; pool.free_block_queue.remove(t)          # a lent block: its regions score inf
L._TENANTS["lora"] = {"blocks": [t], "kind": "x", "bytes": 0, "since": 0}
assert L.score_region(pool, 2, n) == float("inf")
st = L._compact_for_run(pool, n)
assert st is not None and all(pool.blocks[i].ref_cnt == 0 and pool.blocks[i].block_hash is None for i in range(st, st + n))
c2 = cached(pool)
assert all(h in c2 for h in hb) and all(content(c2[h].block_id) == hb[h] for h in hb), "high-value chain intact"
lsurv = [k for k in range(4) if "l%d" % k in c2]
assert lsurv == list(range(len(lsurv))), ("no orphans: surviving low-chain nodes form a prefix", lsurv)
assert len(lsurv) < 4 and L._ST["n"]["compact_dropped_blocks"] == 4 - len(lsurv)
assert [content(b.block_id) for b in reqs["R"]] == rb and t.block_id == 4
print("B CHAIN-DROP OK: window %d-%d, dropped low-chain tail l%d..l3, score %.3f, H intact" % (
    st, st + n - 1, len(lsurv), L._ST["n"]["compact_last_score"]))
print("ALL OK")

# (C) full pool, no free block at all: clearing a region evicts the lowest-value tails, moved blocks land in their slots
lay = {}
for k in range(5):
    lay[1 + 3 * k] = ("use", "R"); lay[2 + 3 * k] = ("chain", "l", k); lay[3 + 3 * k] = ("chain", "h", k)
pool, reqs = setup(lay, {"l": "lowest", "h": "agent"})
assert pool.get_num_free_blocks() == 10 and all(b.ref_cnt or b.block_hash or b.is_null for b in pool.blocks)
for h in ("h0", "h1", "h2", "h3", "h4"): L._CH[h][3] = 10
before = {h: content(b.block_id) for h, b in cached(pool).items()}
rb = [content(b.block_id) for b in reqs["R"]]
victim_slots = {b.block_id for h, b in cached(pool).items() if h.startswith("l")}
st = L._compact_for_run(pool, n, at="top")
assert st == NB - n and all(pool.blocks[i].ref_cnt == 0 and pool.blocks[i].block_hash is None for i in range(st, NB))
c2 = cached(pool)
assert all(h in c2 and content(c2[h].block_id) == before[h] for h in before if h.startswith("h")), "H intact"
lsurv = [k for k in range(5) if "l%d" % k in c2]
assert lsurv == list(range(len(lsurv))) and len(lsurv) < 5, ("low chain consumed tail-first", lsurv)
assert all(content(c2[h].block_id) == before[h] for h in c2), "surviving cached KV identical"
assert [content(b.block_id) for b in reqs["R"]] == rb, "request KV identical"
moved_to = {b.block_id for b in reqs["R"]} | {c2[h].block_id for h in c2}
assert all(i < st for i in moved_to)
print("C FULL-POOL OK: top %d-%d cleared, low chain left l0..l%d, score %.3f" % (st, NB - 1, len(lsurv) - 1,
      L._ST["n"]["compact_last_score"]))
print("ALL OK")

# lent-block counter
pool, _ = setup({i: "free" for i in range(1, NB)}, {})
L._ST["to_return"] = []
got = L._lend(pool, "vision", 3, "t", contiguous=True)
assert got and L._lent_count(pool) == 3 and pool.llmt_lent_blocks == 3
L._unlend("vision", got); assert L._lent_count(pool) == 3, "returned but not yet freed still counts"
L._ST["to_return"] = []; L._TENANTS.clear(); assert L._lent_count(pool) == 0
print("LENT-COUNT OK")
