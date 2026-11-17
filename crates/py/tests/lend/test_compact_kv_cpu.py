"""CPU check of block compaction: a fragmented pool (no free run of n) -> compaction moves in-use blocks out of the
cheapest window (KV pages copied, block identities swapped, hash map / block tables remapped) -> a run of n is lent;
every request reads the same KV content through its block objects and its block-table rows; lent tenant blocks stay."""
import os, types
os.environ["EXL3_LORA_HOT"] = "1"; os.environ["EXL3_LEND_COMPACT"] = "1"; os.environ["EXL3_LEND_MOVE_INUSE"] = "1"
import numpy as np, torch
from llmt_kv import lend as L

NB, P, NS = 24, 4096, 3
raw = [torch.zeros(NB * P, dtype=torch.int8) for _ in range(NS)]
L._kv_layout_init({f"l{i}": raw[i].view(torch.float16).view(NB, P // 2) for i in range(NS)}, NB)
L._ST.update(layers=[], dev=torch.device("cpu"), ready=True, max_slots=1, slot_bytes=1)
for s_, (u8, PP) in enumerate(L._KVL["stores"]):        # page content = (block id, storage) marker
    for b in range(NB): u8[b * PP:(b + 1) * PP] = (b * 7 + s_) % 251

class Blk:
    def __init__(self, i): self.block_id, self.ref_cnt, self.is_null, self.block_hash = i, 0, i == 0, None
class FQ:
    def __init__(self, bl): self.l = [b for b in bl if not b.is_null]
    def remove(self, b): self.l.remove(b)
    @property
    def num_free_blocks(self): return len(self.l)
class HM: _cache = {}
class Pool:
    enable_caching = True
    def __init__(self):
        self.blocks = [Blk(i) for i in range(NB)]; self.free_block_queue = FQ(self.blocks)
        self.cached_block_hashes_by_block = {}; self.cached_block_hash_to_block = HM()
    def get_num_free_blocks(self): return self.free_block_queue.num_free_blocks
    def _maybe_evict_cached_block(self, b):
        if b.block_hash is not None: HM._cache.pop(b.block_hash, None); b.block_hash = None
pool = Pool()
used = [1, 5, 8, 11, 14, 17, 20, 23, 2, 3]                # no free run of 4; cheapest windows hold block 5
reqs = {"r1": [pool.blocks[i] for i in used[:4]], "r2": [pool.blocks[i] for i in used[4:8]], "r3": [pool.blocks[i] for i in used[8:]]}
for b in pool.blocks:
    if b.block_id in used: b.ref_cnt = 1; pool.free_block_queue.remove(b)
pool.blocks[5].block_hash = ("h5", 0); HM._cache[("h5", 0)] = {5: pool.blocks[5]}   # a cached in-use block
tenant_blk = pool.blocks[13]; tenant_blk.ref_cnt = 1; pool.free_block_queue.remove(tenant_blk)
L._TENANTS["lora"] = {"blocks": [tenant_blk], "kind": "x", "bytes": 0, "since": 0}
# worker block table + runner cached ids
bt = types.SimpleNamespace(block_table=types.SimpleNamespace(np=np.zeros((3, 8), dtype=np.int32)), blocks_per_kv_block=1)
bt.block_table.np[0, :4] = used[:4]; bt.block_table.np[1, :4] = used[4:8]; bt.block_table.np[2, :2] = used[8:]
runner = types.SimpleNamespace(input_batch=types.SimpleNamespace(block_table=types.SimpleNamespace(block_tables=[bt])),
                               requests={"r1": types.SimpleNamespace(block_ids=([*used[:4]],)), "r2": types.SimpleNamespace(block_ids=([*used[4:8]],)), "r3": types.SimpleNamespace(block_ids=([*used[8:]],))})
L._ST["runner"] = runner
def content(bid): return [int(u8[bid * PP]) for u8, PP in L._KVL["stores"]]
before = {r: [content(b.block_id) for b in bl] for r, bl in reqs.items()}
assert L._take_run(pool, 4) is None, "fragmented: no free run"
L._ST["n"] = {}
got = L._lend(pool, "vision", 4, "vision tower", contiguous=True)
assert got and [b.block_id for b in got] == list(range(got[0].block_id, got[0].block_id + 4)), got
assert tenant_blk.block_id == 13 and tenant_blk in L._TENANTS["lora"]["blocks"], "lent tenant block not moved"
for r, bl in reqs.items():
    assert [content(b.block_id) for b in bl] == before[r], ("request content changed", r)
    row = {"r1": 0, "r2": 1, "r3": 2}[r]
    assert list(bt.block_table.np[row, :len(bl)]) == [b.block_id for b in bl], "block table follows"
    assert list(runner.requests[r].block_ids[0]) == [b.block_id for b in bl], "runner cached ids follow"
    assert not set(b.block_id for b in bl) & {b.block_id for b in got}, "request blocks outside the run"
b5 = reqs["r1"][1]
assert HM._cache[("h5", 0)].get(b5.block_id) is b5 and (b5.block_id == 5 or 5 not in HM._cache[("h5", 0)]), "hash map follows a moved block"
assert all(pool.blocks[i].block_id == i for i in range(NB)), "pool.blocks index == block_id"
print("COMPACTION OK: %d moves, run %d-%d lent, request KV identical" % (L._ST["n"].get("compacted_blocks", 0), got[0].block_id, got[-1].block_id))
# top-range target (movable boundary)
L._unlend("vision", got); L._ST["to_return"] = []
for b in got: b.ref_cnt = 0; pool.free_block_queue.l.append(b)
pool.blocks[22].ref_cnt = 1; pool.free_block_queue.remove(pool.blocks[22]); reqs["r4"] = [pool.blocks[22]]
before4 = content(22)
assert L._compact_for_run(pool, 4, at="top")
assert all(pool.blocks[i].ref_cnt == 0 for i in range(NB - 4, NB)), "top 4 ids free"
assert content(reqs["r4"][0].block_id) == before4 and reqs["r4"][0].block_id < NB - 4
print("TOP-RANGE OK")
