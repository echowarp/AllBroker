"""CPU checks of KV-block lending: layout check (blocks-first ok, K/V-first refused), packing, lend -> slot views
inside the lent blocks' pages (never in other blocks), growth into a bigger run, release -> blocks back to the
pool at the next schedule(), and the scheduler capacity view."""
import os
os.environ["EXL3_LORA_HOT"] = "1"
import torch
from llmt_kv import lend as L

NB, P = 8, 64 * 1024                     # 8 KV blocks, 2 storages x 64 KiB pages
raw = [torch.zeros(NB * P, dtype=torch.int8) for _ in range(2)]
attn = raw[0].view(torch.float16).view(NB, 2, P // 4)         # blocks-first attention view
mamba = [torch.as_strided(raw[1].view(torch.float32), (NB, 100), (P // 4, 1))]
L._kv_layout_init({"a": attn, "m": mamba}, NB)
assert L._KVL["ok"] and L._KVL["per_block"] == 2 * P, L._KVL
kvfirst = raw[0].view(torch.float16).view(2, NB, P // 4)
L._KVL["ok"] = False; L._kv_layout_init({"a": kvfirst}, NB); assert not L._KVL["ok"], "K/V-first must be refused"
L._kv_layout_init({"a": attn, "m": mamba}, NB)

class Layer: pass
N = 4
layers = []
for i in range(3):
    l = Layer()
    l.lora_a_stacked = tuple(torch.empty(N, 1, 16, 256, device="meta", dtype=torch.float16) for _ in range(2))
    l.lora_b_stacked = tuple(torch.empty(N, 1, 512, 16, device="meta", dtype=torch.float16) for _ in range(2))
    l.adapter_enabled_cpu = [False] * N; l.adapter_ranks_cpu = [0] * N
    layers.append(l)
L._ST.update(layers=layers, dev=torch.device("cpu"), max_slots=0, slot_bytes=0, where="off", k=0, ready=True,
             act_reserve=0, base_alloc=None, cap_cache=None)
row = L._slot_bytes()
print("slot row %d B, per block %d B" % (row, L._KVL["per_block"]))
L._room = lambda: 0                       # no headroom at all: lending is the only way

class Blk:
    def __init__(self, i): self.block_id, self.ref_cnt = i, 0
class Pool:
    def __init__(self): self.free = [Blk(i) for i in range(1, NB)]   # block 0 = null block
    def get_num_free_blocks(self): return len(self.free)
    def get_new_blocks(self, n):
        out, self.free = self.free[:n], self.free[n:]
        for b in out: b.ref_cnt += 1
        return out
    def free_blocks(self, bl):
        for b in bl: b.ref_cnt -= 1; self.free.insert(0, b)
class Req:
    def __init__(self, i): self.lora_request = type("R", (), {"lora_int_id": i})()
class Sched:
    def __init__(self): self.kv_cache_manager = type("K", (), {"block_pool": Pool()})(); self.running = []; self.waiting = []
s = Sched(); pool = s.kv_cache_manager.block_pool
s.waiting = [Req(7)]
L._lend_tick(s)
pend = L._ST["lent_pending"]; assert pend and pend[1] == 1, pend
nb1 = len(pend[0]); assert pool.get_num_free_blocks() == NB - 1 - nb1
assert L._slot_cap() >= 1
L._ensure_slots(1)
assert L._ST["where"] == "device" and L._ST["lent"] and L._ST["k"] == 1
ids = {b.block_id for b in L._ST["lent"]}
def owner(t):                              # which block of which storage holds this tensor
    for si, (u8, PP) in enumerate(L._KVL["stores"]):
        lo = u8.data_ptr()
        if lo <= t.data_ptr() < lo + u8.numel():
            off = t.data_ptr() - lo; b = off // PP
            assert (off + t.numel() * t.element_size() - 1) // PP == b, "tensor spans two blocks"
            return b
    raise AssertionError("not inside the KV storage")
for l in layers:
    for t in l.lora_a_stacked + l.lora_b_stacked:
        assert owner(t) in ids and t.shape[0] == 1
layers[0].lora_a_stacked[0][0].fill_(3)
# a second adapter: the scheduler lends a bigger run, the worker moves the slots (row 0 kept)
s.waiting = [Req(7), Req(8)]
L._lend_tick(s); assert L._ST["lent_pending"][1] == 2
assert L._grow_to(2) and L._ST["k"] == 2
assert float(layers[0].lora_a_stacked[0][0].float().mean()) == 3.0, "row 0 kept across the move"
ids2 = {b.block_id for b in L._ST["lent"]}
for l in layers:
    for t in l.lora_a_stacked + l.lora_b_stacked:
        assert owner(t) in ids2
L._lend_tick(s)                            # the old run goes back
assert pool.get_num_free_blocks() == NB - 1 - len(ids2), pool.get_num_free_blocks()
class Mgr:
    def __init__(self): self._active_adapters = type("C", (), {"cache": {}})(); self.lora_index_to_id = [None] * N
L._ST["mgr"] = Mgr()
L._release("test")
assert L._ST["where"] == "off" and L._ST["lent"] is None and layers[0].lora_a_stacked[0].device.type == "meta"
s.waiting = []
L._lend_tick(s)
assert pool.get_num_free_blocks() == NB - 1, "all lent blocks back"
# not enough free blocks: no lending, the request waits (capacity 0)
pool.free = pool.free[:1]; s.waiting = [Req(9)]
L._ST.update(lent_pending=None, cap_cache=None)
L._lend_tick(s); assert L._ST.get("lent_pending") is None and L._slot_cap() == 0
print("ALL OK")
# buffers made inside inference mode (a step) must stay writable outside it (API-time add_lora)
L._ST.update(where="off", k=0, lent=None, lent_pending=None, cap_cache=None)
pool.free = [Blk(i) for i in range(1, NB)]
s.waiting = [Req(11)]; L._lend_tick(s)
with torch.inference_mode():
    L._ensure_slots(1)
layers[0].lora_a_stacked[0][0] = 0      # raises on an inference tensor
print("INFERENCE-MODE OK")
