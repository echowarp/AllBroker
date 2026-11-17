"""A LoRA request the scheduler skipped (over max_loras = 0 device slots while KV was full) sits in Scheduler.skipped_waiting
(vLLM 0.26). The lend tick must still see it: when KV frees up, blocks are lent, the slot capacity becomes >= 1 and the slots
move into the lent blocks; a pending lend is kept while its LoRA request still waits. EXL3_LORA_HOT=1."""
import os
os.environ["EXL3_LORA_HOT"] = "1"
import torch
from llmt_kv import lend as L

NB, P = 40, 64 * 1024
raw = [torch.zeros(NB * P, dtype=torch.int8) for _ in range(2)]
L._kv_layout_init({"a": raw[0].view(torch.float16).view(NB, 2, P // 4),
                   "m": [torch.as_strided(raw[1].view(torch.float32), (NB, 100), (P // 4, 1))]}, NB)
assert L._KVL["ok"]
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
L._room = lambda: 0                       # no headroom: lending is the only way
class Blk:
    def __init__(self, i): self.block_id, self.ref_cnt = i, 0
class Pool:
    def __init__(self): self.free = [Blk(i) for i in range(1, NB)]
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
    def __init__(self):
        self.kv_cache_manager = type("K", (), {"block_pool": Pool()})(); self.running = []; self.waiting = []
        self.skipped_waiting = []
s = Sched(); pool = s.kv_cache_manager.block_pool
# KV full: base streams hold almost everything -> no lend
busy = pool.get_new_blocks(NB - 1 - 3)
s.skipped_waiting = [Req(7)]              # the adapter request, skipped by max_loras
L._lend_tick(s)
assert not L._ST.get("lent_pending"), "nothing to lend while KV is full"
assert L._slot_cap() == 0, "no capacity yet"
# the base streams finish: their blocks are free again (as cached-evictable / free)
pool.free_blocks(busy)
L._ST["cap_cache"] = None
L._lend_tick(s)
pend = L._ST.get("lent_pending"); assert pend and pend[1] >= 1, "the skipped request is seen and blocks are lent"
L._ST["cap_cache"] = None
assert L._slot_cap() >= 1, "the scheduler may now take the request"
L._ensure_slots(1)
assert L._ST["where"] == "device" and L._ST["lent"] and L._ST["k"] >= 1
# pending lend kept while the request still waits (it used to be given back after 30 s and re-lent)
L._release("test reset") if L._ST["where"] != "off" else None
L._ST.update(lent=None, lent_pending=None, where="off", k=0, cap_cache=None)
for t in list(L._TENANTS): L._unlend(t)
L._lend_tick(s)                           # returns the released blocks, then lends again for the waiter
pend = L._ST.get("lent_pending"); assert pend, "lent again for the waiter"
L._ST["lent_pending"] = (pend[0], pend[1], pend[2] - 1000)          # long past LEND_PENDING_S
L._lend_tick(s)
assert L._ST.get("lent_pending") and L._ST["lent_pending"][0] is pend[0], "kept while the LoRA request waits"
assert L._ST["n"].get("lend_pending_kept") == 1
s.skipped_waiting = []                    # the request is gone
L._ST["lent_pending"] = (pend[0], pend[1], pend[2] - 1000)
L._lend_tick(s)
assert not L._ST.get("lent_pending"), "given back once nobody waits"
print("PENDING-KEEP OK: a pending lend stays while its LoRA request waits, goes back once it is gone")
print("SKIPPED-WAITING OK: a LoRA request in skipped_waiting gets a lend once KV frees up")
print("ALL OK")
