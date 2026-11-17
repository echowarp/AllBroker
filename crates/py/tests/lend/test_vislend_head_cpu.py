"""Vision lending with the WHOLE tower in lent KV pages (head unit / merger tensor by tensor), one-tower guard; blocks
back after release."""
import os, sys, types, time
os.environ["EXL3_LORA_HOT"] = "1"
import torch
from llmt_kv import lend as L

NB, P, NS = 40, 64 * 1024, 3
raw = [torch.zeros(NB * P, dtype=torch.int8) for _ in range(NS)]
L._kv_layout_init({f"l{i}": raw[i].view(torch.float16).view(NB, P // 2) for i in range(NS)}, NB)
assert L._KVL["ok"]
L._ST.update(layers=[], dev=torch.device("cpu"), ready=True, max_slots=1, slot_bytes=1, where="off", k=0)

# fake lazyvis module (the shape of the real LazyVision's relevant parts)
m = types.ModuleType("exl3xpu.lazyvis"); m.MARGIN = 0; m.MODE = "auto"; m._STATE = {"towers": []}
class U:
    def __init__(self, idx, n): self.idx, self.n, self.off, self.entry, self.extra, self.on_dev, self.dev = idx, n, 0, [], [], False, None
class R:
    def __init__(self, off, n): self.off, self.nbytes, self.dtype, self.shape = off, n, torch.float16, (n // 2,)
class Src:
    def copy_unit(self, u, buf):
        if u.idx == 0:       # head: every byte = its offset in the unit mod 251 (checks the per-tensor copies)
            buf.copy_((torch.arange(u.n, dtype=torch.int64) % 251).to(torch.uint8)[:buf.numel()])
        else:
            buf.fill_(u.idx)
class LazyVision:
    def __init__(self):
        self.units = [U(0, 0)] + [U(i, 90 * 1024) for i in range(1, 7)]   # 6 block units of 90 KiB (> one 64 KiB page)
        h = self.units[0]                 # head: merger-like tensors of 100 KiB and 60 KiB + a small one (aligned offsets)
        h.entry = [R(0, 100 * 1024), R(100 * 1024, 60 * 1024), R(160 * 1024, 4096)]; h.n = 160 * 1024 + 4096
        self.set_calls = {}
        self.bytes = sum(u.n for u in self.units); self.resident = False; self.master = None; self.dev = torch.device("cpu")
        self.free_room = 10_000   # the allocator has no room for the tower
    def _set(self, r, t): self.set_calls[id(r)] = t
    def _unit_on(self, u, src):
        u.dev, u.on_dev = torch.empty(u.n, dtype=torch.uint8), True     # "allocator" copy
    def _room(self): return self.free_room
    def _enter(self):
        if m.MODE == "resident" or self._room() >= self.bytes + m.MARGIN:
            for u in self.units: self._unit_on(u, Src())
            self.resident = True
    def maybe_release(self):
        for u in self.units: u.dev, u.on_dev = None, False
        self.resident = False
m.LazyVision = LazyVision; sys.modules["exl3xpu.lazyvis"] = m
lv = LazyVision(); m._STATE["towers"].append(lv)

class Blk:
    def __init__(self, i): self.block_id, self.ref_cnt, self.is_null = i, 0, i == 0
class FQ:
    def __init__(self, blocks): self.l = [b for b in blocks if not b.is_null]
    def remove(self, b): self.l.remove(b)
    @property
    def num_free_blocks(self): return len(self.l)
class Pool:
    def __init__(self):
        self.blocks = [Blk(i) for i in range(NB)]; self.free_block_queue = FQ(self.blocks); self.enable_caching = False
    def get_num_free_blocks(self): return self.free_block_queue.num_free_blocks
    def get_new_blocks(self, n):
        out = self.free_block_queue.l[:n]; del self.free_block_queue.l[:n]
        for b in out: b.ref_cnt += 1
        return out
    def free_blocks(self, bl):
        for b in bl: b.ref_cnt -= 1; self.free_block_queue.l.insert(0, b)
pool = Pool()
for i in (3, 4, 9): pool.blocks[i].ref_cnt = 1; pool.free_block_queue.remove(pool.blocks[i])   # owned by requests
class Req: mm_features = [1]
sched = types.SimpleNamespace(kv_cache_manager=types.SimpleNamespace(block_pool=pool), running=[], waiting=[Req()])

L._lend_tick(sched)
pend = L._ST.get("vis_pending"); assert pend, "a run was lent"
ids = [b.block_id for b in pend[0]]
assert ids == list(range(ids[0], ids[0] + len(ids))) and not set(ids) & {3, 4, 9}, ids
print("lent run", ids[0], "..", ids[-1], "for", sum(u.n for u in lv.units), "B (block units + head)")
lv._enter()
assert lv.resident
def in_run(ptr, n):
    si = [i for i, (u8, PP) in enumerate(L._KVL["stores"]) if u8.data_ptr() <= ptr < u8.data_ptr() + u8.numel()]
    assert si, "not in a KV storage"
    u8, PP = L._KVL["stores"][si[0]]
    b0, b1 = (ptr - u8.data_ptr()) // PP, (ptr - u8.data_ptr() + n - 1) // PP
    assert ids[0] <= b0 and b1 <= ids[-1], (b0, b1)
for u in lv.units[1:]:
    in_run(u.dev.data_ptr(), u.n)
    assert int(u.dev[0]) == u.idx, "copied in"
spans = []
for r in lv.units[0].entry:
    t = lv.set_calls[id(r)]
    in_run(t.data_ptr(), r.nbytes)
    b = t.view(torch.uint8)
    want = (torch.arange(r.off, r.off + r.nbytes, dtype=torch.int64) % 251).to(torch.uint8)
    assert torch.equal(b, want), "head tensor bytes"
    spans.append((t.data_ptr(), t.data_ptr() + r.nbytes))
allu = spans + [(u.dev.data_ptr(), u.dev.data_ptr() + u.n) for u in lv.units[1:]]
allu.sort()
assert all(a[1] <= b[0] for a, b in zip(allu, allu[1:])), "placements overlap"
assert L._ST["n"].get("vis_head_lent") == 1
need = -(-sum(u.n for u in lv.units) // sum(PP for _, PP in L._KVL["stores"]))
print("run", len(ids), "blocks, minimum", need)
assert len(ids) <= need + 2, "run not much bigger than the tower"
assert L.lend_status().get("vision", (0, 0))[1] == len(ids)
free_before = pool.get_num_free_blocks()
lv.maybe_release()
L._lend_tick(types.SimpleNamespace(kv_cache_manager=types.SimpleNamespace(block_pool=pool), running=[], waiting=[]))
assert pool.get_num_free_blocks() == free_before + len(ids) and "vision" not in L.lend_status()
print("VIS-LEND HEAD OK: block units + head tensors (merger) in lent pages, bytes exact, blocks back after release")
print("ALL OK")
