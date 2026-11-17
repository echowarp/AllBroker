"""CPU checks of the multi-slot bookkeeping: N-row meta buffers -> k device rows, grow keeps contents, shrink drops
trailing rows, release back to N-row meta, scheduler view of max_loras."""
import os
os.environ["EXL3_LORA_HOT"] = "1"
import torch
from llmt_kv import lend as L

class Layer: pass
N = 4
layers = []
for i in range(3):
    l = Layer()
    l.lora_a_stacked = tuple(torch.empty(N, 1, 16, 64, device="meta") for _ in range(2))
    l.lora_b_stacked = tuple(torch.empty(N, 1, 32, 16, device="meta") for _ in range(2))
    l.adapter_enabled_cpu = [False] * N; l.adapter_ranks_cpu = [0] * N
    layers.append(l)
L._ST.update(layers=layers, dev=torch.device("cpu"), max_slots=0, ready=False, where="off", k=0, slot_bytes=0)
per = L._slot_bytes()                      # before the slot count is known (scheduler asks first): per-row bytes
assert L._max_slots() == N
assert per == (2 * 16 * 64 + 2 * 32 * 16) * 4 * 3, per
L._ensure_slots(1)
assert L._ST["where"] == "device" and L._ST["k"] == 1 and layers[0].lora_a_stacked[0].shape[0] == 1
assert L._ST["bytes"] == per
layers[0].lora_a_stacked[0][0].fill_(7)
assert L._grow_to(3) and L._ST["k"] == 3 and layers[0].lora_a_stacked[1].shape[0] == 3
assert float(layers[0].lora_a_stacked[0][0].mean()) == 7.0, "grow keeps slot 0"
L._shrink_to(2, "test")
assert L._ST["k"] == 2 and layers[2].lora_b_stacked[0].shape[0] == 2 and float(layers[0].lora_a_stacked[0][0].mean()) == 7.0
class Mgr:
    def __init__(self): self._active_adapters = type("C", (), {"cache": {}})(); self.lora_index_to_id = [None] * N
    def deactivate_adapter(self, i): pass
L._ST["mgr"] = Mgr()
L._release("test")
assert L._ST["where"] == "off" and L._ST["k"] == 0 and layers[0].lora_a_stacked[0].shape[0] == N and layers[0].lora_a_stacked[0].device.type == "meta"
class Cfg: max_loras = N; max_lora_rank = 16
v = L._LoraCfgView(Cfg())
assert v.max_loras == N and v.max_lora_rank == 16, "not ready: N"
L._ST.update(ready=True, where="device", k=1, cap_cache=None, act_reserve=0, base_alloc=None)
L._room = lambda: L.MARGIN + per // 2           # room for less than one more slot row
assert L._slot_cap() == 1 and v.max_loras == 1
v._floor = 3; assert v.max_loras == 3, "never below running adapters"
L._ST["cap_cache"] = None; L._room = lambda: L.MARGIN + 5 * per
v._floor = 0; assert v.max_loras == N
# host mode: only the rows needed, grows without a room check
L._release("test2") if L._ST["where"] != "off" else None
L._ST.update(where="off", k=0, ready=False)
L.MODE = "host"
L._ensure_slots(1)
assert L._ST["where"] == "host" and L._ST["k"] == 1 and layers[0].lora_a_stacked[0].shape[0] == 1 and L._ST["bytes"] == per
assert L._grow_to(2) and L._ST["k"] == 2
L.MODE = "auto"
print("ALL OK")

# --- no host fallback (default): no room -> _NoRoom, cap can be 0, the scheduler view follows it
L.HOST_FALLBACK = False
L._ST.update(where="off", k=0, ready=True, act_reserve=0, cap_cache=None, force_room=False)
L._room = lambda: 10                      # nothing fits
try:
    L._ensure_slots(1); raise SystemExit("expected _NoRoom")
except L._NoRoom:
    pass
assert L._ST["where"] == "off"
assert L._slot_cap() == 0
v = L._LoraCfgView(Cfg()); v._floor = 0
assert v.max_loras == 0, "no device room: LoRA requests wait (max_loras 0)"
v._floor = 1; assert v.max_loras == 1, "running adapters keep their slots"
L._ST["force_room"] = True; L._ensure_slots(1); L._ST["force_room"] = False
assert L._ST["where"] == "device", "already-scheduled step: allocated beyond the room, never host"
print("NO-FALLBACK OK")
