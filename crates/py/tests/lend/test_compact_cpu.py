"""CPU check of LoRA-row compaction: with a stand-in for the punica LoRA term (row r gets x[r] @ W[slot[r]] where slot
comes from the token mapping), the compacted path (gather LoRA rows, mapping compacted to the front, scatter back)
equals the full path, and non-LoRA rows stay exactly the base output."""
import os
os.environ["EXL3_LORA_HOT"] = "1"
import numpy as np, torch
from llmt_kv import lend as L
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
L._patch_lazy_slots()
torch.manual_seed(0)
T, K, O, NS = 64, 32, 24, 3
W = torch.randn(NS, K, O)
class PW:
    def __init__(self): self._token_lora_indices = torch.full((128,), -1, dtype=torch.long)
pw = PW()
def fake_orig(self, x, out):                        # the LoRA term as punica computes it, by token mapping
    m = pw._token_lora_indices[: x.shape[0]]
    for r in range(x.shape[0]):
        if m[r] >= 0: out[r] += x[r] @ W[m[r]]
    return out
import types
lay = BaseLinearLayerWithLoRA.__new__(BaseLinearLayerWithLoRA)
object.__setattr__(lay, "lora_a_stacked", (torch.zeros(NS, 1, 2, K),)); object.__setattr__(lay, "lora_b_stacked", (torch.zeros(NS, 1, O, 2),))
# reach the closure: _apply_lora_to_output -> _eager -> _eager_rows -> orig_out (patch orig_out via its cell)
f = BaseLinearLayerWithLoRA._apply_lora_to_output
eager = [c.cell_contents for c in f.__closure__ if callable(c.cell_contents) and c.cell_contents.__name__ == "_eager"][0]
rows_fn = [c.cell_contents for c in eager.__closure__ if callable(c.cell_contents) and c.cell_contents.__name__ == "_eager_rows"][0]
for c in rows_fn.__closure__:
    if callable(c.cell_contents) and getattr(c.cell_contents, "__name__", "") == "_apply_lora_to_output":
        c.cell_contents = fake_orig
ids = np.zeros(T, dtype=np.int64); lora_rows = [3, 17, 18, 40]; ids[lora_rows] = [11, 12, 11, 13]
slot = {11: 0, 12: 1, 13: 2}
x = torch.randn(T, K); base = torch.randn(T, O)
full_map = torch.tensor([slot.get(int(i), -1) if i > 0 else -1 for i in ids]); pw._token_lora_indices[:T] = full_map
L._ST["compact"] = None
ref = eager(lay, x, base.clone())
class Mgr: lora_index_to_id = [11, 12, 13, None]; punica_wrapper = pw; _last_mapping = 1
L._ST.update(mgr=Mgr(), dev=torch.device("cpu"), lrows=(np.array(lora_rows), ids[lora_rows], T))
L._compact_on({})
assert L._ST["compact"] is not None and L._ST["compact"][1] == 4
got = eager(lay, x, base.clone())
assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
other = [r for r in range(T) if r not in lora_rows]
assert torch.equal(got[other], base[other]), "base rows untouched"
print("COMPACT OK: 4 of 64 rows, max err %.2e" % (got - ref).abs().max())
# kernel hook: called for LoRA linears (here on the compacted rows), its result is used; None falls back
calls = []
def hk(layer, x, out):
    calls.append(x.shape[0]); return fake_orig(layer, x, out)
L.set_hooks(linear=hk); L._ST["where"] = "device"
object.__setattr__(lay, "is_lora_active", lambda: True)
L._compact_on({})
got2 = eager(lay, x, base.clone())
assert calls == [4] and torch.allclose(got2, ref, atol=1e-5), calls
L.set_hooks(linear=lambda *a: None)
assert torch.allclose(eager(lay, x, base.clone()), ref, atol=1e-5), "None -> punica fallback"
assert L.lora_rows()[1] == 4
print("HOOKS OK")
# linear_full: replaces base + LoRA on LoRA steps (not in compacted steps); None -> base + current LoRA path
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA as BL
fc = []
def full(layer, x):
    fc.append(x.shape[0]); return torch.full((x.shape[0], O), 7.0)
L.set_hooks(linear_full=full)
L._ST.update(step_lora=True, compact=None, where="device")
r = BL._apply_sync(lay, x, None)
assert fc == [T] and float(r.mean()) == 7.0, "linear_full replaced base + LoRA"
L._ST["compact"] = (torch.tensor([3]), 1, T, 3)
assert L._full_hook(lay, x) is None, "not in compacted steps"
L._ST.update(compact=None, step_lora=False)
L.set_hooks(linear_full=lambda *a: None)
assert L._full_hook(lay, x) is None
print("LINEAR_FULL OK")
