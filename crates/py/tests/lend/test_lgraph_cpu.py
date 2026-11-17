"""CPU checks of the lazy LoRA graph bookkeeping (no GPU): dispatch seen -> want -> captured, capture-time key,
no-LoRA-token swap to the base graph, drop of has_lora entries only."""
import os
os.environ["EXL3_LORA_HOT"] = "1"
import contextlib, dataclasses
import torch
from llmt_kv import lend as L
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.compilation.cuda_graph import CUDAGraphWrapper
import vllm.v1.worker.gpu_model_runner as gmr

def stub(self, num_tokens, uniform_decode=False, has_lora=False, num_active_loras=0, valid_modes=None, invalid_modes=None):
    if uniform_decode and not has_lora:
        return CUDAGraphMode.FULL, BatchDescriptor(num_tokens=num_tokens, num_reqs=num_tokens // 8, uniform=True)
    return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
CudagraphDispatcher.dispatch = stub
seen_ctx = []
@contextlib.contextmanager
def fake_sfc(*a, **k):
    seen_ctx.append(k.get("batch_descriptor")); yield
gmr.set_forward_context = fake_sfc
L._patch_steps(); L._patch_lgraph()
d = CudagraphDispatcher.__new__(CudagraphDispatcher)
L._ST["where"] = "device"
m, b = d.dispatch(16, True, True, 1)
assert m == CUDAGraphMode.NONE and len(L._LG["seen"]) == 1 and not L._LG["want"], "first LoRA step: seen only"
m, b = d.dispatch(16, True, True, 1)
ld = BatchDescriptor(num_tokens=16, num_reqs=2, uniform=True, has_lora=True, num_active_loras=1)
assert m == CUDAGraphMode.NONE and ld in L._LG["want"], "second: wanted"
L._LG["capturing"] = ld
assert d.dispatch(16, True, True, 1) == (CUDAGraphMode.FULL, ld), "capture run gets the LoRA key"
L._LG["capturing"] = None; L._LG["captured"].add(ld); L._LG["want"].clear()
assert d.dispatch(16, True, True, 1) == (CUDAGraphMode.FULL, ld), "captured: FULL with the LoRA key"
assert d.dispatch(16, True, False, 0)[1].has_lora is False, "no LoRA request: base key"
assert d.dispatch(24, False, True, 1)[0] == CUDAGraphMode.NONE, "non-uniform LoRA step: no graph"
L._ST["step_lora"] = False
with gmr.set_forward_context(batch_descriptor=ld): pass
assert seen_ctx[-1] == dataclasses.replace(ld, has_lora=False, num_active_loras=0), "no LoRA token -> base graph key"
L._ST["step_lora"] = True
with gmr.set_forward_context(batch_descriptor=ld): pass
assert seen_ctx[-1] == ld, "LoRA tokens -> LoRA graph key"
L._ST["step_lora"] = False
class W: pass
w = W(); base = dataclasses.replace(ld, has_lora=False, num_active_loras=0)
w.concrete_cudagraph_entries = {ld: 1, base: 2}
CUDAGraphWrapper._all_instances.add(w)
L._LG["pool"] = object(); L._LG["bytes"] = 5 << 20
L._lg_drop("test")
assert list(w.concrete_cudagraph_entries) == [base] and not L._LG["captured"] and L._LG["pool"] is None and L._LG["bytes"] == 0
print("ALL OK")
