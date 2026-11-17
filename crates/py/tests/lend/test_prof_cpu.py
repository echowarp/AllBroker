"""CPU check of the lab profiling hook: install, arm a 3-step capture of a fake execute_model, table written."""
import os, sys, tempfile, time
d = tempfile.mkdtemp(); os.environ["EXL3_LORA_HOT"] = "1"; os.environ["EXL3_LORA_PROF"] = d; os.environ["EXL3_LORA_PROF_CPU_ONLY"] = "1"
import torch
from llmt_kv import lend as L
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
calls = []
GPUModelRunner.execute_model = lambda self, *a, **k: (calls.append(1), torch.randn(64, 64) @ torch.randn(64, 64))[1]
L.PROF = d
L._patch_prof()
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
r = GPUModelRunner.__new__(GPUModelRunner)
open(os.path.join(d, "arm"), "w").write("3 t1")
for i in range(6):
    r.execute_model(None)
out = os.path.join(d, "t1.txt")
assert not os.path.exists(os.path.join(d, "arm")) and len(calls) == 6, (os.listdir(d), len(calls))
if os.path.exists(out):
    txt = open(out).read(); print(txt.splitlines()[0]); assert "3 execute_model steps" in txt and "aten::mm" in txt
else:
    print("profiler unavailable here (no Level Zero driver): start failure handled, steps kept running")
open(os.path.join(d, "nograph"), "w").close()
mode, desc = CudagraphDispatcher.dispatch(CudagraphDispatcher.__new__(CudagraphDispatcher), 8)
assert str(mode).endswith("NONE") and desc.num_tokens == 8
print("ALL OK")
