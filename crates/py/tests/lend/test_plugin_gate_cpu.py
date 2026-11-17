"""llmt_kv.plugin.register() starts the lend manager for EXL3_LORA_HOT=1 and for EXL3_LEND_ONLY=1 (vision tower only,
no --enable-lora), and nothing without either. The lend module is replaced by a stub: no vLLM needed."""
import os, sys, types
import llmt_kv
calls = []
stub = types.ModuleType("llmt_kv.lend")
stub.register = lambda: calls.append("lend")
sys.modules["llmt_kv.lend"] = stub
llmt_kv.lend = stub
from llmt_kv import plugin
for env, want in (({}, []), ({"EXL3_LEND_ONLY": "1"}, ["lend"]), ({"EXL3_LORA_HOT": "1"}, ["lend"])):
    calls.clear()
    plugin._DONE[0] = False
    for k in ("EXL3_LEND_ONLY", "EXL3_LORA_HOT", "EXL3_WHX", "EXL3_LAZY_VISION", "LLMT_KV_PLUGIN"):
        os.environ.pop(k, None)
    os.environ.update(env)
    plugin.register()
    assert calls == want, (env, calls)
print("plugin gate: 3 checks passed")
