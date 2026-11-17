"""vLLM plugin entry point for llmt_kv (``vllm.general_plugins``: ``llmt = llmt_kv.plugin:register``).

vLLM calls register() in every process it starts (API server, engine core, workers). Each component turns itself on
from its own environment variable and does nothing otherwise, so installing the package changes nothing by itself:

  EXL3_WHX=<keeper socket>      weight keeper loader (llmt_kv.whx_loader): weights stay in GPU memory across restarts
  EXL3_LAZY_VISION=1            lazy vision tower with a broker-managed host copy (llmt_kv.lazyvis)
  EXL3_LORA_HOT=1               LoRA hot loading in lent KV blocks (llmt_kv.lend)
  EXL3_LEND_ONLY=1              KV-block lending for the vision tower only, on an engine without --enable-lora (llmt_kv.lend)
  LLMT_KV_PLUGIN=0              skip everything below (for an engine that installs the parts itself)

The KV tier itself is not part of this entry point: it is vLLM's OffloadingConnector with
``"spec_module_path": "llmt_kv.cuda_tier"`` (NVIDIA) or ``"llmt_kv.xpu_tier"`` (Intel XPU), see docs/vllm-plugin.md.
"""
import logging
import os

_log = logging.getLogger("llmt_kv.plugin")
_DONE = [False]


def _try(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001  (a failing component never stops the engine; it stays off)
        _log.warning("llmt_kv.plugin: %s not installed: %r", name, e)


def register():
    if _DONE[0] or os.environ.get("LLMT_KV_PLUGIN", "1") == "0":
        return
    _DONE[0] = True
    env = os.environ.get
    if env("EXL3_WHX"):
        from llmt_kv import whx_loader
        _try("weight keeper loader", whx_loader.install)
    if env("EXL3_LAZY_VISION") == "1":
        from llmt_kv import lazyvis
        _try("lazy vision", lazyvis.install)
    if env("EXL3_LORA_HOT") == "1" or env("EXL3_LEND_ONLY") == "1":
        from llmt_kv import lend
        _try("lend manager", lend.register)
