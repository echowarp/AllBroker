"""Intel XPU (Arc B70) RAM KV tier through llmt-kvbroker: the shared vLLM tier
(llmt_kv.vllm_tier: manager, keys, reconnects, fallbacks) plus Level Zero pinning of the broker pool (vLLM's
pin_mmap_region only knows cudaHostRegister). Measured on the B70: stores 9.1 GiB/s pinned vs 6.1 pageable, the same
as torch's own pinned memory (9.9); loads 4.9 GiB/s either way.

  --set 'vllm.kv_transfer_config={"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":
    {"spec_name":"BrokerOffloadingSpec","spec_module_path":"llmt_kv.xpu_tier","cpu_bytes_to_use":1,
     "broker_socket":"/run/llmt-kv-arc/broker.sock","broker_namespace":"arc-b70-private"}}'

Pinning is OFF unless LLMT_KV_PIN=1 (--set env.LLMT_KV_PIN=1): an imported pool page that the kernel swaps
out or migrates (the pool is shmem in /dev/shm and the broker does not mlock it) faults the copy engine ("Engine memory
CAT error, class=bcs", fault response -EBUSY/-ENOENT); xe resets the copy engine and the engine hangs for good in
swap_blocks_batch (seen with 5.5 GB of swap in use). Pageable copies go through the driver's own staging memory.
"""
import os

from llmt_kv.vllm_tier import BrokerOffloadingSpecBase, logger

from llmt_kv import xpu_hostreg


class BrokerOffloadingSpec(BrokerOffloadingSpecBase):
    DEFAULT_NAMESPACE = "arc-b70-private"
    DEFAULT_ENGINE_ID = 1
    DEFAULT_NAME = "arc-b70"
    # Level Zero refuses one 24 GiB import (and stops somewhere between 16 and 22 GiB of imports in total while an
    # engine runs): import in slot-aligned chunks and keep what succeeds. A copy never crosses a slot, so
    # pages in imported chunks take the pinned path and the rest are copied as pageable memory (stores 6.1 vs 9.1
    # GiB/s; loads 4.9 either way).
    CHUNK_SLOTS = 36                       # 36 x 58.8 MB = 2.1 GB

    def pin(self, ptr: int, size: int) -> bool:
        if os.environ.get("LLMT_KV_PIN", "0") != "1":
            logger.info("kvbroker: Level Zero import off (LLMT_KV_PIN != 1): pageable copies")
            return False
        step = self.CHUNK_SLOTS * (self.kv_bytes_per_chunk or size)
        self._pinned = []
        off = 0
        while off < size:
            n = min(step, size - off)
            if not xpu_hostreg.register(ptr + off, n):
                break
            self._pinned.append(ptr + off)
            off += n
        logger.info("kvbroker: Level Zero imported %.1f of %.1f GiB in %d chunks", off / 2**30, size / 2**30,
                    len(self._pinned))
        return bool(self._pinned)

    def unpin(self, ptr: int) -> None:
        for p in getattr(self, "_pinned", []):
            xpu_hostreg.unregister(p)
        self._pinned = []
