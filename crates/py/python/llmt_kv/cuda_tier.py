"""CUDA (RTX 5090) RAM KV tier through llmt-kvbroker: the shared vLLM tier (llmt_kv.vllm_tier: manager, keys +
compat tag, reconnects, unleased-load fallback, start without a broker) plus CUDA defaults and pinning.

Pinning: vLLM's CPUOffloadingWorker registers the whole pool with ONE cudaHostRegister
(pin_mmap_region). Measured on an RTX 5090 host: a 7.95 GiB pool fails as a whole with cudaErrorInvalidValue while the Arc's
24 GiB pool is resident (3.95 GiB worked; in 2.1 GB chunks 5.9 of 7.95 GiB registered), and the failure is left as the
pending CUDA error, so the next kernel launch aborts engine start (v1_block_table_warmup: "CUDA error: invalid
argument"). Here the pool is registered in slot-aligned chunks (CHUNK_SLOTS slots each); chunks that fail stay pageable
(copies still work, slower), the pending error is cleared (cudaGetLastError), and vLLM's own whole-pool attempt is
skipped. Same approach as the Arc tier (Level Zero).

  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":
    {"spec_name":"BrokerOffloadingSpec","spec_module_path":"llmt_kv.cuda_tier","cpu_bytes_to_use":1,
     "broker_socket":"/run/llmt-kv/broker.sock","broker_namespace":"rtx5090-private"}}'
"""
import ctypes
import os
import threading
import time

import torch

from llmt_kv.vllm_tier import BrokerOffloadingSpecBase, _NoBroker, logger


def _cudart():
    for line in open("/proc/self/maps"):
        if "libcudart" in line:
            return ctypes.CDLL(line.split()[-1])
    return None


class BrokerOffloadingSpec(BrokerOffloadingSpecBase):
    DEFAULT_NAMESPACE = "rtx5090-private"
    DEFAULT_ENGINE_ID = 2
    DEFAULT_NAME = "rtx5090"
    CHUNK_SLOTS = 36                       # 36 x 58.8 MB = 2.1 GB per cudaHostRegister

    def _get_client(self):
        """Connect, then make sure the pool file is fully backed. /dev/shm is a size-limited tmpfs and pool files are
        sparse: writing a page tmpfs can't allocate raises SIGBUS and kills the engine with no traceback (seen
        with a 24 GiB Arc pool + 8 GiB 5090 pool on a 31 GB /dev/shm). posix_fallocate reserves every page up front
        (no-op for pages already written); if it fails, the tier is OFF for this engine instead (scheduler and worker
        both run this check, so they agree)."""
        if self._client_obj is None:
            c = super()._get_client()
            if not isinstance(c, _NoBroker):
                try:
                    fd = os.open(c.pool_path, os.O_RDWR)
                    try:
                        os.posix_fallocate(fd, 0, c.n_slots * c.slot_bytes)
                    finally:
                        os.close(fd)
                except OSError as e:
                    st = os.statvfs(os.path.dirname(c.pool_path))
                    logger.error("kvbroker: cannot back the pool %s (%.1f GiB) on its filesystem (%s; %.1f GiB free): "
                                 "RAM tier OFF until the engine restarts", c.pool_path, c.n_slots * c.slot_bytes / 2**30,
                                 e, st.f_bavail * st.f_frsize / 2**30)
                    self._client_obj = _NoBroker()
        return self._client_obj

    def pin(self, ptr: int, size: int) -> bool:
        if os.environ.get("LLMT_KV_PIN", "1") == "0":      # as on the Arc: pool stays pageable
            logger.info("kvbroker: LLMT_KV_PIN=0: pool not pinned")
            self._pinned = []
            return False
        if os.environ.get("EXL3_DYN_PIN", "1") == "1" and self.kv_bytes_per_chunk:
            try:
                self._slotpin = SlotPinner(ptr, size, self.kv_bytes_per_chunk, self.extra_config)
                return True
            except Exception as e:  # noqa: BLE001
                logger.warning("kvbroker: SLOT-PIN failed (%s): chunked whole-pool pinning instead (pool can't shrink)", e)
        logger.info("kvbroker: pinning in thread %s", threading.current_thread().name)
        rt = _cudart()
        step = self.CHUNK_SLOTS * (self.kv_bytes_per_chunk or size)
        self._pinned, off, t0 = [], 0, time.monotonic()
        nbytes = 0
        while off < size:
            n = min(step, size - off)
            r = torch.cuda.cudart().cudaHostRegister(ptr + off, n, 0)
            if int(r.value) == 0:
                self._pinned.append(ptr + off)
                nbytes += n
            elif rt is not None:
                rt.cudaGetLastError()      # clear the pending error, or the next kernel launch reports it
            off += n
        logger.info("kvbroker: cudaHostRegister %d of %d chunks (%.1f of %.1f GiB pinned, %.1f s); the rest is copied "
                    "as pageable memory", len(self._pinned), -(-size // step), nbytes / 2**30, size / 2**30,
                    time.monotonic() - t0)
        return bool(self._pinned)

    def unpin(self, ptr: int) -> None:
        sp = getattr(self, "_slotpin", None)
        if sp is not None:
            sp.close()
            self._slotpin = None
        for p in getattr(self, "_pinned", []):
            torch.cuda.cudart().cudaHostUnregister(p)
        self._pinned = []

    def create_worker(self, kv_caches):
        import vllm.v1.kv_offload.cpu.gpu_worker as gw
        orig = gw.pin_mmap_region
        gw.pin_mmap_region = lambda region: None       # pinned above, in chunks
        try:
            return super().create_worker(kv_caches)
        finally:
            gw.pin_mmap_region = orig


class SlotPinner:
    """Per-slot pinning, so the broker can give memory back at runtime: the chunked whole-pool
    registration above pins every page, so a broker shrink (park + hole-punch) freed nothing (pages stay pinned as
    hugetlb surplus) and a later regrow would hand the parked slots new pages while the GPU still DMAs to the old
    ones. Here each broker slot gets its own cudaHostRegister, parked slots are never registered (registering a punched
    hole would fault it back in behind the broker), and a thread follows the broker's parked set: unregister newly
    parked slots, register slots that came back, then pool_ack the generation. The broker punches only after every
    announced client acked; it parks only free slots (never leased/reserved), so no copy is in flight on them.
    Replaces gpu_worker.DynamicPoolPinner, which this tier bypassed (create_worker no-ops pin_mmap_region)."""

    POLL_S = 1.0

    def __init__(self, base: int, size: int, stride: int, extra: dict):
        import llmt_kv
        self.base, self.stride, self.n = base, int(stride), int(size // stride)
        self.rt = _cudart()
        self.sock = extra.get("broker_socket", "/run/llmt-kv/broker.sock")
        self.engine_id = int(extra.get("broker_engine_id", 2))
        self.client = llmt_kv.Client(socket=self.sock, engine_id=self.engine_id,
                                     name=os.environ.get("LLMT_KV_PIN_NAME", "rtx5090-pin"), timeout_ms=200.0)
        if not hasattr(self.client, "pool_state"):
            raise RuntimeError("llmt_kv client without pool_state")
        self.gen, runs = self.client.pool_state()
        self.parked = {j for s, k in runs for j in range(s, s + k)}
        self.registered: set[int] = set()
        self.failed = 0
        t0 = time.monotonic()
        for i in range(self.n):
            if i not in self.parked:
                self._register(i)
        self.client.pool_ack(self.gen)
        logger.info("kvbroker: SLOT-PIN registered %d of %d slots (%.1f MB each, %d parked, %d failed) in %.1f s; gen %d acked",
                    len(self.registered), self.n, self.stride / 1e6, len(self.parked), self.failed,
                    time.monotonic() - t0, self.gen)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="slot-pin", daemon=True)
        self.thread.start()

    def _register(self, i: int) -> bool:
        r = torch.cuda.cudart().cudaHostRegister(self.base + i * self.stride, self.stride, 0)
        if int(r.value) != 0:
            if self.rt is not None:
                self.rt.cudaGetLastError()
            self.failed += 1
            if self.failed <= 3:
                logger.warning("kvbroker: SLOT-PIN cudaHostRegister slot %d failed (code %d): pageable", i, int(r.value))
            return False
        self.registered.add(i)
        return True

    def _unregister(self, i: int) -> None:
        if i in self.registered:
            r = torch.cuda.cudart().cudaHostUnregister(self.base + i * self.stride)
            if int(r.value) != 0 and self.rt is not None:
                self.rt.cudaGetLastError()
                logger.warning("kvbroker: SLOT-PIN cudaHostUnregister slot %d failed (code %d)", i, int(r.value))
            self.registered.discard(i)

    def _run(self) -> None:
        import llmt_kv
        while not self.stop.wait(self.POLL_S):
            try:
                if self.client is None:
                    self.client = llmt_kv.Client(socket=self.sock, engine_id=self.engine_id,
                                                 name=os.environ.get("LLMT_KV_PIN_NAME", "rtx5090-pin"), timeout_ms=200.0)
                    self.gen = -1            # re-announce after a reconnect (the broker forgot us)
                gen, runs = self.client.pool_state()
                if gen == self.gen:
                    continue
                parked = {j for s, k in runs for j in range(s, s + k)}
                with self.lock:
                    gone, back = sorted(parked - self.parked), sorted(self.parked - parked)
                    for i in gone:
                        self._unregister(i)
                    for i in back:
                        if i < self.n:
                            self._register(i)
                    self.parked = parked
                self.client.pool_ack(gen)
                if gone or back or self.gen < 0:
                    logger.info("kvbroker: SLOT-PIN gen %d: %d parked (-%d unregistered, +%d registered), %d pinned; acked",
                                gen, len(parked), len(gone), len(back), len(self.registered))
                self.gen = gen
            except Exception as e:  # noqa: BLE001
                logger.warning("kvbroker: SLOT-PIN broker call failed (%s); reconnecting", e)
                self.client = None

    def close(self) -> None:
        self.stop.set()
        with self.lock:
            for i in sorted(self.registered):
                self._unregister(i)
