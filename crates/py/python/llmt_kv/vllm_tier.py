"""vLLM offloading tier over llmt-kvbroker, shared by the engines (Arc B70: llmt_kvbroker_arc.tier, RTX 5090:
exl3cuda.kvbroker_tier). Engine-specific: only pool pinning (cudaHostRegister / Level Zero import) and defaults, in a
subclass of BrokerOffloadingSpecBase. Imported by the engines only (needs vllm and torch); `import llmt_kv` alone
does not import it.

vLLM's CPU offloading copies GPU blocks <-> rows of one mapped region; here the region is the broker's pool (one broker
slot = one vLLM CPU block = all KV tensors' pages for one block number; Qwen3.8-27B, 1,664-token pages, int8 KV:
58,839,040 B on both GPUs -> broker --slot-kib 57460) and the bookkeeping is the broker's:

  lookup(key)      broker lookup with a lease (the broker never evicts a leased entry). The lease belongs to the
                   requests that looked the key up and is released when each of them has loaded it, was admitted
                   (ScheduleEndContext.new_req_ids: vLLM looks up and calls prepare_load in the same schedule()
                   call, so whatever an admitted request did not load it never will) or finished. A request still
                   waiting keeps its leases, so its lookup every scheduling attempt is answered locally (`local`)
                   instead of by a broker round trip (97% of broker lookups in the 5090's 5-agent replay were such
                   re-lookups). Without the release on admission the hybrid model's lookups held leases they never
                   load (chunks past the hit boundary the Mamba-group suffix lookup sets) for the whole request: 331
                   of 986 in one Arc run. Safety net: a lease idle for `broker_lease_steps` steps
                   (default 1024) is released; without new_req_ids (older vLLM) every unloaded lease is released at
                   the end of its step. At most
                   `broker_lease_frac` of the pool (default 0.25) is leased: beyond that, new lookups miss (those
                   requests recompute) instead of pinning more.
  prepare_load     held leases -> CPU block numbers (= broker slots)
  complete_load    release
  prepare_store    reserve: OK -> slot; EXISTS / BUSY (stored or being written elsewhere) -> skipped; FULL -> partial
  complete_store   commit (statuses checked: a reservation the broker dropped is counted as lost), or abort
  touch            touch

Failures never stop the engine; they skip the tier:
  - no broker at engine start (waits `broker_wait_s`, default 30), or a pool that can't be fully allocated (full
    tmpfs; writing an unallocated page would SIGBUS the engine): tier off until the engine restarts
  - BrokerTimeout / BrokerGone: this step runs without the tier; the client reconnects by itself
  - BrokerReset (this call reconnected): reservations and leases are gone; pending stores are dropped; slots of
    held leases stay readable (a restarted broker reuses the pool file and hands out no slots for a grace period)
  - BrokerPoolChanged: another pool file; tier off until the engine restarts
  - a load whose lease is gone and can't be re-acquired (broker restarted or unreachable between lookup and load):
    the slot remembered from the lookup is loaded unleased (safe: a restarted broker reuses the pool and hands out no
    slots for a grace period). If the entry was instead evicted with the broker up (only if the lease was
    released before the load: a scheduler that loads in a later step than it looks up), the slot may hold another
    block: counted as `unleased_evicted` and logged as an error. vLLM 0.26's offloading connector asserts that loads
    succeed, so failing the load (kv_load_failure_policy=recompute) would need a vLLM patch.

Per request (kv_transfer_params in the request body, passed to the connector by vLLM):
  llmt_ns    use this namespace instead of the engine's own, if it is in `broker_ns_allow` (default ["xgpu-test"]):
             cross-GPU tests on live engines without restarting them onto a test namespace.
  llmt_conv  a conversation id (set by the router): when the request finishes, the number of its chunks that are in
             RAM (stored, already there, or loaded) is written to `broker_hints_dir`/<broker_name>.json (default
             /run/llmt-kv/hints; at most every 5 s, atomic rename): {"client", "ns", "t", "chunk_tokens",
             "convs": {conv: {"chunks", "t", "ns"}}}, newest 2000 conversations of the last hour. The router uses it to
             credit the other GPU with a RAM load instead of a prefill when it moves a conversation.
  llmt_anchor  32 hex chars (set by the router, docs/prefix-hints.md): when the request finishes, its KV keys per group in
             token order go to the broker as one HINT under this anchor (if the broker has FEAT_PREFIX_HINTS), so a
             router's WHERE can tell which engine holds how many leading chunks of a conversation in RAM.
             `broker_hint_chunk_tokens` (default: the offloaded block size, else 1664) = tokens per key.
  llmt_nostore  true: this request stores nothing in RAM (it may still load). For test traffic that must not displace
             production entries.

Keys: blake2b-256(namespace | key format version | compat tag | vLLM offload key = block hash + group index). Engines
share entries only with the same namespace and compat tag; cross-GPU sharing also needs the same block hash function
(--prefix-caching-hash-algo sha256 and PYTHONHASHSEED), block size and KV format.

  kv_connector_extra_config: {"spec_name": "BrokerOffloadingSpec", "spec_module_path": "<engine module>",
    "cpu_bytes_to_use": 1, "broker_socket": ..., "broker_namespace": ..., "broker_engine_id": ..., "broker_name": ...,
    "broker_timeout_ms": 2, "broker_reconnect_ms": 1000, "broker_wait_s": 30, "broker_lease_frac": 0.25,
    "broker_lease_steps": 1024}
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import time
from collections import OrderedDict
from collections.abc import Collection, Iterable

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (LoadStoreSpec, LookupResult, OffloadingManager, OffloadKey, PrepareStoreOutput,
                                     ReqContext, RequestOffloadingContext)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

import llmt_kv

logger = init_logger("vllm.llmt_kv")
_DBG = os.environ.get("EXL3_OFFLOAD_DEBUG") == "1"

KEY_FORMAT = 1          # bump when the key derivation or the stored layout changes
FORMAT_ID = 1           # entry format: vLLM CPU block rows, int8_per_token_head KV + GDN state pages
STATS_EVERY_S = 300     # one stats line per 5 min
# Entry metadata (broker `meta`, opaque to the broker except the event journal): b"L1" + parent (16 B prefix of the
# previous stored chunk's broker key in the same group, zeros if none) + fingerprint (8 B blake2b of the vLLM offload
# key: namespace-independent, for cross-GPU miss analysis) + model id (6 B)
META_V1 = b"L1"
HINT_EVERY_S = 5        # conversation hints file: at most every 5 s, when something changed
HINT_MAX_CONVS = 2000
HINT_MAX_AGE_S = 3600
HINT_CHUNK_TOKENS = 1664  # tokens per chunk (one broker slot) on both GPUs today



# SESSION + PRIORITY TAGS (router session_policy). The router puts the request's class and session tags in
# kv_transfer_params: llmt_cls lowest / low / normal / agent, llmt_term 1 (the session's terminal request), llmt_sess / llmt_parent (16 hex: the router's keyed 64-bit hashes,
# never the raw ids), llmt_agent (letter). Without them the X-Request-Id session tag ("arcs" + 12 hex + 12 hex +
# letter, vLLM's request id starts with it) gives the session. Low / lowest stores carry ENTRY_LOW / ENTRY_LOWEST
# (reserve "low" / "lowest": the broker takes them first under room pressure, ranks them last in the hot set); one TOUCH_TAGGED per request carries
# the group kinds + tags (a tagged session links from its first request). Old llmt_kv build: TOUCH_REQUEST or TOUCH;
# old broker: the native client falls back by itself. LLMT_KV_TAGS=0: no tags, no low flag (TOUCH_REQUEST only);
# LLMT_KV_TOUCH_REQUEST=0: plain TOUCH batches as before (full rollback without an image change).
_TAG_CLS = {"low": 1, "normal": 2, "agent": 3, "lowest": 4}
_TAG_RE = re.compile(r"arcs([0-9a-f]{12})([0-9a-f]{12})([a-z]?)-")


def _tags_on() -> bool:
    return os.environ.get("LLMT_KV_TAGS", "1") != "0"


def _touch_request_on() -> bool:
    return os.environ.get("LLMT_KV_TOUCH_REQUEST", "1") != "0"


def req_tags(req_context) -> tuple[int, int, int, int, int]:
    """(class 0 unknown / 1 low / 2 normal / 3 agent / 4 lowest, session, parent session (64-bit, 0 = none), agent
    letter byte, flags (bit 0: terminal request))."""
    p = getattr(req_context, "kv_transfer_params", None)
    p = p if isinstance(p, dict) else {}
    cls = _TAG_CLS.get(str(p.get("llmt_cls") or ""), 0)

    def h64(v) -> int:
        try:
            return int(str(v), 16) & ((1 << 64) - 1) if v else 0
        except ValueError:
            return 0
    sess, par = h64(p.get("llmt_sess")), h64(p.get("llmt_parent"))
    ag = str(p.get("llmt_agent") or "")[:1]
    if not sess:
        m = _TAG_RE.search(str(getattr(req_context, "req_id", "") or ""))
        if m:
            sess, par = int(m.group(1), 16), int(m.group(2), 16)
            ag = ag or m.group(3)
    return cls, sess, par, (ord(ag) if ag else 0), (1 if p.get("llmt_term") else 0)

class _NoBroker:
    """Stand-in client when no broker was reachable at engine start: every call is a no-op, the tier stays off."""
    connected = False
    reconnects = 0
    slot_bytes = 0
    n_slots = 0
    pool_path = ""

    def __getattr__(self, name):
        return lambda *a, **k: None


def _pool_backed(path: str, size: int, slot_bytes: int = 0, parked=()) -> bool:
    """Make sure every pool page the broker can hand out is allocated (a no-op when the broker already did, as it does
    by default). An unallocated page on a full tmpfs gives the engine SIGBUS on its first write: a silent death
    with no traceback. Not allocatable: tier off.
    Dynamic pool: `parked` = the broker's parked runs [(start_slot, n_slots)], hole-punched and their huge
    pages returned to the kernel. Allocate only around them: filling the holes fails with ENOSPC on a shrunk hugetlbfs
    pool (the tier then stays off) and would undo the shrink behind the broker's back."""
    runs, pos = [], 0
    for start, n in sorted(parked):
        if start * slot_bytes > pos:
            runs.append((pos, start * slot_bytes - pos))
        pos = max(pos, (start + n) * slot_bytes)
    if pos < size:
        runs.append((pos, size - pos))
    try:
        fd = os.open(path, os.O_RDWR)
        try:
            for off, ln in runs:
                os.posix_fallocate(fd, off, ln)
        finally:
            os.close(fd)
        return True
    except OSError as e:
        logger.error("kvbroker: cannot allocate the pool %s (%s): RAM tier OFF until the engine restarts", path, e)
        return False


def connect(extra: dict, engine_id: int, name: str):
    """Connect, waiting up to broker_wait_s for a broker that is still starting. A _NoBroker when none answers: the
    engine then starts with the tier off (logged) instead of failing to start."""
    sock = extra.get("broker_socket", "/run/llmt-kv/broker.sock")
    deadline = time.monotonic() + float(extra.get("broker_wait_s", 30))
    while True:
        try:
            c = llmt_kv.Client(socket=sock, engine_id=int(extra.get("broker_engine_id", engine_id)),
                                  name=str(extra.get("broker_name", name)),
                                  timeout_ms=float(extra.get("broker_timeout_ms", 2)),
                                  reconnect_ms=float(extra.get("broker_reconnect_ms", 1000)))
            try:
                parked = c.pool_state()[1] if hasattr(c, "pool_state") else []
            except llmt_kv.BrokerError:  # broker without the dynamic pool: nothing is parked
                parked = []
            return c if _pool_backed(c.pool_path, c.n_slots * c.slot_bytes, c.slot_bytes, parked) else _NoBroker()
        except llmt_kv.BrokerError as e:
            if time.monotonic() > deadline:
                logger.error("kvbroker: no broker on %s (%s): RAM tier OFF until the engine restarts", sock, e)
                return _NoBroker()
            time.sleep(1)


class BrokerPoolRegion(SharedOffloadRegion):
    """The broker's pool file as vLLM's shared offload region: rows = slots (row stride = slot size), one worker
    (rank 0) owning the whole row. `pin(ptr, size) -> bool` / `unpin(ptr)` register it with the GPU runtime. Never
    creates, truncates or unlinks the pool (the broker owns it)."""

    def __init__(self, pool_path: str, n_slots: int, slot_bytes: int, pin=None, unpin=None):  # noqa: no super()
        self.page_size = mmap.PAGESIZE
        self.num_blocks = n_slots
        self._row_stride = slot_bytes
        self.total_size_bytes = n_slots * slot_bytes
        self.mmap_path = pool_path
        self._creator = False
        self.rank = 0
        self._worker_offset = 0
        self._worker_area_end = slot_bytes
        self._unpin = unpin
        self.fd = os.open(pool_path, os.O_RDWR)
        size = os.fstat(self.fd).st_size
        assert size >= self.total_size_bytes, f"broker pool {pool_path} is {size} B < {self.total_size_bytes}"
        # MAP_NORESERVE: a shared hugetlbfs mapping otherwise reserves a huge page for every
        # unallocated page of the file, i.e. the parked slots' punched holes, whose pages went back to the kernel:
        # ENOSPC on a shrunk pool. We only touch broker-assigned (allocated) slots. Python's mmap module lacks the name.
        self.mmap_obj = mmap.mmap(self.fd, self.total_size_bytes,
                                  flags=mmap.MAP_SHARED | getattr(mmap, "MAP_NORESERVE", 0x4000),
                                  prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views = []
        t0 = time.monotonic()
        self.is_pinned = bool(pin and pin(self._base.data_ptr(), self.total_size_bytes))
        logger.info("kvbroker: mapped pool %s (%d slots x %.1f MB), pinned: %s (%.1f s)", pool_path, n_slots,
                    slot_bytes / 1e6, self.is_pinned, time.monotonic() - t0)

    def cleanup(self) -> None:
        if self.is_pinned and self._unpin:
            try:
                self._unpin(self._base.data_ptr())
            except Exception:  # noqa
                pass
        self.is_pinned = False
        self._views.clear()
        try:
            self._base = None
            if self.mmap_obj is not None:
                self.mmap_obj.close()
        except Exception:  # noqa (views still alive at shutdown)
            pass
        self.mmap_obj = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def model_id() -> str:
    """Model identity for the compat tag and entry metadata: sha256 over config.json,
    quantization_config.json and the shards' crc32.txt in the model directory (LLMT_KV_MODEL_DIR, default /models,
    the path both engines mount). Block keys depend only on tokens, so without this a weights change with the same
    tokenizer would load KV computed by the old model."""
    d = os.environ.get("LLMT_KV_MODEL_DIR", "/models")
    h = hashlib.sha256()
    found = False
    for name in ("config.json", "quantization_config.json", "crc32.txt", "model.safetensors.index.json"):
        try:
            h.update(name.encode() + open(os.path.join(d, name), "rb").read())
            found = True
        except OSError:
            pass
    return h.hexdigest()[:12] if found else "unknown"


_STATE_TAG: str | None = None

# Layout-neutral GDN state in RAM (so engines on different GPUs, e.g. an Arc and an RTX 5090, share one RAM cache). A GDN state
# page holds the conv state ((kernel - 1) + k rows; rows 0..kernel-2 are the state, the k spec-window rows are scratch
# that GDN recompute never writes) and then the SSM state, so its byte offset depends on the engine's draft depth k
# (Arc 6, 5090 12). RAM keeps only the real ranges, packed: conv rows 0..kernel-2, then the SSM state. The copy path
# splits each GDN page copy into those ranges (no staging, no extra bytes moved); see `_NeutralCopy`.
# LLMT_KV_GDN_NEUTRAL=0 keeps the engine layout in RAM (tag k<k>) as before.
GDN_NEUTRAL = os.environ.get("LLMT_KV_GDN_NEUTRAL", "1") != "0"
GDN_CONV_ROWS = int(os.environ.get("EXL3_KV_NANCHECK_CONV_ROWS", "3"))   # kernel - 1 (Qwen3.x GDN: kernel 4)
_STATE_K: int | None = None


def state_tag() -> str:
    """GDN/Mamba state layout: a state slot holds the conv state ((conv_kernel - 1) + num_speculative_tokens
    rows: MTP scratch rows) and then the SSM state, so the SSM state's byte offset depends on the engine's MTP depth.
    Arc-made states (k=6) loaded on the RTX 5090 (k=7) gave NaN from GDN layer 0 (NAN-PREEMPT). Tag = max draft depth +
    the Mamba state dtypes, from the vLLM config. That config is current while the worker side is built (spec init /
    create_worker) but not when the scheduler-side manager is (first live start: gdn:unknown), so the first value seen
    is cached for the process (one EngineCore holds both sides on a single-GPU engine). LLMT_KV_STATE_TAG overrides;
    "unknown" (never matching a known layout) if nothing was seen."""
    global _STATE_TAG
    t = os.environ.get("LLMT_KV_STATE_TAG")
    if t:
        return t
    if _STATE_TAG is None:
        try:
            from vllm.config import get_current_vllm_config_or_none
            vc = get_current_vllm_config_or_none()
        except Exception:  # noqa: BLE001 - older vLLM without the accessor
            vc = None
        if vc is not None:
            sc = getattr(vc, "speculative_config", None)
            k = int(getattr(sc, "num_speculative_tokens", 0) or 0) if sc is not None else 0
            cc = vc.cache_config
            global _STATE_K
            _STATE_K = k
            lay = f"neutral{GDN_CONV_ROWS}" if GDN_NEUTRAL else f"k{k}"   # neutral: sizes added by _install_neutral
            _STATE_TAG = f"{lay}-ssm:{getattr(cc, 'mamba_ssm_cache_dtype', '?')}-conv:{getattr(cc, 'mamba_cache_dtype', '?')}"
    if _STATE_TAG is None:
        logger.warning("kvbroker: GDN state layout unknown (no vLLM config current yet): entries only match engines "
                       "that also don't know; set LLMT_KV_STATE_TAG")
        return "unknown"
    return _STATE_TAG


def compat_tag(slot_bytes: int) -> str:
    """Entries are interchangeable only between engines with the same block size and KV row format (both in the slot
    size), KV dtype, K-smoothing table, GDN state handling and attention page layout; folded into every key (the
    spec's compat_id). Layout: VLLM_KV_CACHE_LAYOUT if set (the Arc sets NHD
    for its kernels), else HND, which vLLM takes from OffloadingConnector.get_required_kvcache_layout(). An NHD page
    is [token][head][row], an HND page [head][token][row]: same logical shape and slot size, transposed bytes; loading
    one as the other gave KL 0.07-1.2 in the cross-GPU test. The tag describes the bytes in RAM: an engine
    whose VRAM layout differs from the RAM layout must convert on its copy path and set LLMT_KV_RAM_LAYOUT."""
    ks = os.environ.get("EXL3_KSMOOTH")
    try:
        ksh = hashlib.sha256(open(ks, "rb").read()).hexdigest()[:12] if ks else "none"
    except OSError:
        ksh = "unreadable"
    kvd = os.environ.get("EXL3_KV_DTYPE_TAG", "int8_per_token_head")
    rc = os.environ.get("EXL3_GDN_RECOMPUTE", "0")
    # RAM format, which an engine that converts on its copy path declares with LLMT_KV_RAM_LAYOUT (RAM holds
    # the Arc's NHD layout; the RTX 5090 keeps HND in VRAM and converts when it stores/loads)
    lay = (os.environ.get("LLMT_KV_RAM_LAYOUT") or os.environ.get("VLLM_KV_CACHE_LAYOUT") or "HND").upper()
    return f"slot{slot_bytes}|kv:{kvd}|ks:{ksh}|gdnrc:{rc}|lay:{lay}|gdn:{state_tag()}|model:{model_id()}|fmt{FORMAT_ID}"


class BrokerOffloadingManager(OffloadingManager):
    def __init__(self, client, namespace: str, slot_bytes: int, lease_steps: int = 1024, lease_cap: int = 0,
                 ns_allow: Collection[str] = ("xgpu-test",), hints_path: str | None = None, client_name: str = "",
                 group_kinds: Collection[int] = (), hint_chunk_tokens: int = 0, resident_cap: int = 0):
        self.c = client
        # prefix-affinity hints (llmt_anchor): req id -> [anchor, namespace id, {group: [broker key]}, seen keys]
        # (None: the request has no anchor)
        self._anchor: dict[str, list] = {}
        self.hint_chunk_tokens = int(hint_chunk_tokens or HINT_CHUNK_TOKENS)
        self.namespace = namespace
        self.compat = compat_tag(slot_bytes)
        self._model6 = bytes.fromhex(self.compat.split("|model:")[1].split("|")[0][:12].ljust(12, "0")) \
            if "|model:" in self.compat and "unknown" not in self.compat else bytes(6)
        self._nss: dict[str, tuple[int, bytes]] = {}                  # namespace -> (broker id, key tag)
        self.ns, self._ns_tag = self._nsinfo(namespace)
        self.ns_allow = set(ns_allow)                                 # namespaces a request may pick (llmt_ns)
        self.hints_path, self.client_name = hints_path, client_name
        self._conv: OrderedDict[str, dict] = OrderedDict()            # llmt_conv -> {"chunks", "t", "ns"}
        self._req_ram: dict[str, list] = {}                           # req id -> [conv, chunks in RAM, ns]
        self._hints_t = 0.0
        self._hints_dirty = False
        # entry kind per KV group (0 attention, 1 Mamba/GDN state; the broker's kind field; unknown groups: 3)
        self.group_kinds = tuple(group_kinds)
        # (protocol "TOUCH_REQUEST"): a request's touched groups, sent as one tagged TOUCH_REQUEST once its last
        # group is touched (or at the end of the scheduler step). req id -> (namespace id, group -> keys)
        self._touch_buf: dict[str, tuple[int, dict[int, list]]] = {}
        self._req_tags: dict[str, tuple[int, int, int, int, int]] = {}       # req id -> req_tags()
        self._last_stored: dict[str, dict[int, bytes]] = {}           # req id -> group -> broker key of last store
        self.slot_bytes = slot_bytes
        self.lease_steps = lease_steps
        self.lease_cap = lease_cap or 1 << 30                          # max LOAD-class leases (0 = no cap)
        # LEASE-CLASS: leases have a class by holder. "load" = a request's lookup before its load
        # (transient; lease_cap applies, but a request that is the only loader may go past it: one request alone can
        # always restore what the pool holds), "resident" = a long-context session's middle chunks (lookup_cls), "hold" = an
        # awaiting session's hold (holder ids "arcH|..." from a long-context residency plugin). resident + hold share resident_cap. Every class is renewed
        # on the broker before its lease deadline (LLMT_KV_LEASE_RENEW_S, default 20 s; broker default 60 s).
        self.resident_cap = resident_cap or 1 << 30
        self._res_rids: set[str] = set()                              # holders that lease as "resident"
        self.renew_s = float(os.environ.get("LLMT_KV_LEASE_RENEW_S", "20") or 0)
        self.renew_max = int(os.environ.get("LLMT_KV_LEASE_RENEW_MAX", "256") or 256)   # renewals per schedule
        self._step = 0
        self._leases: dict[OffloadKey, list] = {}                     # key -> [lease_id, slot, last step, {req ids}, t]
        # (a request held back for a while hallucinated a user turn): a lease answered
        # locally for longer than the broker's lease timeout may already be gone on the broker, which may then have
        # reused the slot. A local hit older than LLMT_KV_LEASE_FRESH_S (30; broker timeout 60 s, now
        # 1800 s) re-acquires the lease (miss -> MISS); a load always uses a lease younger than that.
        self.lease_fresh_s = float(os.environ.get("LLMT_KV_LEASE_FRESH_S", "30"))
        self._req_keys: dict[str, set[OffloadKey]] = {}               # req id -> keys it holds leases on
        # key -> (lease_id, slot, req id) of in-flight loads; every load holds its own broker lease (a
        # shared lease could be released while another request's copy of the slot was still running)
        self._loading: dict[OffloadKey, list[tuple[int, int, str]]] = {}
        self._pending_store: set[OffloadKey] = set()
        self._hint: OrderedDict[OffloadKey, int] = OrderedDict()     # slot of recent hits, for unleased loads
        self._off = isinstance(client, _NoBroker)                     # no broker at start / BrokerPoolChanged
        self.n = dict.fromkeys(("lease_refresh", "lease_refresh_moved", "lease_refresh_lost", "stale_cleared",
                                "lookups", "hits", "loads", "stores", "exists", "busy", "full", "committed", "lost",
                                "errors", "resets", "unleased", "unleased_evicted", "lease_expired", "lease_admitted", "lease_cap", "local",
                                "hits_attn", "hits_state", "loads_attn", "loads_state", "committed_attn", "committed_state",
                                "admitted_attn", "admitted_state", "lease_cap_res", "lease_cap_bypass",
                                "renewed", "renew_lost", "renew_moved"), 0)
        self._stats_t = time.monotonic()
        logger.info("kvbroker: namespace %r, compat %s, tier %s", namespace, self.compat, "OFF" if self._off else "on")

    # ---- helpers

    def _nsinfo(self, name: str) -> tuple[int, bytes]:
        v = self._nss.get(name)
        if v is None:
            v = self._nss[name] = (int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "little"),
                                   f"{name}|{KEY_FORMAT}|{self.compat}|".encode())
        return v

    def _req_ns(self, req_context) -> str | None:
        """Namespace a request asks for (kv_transfer_params.llmt_ns, allowlisted); None = the engine's own."""
        p = getattr(req_context, "kv_transfer_params", None)
        n = p.get("llmt_ns") if isinstance(p, dict) else None
        return n if n and n != self.namespace and n in self.ns_allow else None

    def _ik(self, key: OffloadKey, req_context):
        """Internal key: the vLLM key in the engine's namespace, (namespace, key) in a requested one."""
        n = self._req_ns(req_context)
        return key if n is None else (n, key)

    def _bk(self, ik) -> bytes:
        if isinstance(ik, tuple):
            return hashlib.blake2b(self._nsinfo(ik[0])[1] + bytes(ik[1]), digest_size=32).digest()
        return hashlib.blake2b(self._ns_tag + bytes(ik), digest_size=32).digest()

    def _kind(self, ik) -> int:
        """Entry kind of a (internal) key from its KV group: 0 attention, 1 GDN state, 3 unknown."""
        k = ik[1] if isinstance(ik, tuple) else ik
        g = int.from_bytes(k[-4:], "big")
        return self.group_kinds[g] if g < len(self.group_kinds) else 3

    def _count(self, what: str, keys) -> None:
        """Per-kind counters (what_attn / what_state) for diagnosing hits that need a missing state."""
        for k in keys:
            kd = self._kind(k)
            if kd in (0, 1):
                self.n[f"{what}_{'attn' if kd == 0 else 'state'}"] += 1

    def _nsid(self, ik) -> int:
        return self._nsinfo(ik[0])[0] if isinstance(ik, tuple) else self.ns

    def _ram(self, req_context, n: int) -> None:
        """Count chunks of a request that are in RAM (stored, already there, or loaded), for conversation hints."""
        if not self.hints_path or n <= 0:
            return
        r = self._req_ram.get(req_context.req_id)
        if r is None:
            p = getattr(req_context, "kv_transfer_params", None)
            conv = p.get("llmt_conv") if isinstance(p, dict) else None
            if not conv:
                return
            r = self._req_ram[req_context.req_id] = [str(conv)[:128], 0, self._req_ns(req_context) or self.namespace]
        r[1] += n

    def _note_keys(self, req_context, iks) -> None:
        """Prefix hints: remember a request's keys per group in token order (llmt_anchor requests only)."""
        rid = req_context.req_id
        a = self._anchor.get(rid)
        if a is None and rid in self._anchor:           # no anchor (checked once per request)
            return
        if a is None:
            p = getattr(req_context, "kv_transfer_params", None)
            h = p.get("llmt_anchor") if isinstance(p, dict) else None
            try:
                anchor = bytes.fromhex(h) if isinstance(h, str) else b""
            except ValueError:
                anchor = b""
            if len(anchor) != 16:
                self._anchor[rid] = None
                return
            if not iks:
                return
            a = self._anchor[rid] = [anchor, self._nsid(iks[0]), {}, set()]
        for ik in iks:
            if ik in a[3]:
                continue
            a[3].add(ik)
            raw = ik[1] if isinstance(ik, tuple) else ik
            a[2].setdefault(int.from_bytes(raw[-4:], "big"), []).append(self._bk(ik))

    def _send_hint(self, req_id: str) -> None:
        a = self._anchor.pop(req_id, None)
        if a is None or not a[2] or not (getattr(self.c, "features", 0) & getattr(llmt_kv, "FEAT_PREFIX_HINTS", 1 << 5)):
            return
        kinds = self.group_kinds
        groups = [(kinds[g] if g < len(kinds) else 3, a[2][g]) for g in sorted(a[2])]
        if self._call(lambda: self.c.hint(a[1], a[0], self.hint_chunk_tokens, groups, 1)) is not None:
            self.n["hints"] = self.n.get("hints", 0) + 1

    def _slot(self, entry) -> int:
        off, _ = entry.byte_ranges[0]
        assert len(entry.byte_ranges) == 1 and off % self.slot_bytes == 0, "kvbroker: entry spans several slots"
        return off // self.slot_bytes

    def _call(self, fn, *a):
        """None = skip the tier this step (timeout, not connected, just reconnected, pool changed)."""
        if self._off:
            return None
        try:
            return fn(*a)
        except llmt_kv.BrokerPoolChanged as e:
            self._off = True
            self._pending_store.clear()
            self._forget_leases()
            logger.error("kvbroker: %s; RAM tier OFF until the engine restarts", e)
        except llmt_kv.BrokerReset as e:
            self.n["resets"] += 1
            self._pending_store.clear()
            # broker-side leases are gone: its slots may be handed out again, so no
            # lease, hint or held slot of before the reset may be loaded any more (lookups ask the broker afresh)
            self._forget_leases()
            for ls in self._loading.values():
                ls[:] = [(0, slot, rid) for _, slot, rid in ls]
            logger.warning("kvbroker: %s (reset %d)", e, self.n["resets"])
        except llmt_kv.BrokerError as e:      # timeout / broker gone: run without the tier this step
            self.n["errors"] += 1
            if self.n["errors"] <= 5 or self.n["errors"] % 1000 == 0:
                logger.warning("kvbroker: %s: %s (%d errors)", type(e).__name__, e, self.n["errors"])
        return None

    def _acquire(self, key: OffloadKey):
        """Lease `key` on the broker. (lease_id, slot), or None on miss / no broker."""
        r = self._call(lambda ks: self.c.lookup(self._nsid(key), ks, acquire=True), [self._bk(key)])
        if _DBG:
            logger.info("KVB-DBG lookup %s -> %s", self._bk(key)[:6].hex(), "err" if r is None else r[0])
        if r is None:
            return None
        if r[0] == 0:
            return 0
        held = (r[1], self._slot(r[2][0]))
        self._hint[key] = held[1]
        self._hint.move_to_end(key)
        if len(self._hint) > 8192:
            self._hint.popitem(last=False)
        return held

    # ---- OffloadingManager

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def _own_lease(self, key: OffloadKey, slot: int) -> tuple[int, int]:
        """A broker lease of this load's own on `key`, which must still be at `slot`; raise otherwise."""
        h = self._acquire(key)
        if not h or h[1] != slot:
            if h:
                self._call(self.c.release, [h[0]])
            raise RuntimeError(f"kvbroker: block {self._bk(key)[:6].hex()} cannot be leased at slot {slot} for its "
                               "load; refusing to load it unleased")
        return h

    def _forget_leases(self) -> None:
        """Drop every lease, holder and hint this engine keeps (tier off or broker reset): never load them."""
        self.n["stale_cleared"] += len(self._leases)
        self._leases.clear()
        self._req_keys.clear()
        self._hint.clear()

    def _fresh(self, key: OffloadKey, held: list) -> bool:
        """Re-acquire a lease older than lease_fresh_s from the broker. False: the entry is gone (lease dropped)."""
        if time.monotonic() - held[4] < self.lease_fresh_s:
            return True
        self.n["lease_refresh"] += 1
        h = self._acquire(key)
        if not h:
            self.n["lease_refresh_lost"] += 1
            for rid in held[3]:
                self._req_keys.get(rid, set()).discard(key)
            self._leases.pop(key, None)
            if held[0]:
                self._call(self.c.release, [held[0]])
            return False
        if h[1] != held[1]:
            self.n["lease_refresh_moved"] += 1
        old = held[0]
        held[0], held[1], held[4] = h[0], h[1], time.monotonic()
        if old:
            self._call(self.c.release, [old])
        return True

    def _hold(self, key: OffloadKey, req_id: str) -> None:
        self._leases[key][3].add(req_id)
        self._req_keys.setdefault(req_id, set()).add(key)

    def _drop(self, key: OffloadKey, req_id: str) -> None:
        """`req_id` no longer needs the lease on `key`; release it when nobody does."""
        held = self._leases.get(key)
        if held is None:
            return
        held[3].discard(req_id)
        if not held[3]:
            del self._leases[key]
            if held[0]:
                self._call(self.c.release, [held[0]])

    def _hcls(self, rid: str) -> str:
        return "hold" if rid.startswith("arcH|") else ("resident" if rid in self._res_rids else "load")

    def lease_class(self, key) -> str:
        """Class of a held lease (internal key): the strongest of its holders' classes."""
        held = self._leases.get(key)
        if held is None:
            return "none"
        cl = {self._hcls(r) for r in held[3]}
        return "hold" if "hold" in cl else ("resident" if "resident" in cl else "load")

    def class_counts(self) -> dict[str, int]:
        out = {"load": 0, "resident": 0, "hold": 0}
        for k in self._leases:
            c = self.lease_class(k)
            if c in out:
                out[c] += 1
        return out

    def lookup_cls(self, key: OffloadKey, req_context: ReqContext, cls: str = "load") -> LookupResult:
        """lookup() under a lease class: "resident" (long-context session) or "hold" (awaiting session), see LEASE-CLASS."""
        if cls == "resident":
            self._res_rids.add(req_context.req_id)
        return self.lookup(key, req_context)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        self.n["lookups"] += 1
        if self._off:
            return LookupResult.MISS
        key = self._ik(key, req_context)
        self._note_keys(req_context, [key])
        held = self._leases.get(key)
        if held is not None and not self._fresh(key, held):
            held = None
        if held is not None:
            held[2] = self._step
            self.n["local"] += 1
            self._hold(key, req_context.req_id)
            return LookupResult.HIT
        # (Hypothesis state machine: a key being loaded for another request used to be a HIT without
        # asking the broker; after a broker reset that HIT could not be honoured and the load was refused. It now
        # takes a lease of its own like any other hit.)
        rid = req_context.req_id
        if self._hcls(rid) == "load":
            n_load, others = 0, 0
            for v in self._leases.values():
                if all(self._hcls(r) == "load" for r in v[3]):
                    n_load += 1
                    if v[3] - {rid}:
                        others += 1
            if n_load >= self.lease_cap:
                if others:
                    self.n["lease_cap"] += 1
                    return LookupResult.MISS
                self.n["lease_cap_bypass"] += 1          # the only loader: may restore past the cap
        else:
            n_res = sum(1 for k in self._leases if self.lease_class(k) in ("resident", "hold"))
            if n_res >= self.resident_cap:
                self.n["lease_cap_res"] += 1
                return LookupResult.MISS
        h = self._acquire(key)
        if not h:
            return LookupResult.MISS
        self._leases[key] = [h[0], h[1], self._step, set(), time.monotonic()]
        self._hold(key, req_context.req_id)
        self.n["hits"] += 1
        self._count("hits", (key,))
        return LookupResult.HIT

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        slots = []
        rid = req_context.req_id
        keys = [self._ik(k, req_context) for k in keys]
        self._ram(req_context, len(keys))
        for key in keys:
            held = self._leases.get(key)
            if held is not None and not self._fresh(key, held):
                raise RuntimeError(f"kvbroker: block {self._bk(key)[:6].hex()} left the RAM tier between lookup and "
                                   "load (lease too old); refusing to load its slot")
            if held is not None:
                held[3].discard(rid)
                self._req_keys.get(rid, set()).discard(key)
                if held[3]:                                # other waiting requests still need it: keep their lease,
                    held = self._own_lease(key, held[1])   # and this load takes one of its own
                else:                                      # last user: the lease moves to the load
                    del self._leases[key]
                    held = (held[0], held[1])
            elif self._loading.get(key):                   # being loaded for another request: same slot, own lease
                held = self._own_lease(key, self._loading[key][-1][1])
            else:                                          # lease released or never taken: pin it again
                h = self._acquire(key)
                if h:
                    held = h
                else:
                    # never load an unleased slot (its bytes may belong to another
                    # block, or be zeros after a broker restart); a loud failure beats a corrupted context
                    self.n["unleased"] += 1
                    raise RuntimeError(f"kvbroker: block {self._bk(key)[:6].hex()} has no lease at load time "
                                       f"({'broker unreachable' if h is None else 'evicted since the lookup'}); "
                                       "refusing to load an unleased slot")
            self._loading.setdefault(key, []).append((held[0], held[1], rid))
            slots.append(held[1])
        self.n["loads"] += len(slots)
        self._count("loads", keys)
        return CPULoadStoreSpec(slots)

    def _tags(self, req_context) -> tuple[int, int, int, int, int]:
        rid = req_context.req_id
        t = self._req_tags.get(rid)
        if t is None:
            t = self._req_tags[rid] = req_tags(req_context) if _tags_on() else (0, 0, 0, 0, 0)
            if t[0] in (1, 4):
                self.n["low_requests"] = self.n.get("low_requests", 0) + 1
        return t

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        if keys:
            iks = [self._ik(k, req_context) for k in keys]
            if not _touch_request_on() or not hasattr(self.c, "touch_request"):    # an llmt_kv build without it
                self._call(self.c.touch, self._nsid(iks[0]), [self._bk(k) for k in iks])
                return
            g = int.from_bytes((iks[0][1] if isinstance(iks[0], tuple) else iks[0])[-4:], "big")
            self._touch_buf.setdefault(req_context.req_id, (self._nsid(iks[0]), {}, self._tags(req_context)))[1][g] = iks
            if g >= len(self.group_kinds) - 1:          # the scheduler touches the groups in order: this was the last
                self._flush_touch(req_context.req_id)

    def _flush_touch(self, rid: str) -> None:
        """Send a request's buffered groups as one TOUCH_TAGGED (tags known and the build has it) or TOUCH_REQUEST; the
        native client sends what the broker understands (TOUCH batches to an older one)."""
        b = self._touch_buf.pop(rid, None)
        if b:
            ns, groups, (cls, sess, par, ag, fl) = b
            gs = [(self.group_kinds[g] if g < len(self.group_kinds) else 3, [self._bk(k) for k in groups[g]])
                  for g in sorted(groups)]
            if (cls or sess) and hasattr(self.c, "touch_tagged"):
                self._call(self.c.touch_tagged, ns, gs, sess, par, cls, ag, fl)
                self.n["touch_tagged"] = self.n.get("touch_tagged", 0) + 1
            else:
                self._call(self.c.touch_request, ns, gs)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        rel = []
        for key in keys:
            key = self._ik(key, req_context)
            ls = self._loading.get(key)
            if ls:
                i = next((j for j in range(len(ls) - 1, -1, -1) if ls[j][2] == req_context.req_id), len(ls) - 1)
                lease, _, _ = ls.pop(i)               # this request's own load (not the newest one of the key)
                if lease:
                    rel.append(lease)
                if not ls:
                    del self._loading[key]
        if rel:
            self._call(self.c.release, rel)

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext) -> PrepareStoreOutput | None:
        p = getattr(req_context, "kv_transfer_params", None)
        if isinstance(p, dict) and p.get("llmt_nostore"):
            self.n["nostore"] = self.n.get("nostore", 0) + len(keys)
            return PrepareStoreOutput(keys_to_store=[], store_spec=CPULoadStoreSpec([]), evicted_keys=[])
        all_iks = [self._ik(k, req_context) for k in keys]
        self._note_keys(req_context, all_iks)
        pairs = [(k, ik) for k, ik in zip(keys, all_iks) if ik not in self._pending_store]
        if not pairs:
            return PrepareStoreOutput(keys_to_store=[], store_spec=CPULoadStoreSpec([]), evicted_keys=[])
        keys = [ik for _, ik in pairs]
        # Parent labels (journal metadata): the chain advances only over blocks the broker accepted (OK / EXISTS) or
        # that are already pending. Advancing before the reply made a store retried after a broker restart its own
        # parent (a cycle in the journal's chains).
        last = self._last_stored.setdefault(req_context.req_id, {})
        grp = lambda k: int.from_bytes((k[1] if isinstance(k, tuple) else k)[-4:], "big")
        tmp, items = dict(last), []
        low = self._tags(req_context)[0]
        low = low if low in (1, 4) else 0                # the request's value class: ENTRY_LOW / ENTRY_LOWEST
        if low:
            self.n["low_stores"] = self.n.get("low_stores", 0) + len(all_iks)
        for k in all_iks:
            bk = self._bk(k)
            if k in self._pending_store:
                tmp[grp(k)] = bk
                continue
            parent = tmp.get(grp(k))
            raw = k[1] if isinstance(k, tuple) else k
            items.append({"key": bk, "parent": None, "nbytes": self.slot_bytes, "kind": self._kind(k), "low": low == 1,
                          "lowest": low == 4,
                          "format": FORMAT_ID, "meta": META_V1 + (parent[:16] if parent else bytes(16))
                          + hashlib.blake2b(bytes(raw), digest_size=8).digest() + self._model6})
            tmp[grp(k)] = bk
        r = self._call(self.c.reserve, self._nsid(keys[0]), items)
        if r is None:
            return None
        status = dict(zip(keys, (st for st, _ in r)))
        for k in all_iks:
            if k in self._pending_store or status.get(k) in (llmt_kv.OK, llmt_kv.EXISTS):
                last[grp(k)] = self._bk(k)
        if _DBG:
            logger.info("KVB-DBG reserve %s", " ".join(f"{self._bk(k)[:6].hex()}:{st}" for k, (st, _) in zip(keys, r)))
        to_store, slots = [], []
        for (vkey, key), (status, ranges) in zip(pairs, r):
            if status == llmt_kv.OK:
                off, _ = ranges[0]
                assert len(ranges) == 1 and off % self.slot_bytes == 0
                to_store.append(vkey)
                slots.append(off // self.slot_bytes)
                self._pending_store.add(key)
            elif status == llmt_kv.EXISTS:
                self.n["exists"] += 1
                self._ram(req_context, 1)
            elif status == llmt_kv.BUSY:
                self.n["busy"] += 1
            else:
                self.n["full"] += 1
        self.n["stores"] += len(to_store)
        if not to_store and all(s not in (llmt_kv.OK, llmt_kv.EXISTS, llmt_kv.BUSY) for s, _ in r):
            return None
        return PrepareStoreOutput(keys_to_store=to_store, store_spec=CPULoadStoreSpec(slots), evicted_keys=[])

    def complete_store(self, keys: Collection[OffloadKey], req_context: ReqContext, success: bool = True) -> None:
        mine = [ik for ik in (self._ik(k, req_context) for k in keys) if ik in self._pending_store]
        if not mine:
            return
        for k in mine:
            self._pending_store.discard(k)
        if _DBG:
            logger.info("KVB-DBG %s %s", "commit" if success else "abort", " ".join(self._bk(k)[:6].hex() for k in mine))
        st = self._call(self.c.commit if success else self.c.abort, self._nsid(mine[0]), [self._bk(k) for k in mine])
        if success and st is not None:
            ok = sum(1 for s in st if s == llmt_kv.OK)
            self.n["committed"] += ok
            self._count("committed", [k for k, s_ in zip(mine, st) if s_ == llmt_kv.OK])
            self._ram(req_context, ok)
            if ok < len(st):
                self.n["lost"] += len(st) - ok
                logger.warning("kvbroker: %d of %d commits refused (%s): reservation expired or dropped (%d lost)",
                               len(st) - ok, len(st), ",".join(sorted({llmt_kv.status_name(s) for s in st
                                                                         if s != llmt_kv.OK})), self.n["lost"])

    def on_schedule_end(self, context) -> None:
        for rid in list(self._touch_buf):                # requests whose last group had nothing to touch
            self._flush_touch(rid)
        self._step += 1
        admitted = getattr(context, "new_req_ids", None)
        limit = self.lease_steps
        if admitted is None:                            # no admission info: release every unloaded lease now
            limit = 0
        elif self._req_keys:
            for rid in admitted:                        # admitted: its loads are prepared; the rest won't be loaded
                keys = self._req_keys.pop(rid, ())
                unloaded = [k for k in keys if k in self._leases]
                self.n["lease_admitted"] += len(unloaded)
                self._count("admitted", unloaded)
                for key in keys:
                    self._drop(key, rid)
        self._renew()
        if self._leases and (limit < 64 or self._step % 64 == 0):   # safety net: leases nobody loaded or released
            old = [k for k, v in self._leases.items() if self._step - 1 - v[2] >= limit
                   and self.lease_class(k) == "load"]          # LEASE-CLASS: resident / hold end with their holder
            if old:
                rel = []
                for k in old:
                    v = self._leases.pop(k)
                    for r in v[3]:
                        self._req_keys.get(r, set()).discard(k)
                    if v[0]:
                        rel.append(v[0])
                self.n["lease_expired"] += len(old)
                if limit:
                    logger.warning("kvbroker: released %d leases idle for >= %d steps (%d so far)", len(old),
                                   limit, self.n["lease_expired"])
                if rel:
                    self._call(self.c.release, rel)
        if self.hints_path and self._hints_dirty and time.monotonic() - self._hints_t > HINT_EVERY_S:
            self._hints_t, self._hints_dirty = time.monotonic(), False
            self._write_hints()
        if time.monotonic() - self._stats_t > STATS_EVERY_S:
            self._stats_t = time.monotonic()
            logger.info("kvbroker: stats %s, leases held %d, connected %s, reconnects %d", self.n, len(self._leases),
                        self.c.connected, self.c.reconnects)

    def _renew(self) -> None:
        """Re-acquire leases older than renew_s (new broker lease first, then release the old one), so a long-lived
        lease never hits the broker's lease deadline while its slot is still read."""
        if not self.renew_s or not self._leases or self._off:
            return
        now = time.monotonic()
        due = [k for k, v in self._leases.items() if v[0] and now - v[4] >= self.renew_s]
        for k in due[:self.renew_max]:
            v = self._leases[k]
            h = self._acquire(k)
            if h is None:                                   # broker error: retry next schedule
                break
            if h == 0:                                      # entry gone (evicted after a lapse / reset): mark lost
                self.n["renew_lost"] += 1
                v[0] = 0
                continue
            old = v[0]
            if h[1] != v[1]:
                self.n["renew_moved"] += 1
            v[0], v[1], v[4] = h[0], h[1], now
            self.n["renewed"] += 1
            self._call(self.c.release, [old])

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._send_hint(req_context.req_id)
        self._res_rids.discard(req_context.req_id)
        for key in self._req_keys.pop(req_context.req_id, ()):
            self._drop(key, req_context.req_id)
        self._last_stored.pop(req_context.req_id, None)
        self._flush_touch(req_context.req_id)
        self._req_tags.pop(req_context.req_id, None)
        r = self._req_ram.pop(req_context.req_id, None)
        if r is not None:                               # conversation hint: chunks of its latest request in RAM
            self._conv[r[0]] = {"chunks": r[1], "t": round(time.time(), 1), "ns": r[2]}
            self._conv.move_to_end(r[0])
            while len(self._conv) > HINT_MAX_CONVS:
                self._conv.popitem(last=False)
            self._hints_dirty = True

    def _write_hints(self) -> None:
        """Conversation -> chunks in RAM, for the router (atomic rename; dropped after HINT_MAX_AGE_S)."""
        now = time.time()
        while self._conv and now - next(iter(self._conv.values()))["t"] > HINT_MAX_AGE_S:
            self._conv.popitem(last=False)
        doc = {"client": self.client_name, "ns": self.namespace, "t": round(now, 1), "chunk_tokens": HINT_CHUNK_TOKENS,
               "convs": self._conv}
        try:
            os.makedirs(os.path.dirname(self.hints_path), exist_ok=True)
            tmp = f"{self.hints_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(doc, f)
            os.replace(tmp, self.hints_path)
        except OSError as e:
            logger.warning("kvbroker: cannot write hints %s: %s", self.hints_path, e)
            self.hints_path = None

    def reset_cache(self) -> None:
        pass

    def take_events(self) -> Iterable:
        return []


_POOL_REGION = None      # the worker's mapped (and on CUDA pinned) pool, set by create_worker; TailStore copies through it

TAIL_KIND = 2            # broker entry kind of a tail spill ("attn+state" in broker logs and metrics)
TAIL_FORMAT = 2          # payload format id: raw engine layout (raw-nv); the caller's meta describes it


class TailStore:
    """Partial-block ("tail") spills of the RTX 5090's tail holds: the engine copies a
    released hold's partial attention rows + GDN states into broker slots and a follow-up request copies them back.
    Ephemeral broker entries (kind TAIL_KIND, TTL; evicted first once read). Keys use the tier's namespace/compat
    prefix, so they only match engines with the same compat tag. The caller does the GPU copies through
    `host_tensor()` and owns meta (<= 1024 B; checked by the caller on load). Every broker call goes through the
    manager's `_call`: a broker outage or restart is a miss (None), never an exception.

        ts = TailStore(manager)
        ns = ts.ns_for(request.kv_transfer_params); k = ts.key(ns, prev_full_block_hash)
        ranges = ts.reserve(ns, k, nbytes, ttl_ms, meta)      # [(host_ptr, length)] | "exists" | None
        ... D2H into those ranges ...; ts.commit(ns, k)  (or ts.abort(ns, k))
        hit = ts.lookup(ns, k)                                # (lease_id, [(host_ptr, length)], meta, nbytes) | None
        ... H2D ...; ts.release(hit[0])
    """
    KEY_TAG = b"tail1|"
    META_MAX = 1024

    def __init__(self, manager, region=None):
        self.m = manager
        self.region = region if region is not None else _POOL_REGION
        self.n = {"reserved": 0, "exists": 0, "full": 0, "committed": 0, "aborted": 0, "lookups": 0, "hits": 0,
                  "released": 0}

    def namespace_for(self, llmt_ns: str | None) -> str:
        """The namespace a request with kv_transfer_params.llmt_ns = `llmt_ns` uses, resolved like the tier does
        (allowlisted name, else the engine's own)."""
        m = self.m
        return llmt_ns if llmt_ns and llmt_ns != m.namespace and llmt_ns in m.ns_allow else m.namespace

    def ns_for(self, kv_transfer_params) -> str:
        """Namespace for a request's kv_transfer_params (its llmt_ns if allowlisted, else the engine's own)."""
        n = kv_transfer_params.get("llmt_ns") if isinstance(kv_transfer_params, dict) else None
        return self.namespace_for(n)

    def key(self, ns_name: str, prev_full_block_hash: bytes, tail_hash: bytes = b"") -> bytes:
        """Usually key(ns, prev_full_block_hash): one tail per block chain; the caller keeps L and the tail-token hash
        in meta and verifies them after lookup (a newer tail for the same chain finds "exists": the older one stays)."""
        return hashlib.blake2b(self.m._nsinfo(ns_name)[1] + self.KEY_TAG + bytes(prev_full_block_hash) + b"|"
                               + bytes(tail_hash), digest_size=32).digest()

    def _base(self) -> int:
        if self.region is None:
            raise RuntimeError("kvbroker: TailStore needs the worker's mapped pool (create_worker has not run)")
        return self.region._base.data_ptr()

    def host_tensor(self) -> torch.Tensor:
        """The mapped pool as one int8 tensor (pinned on the RTX 5090); slice [ptr - base : ptr - base + length]."""
        self._base()
        return self.region._base

    def host_base(self) -> int:
        return self._base()

    def reserve(self, ns_name: str, key: bytes, nbytes: int, ttl_ms: int, meta: bytes):
        if len(meta) > self.META_MAX:
            raise ValueError(f"TailStore meta is {len(meta)} B > {self.META_MAX}")
        r = self.m._call(self.m.c.reserve, self.m._nsinfo(ns_name)[0],
                         [{"key": key, "nbytes": int(nbytes), "kind": TAIL_KIND, "format": TAIL_FORMAT,
                           "ephemeral": True, "ttl_ms": int(ttl_ms), "meta": bytes(meta)}])
        if not r:
            return None
        st, ranges = r[0]
        if st == llmt_kv.EXISTS:
            self.n["exists"] += 1
            return "exists"
        if st != llmt_kv.OK:
            self.n["full"] += 1
            return None
        self.n["reserved"] += 1
        base = self._base()
        return [(base + off, ln) for off, ln in ranges]

    def commit(self, ns_name: str, key: bytes) -> bool:
        st = self.m._call(self.m.c.commit, self.m._nsinfo(ns_name)[0], [key])
        ok = bool(st) and st[0] == llmt_kv.OK
        self.n["committed" if ok else "aborted"] += 1
        return ok

    def abort(self, ns_name: str, key: bytes) -> None:
        self.m._call(self.m.c.abort, self.m._nsinfo(ns_name)[0], [key])
        self.n["aborted"] += 1

    def lookup(self, ns_name: str, key: bytes):
        self.n["lookups"] += 1
        r = self.m._call(self.m.c.lookup, self.m._nsinfo(ns_name)[0], [key], True)
        if not r:
            return None
        n_hit, lease_id, entries = r
        if not n_hit or not entries:
            if lease_id:
                self.m._call(self.m.c.release, [lease_id])
            return None
        e = entries[0]
        self.n["hits"] += 1
        base = self._base()
        return lease_id, [(base + off, ln) for off, ln in e.byte_ranges], bytes(e.meta), e.nbytes

    def release(self, lease_id: int) -> None:
        if lease_id:
            self.m._call(self.m.c.release, [lease_id])
            self.n["released"] += 1


def neutral_plan(group_segs, group_kinds, page_sizes):
    """Per KV group: None (copied as whole pages) or, per data ref, [(engine offset, RAM offset, bytes)] = the ref's
    state segments (from the engine's page layout, `group_segs`: group -> ref -> [(offset, bytes, dtype)]) packed from
    RAM offset 0. Raises ValueError on a layout that can't be converted safely."""
    plan = []
    for g, refs in enumerate(group_segs):
        if g >= len(group_kinds) or group_kinds[g] != 1:
            plan.append(None)
            continue
        if not refs or not all(refs):
            raise ValueError(f"GDN state group {g} has a ref without state segments")
        gp = []
        for r, segs in enumerate(refs):
            out, ram, end = [], 0, 0
            for off, n, _dt in segs:
                if off < end or n <= 0:
                    raise ValueError(f"group {g} ref {r}: segments overlap or are empty: {segs}")
                out.append((int(off), ram, int(n)))
                ram += int(n)
                end = off + n
            if end > page_sizes[g][r]:
                raise ValueError(f"group {g} ref {r}: segments end at {end} > page {page_sizes[g][r]}")
            gp.append(out)
        plan.append(gp)
    return plan


class _NeutralCopy:
    """Wraps one SingleDirectionOffloadingHandler: GDN page copies are split into the plan's ranges (GPU engine offsets
    <-> packed RAM offsets) right before the batched copy. Non-GDN ops pass through unchanged. The rewritten
    descriptor buffers stay referenced until their job is reported finished (the copy may read them asynchronously).

    Loads also zero the rest of each GDN page: the k spec-window conv rows and the page tail are not in
    RAM, and the GPU block a restore lands in still holds its previous owner's bytes there. Decode reads those rows
    before writing them (MAMBA-ZERO's finding; MAMBA-ZERO zeroes new blocks but skips the restored prefix), so a
    restored state ran its first speculative step on a foreign request's scratch rows: NaN when that owner left NaN
    (NAN-PREEMPT, which frees the poisoned blocks for the next restore). A cold-computed state has zeros there."""

    def __init__(self, handler, plan, gpu_to_cpu: bool):
        self.h, self.plan, self.g2c = handler, plan, gpu_to_cpu
        # loads: per GDN group / ref, the engine page ranges the plan does not write -> zero-filled
        self.gaps: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.zero = None
        if not gpu_to_cpu:
            big = 0
            for g, gp in enumerate(plan):
                if gp is None:
                    continue
                for r, rp in enumerate(gp):
                    page = int(handler.kv_cache_groups_data_refs[g][r].page_size_bytes)
                    gaps, at = [], 0
                    for eoff, _roff, nb in sorted(rp):
                        if eoff > at:
                            gaps.append((at, eoff - at))
                        at = max(at, eoff + nb)
                    if at < page:
                        gaps.append((at, page - at))
                    self.gaps[(g, r)] = gaps
                    big = max([big] + [n for _o, n in gaps])
            if big:
                z = torch.zeros(big, dtype=torch.int8)
                try:
                    z = z.pin_memory()                  # a direct (non-staged) copy reads it by DMA
                except Exception:  # noqa: BLE001 - no accelerator: staged / test copies read host memory
                    pass
                self.zero = z
        self.zero_ptr = int(self.zero.data_ptr()) if self.zero is not None else None   # tests: a fake address space
        self.zeroed_ops = 0
        self.orig_transfer, self.orig_swap = handler.transfer_async, handler._swap_blocks_batch
        self.orig_finished = handler.get_finished
        self.group_sizes = None
        self.job = None
        self.inflight: dict[int, tuple] = {}
        self.ops_split = 0
        handler.transfer_async = self.transfer_async
        handler._swap_blocks_batch = self.swap
        handler.get_finished = self.get_finished

    def transfer_async(self, job_id, src_spec, dst_spec):
        gpu_spec = src_spec if self.g2c else dst_spec
        self.group_sizes, self.job = list(gpu_spec.group_sizes), job_id
        try:
            return self.orig_transfer(job_id, src_spec, dst_spec)
        finally:
            self.group_sizes, self.job = None, None

    def get_finished(self):
        res = self.orig_finished()
        for r in res:
            self.inflight.pop(getattr(r, "job_id", r[0] if isinstance(r, tuple) else None), None)
        return res

    def rewrite(self, src, dst, sizes):
        """numpy int64 arrays in transfer_async's op order -> rewritten arrays, or None if nothing to split."""
        import numpy as np
        gs = self.group_sizes
        if gs is None or not any(p is not None and n for p, n in zip(self.plan, gs)):
            return None
        parts_s, parts_d, parts_z = [], [], []
        op = 0
        for g, n in enumerate(gs):
            if not n:
                continue
            for r in range(len(self.h.kv_cache_groups_data_refs[g])):
                a, b = op, op + n
                if self.plan[g] is None:
                    parts_s.append(src[a:b]); parts_d.append(dst[a:b]); parts_z.append(sizes[a:b])
                else:
                    for eoff, roff, nb in self.plan[g][r]:
                        so, do = (eoff, roff) if self.g2c else (roff, eoff)
                        parts_s.append(src[a:b] + so); parts_d.append(dst[a:b] + do)
                        parts_z.append(np.full(b - a, nb, dtype=sizes.dtype))
                    if not self.g2c and self.zero_ptr is not None:
                        zp = self.zero_ptr
                        for goff, glen in self.gaps.get((g, r), ()):
                            parts_s.append(np.full(b - a, zp, dtype=src.dtype)); parts_d.append(dst[a:b] + goff)
                            parts_z.append(np.full(b - a, glen, dtype=sizes.dtype))
                            self.zeroed_ops += b - a
                    self.ops_split += b - a
                op = b
        if op != len(src):
            raise RuntimeError(f"kvbroker: GDN-neutral copy: {op} ops expected, transfer has {len(src)}")
        return np.concatenate(parts_s), np.concatenate(parts_d), np.concatenate(parts_z)

    def swap(self, src, dst, sizes, is_src_access_order_any: bool = False, **kw):
        new = self.rewrite(src.numpy(), dst.numpy(), sizes.numpy())
        if new is None:
            return self.orig_swap(src, dst, sizes, is_src_access_order_any=is_src_access_order_any, **kw)
        pin = src.is_pinned()
        ts = tuple(torch.from_numpy(x).pin_memory() if pin else torch.from_numpy(x) for x in new)
        if self.job is not None:
            self.inflight[self.job] = ts
        return self.orig_swap(*ts, is_src_access_order_any=is_src_access_order_any, **kw)


def _install_neutral(worker, kinds) -> bool:
    """Wrap the worker's store/load handlers with the GDN-neutral copy, finish the state tag with the packed range sizes
    (engines whose ranges differ never match), and point the NaN scan of stored RAM pages at the packed offsets. Runs at
    worker creation, before the scheduler-side manager computes its compat tag. Without the engine's page layout
    (gpu_worker.set_layout, our offloading worker patch) or with an unexpected one, RAM keeps the engine layout and the
    tag falls back to k<k> (shares only with the same draft depth, as before)."""
    global _STATE_TAG
    import vllm.v1.kv_offload.cpu.gpu_worker as gw
    tag = state_tag()
    if os.environ.get("LLMT_KV_STATE_TAG") or not tag.startswith(f"neutral{GDN_CONV_ROWS}-"):
        return False
    st, ld = worker._store_handler, worker._load_handler
    try:
        segs = getattr(gw, "_NC_LAYOUT", None)
        if segs is None:
            raise ValueError("no engine page layout (gpu_worker.set_layout)")
        pages = [[r.page_size_bytes for r in g] for g in st.kv_cache_groups_data_refs]
        plan = neutral_plan(segs, list(kinds), pages)
        sizes = {tuple(n for _e, _r, n in rp) for gp in plan if gp is not None for rp in gp}
        if not sizes:
            return False
        if len(sizes) != 1 or len(next(iter(sizes))) != 2:
            raise ValueError(f"expected one (conv rows, SSM) range pair for every GDN page, got {sorted(sizes)}")
    except ValueError as e:
        _STATE_TAG = tag.replace(f"neutral{GDN_CONV_ROWS}-", f"k{_STATE_K}-", 1)
        logger.error("kvbroker: GDN-neutral RAM layout unavailable (%s): RAM keeps this engine's layout, tag %s", e,
                     _STATE_TAG)
        return False
    _NeutralCopy(st, plan, gpu_to_cpu=True)
    _NeutralCopy(ld, plan, gpu_to_cpu=False)
    gw._NC_LAYOUT = [segs[g] if gp is None else [[(roff, nb, dt) for (_e, roff, nb), (_o, _n, dt) in zip(rp, rs)]
                                                 for rp, rs in zip(gp, segs[g])] for g, gp in enumerate(plan)]
    _STATE_TAG = tag.replace(f"neutral{GDN_CONV_ROWS}-", f"neutral{GDN_CONV_ROWS}:{'+'.join(map(str, next(iter(sizes))))}-", 1)
    g0 = next(g for g, p in enumerate(plan) if p is not None)
    logger.info("kvbroker: GDN state in RAM is layout-neutral (tag %s): engine ranges %s -> RAM %s per page",
                _STATE_TAG, [(e, n) for e, _r, n in plan[g0][0]], [(r, n) for _e, r, n in plan[g0][0]])
    return True


class BrokerOffloadingSpecBase(CPUOffloadingSpec):
    """Engine modules subclass this as `BrokerOffloadingSpec`, set the defaults and implement pin/unpin."""
    DEFAULT_NAMESPACE = "private"
    DEFAULT_ENGINE_ID = 0
    DEFAULT_NAME = "engine"

    def pin(self, ptr: int, size: int) -> bool:
        return False

    def unpin(self, ptr: int) -> None:
        pass

    def __init__(self, config):
        super().__init__(config)
        state_tag()                                     # capture the GDN state layout while vLLM's config is current
        self._client_obj = None
        self._region = None
        if self.kv_bytes_per_chunk:
            c = self._get_client()
            if isinstance(c, _NoBroker):
                self.num_blocks = 1                            # placeholder CPU block; the manager never uses it
                return
            if c.slot_bytes != self.kv_bytes_per_chunk:
                raise RuntimeError(f"kvbroker: broker slot {c.slot_bytes} B != vLLM CPU block {self.kv_bytes_per_chunk}"
                                   f" B: start the broker with --slot-kib {self.kv_bytes_per_chunk // 1024}")
            self.num_blocks = c.n_slots
            logger.info("kvbroker: %d slots of %.1f MB in namespace %r", c.n_slots, c.slot_bytes / 1e6,
                        self._namespace())

    def _namespace(self) -> str:
        return str(self.extra_config.get("broker_namespace", self.DEFAULT_NAMESPACE))

    def _get_client(self):
        if self._client_obj is None:
            self._client_obj = connect(self.extra_config, self.DEFAULT_ENGINE_ID, self.DEFAULT_NAME)
        return self._client_obj

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            c = self._get_client()
            name = str(self.extra_config.get("broker_name", self.DEFAULT_NAME))
            kinds = self._group_kinds()
            logger.info("kvbroker: entry kinds per KV group (0 attention, 1 state): %s", list(kinds))
            hints = self.extra_config.get("broker_hints_dir", "/run/llmt-kv/hints")
            self._manager = BrokerOffloadingManager(
                c, self._namespace(), c.slot_bytes or 1, int(self.extra_config.get("broker_lease_steps", 1024)),
                max(1, int(c.n_slots * float(self.extra_config.get("broker_lease_frac", 0.25)))) if c.n_slots else 0,
                ns_allow=self.extra_config.get("broker_ns_allow", ["xgpu-test"]),
                hints_path=f"{hints}/{name}.json" if hints else None, client_name=name, group_kinds=kinds,
                hint_chunk_tokens=int(self.extra_config.get("broker_hint_chunk_tokens", 0)
                                      or getattr(self, "offloaded_block_size", 0) or 0),
                resident_cap=max(1, int(c.n_slots * float(self.extra_config.get("broker_resident_frac", 0.9))))
                if c.n_slots else 0)
        return self._manager

    def create_worker(self, kv_caches) -> CPUOffloadingWorker:
        state_tag()
        c = self._get_client()
        if isinstance(c, _NoBroker):
            return CPUOffloadingWorker(kv_caches=kv_caches, blocks_per_chunk=self.blocks_per_chunk, num_cpu_blocks=1)
        global _POOL_REGION
        self._region = _POOL_REGION = BrokerPoolRegion(c.pool_path, c.n_slots, c.slot_bytes, self.pin, self.unpin)
        w = CPUOffloadingWorker(kv_caches=kv_caches, blocks_per_chunk=self.blocks_per_chunk,
                                num_cpu_blocks=c.n_slots, mmap_region=self._region)
        if GDN_NEUTRAL:
            _install_neutral(w, self._group_kinds())
        return w

    def _group_kinds(self) -> list[int]:
        kinds = self.extra_config.get("broker_group_kinds")
        if kinds is None:   # GDN/Mamba groups (linear attention) store states; the rest attention pages
            kinds = [1 if any(t in ln for ln in g.layer_names for t in ("linear_attn", "mamba", "gdn")) else 0
                     for g in getattr(self.config, "groups", ())]
        return list(kinds)
