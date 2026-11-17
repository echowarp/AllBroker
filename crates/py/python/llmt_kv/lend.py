"""llmt_kv.lend: the lend manager. VRAM that vLLM sized as KV cache is lent, block by block, to other tenants while
they need it, and returned to KV when they are done. KV always wins.

Tenants (borrowers):
  LoRA adapters   slot buffers (punica lora_a/b_stacked) live in lent KV blocks only while an adapter is active
                  (EXL3_LORA_HOT=1; vLLM started with --enable-lora --max-loras N --max-lora-rank R and
                  VLLM_ALLOW_RUNTIME_LORA_UPDATING=1). The adapter bytes are a broker kind-5 (weights) entry shared by
                  every engine with the same adapter file, read from the broker's pool on activation.
  vision tower    the lazy vision tower (llmt_kv.lazyvis) is placed into lent pages when it is needed (VIS-LEND).

Policy:
  chain-aware compaction   prefix-cache chains are valued (request class, session liveness, reuse); before a lend the
                           manager scores candidate regions and moves the cheapest blocks out of the way
  region scoring           score_region(pool, lo, n): the cost of clearing a block range
  victim eviction          free blocks first, then cached blocks of the lowest-value chains
  move guard               in-use block moves are deferred while an async-scheduled step or a pending KV-RING copy
                           may still read the block
  regret counters          chain blocks dropped by a lend and asked for again soon after are counted
  LEND-STATS               lending / compaction / regret / fragmentation counters, logged periodically

vLLM integration: a vllm.general_plugins entry point (llmt_kv.plugin:register). Scheduler, block-pool, model-runner
and worker points go through llmt_kv.vllm_hooks: echowarp/vllm's extension hooks natively, a one-time method wrapper on
stock vLLM. LoRA internals (punica wrapper, LoRA layers, cudagraph dispatch for LoRA steps, the LoRA model-runner mixin)
are still wrapped directly; see docs/vllm-plugin.md for the list and the hooks that would replace them.
Admission hold / priority preemption for a full KV pool: extension point set_admission_policy() (hook scheduler.admit,
echowarp/vllm only); no policy ships yet.

Why stock vLLM LoRA applies nothing in compiled mode: the punica wrapper and the LoRA layers decide in Python whether a
LoRA is active. Dynamo traces the model once, at the startup profile run with no adapter active, and vLLM drops all
guards, so the compiled graph has the LoRA terms traced out. This module keeps the compiled + graph path exactly the
base model (the LoRA layers return the base output while Dynamo traces, the dispatcher captures only no-LoRA graphs)
and runs a step with any LoRA token eagerly (forward_context.skip_compiled), or from a separate LoRA graph pool.

Knobs (all optional): EXL3_LORA_TTL (300), EXL3_LORA_MARGIN_MB (256), EXL3_LORA_MODE auto|device|host,
EXL3_LORA_BROKER auto|off|<sock>, EXL3_LORA_NS, EXL3_LORA_ENGINE_ID, EXL3_LORA_STORE (1), EXL3_LORA_VERIFY_HIT (0),
EXL3_LORA_DIR (adapter directory for load-by-name), EXL3_LORA_TIMEOUT_MS (100), EXL3_LORA_BUSY_WAIT_MS (5000),
EXL3_LEND_* (see docs/lend.md).
Ported from an internal tree with tools/lend-port/port.py; SYNC.json there names the source commit.
Logs counts / timings / sizes only (no request data)."""
import contextlib
import ctypes
import sys
import hashlib
import json
import os
import struct
import threading
import math
import time
import zlib

import torch

_ENV = os.environ.get
ON = _ENV("EXL3_LORA_HOT", "0") == "1"
TTL = float(_ENV("EXL3_LORA_TTL", "300"))
MARGIN = int(float(_ENV("EXL3_LORA_MARGIN_MB", "256")) * 2**20)
MODE = _ENV("EXL3_LORA_MODE", "auto")
# A LoRA request must never slow base traffic down: no host-slot fallback by default.
# A LoRA request runs only from a device slot; with no room it waits in the queue (the scheduler's max_loras is the
# device capacity, 0 if nothing fits; base requests keep running). EXL3_LORA_HOST_FALLBACK=1: the old streamed host
# slots (lab only: every request in such a step runs at the host-slot speed).
HOST_FALLBACK = _ENV("EXL3_LORA_HOST_FALLBACK", "0") == "1"


class _NoRoom(Exception):
    pass


# ------------------------------------------------------------------------------------------ kernel hooks
# LoRA decode kernels live in a separate module (exl3xpu lorak/, loaded by name).
# EXL3_LORA_KERNELS=<importable module>: at registration this module calls <module>.install(llmt_lora), which calls
# llmt_lora.set_hooks(linear=..., mlp=...). Both hooks run only on LoRA steps, outside torch.compile tracing (eagerly,
# or inside a LoRA cudagraph capture: then they must be graph-safe: no host syncs, no allocations that depend on data).
#   linear(layer, x, out) -> out | None
#       layer: a vLLM BaseLinearLayerWithLoRA (lora_a_stacked / lora_b_stacked tuples [k, 1, r, in] / [k, 1, out, r],
#       output_slices, punica_wrapper); x [n, in], out [n, sum(output_slices)] = the base output; add the LoRA terms
#       of the rows' adapters in place and return out. The rows' slot ids: layer.punica_wrapper._token_lora_indices[:n]
#       (-1 = no adapter; compacted steps hold only LoRA rows, see lora_rows()). None = not handled (punica runs).
#   mlp(module, x) -> tensor | None
#       module: the Qwen2MoeMLP (gate_up_proj / down_proj are LoRA wrappers, .base_layer = the exl3 linears); return the
#       whole MLP output incl. the LoRA terms (e.g. the fused exl3 decode MLP + per-row deltas). None = not handled
#       (the plain MLP through the LoRA wrappers runs).
#   linear_full(layer, x) -> out | None
#       REPLACES base linear + LoRA for that layer on a LoRA step (called instead of the base quant-method apply):
#       return the full output [n, sum(output_slices)] = base + the rows' LoRA terms. Not called in compacted steps
#       (mixed prefill without graphs: base + the compacted linear()/punica path runs there). None = the current path.
_HOOKS = {"linear": None, "mlp": None, "linear_full": None}


def set_hooks(linear=None, mlp=None, linear_full=None):
    _HOOKS.update(linear=linear, mlp=mlp, linear_full=linear_full)
    _log("kernel hooks: linear %s, mlp %s, linear_full %s", *(getattr(f, "__qualname__", f) for f in
                                                               (linear, mlp, linear_full)))


def _full_hook(layer, x):
    hk = _HOOKS["linear_full"]
    if hk is None or _ST["where"] != "device" or _ST.get("compact") is not None or x.dim() != 2:
        return None
    a = layer.lora_a_stacked
    t0 = a[0] if isinstance(a, (tuple, list)) else a
    if t0.device.type == "meta" or not layer.is_lora_active():
        return None
    return hk(layer, x)


def lora_rows():
    """(row index tensor, n) of this step's LoRA rows when the step is compacted, else None (all rows, full mapping)."""
    c = _ST.get("compact")
    return (c[0], c[1]) if c is not None else None


def slot_buffers(layer):
    """The layer's (lora_a_stacked, lora_b_stacked, output_slices); device tensors while slots are resident."""
    return layer.lora_a_stacked, layer.lora_b_stacked, tuple(layer.output_slices)


def _load_kernels():
    name = _ENV("EXL3_LORA_KERNELS", "")
    if not name:
        return
    try:
        import importlib
        importlib.import_module(name).install(__import__(__name__))
    except Exception as e:  # noqa: BLE001
        _log("LoRA kernels %r not loaded (%r): punica path", name, e, level="warning")
BROKER = _ENV("EXL3_LORA_BROKER", "auto")
NS_NAME = _ENV("EXL3_LORA_NS", "weights:lora")
STORE = _ENV("EXL3_LORA_STORE", "1") == "1"
VERIFY_HIT = _ENV("EXL3_LORA_VERIFY_HIT", "0") == "1"
LORA_DIR = _ENV("EXL3_LORA_DIR", "/loras")
TIMEOUT_MS = float(_ENV("EXL3_LORA_TIMEOUT_MS", "100"))
BUSY_WAIT_MS = int(_ENV("EXL3_LORA_BUSY_WAIT_MS", "5000"))
PRESSURE_IDLE_S = 1.0
# broker cost_hint of an adapter entry = what a miss costs: a disk read at EXL3_LORA_DISK_MBPS (100; the root stick
# reads ~34 MB/s cold, the USB SSD ~500) + parse/merge (~750 ms measured for a 223 MiB rank-16 adapter)
DISK_MBPS = float(_ENV("EXL3_LORA_DISK_MBPS", "100"))
PARSE_MS = float(_ENV("EXL3_LORA_PARSE_MS", "1000"))
# compiled (default): LoRA steps run the compiled model without cudagraphs; the LoRA terms are opaque custom ops
# (llmt_lora_delta per LoRA linear, llmt_lora_mlp per MLP) whose Python decides at run time. eager: the whole language
# model runs eagerly on LoRA steps (v1; measured 16 tok/s on the Arc, and every request in such a step runs at that
# speed).
PATH = _ENV("EXL3_LORA_PATH", "compiled")
# LGRAPH (default on with the compiled path): FULL decode graphs WITH the LoRA terms, captured lazily for the decode
# batch shapes LoRA traffic actually hits (a shape is wanted after its first LoRA step ran without a graph), in their
# own graph pool, only while the slots are on the device and the room (same rule as the slots) covers
# EXL3_LORA_GRAPH_EST_MB per graph + margin. Dropped together with the slots (TTL / pressure / host move) and whenever
# the slot contents change (another adapter activated). Non-LoRA graphs are never touched.
LGRAPH = _ENV("EXL3_LORA_GRAPHS", "1") == "1" and PATH == "compiled"
LG_EST = int(float(_ENV("EXL3_LORA_GRAPH_EST_MB", "48")) * 2**20)
# lab lp3: a separate LoRA graph pool keeps every LoRA graph's intermediates private (taken out of the
# allocator's cache; the reserved delta showed 0 MiB), shrinking the prefill headroom: XPU OOM on a 544 MiB MLP
# buffer under 6 x 60k uncached prefills. Default now: the GLOBAL graph pool (vLLM shares one pool across all graphs;
# they never run concurrently). EXL3_LORA_GRAPH_POOL=own: the old separate pool.
LG_OWN_POOL = _ENV("EXL3_LORA_GRAPH_POOL", "global") == "own"
LG_MAX_PER_STEP = int(_ENV("EXL3_LORA_GRAPH_MAX_PER_STEP", "2"))
# margin above the activation reserve for a graph capture (the slots use EXL3_LORA_MARGIN_MB); lab: with the
# slots resident the room was 294 MiB, so 48 + 256 never captured
LG_MARGIN = int(float(_ENV("EXL3_LORA_GRAPH_MARGIN_MB", "64")) * 2**20)
_LG = {"captured": set(), "want": set(), "seen": set(), "pool": None, "bytes": 0, "capturing": None, "n": 0,
       "fail": 0, "dropped": 0}
KIND_WEIGHTS = 5
FORMAT = 0x4C01            # "L" v1: the raw adapter_model.safetensors bytes
_ST2T = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

_LOCK = threading.RLock()
_ST = {"k": 0, "ready": False, "act_reserve": None, "base_alloc": None, "step_lora": False, "last_use": 0.0,
       "where": "off", "bytes": 0, "layers": [], "mgr": None, "worker": None, "dev": None, "broker": None,
       "n": {"activations": 0, "dev_allocs": 0, "host_allocs": 0, "promotions": 0, "releases": 0, "lora_steps": 0,
             "pool_hits": 0, "disk_loads": 0, "stored": 0, "tmp_loads": 0}}


def _log(msg, *a, level="info"):
    try:
        from vllm.logger import init_logger
        getattr(init_logger("vllm.llmt_lora"), level)("LLMT-LORA: " + msg, *a)
    except Exception:  # noqa: BLE001
        print("LLMT-LORA: " + (msg % a), flush=True)


def _acc():
    return torch.xpu if (hasattr(torch, "xpu") and torch.xpu.is_available()) else torch.cuda


def _sync():
    try:
        _acc().synchronize()
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------ slot buffers (VRAM)

_SLOT_ATTRS = ("lora_a_stacked", "lora_b_stacked")


def _tensors_of(layer):
    for a in _SLOT_ATTRS:
        v = getattr(layer, a, None)
        if isinstance(v, torch.Tensor):
            yield a, None, v
        elif isinstance(v, (tuple, list)):
            for i, t in enumerate(v):
                if isinstance(t, torch.Tensor):
                    yield a, i, t


def _layer_bytes(layer):
    return sum(t.numel() * t.element_size() for _, _, t in _tensors_of(layer))


@torch.inference_mode(False)
def _place(layer, device, k=None):
    """Re-create the layer's slot buffers on `device` (meta = freed) with k slot rows (default: as many as now; the
    full N-slot shape on meta). Contents are not kept: the caller re-activates."""
    for a in _SLOT_ATTRS:
        v = getattr(layer, a, None)

        def mk(t):
            rows = t.shape[0] if k is None else k
            return torch.zeros((rows,) + tuple(t.shape[1:]), dtype=t.dtype, device=device)
        if isinstance(v, torch.Tensor):
            setattr(layer, a, mk(v))
        elif isinstance(v, (tuple, list)):
            setattr(layer, a, type(v)(mk(t) for t in v))


@torch.inference_mode(False)
def _resize(layer, k):
    """k slot rows on the same device, keeping the contents of the first min(old, k) rows."""
    for a in _SLOT_ATTRS:
        v = getattr(layer, a, None)

        def rs(t):
            n = torch.zeros((k,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
            m = min(k, t.shape[0])
            n[:m].copy_(t[:m])
            return n
        if isinstance(v, torch.Tensor):
            setattr(layer, a, rs(v))
        elif isinstance(v, (tuple, list)):
            setattr(layer, a, type(v)(rs(t) for t in v))


def _move(layer, device):
    """Move the layer's slot buffers with their contents (host -> device promotion)."""
    for a in _SLOT_ATTRS:
        v = getattr(layer, a, None)
        if isinstance(v, torch.Tensor):
            setattr(layer, a, v.to(device))
        elif isinstance(v, (tuple, list)):
            setattr(layer, a, type(v)(t.to(device) for t in v))


def _cap_bytes():
    """The torch allocator's effective cap. With the weight keeper (whx_loader capfix) the held weights live outside
    torch and the cap is TORCH_CAP_MIB - held (LLMT_WHX_HELD_MIB) -- without this the tower and LoRA slots
    were placed in the activation reserve (device free 1.6 GiB, cap room 0.6 GiB) and a 60k prefill OOMed."""
    cap = int(os.environ.get("TORCH_CAP_MIB", "0") or 0)
    if not cap:
        return None
    held = int(os.environ.get("LLMT_WHX_HELD_MIB", "0") or 0)
    return max(1024, cap - held) << 20


def _cap_clamp(free):
    """Device free memory clamped to what the torch cap still allows."""
    cap = _cap_bytes()
    if cap is None:
        return free
    try:
        return max(0, min(free, cap - _acc().memory_reserved(_ST["dev"])))
    except Exception:  # noqa: BLE001
        return free


def _lazyvis_resident_bytes():
    n = 0
    for modname in ("llmt_kv.lazyvis", "exl3xpu.lazyvis", "llmt_lazyvis"):
        import sys
        m = sys.modules.get(modname)
        for lv in (getattr(m, "_STATE", {}) or {}).get("towers", []) if m else []:
            if getattr(lv, "resident", False):
                n += int(getattr(lv, "bytes", 0) or getattr(lv, "entry_bytes", 0) or 0)
    return n


def _room():
    """Device memory left after the engine's activation reserve: free on the device (within the torch allocator cap),
    cached-unused in torch, plus activations in flight now (they come out of the reserve). Same rule as the lazy
    vision tower; our own device buffers and a resident tower count as persistent."""
    acc, dev = _acc(), _ST["dev"]
    free, _tot = acc.mem_get_info(dev)
    alloc = acc.memory_allocated(dev)
    reserved = acc.memory_reserved(dev)
    cap = _cap_bytes()
    if cap is not None:
        free = max(0, min(free, cap - reserved))
    transient = 0
    if _ST["base_alloc"] is not None:
        mine = (_ST["bytes"] if _ST["where"] == "device" and not _ST.get("lent") else 0) + _LG["bytes"]
        transient = max(0, alloc - _ST["base_alloc"] - mine - _lazyvis_resident_bytes())
    return free + (reserved - alloc) + transient - (_ST["act_reserve"] or 0)


def _lazyvis_base_adjust(delta):
    """The lazy vision tower counts allocations above its idle snapshot as transient activations: tell it about ours."""
    import sys
    for modname in ("llmt_kv.lazyvis", "exl3xpu.lazyvis", "llmt_lazyvis"):
        m = sys.modules.get(modname)
        st = getattr(m, "_STATE", None) if m else None
        if isinstance(st, dict) and st.get("base_alloc") is not None:
            st["base_alloc"] += delta


def _max_slots():
    """N = the slot rows the layers were created with (meta tensors carry the full --max-loras shape)."""
    if not _ST.get("max_slots"):
        for l in _ST["layers"]:
            for _, _, t in _tensors_of(l):
                if t.device.type == "meta":
                    _ST["max_slots"] = int(t.shape[0])
                    break
            if _ST.get("max_slots"):
                break
    return _ST.get("max_slots") or 1


def _slot_bytes():
    """Bytes of ONE slot row over all LoRA layers (fix: computed from the per-row size, not from a slot
    count that may not be known yet)."""
    if not _ST.get("slot_bytes"):
        b = 0
        for l in _ST["layers"]:
            for _, _, t in _tensors_of(l):
                b += (t.numel() // max(1, int(t.shape[0]))) * t.element_size()
        _ST["slot_bytes"] = b
    return _ST["slot_bytes"]


# ------------------------------------------------------------------------------------------ KV-block lending
# LoRA slots are dynamic like the vision tower, and KV is unloaded only while a LoRA needs the room: when the room does not
# cover a slot, the scheduler takes whole KV blocks out of its free pool (BlockPool.get_new_blocks: coldest first,
# cached prefix blocks are evicted from the GPU as on any allocation; the RAM tier keeps their copies) and the slot
# buffers become views into those blocks' pages. The blocks go back (BlockPool.free_blocks) when all slots are idle.
# KV capacity is reduced only while adapters are resident. Running requests' blocks are never touched; a reserve of
# free blocks (EXL3_LORA_LEND_KEEP_FRAC of the pool, >= 2) stays for them; without enough free blocks the LoRA request
# waits. Needs scheduler and worker in one process (UniProcExecutor) and a blocks-first KV layout (hybrid models).
LEND = _ENV("EXL3_LORA_LEND", "1") == "1"
LEND_KEEP_FRAC = float(_ENV("EXL3_LORA_LEND_KEEP_FRAC", "0.10"))
LEND_PENDING_S = float(_ENV("EXL3_LORA_LEND_PENDING_S", "30"))
_ALIGN_B = 512
_KVL = {"ok": False, "why": "KV cache not initialized", "num_blocks": 0, "stores": [], "per_block": 0}


def _kv_layout_init(kv_caches, num_blocks):
    """Map block id -> byte ranges: every KV view must be blocks-first with one page of P bytes per block on its
    storage (P = storage bytes / num_blocks; kernel blocks per block folded in). Any other layout: lending off."""
    stores = {}
    try:
        for name, v in kv_caches.items():
            for t in (v if isinstance(v, (list, tuple)) else [v]):
                st = t.untyped_storage()
                nbytes = st.nbytes()
                if num_blocks <= 0 or nbytes % num_blocks or t.dim() == 0 or t.shape[0] % num_blocks:
                    raise ValueError("%s: storage %d B / %d blocks, dim0 %s" % (name, nbytes, num_blocks,
                                                                                tuple(t.shape[:1])))
                P = nbytes // num_blocks
                kf = t.shape[0] // num_blocks
                if t.stride(0) * t.element_size() * kf != P:
                    raise ValueError("%s: not blocks-first (stride0 %d B x %d, page %d B)" % (
                        name, t.stride(0) * t.element_size(), kf, P))
                stores.setdefault(st.data_ptr(), (st, P, t.device))
    except Exception as e:  # noqa: BLE001
        _KVL.update(ok=False, why=str(e))
        _log("KV-block lending off: %s", e, level="warning")
        return
    us = []
    for ptr, (st, P, dev) in stores.items():
        u8 = torch.empty(0, dtype=torch.uint8, device=dev)
        u8.set_(st, 0, (st.nbytes(),), (1,))
        us.append((u8, P))
    _KVL.update(ok=True, why="", num_blocks=num_blocks, stores=us, per_block=sum(P for _, P in us))
    _log("KV-block lending ready: %d blocks, %d KV storages, %.1f MiB per block (a LoRA slot row = %.2f blocks)",
         num_blocks, len(us), _KVL["per_block"] / 2**20, _slot_bytes() / max(1, _KVL["per_block"]))


def _slot_specs(rows):
    out = []
    for l in _ST["layers"]:
        for a, i, t in _tensors_of(l):
            row = t.numel() // max(1, t.shape[0])
            out.append((l, a, i, (rows,) + tuple(t.shape[1:]), t.dtype, rows * row * t.element_size()))
    return out


def _pack(specs, nblocks):
    """First-fit-decreasing of the slot tensors into nblocks x (one page per KV storage). [(spec, j, s, off)] or None."""
    ranges = [[j, s, 0, P] for j in range(nblocks) for s, (_, P) in enumerate(_KVL["stores"])]
    plan = []
    for sp in sorted(specs, key=lambda x: -x[5]):
        nb = sp[5]
        for r in ranges:
            off = (r[2] + _ALIGN_B - 1) // _ALIGN_B * _ALIGN_B
            if off + nb <= r[3]:
                plan.append((sp, r[0], r[1], off))
                r[2] = off + nb
                break
        else:
            return None
    return plan


def _blocks_for(rows):
    specs = _slot_specs(rows)
    n = max(1, -(-sum(x[5] for x in specs) // max(1, _KVL["per_block"])))
    while n <= _KVL["num_blocks"]:
        if _pack(specs, n) is not None:
            return n
        n += 1
    return None


@torch.inference_mode(False)
def _slot_views(block_ids, rows):
    """Slot tensors (rows slot rows) as zeroed views into the lent blocks: {(layer, attr): [tensors by slice]}."""
    plan = _pack(_slot_specs(rows), len(block_ids))
    if plan is None:
        raise RuntimeError("LoRA slots do not pack into %d lent blocks" % len(block_ids))
    views = {}
    for (l, a, i, shape, dtype, nb), j, si, off in plan:
        u8, P = _KVL["stores"][si]
        base = block_ids[j] * P + off
        t = u8[base:base + nb].view(dtype).view(shape)
        t.zero_()
        views.setdefault((id(l), a), {})[i] = t
    return views


@torch.inference_mode(False)
def _install_views(views, copy_from_rows=0):
    """Point the layers' slot attributes at the views (copying the first copy_from_rows rows of the old buffers)."""
    for l in _ST["layers"]:
        for a in _SLOT_ATTRS:
            v = getattr(l, a, None)
            nv = views.get((id(l), a))
            if nv is None:
                continue
            if isinstance(v, torch.Tensor):
                n = nv[None]
                if copy_from_rows and v.device.type != "meta":
                    n[:copy_from_rows].copy_(v[:copy_from_rows])
                setattr(l, a, n)
            else:
                lst = []
                for i, t in enumerate(v):
                    n = nv[i]
                    if copy_from_rows and t.device.type != "meta":
                        m = min(copy_from_rows, t.shape[0], n.shape[0])
                        n[:m].copy_(t[:m])
                    lst.append(n)
                setattr(l, a, type(v)(lst))


def _use_lent(rows_needed):
    """With _LOCK held (worker): move the slots into the pending lent blocks if they cover rows_needed. True if done."""
    pend = _ST.get("lent_pending")
    if not pend or pend[1] < rows_needed:
        return False
    blocks, rows, _t = pend
    ids = [b.block_id for b in blocks]
    t0 = time.perf_counter()
    cur = _ST["k"] if _ST["where"] == "device" else 0
    old_lent, was_room = _ST.get("lent"), (_ST["where"] == "device" and not _ST.get("lent"))
    _lg_drop("slot buffers move into %d lent KV blocks" % len(ids))
    views = _slot_views(ids, rows)
    _install_views(views, copy_from_rows=cur)
    _sync()
    if was_room:
        _lazyvis_base_adjust(-_ST["bytes"])
        try:
            _acc().empty_cache()
        except Exception:  # noqa: BLE001
            pass
    if old_lent:
        _unlend("lora", old_lent)
    _ST.update(lent=blocks, lent_pending=None, where="device", k=rows, bytes=rows * _slot_bytes(), cap_cache=None)
    _ST["n"]["lends"] = _ST["n"].get("lends", 0) + 1
    _log("LoRA slots: %d rows in %d lent KV blocks (%.0f KV tokens' worth while resident), %.1f ms", rows, len(ids),
         len(ids) * _ST.get("block_tokens", 0), (time.perf_counter() - t0) * 1e3)
    return True


def _sched_queues(sched):
    """Every queue that can hold a request for this engine: running, waiting and (vLLM 0.26) skipped_waiting."""
    qs = [getattr(sched, "running", None), getattr(sched, "waiting", None), getattr(sched, "skipped_waiting", None)]
    return [q for q in qs if q is not None]


LORA_MLP_ROWS = int(_ENV("EXL3_LORA_MLP_ROWS", "2048"))


def _mlp_mapping_tensors():
    mgr = _ST.get("mgr")
    pw = getattr(mgr, "punica_wrapper", None) or next(iter(getattr(mgr, "punica_wrapper_mapping", {}).values()), None)
    if pw is None:
        return None
    ts = [pw._token_lora_indices]
    tm = getattr(getattr(pw, "token_mapping_meta", None), "token_lora_mapping", None)
    if tm is not None:
        ts.append(tm)
    return ts


def _mlp_split_plan(lr, T):
    """Per step (cached on the lrows tuple): device row ids of the adapter rows, their slot ids, the base rows."""
    c = _ST.get("msplit")
    if c is not None and c[0] is lr:
        return c[1]
    plan = None
    try:
        import numpy as np
        rows, ids, total = lr
        mgr, ts = _ST.get("mgr"), _mlp_mapping_tensors()
        n = len(rows)
        if mgr is not None and ts is not None and total == T and n > 0:
            slot = {lid: j for j, lid in enumerate(mgr.lora_index_to_id) if lid is not None}
            sl = np.fromiter((slot[i] for i in ids), dtype=np.int64, count=n)
            dev = _ST["dev"]
            idx = torch.from_numpy(rows.astype(np.int64)).to(dev)
            base = None if n == T else torch.from_numpy(np.setdiff1d(np.arange(T), rows).astype(np.int64)).to(dev)
            plan = (idx, torch.from_numpy(sl).to(dev), base, ts)
    except Exception:  # noqa: BLE001  (unknown adapter id etc.: old all-plain path)
        plan = None
    _ST["msplit"] = (lr, plan)
    return plan


def _mlp_split(m, x, plain, fused, rows_cap, plan):
    """Row split + chunked plain MLP; see item 9. plain/fused take (m, x)."""
    idx, s64, base, ts = plan
    T, n = x.shape[0], idx.shape[0]
    saved = [t[:T].clone() for t in ts]
    comp = _ST.get("compact")
    _ST["compact"] = None           # _eager must not re-gather: x here is already the adapter rows
    out = None
    try:
        if base is not None:
            yb = fused(m, x.index_select(0, base))
            out = x.new_empty((T, yb.shape[-1]))
            out.index_copy_(0, base, yb)
            del yb
            xa = x.index_select(0, idx)
        else:
            xa = x
        for s in range(0, n, rows_cap):
            e = min(n, s + rows_cap)
            for t in ts:
                t[:e - s].copy_(s64[s:e].to(t.dtype))
            y = plain(m, xa[s:e])
            if out is None:
                out = x.new_empty((T, y.shape[-1]))
            if base is None:
                out[s:e] = y
            else:
                out.index_copy_(0, idx[s:e], y)
        _ST["n"]["mlp_split"] = _ST["n"].get("mlp_split", 0) + 1
        _ST["mlp_split_rows"] = max(_ST.get("mlp_split_rows", 0), min(n, rows_cap))
        if _ST["n"]["mlp_split"] in (1, 100) or _ST["n"]["mlp_split"] % 2000 == 0:
            _log("mlp_split %d MLP calls, max plain rows %d (cap %d), adapter rows in last call %d of %d",
                 _ST["n"]["mlp_split"], _ST["mlp_split_rows"], rows_cap, n, T)
    finally:
        for t, sv in zip(ts, saved):
            t[:T].copy_(sv)
        _ST["compact"] = comp
    return out


def _lend_tick(sched):
    """Scheduler thread, with _LOCK held, before each schedule(): give back blocks the worker released; lend blocks
    for the adapters the queue needs when the room cannot hold them."""
    pool = getattr(getattr(sched, "kv_cache_manager", None), "block_pool", None)
    if pool is None:
        return
    rt = _ST.get("to_return") or []
    while rt:
        blocks = rt.pop()
        pool.free_blocks(blocks)
        _ST["n"]["lend_returns"] = _ST["n"].get("lend_returns", 0) + 1
        _log("%d lent KV blocks back to the pool (free now %d)", len(blocks), pool.get_num_free_blocks())
    try:
        _patch_vision()
        _vis_tick(sched, pool)
    except Exception as e:  # noqa: BLE001
        _log("vision lend: %r", e, level="warning")
    if not (LEND and _KVL["ok"] and _ST["ready"] and MODE == "auto"):
        return
    pend = _ST.get("lent_pending")
    if pend:
        if time.monotonic() - pend[2] > LEND_PENDING_S:
            wait_ids = set()
            for q in _sched_queues(sched):
                try:
                    for r in q:
                        lr = getattr(r, "lora_request", None)
                        if lr and lr.lora_int_id > 0:
                            wait_ids.add(lr.lora_int_id)
                except Exception:  # noqa: BLE001
                    pass
            cur = _ST["k"] if _ST["where"] == "device" else 0
            if min(_max_slots(), len(wait_ids)) > cur:      # still needed: keep it, look again later
                _ST["lent_pending"] = (pend[0], pend[1], time.monotonic())
                _ST["n"]["lend_pending_kept"] = _ST["n"].get("lend_pending_kept", 0) + 1
                return
            _unlend("lora", pend[0])                         # nobody used it (request gone): give it back
            _ST["lent_pending"] = None
            _log("pending lent KV blocks unused for %.0f s: back to the pool", LEND_PENDING_S)
        return
    ids = set()
    for q in _sched_queues(sched):
        try:
            for r in q:
                lr = getattr(r, "lora_request", None)
                if lr and lr.lora_int_id > 0:
                    ids.add(lr.lora_int_id)
        except Exception:  # noqa: BLE001
            pass
    N = _max_slots()
    demand = min(N, len(ids))
    cur = _ST["k"] if _ST["where"] == "device" else 0
    if demand <= cur:
        return
    if not _ST.get("lent"):
        room_rows = max(0, int((_room() - MARGIN) // max(1, _slot_bytes())))
        if cur + room_rows >= demand:
            return                       # the room covers it: no KV taken
    keep = _keep_blocks()
    for target in range(demand, cur, -1):
        nb = _blocks_for(target)
        if nb is None:
            continue
        if pool.get_num_free_blocks() - keep >= nb:
            blocks = _lend(pool, "lora", nb, "LoRA slot rows")
            if blocks is None:
                break
            _ST["lent_pending"] = (blocks, target, time.monotonic())
            _ST["cap_cache"] = None
            _log("lending %d KV blocks (free %d, keep %d) for %d LoRA slot rows", nb, pool.get_num_free_blocks(),
                 keep, target)
            return
    now = time.monotonic()
    if now - _ST.get("lend_log", 0) > 30:
        _ST["lend_log"] = now
        _log("no KV blocks to lend for another LoRA slot (free %d, keep %d, a slot row needs %s): LoRA requests wait",
             pool.get_num_free_blocks(), keep, _blocks_for(cur + 1))


# ------------------------------------------------------------------------------------------ the lend manager
# Use up to 100% of the GPU for KV at all times and evict KV only when another use needs the memory: one manager owns every KV lend. Tenants (LoRA slot rows, the vision tower, later encoder working
# memory) register a size, ask for blocks (any blocks, or a run of consecutive block ids when they
# need contiguous spans), and give them back after their idle TTL. Same coldest-first selection (the pool's free queue
# = LRU: evicts cached prefix blocks, RAM tier keeps them), same reserve floor (EXL3_LORA_LEND_KEEP_FRAC of the pool),
# same return path (queued, freed by the scheduler before its next schedule()), same log lines and counters.
_TENANTS = {}     # name -> {"blocks": [...], "kind": str, "bytes": int, "since": t}


def _keep_blocks():
    return max(2, -(-int(LEND_KEEP_FRAC * max(1, _KVL["num_blocks"]) * 1000) // 1000))



# ---- lend fences (lend-only deploy 11-15; ported from the 11-14 _move_ctx / _patch_load_fence, review V1). A lend
# takes ref_cnt-0 blocks (free or cached-evictable). Some of those are still the target or source of a KV-connector
# transfer that vLLM no longer counts as a reference:
#   - stores whose jobs are built (_jobs / _block_id_to_pending_jobs): a finished request's blocks are freed at once
#     (request_finished returns False) while the paced KV-RING store still reads them;
#   - stores not yet built: the connector builds a finished request's last chunk stores at the NEXT
#     build_connector_meta from the ids remembered in group_states (review V1). A lend in that gap would make the
#     store save tower / LoRA bytes under a valid KV key, and the RAM tier would later serve them as KV;
#   - in-flight loads: once build_connector_meta hands a load job to the worker, the connector forgets its GPU
#     destination ids, while the KV-RING may copy into them for several steps (aborted / preempted requests).
# Those ids are never lent. If the connector state cannot be read, nothing is lent that tick.
_LOADS = {}               # load job id -> its GPU destination block ids (captured at build_connector_meta)


def _patch_load_fence():
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as osch
    except Exception as e:  # noqa: BLE001
        _log("load fence: offloading scheduler not found (%r)", e, level="warning")
        return
    cls = getattr(osch, "OffloadingConnectorScheduler", None)
    if cls is None or getattr(cls, "_llmt_load_fence", False):
        return
    orig = cls.build_connector_meta

    def build_connector_meta(self, *a, **k):
        try:
            for jid, tj in list((getattr(self, "_current_batch_load_jobs", None) or {}).items()):
                ids = {int(x) for x in getattr(getattr(tj, "dst_spec", None), "block_ids", ())}
                if ids:
                    _LOADS[jid] = ids
                    _ST["n"]["fence_load_jobs"] = _ST["n"].get("fence_load_jobs", 0) + 1
        except Exception as e:  # noqa: BLE001
            _ST["n"]["fence_load_errors"] = _ST["n"].get("fence_load_errors", 0) + 1
            if _ST["n"]["fence_load_errors"] == 1:
                _log("load fence: %r", e, level="warning")
        return orig(self, *a, **k)
    cls.build_connector_meta = build_connector_meta
    cls._llmt_load_fence = True
    _log("load fence: on (in-flight load destinations are never lent until the job completes)")


def _lend_fence(sched=None):
    """(block ids that must not be lent now, state readable)."""
    sched = sched or _ST.get("sched")
    blocked, ok = set(), True
    c = getattr(sched, "connector", None)
    conns = ([c] + list(getattr(c, "_connectors", None) or [])) if c is not None else []
    for c in conns:
        cs = getattr(c, "connector_scheduler", None)
        if cs is None:
            continue
        try:
            blocked |= set((getattr(cs, "_block_id_to_pending_jobs", None) or {}).keys())
            jobs = getattr(cs, "_jobs", None) or {}
            for js in jobs.values():
                for lst in (getattr(js, "non_sliding_window_block_ids", None),
                            getattr(js, "sliding_window_block_ids", None)):
                    blocked |= set(lst or ())
            for js in (getattr(cs, "_current_batch_load_jobs", None) or {}).values():
                for attr in ("block_ids", "dst_block_ids"):
                    blocked |= set(getattr(js, attr, None) or ())
                try:
                    blocked |= {int(x) for x in getattr(js.dst_spec, "block_ids", ())}
                except Exception:  # noqa: BLE001
                    pass
            for jid in [j for j in _LOADS if j not in jobs]:
                del _LOADS[jid]
            for ids in _LOADS.values():
                blocked |= ids
            nf = 0
            bpc = int(getattr(getattr(cs, "config", None), "blocks_per_chunk", 0) or 0)
            for rs in list((getattr(cs, "_req_status", None) or {}).values()):
                rq = getattr(rs, "req", None)
                fin = getattr(rq, "is_finished", None)
                if rq is None or not callable(fin) or not fin():
                    continue
                for gs in getattr(rs, "group_states", ()) or ():
                    st0 = int(getattr(gs, "next_stored_chunk_idx", 0) or 0) * bpc if bpc else 0
                    ids = [b for b in (getattr(gs, "block_ids", None) or ())[st0:] if b]
                    blocked.update(ids)
                    nf += len(ids)
            if nf:
                _ST["n"]["fence_finished_calls"] = _ST["n"].get("fence_finished_calls", 0) + 1
                _ST["n"]["fence_finished_blocks_max"] = max(_ST["n"].get("fence_finished_blocks_max", 0), nf)
        except Exception as e:  # noqa: BLE001
            ok = False
            _ST["n"]["fence_errors"] = _ST["n"].get("fence_errors", 0) + 1
            if _ST["n"]["fence_errors"] == 1:
                _log("lend fence: connector state unreadable (%r): nothing is lent while it is", e, level="warning")
    if blocked:
        _ST["n"]["fence_blocked_max"] = max(_ST["n"].get("fence_blocked_max", 0), len(blocked))
    return blocked, ok


def _lend(pool, name, n, kind, contiguous=False):
    """Scheduler thread, _LOCK held: n blocks for tenant `name` (or None). contiguous: consecutive block ids, so every
    KV storage gives one span of n pages (tenants whose tensors are bigger than one page)."""
    keep = _keep_blocks()
    if pool.get_num_free_blocks() - keep < n:
        return None
    blocked, ok = _lend_fence()
    if not ok:
        return None
    if not contiguous:
        if blocked:
            cand = [b for b in pool.blocks if b.ref_cnt == 0 and not getattr(b, "is_null", False)
                    and b.block_id not in blocked][:n]
            if len(cand) < n:
                return None
            for b in cand:
                pool.free_block_queue.remove(b)
                if getattr(pool, "enable_caching", False):
                    pool._maybe_evict_cached_block(b)
                b.ref_cnt += 1
            blocks = cand
            _ST["n"]["lends_fenced"] = _ST["n"].get("lends_fenced", 0) + 1
        else:
            blocks = pool.get_new_blocks(n)
    else:
        if LEND_COMPACT:                 # chain-aware: prefer free uncached runs, move/drop by value otherwise
            top = PROF and os.path.exists(os.path.join(PROF, "compact_top"))   # lab: force the top region
            st = _compact_for_run(pool, n, at="top" if top else None)
            blocks = _take_run(pool, n, start=st) if st is not None else None
        else:
            blocks = _take_run(pool, n, blocked=blocked)
        if blocks is None:
            return None
    t = _TENANTS.setdefault(name, {"blocks": [], "kind": kind, "bytes": 0, "since": time.time()})
    t["blocks"] = t["blocks"] + blocks
    _ST["n"]["lends"] = _ST["n"].get("lends", 0) + 1
    _log("lend: %d KV blocks to %s (%s%s; free %d, keep %d)", n, name, kind, ", consecutive" if contiguous else "",
         pool.get_num_free_blocks(), keep)
    return blocks


def _take_run(pool, n, start=None, blocked=()):
    """n consecutive free block ids (ref_cnt 0, i.e. in the free queue: free or cached-evictable), coldest run first
    by the free-queue position of its first block. None if no such run."""
    blocks = getattr(pool, "blocks", None)
    if not blocks:
        return None
    _allow, moving = _move_ctx()
    blocked = set(blocked) | moving
    ok = [(not getattr(b, "is_null", False)) and b.ref_cnt == 0 and b.block_id not in blocked for b in blocks]
    best = start
    if start is not None and not all(ok[start:start + n]):
        return None
    run = 0
    for i, f in enumerate(ok if best is None else ()):
        run = run + 1 if f else 0
        if run >= n:
            best = i - n + 1
            break               # first run found = lowest ids; good enough (all candidates are free / evictable)
    if best is None:
        return None
    out = []
    for b in blocks[best:best + n]:
        pool.free_block_queue.remove(b)
        if getattr(pool, "enable_caching", False):
            pool._maybe_evict_cached_block(b)
        b.ref_cnt += 1
        out.append(b)
    return out


# ---- prefix chains (a high-value chain should stay in VRAM). Prefix-cache blocks
# form chains: block k of a prompt is only reachable through blocks 0..k-1. vLLM keeps no parent links, so we record
# them when blocks get cached (BlockPool.cache_full_blocks: parent = the request's previous block hash), keyed by the
# group-independent block hash (a hybrid model caches one block per group per position; a hit needs all of them).
# Per node: parent, class (router tag llmt_cls via kv_transfer_params), session, hits (BlockPool.touch), last use.
_CH = {}                 # raw block hash -> [parent raw | None, class weight, session, hits, last use (monotonic)]
_CLS_W = {"agent": 4.0, "normal": 1.0, "": 1.0, "low": 0.3, "lowest": 0.1}
CH_TAU = float(_ENV("EXL3_LEND_CHAIN_TAU_S", "300"))      # recency half-weight age
STORE_COST = 0.5        # per block whose RAM-tier copy is not known to exist (a PCIe store, or a re-prefill)
MOVE_COST = 0.02         # one block copy (~ms) vs dropping a block of value ~1 (a re-prefill or a RAM-tier load)


def _rawh(key):
    if key is None:
        return None
    if isinstance(key, tuple):           # CPU tests
        return key[0]
    try:
        from vllm.v1.core.kv_cache_utils import get_block_hash
        return get_block_hash(key)
    except Exception:  # noqa: BLE001
        return key


def _req_tags(request):
    p = getattr(request, "kv_transfer_params", None) or {}
    cls = str(p.get("llmt_cls") or "") if isinstance(p, dict) else ""
    sess = (p.get("llmt_sess") if isinstance(p, dict) else None) or None
    return _CLS_W.get(cls, 1.0), sess


# Session liveness beyond running requests (blocks may belong to an agent waiting on a tool
# result). _SESS[sess] = [last seen running (monotonic), awaiting (bool)]. A session counts as awaiting when its last
# request carried llmt_await=1 (router: the previous response finished with tool_calls) or was not the session's
# terminal request (llmt_term != 1). after tool_calls the next request came within 2 min in
# 97.8 % (p50 0.3 s, p90 3.4 s), within 10 min in 98.9 %. Boost: x3 up to AWAIT_FULL_S after the session left the
# running set, linear down to x1 at AWAIT_END_S.
_SESS = {}
AWAIT_FULL_S = float(_ENV("EXL3_LEND_AWAIT_FULL_S", "120"))
AWAIT_END_S = float(_ENV("EXL3_LEND_AWAIT_END_S", "600"))


def _sess_note(sched):
    now = time.monotonic()
    for rq in (getattr(sched, "running", None) or []):
        _w, sess = _req_tags(rq)
        if not sess:
            continue
        p = getattr(rq, "kv_transfer_params", None) or {}
        aw = isinstance(p, dict) and (str(p.get("llmt_await") or "") == "1" or str(p.get("llmt_term") or "") != "1")
        _SESS[sess] = [now, aw]
    if len(_SESS) > 20000:
        for k in [k for k, v in _SESS.items() if now - v[0] > AWAIT_END_S]:
            del _SESS[k]


def _sess_boost(sess, live, now):
    if not sess:
        return 1.0
    if sess in live:
        return 3.0
    e = _SESS.get(sess)
    if not e or not e[1]:
        return 1.0
    age = now - e[0]
    if age <= AWAIT_FULL_S:
        return 3.0
    if age >= AWAIT_END_S:
        return 1.0
    return 3.0 - 2.0 * (age - AWAIT_FULL_S) / (AWAIT_END_S - AWAIT_FULL_S)


def _patch_block_pool():
    from llmt_kv import vllm_hooks as H

    def cache_full_blocks(pool, request, num_cached_blocks, num_full_blocks):
        try:
            _chain_note(pool, request, num_cached_blocks, num_full_blocks)
        except Exception as e:  # noqa: BLE001
            _log("chain note: %r", e, level="warning")

    def get_cached_block(pool, block_hash):
        if _REGRET:
            try:
                _regret_check(block_hash)
            except Exception:  # noqa: BLE001
                pass

    def touch(pool, blocks):
        try:
            now = time.monotonic()
            for b in blocks:
                m = _CH.get(_rawh(getattr(b, "block_hash", None)))
                if m is not None:
                    m[3] += 1
                    m[4] = now
        except Exception:  # noqa: BLE001
            pass

    try:
        H.register("block_pool.cache_full_blocks", cache_full_blocks)
        H.register("block_pool.get_cached_block", get_cached_block)
        H.register("block_pool.touch", touch)
        H.track_lent_blocks()                      # KV usage of the pool without the lent blocks
    except Exception as e:  # noqa: BLE001
        _log("block pool hooks not installed (%r): compaction runs without chain values", e, level="warning")


def _chain_note(pool, request, lo, hi):
    hs = getattr(request, "block_hashes", None) or []
    w, sess = _req_tags(request)
    now = time.monotonic()
    for i in range(lo, min(hi, len(hs))):
        m = _CH.get(hs[i])
        if m is None:
            _CH[hs[i]] = [hs[i - 1] if i > 0 else None, w, sess, 0, now]
        else:
            m[1] = max(m[1], w)
            m[4] = now
    if len(_CH) > 8 * len(getattr(pool, "blocks", ()) or ()) + 4096:
        _chain_gc(pool)


def _chain_gc(pool):
    live = {_rawh(b.block_hash) for b in (getattr(pool, "blocks", None) or []) if getattr(b, "block_hash", None)}
    for h in [h for h in _CH if h not in live]:
        del _CH[h]


def _chain_view(pool):
    """Snapshot of the cached chains: node -> its cached blocks, node -> children, node value, node -> in use."""
    nodes, inuse = {}, {}
    hs = getattr(pool, "cached_block_hashes_by_block", None) or {}
    for b in pool.blocks:
        if getattr(b, "is_null", False):
            continue
        keys = ([b.block_hash] if getattr(b, "block_hash", None) is not None else []) + list(hs.get(b.block_id, ()))
        for key in keys:
            r = _rawh(key)
            nodes.setdefault(r, []).append(b)
            inuse[r] = inuse.get(r, False) or b.ref_cnt > 0
    kids = {}
    for r in nodes:
        m = _CH.get(r)
        if m and m[0] in nodes:
            kids.setdefault(m[0], []).append(r)
    live = set()
    sch = _ST.get("sched")
    for rq in (getattr(sch, "running", None) or []):
        s = _req_tags(rq)[1]
        if s:
            live.add(s)
    now = time.monotonic()
    val = {}
    for r in nodes:
        m = _CH.get(r)
        if m is None:
            val[r] = 1.0
            continue
        v = m[1] * (1.0 + math.log1p(m[3])) / (1.0 + (now - m[4]) / CH_TAU)
        val[r] = v * _sess_boost(m[2], live, now)
    return nodes, kids, val, inuse


def _subtree(r, kids):
    out, st = [], [r]
    while st:
        x = st.pop()
        out.append(x)
        st.extend(kids.get(x, ()))
    return out


def _is_cached(b, hs):
    return getattr(b, "block_hash", None) is not None or b.block_id in hs


# ---- block compaction (swap the physical locations of pages and update the block tables)
# When no run of n consecutive UNCACHED free ids exists, choose the window of n ids with the lowest cost and clear it:
#  - in-use blocks in the window are MOVED (always: copy every KV storage's page -- attention KV and GDN state -- to a
#    free uncached block outside the window, then swap the two KVCacheBlock objects' identities);
#  - cached-unreferenced blocks in the window are MOVED the same way while free uncached blocks remain outside;
#  - only when that runs out are cached chains DROPPED (evicted as vLLM's LRU would; the RAM tier keeps what it stored),
#    whole chain suffixes, leaf-first, lowest value first: dropping a node drops its cached descendants with it, so
#    nothing is orphaned; a node with an in-use descendant is never dropped. Dropped blocks outside the window go to
#    the free-queue head (reused first).
# Window cost = MOVE_COST x moves + sum of dropped nodes' values (value = class weight x (1 + ln(1 + hits)) /
# (1 + age / tau), x3 for a session with a running request). at="top" fixes the window to the last n ids (the
# movable-boundary design). Identity swap: requests' block lists, the free queue and hash-map values hold the objects
# and follow; id-keyed state is remapped (cached_block_hashes_by_block, the hash map's {id: block} dicts, worker
# block-table rows, the runner's request block_ids). Between steps only (device sync, scheduler thread, _LOCK).
# Default OFF until lab-validated: EXL3_LEND_COMPACT=1.
LEND_COMPACT = _ENV("EXL3_LEND_COMPACT", "0") == "1"


# Regret: chain nodes dropped by compaction that a lookup asks for again within
# 1 / 5 / 15 min (BlockPool.get_cached_block hook). Counters regret_1m / regret_5m / regret_15m, dropped total.
_REGRET = {}


def _regret_check(raw):
    t = _REGRET.pop(raw, None)
    if t is None:
        return
    age = time.monotonic() - t
    c = _ST["n"]
    for lim, key in ((60, "regret_1m"), (300, "regret_5m"), (900, "regret_15m")):
        if age <= lim:
            c[key] = c.get(key, 0) + 1
    if len(_REGRET) > 200000:
        old = time.monotonic() - 900
        for k in [k for k, v in _REGRET.items() if v < old]:
            del _REGRET[k]


def _plan_window(pool, st, n, view, immov, hs):
    """(cost, drop nodes, movers) for clearing ids st..st+n-1, or None if impossible."""
    nodes, kids, val, inuse = view
    blocks = pool.blocks
    W = range(st, st + n)
    if any(immov[i] for i in W):
        return None
    U = [blocks[i] for i in W if blocks[i].ref_cnt > 0]
    if U and not _ST.get("move_allow", True):
        return None                          # in-use moves deferred (see _move_ctx); drop-only regions still go
    C = [blocks[i] for i in W if blocks[i].ref_cnt == 0 and _is_cached(blocks[i], hs)]
    fo = sum(1 for b in blocks if b.ref_cnt == 0 and not immov[b.block_id] and not (st <= b.block_id < st + n)
             and not _is_cached(b, hs))
    deficit = len(U) + len(C) - fo
    # Destinations beyond the free blocks: victims, i.e. chain TAILS (a node none of whose cached descendants stays),
    # lowest value per block first, re-evaluated after each pick (a low-value chain is consumed tail-first). A victim
    # whose RAM-tier copy is unknown costs extra (STORE_COST; store-first is not done here -- the offload connector
    # writes blocks through as they are computed). In-use and lent blocks are never victims; a victim inside the
    # window simply stops needing a move.
    ram_has = _ST.get("ram_has")
    dropped, drop_nodes, cost = set(), [], 0.0
    while deficit > 0:
        best = None
        for r in nodes:
            if r in dropped or inuse.get(r) or any(k not in dropped for k in kids.get(r, ())):
                continue
            if any(immov[b.block_id] for b in nodes[r]):
                continue
            gain = len(nodes[r])
            c = val[r] + (0.0 if ram_has is None or ram_has(r) else STORE_COST * gain)
            if gain and (best is None or c / gain < best[0] / best[2]):
                best = (c, r, gain)
        if best is None:
            return None
        cost += best[0]
        dropped.add(best[1])
        drop_nodes.append(best[1])
        deficit -= best[2]
    dropped_ids = {id(b) for x in drop_nodes for b in nodes[x]}
    movers = U + [b for b in C if id(b) not in dropped_ids]
    return cost + MOVE_COST * len(movers), drop_nodes, movers


# In-use block moves (copy + identity swap of ref_cnt > 0 blocks) stay OFF by default  until the forced-move
# logprob test passes on BOTH GPUs under async scheduling; until then compaction / claims / lends only use free blocks
# and drop cached-unreferenced ones. Labs: EXL3_LEND_MOVE_INUSE=1.
MOVE_INUSE = _ENV("EXL3_LEND_MOVE_INUSE", "0") == "1"


def _move_ctx(sched=None):
    """(in-use moves allowed, block ids that must not be touched) for a compaction / lend right now.
    Race analysis (async scheduling on both engines): EngineCore.step_with_batch_queue calls
    model_executor.execute_model(..., non_block=True) and sample_tokens(..., non_block=True) BEFORE the next
    scheduler.schedule(); with UniProcExecutor (single GPU, our engines) collective_rpc(non_block=True) still runs
    run_method() synchronously in the engine thread (vllm/v1/executor/uniproc_executor.py collective_rpc: only an
    AsyncModelRunnerOutput's D2H copy is deferred), so every kernel of step N is enqueued before schedule(N+1) runs,
    and _sync() (device-wide: torch.cuda/xpu.synchronize wait for all streams) waits for them. What can still be in
    flight across steps are the KV connector's paced transfers (KV-RING stores of running requests' blocks, loads into
    WAITING_FOR_REMOTE_KVS requests' blocks, stores of finished requests' blocks); their block ids are blocked here.
    In-use moves also need the worker in this process (block-table remap): no runner -> no in-use moves."""
    sched = sched or _ST.get("sched")
    blocked = set()
    allow = MOVE_INUSE and _ST.get("runner") is not None
    conns = []
    c = getattr(sched, "connector", None)
    if c is not None:
        conns = [c] + list(getattr(c, "_connectors", None) or [])
    for c in conns:
        cs = getattr(c, "connector_scheduler", None)
        if cs is None:
            continue
        try:
            blocked |= set((getattr(cs, "_block_id_to_pending_jobs", None) or {}).keys())
            for js in (getattr(cs, "_jobs", None) or {}).values():
                for lst in (getattr(js, "non_sliding_window_block_ids", None), getattr(js, "sliding_window_block_ids", None)):
                    blocked |= set(lst or ())
            for js in (getattr(cs, "_current_batch_load_jobs", None) or {}).values():
                for attr in ("block_ids", "dst_block_ids"):
                    blocked |= set(getattr(js, attr, None) or ())
        except Exception:  # noqa: BLE001
            allow = False                    # unknown connector state: no in-use moves
    try:
        from vllm.v1.request import RequestStatus
        km = getattr(sched, "kv_cache_manager", None)
        for r in (getattr(sched, "waiting", None) or []):
            if getattr(r, "status", None) == RequestStatus.WAITING_FOR_REMOTE_KVS and km is not None:
                for lst in km.get_block_ids(r.request_id):
                    blocked |= set(lst)
    except Exception:  # noqa: BLE001
        pass
    return allow, blocked


def _immovable(pool):
    lent = {id(b) for t in _TENANTS.values() for b in t["blocks"]}
    for key in ("lent_pending", "vis_pending"):
        pend = _ST.get(key)
        if pend:
            lent |= {id(b) for b in pend[0]}
    allow, blocked = _move_ctx()
    _ST["move_allow"] = allow
    return [getattr(b, "is_null", False) or id(b) in lent or b.block_id in blocked for b in pool.blocks]


def score_region(pool, lo, n):
    """Cost of clearing ids lo..lo+n-1 for a lend: free 0;
    in use or cached-unreferenced: one copy, onto a free block outside, else onto the least valuable chain tail outside
    (that victim's value is added; it is evicted to the RAM tier); lent / null in the region: inf. Fails only if the
    region holds lent blocks or every block outside it is in use / lent."""
    hs = getattr(pool, "cached_block_hashes_by_block", None) or {}
    p = _plan_window(pool, lo, n, _chain_view(pool), _immovable(pool), hs)
    return float("inf") if p is None else p[0]


def _compact_for_run(pool, n, at=None, below=None):
    """Clear a window of n consecutive ids (see above). at=None: the cheapest window; at="top": the last n ids.
    Returns the window's first id, or None."""
    blocks = pool.blocks
    N = len(blocks)
    hs = getattr(pool, "cached_block_hashes_by_block", None)
    if hs is None:
        hs = {}
    immov = _immovable(pool)   # the null block and blocks lent to tenants (their tensors are views into those pages)
    view = _chain_view(pool)
    best = None
    hi = N if below is None else max(n + 1, min(N, below))
    for st in ([N - n] if at == "top" else range(1, hi - n + 1)):
        p = _plan_window(pool, st, n, view, immov, hs)
        if p is not None and (best is None or p[0] < best[1][0]):
            best = (st, p)
            if p[0] == 0:
                break
    if best is None:
        return None
    st, (cost, drop_nodes, movers) = best
    t0 = time.perf_counter()
    nodes = view[0]
    ndrop = 0
    now_r = time.monotonic()
    for r in drop_nodes:
        _REGRET[r] = now_r
        for b in nodes[r]:
            if _is_cached(b, hs):
                pool._maybe_evict_cached_block(b)
                ndrop += 1
            if not (st <= b.block_id < st + n) and b.ref_cnt == 0:
                try:
                    pool.free_block_queue.remove(b)
                    pool.free_block_queue.prepend_n([b])
                except Exception:  # noqa: BLE001
                    pass
        _CH.pop(r, None)
    dests = [b for b in blocks if b.ref_cnt == 0 and not immov[b.block_id] and not (st <= b.block_id < st + n)
             and not _is_cached(b, hs)]
    if len(dests) < len(movers):
        return None
    if movers:
        _sync()
    remap = {}
    for U, F in zip(movers, dests):
        u, f = U.block_id, F.block_id
        for u8, P in _KVL["stores"]:
            u8[f * P:(f + 1) * P].copy_(u8[u * P:(u + 1) * P])
        U.block_id, F.block_id = f, u
        blocks[f], blocks[u] = U, F
        if u in hs:
            hs[f] = hs.pop(u)
        hm = getattr(getattr(pool, "cached_block_hash_to_block", None), "_cache", None)
        if hm is not None:
            keys = ([U.block_hash] if getattr(U, "block_hash", None) is not None else []) + list(hs.get(f, ()))
            for key in keys:
                e = hm.get(key)
                if isinstance(e, dict) and u in e:
                    e[f] = e.pop(u)
        remap[u] = f
    if movers:
        _sync()
    _remap_worker_tables(remap)
    c = _ST["n"]
    c["compactions"] = c.get("compactions", 0) + 1
    c["compacted_blocks"] = c.get("compacted_blocks", 0) + len(remap)
    c["compact_dropped_blocks"] = c.get("compact_dropped_blocks", 0) + ndrop
    c["compact_last_score"] = round(cost, 4)
    dt = (time.perf_counter() - t0) * 1e3
    _log("lend: compaction cleared ids %d-%d: moved %d blocks (%d in use), dropped %d cached blocks of %d chain "
         "nodes (cost %.2f) in %.1f ms (%.2f ms/move)", st, st + n - 1, len(remap),
         sum(1 for b in movers if b.ref_cnt > 0), ndrop, len(drop_nodes), cost, dt, dt / max(1, len(remap)))
    return st


_STATS = {}


def lend_stats():
    st = {"lent_by_tenant": {n: len(t["blocks"]) for n, t in _TENANTS.items()},
          **{k: v for k, v in _ST["n"].items() if k.startswith(("regret_", "compact", "lend"))}}
    try:                                   # fragmentation of the normal allocator
        acc = _acc()
        ms = acc.memory_stats()
        st.update(torch_alloc_mib=acc.memory_allocated() >> 20, torch_reserved_mib=acc.memory_reserved() >> 20,
                  torch_inactive_split_mib=int(ms.get("inactive_split_bytes.all.current", 0)) >> 20)
    except Exception:  # noqa: BLE001
        pass
    return st


def _stats_tick():
    """LEND-STATS <json> every EXL3_LEND_STATS_S (30 s) from the scheduler: lending / compaction / regret counters
    (cumulative since engine start) for the exporter (llmt_lend_* metrics)."""
    now = time.monotonic()
    if now - _STATS.get("t", 0) < float(_ENV("EXL3_LEND_STATS_S", "30")):
        return
    _STATS["t"] = now
    if not (_TENANTS or _ST["n"]):
        return
    _log("LEND-STATS %s", json.dumps(lend_stats(), sort_keys=True))


def _remap_worker_tables(remap):
    """Rewrite block ids in the worker's block tables (cpu copies; the next step commits them) and the runner's cached
    request block_ids."""
    if not remap:
        return
    import numpy as np
    runner = _ST.get("runner")
    if runner is None:
        return
    ib = getattr(runner, "input_batch", None)
    mg = getattr(ib, "block_table", None)
    for bt in getattr(mg, "block_tables", []) or []:
        arr = bt.block_table.np
        k = int(getattr(bt, "blocks_per_kv_block", 1) or 1)
        for u, f in remap.items():
            for j in range(k):
                arr[arr == u * k + j] = -(f * k + j) - 1        # two-phase (no chains through other remapped ids)
        neg = arr < 0
        arr[neg] = -arr[neg] - 1
    for rs in (getattr(runner, "requests", {}) or {}).values():
        bids = getattr(rs, "block_ids", None)
        if bids:
            for lst in bids:
                for i, b in enumerate(lst):
                    if b in remap:
                        lst[i] = remap[b]


def _unlend(name, blocks=None):
    """Any thread: give a tenant's blocks (or the given subset) back; the scheduler frees them before its next
    schedule()."""
    with _LOCK:
        t = _TENANTS.get(name)
        if blocks is None:
            blocks = t["blocks"] if t else []
        if t:
            ids = {id(b) for b in blocks}
            t["blocks"] = [b for b in t["blocks"] if id(b) not in ids]
            if not t["blocks"]:
                _TENANTS.pop(name, None)
        if blocks:
            _ST.setdefault("to_return", []).append(list(blocks))
            _log("lend: %s gives back %d KV blocks", name, len(blocks))


def _lent_count(pool):
    """pool.num_lent_blocks: blocks out of the KV pool for tenants (and returned ones not yet freed). The KV usage
    metric, the scheduler's length-cap wait and its tail-hold watermark leave them out (echowarp/vllm; the
    llmt_lent_blocks alias is kept for older engine builds)."""
    if pool is None:
        return 0
    n = sum(len(t["blocks"]) for t in _TENANTS.values()) + sum(len(b) for b in (_ST.get("to_return") or []))
    pool.num_lent_blocks = pool.llmt_lent_blocks = n
    return n


def lend_status():
    return {n: (t["kind"], len(t["blocks"])) for n, t in _TENANTS.items()}


# ---- vision tower tenant (VIS-LEND: the tower's VRAM is KV until an image arrives, then KV is evicted for it)
VIS_LEND = _ENV("EXL3_VIS_LEND", "1") == "1"
VIS_MAX_RUN = int(_ENV("EXL3_VIS_LEND_MAX_RUN", "48"))


def _vis_tower():
    import sys
    found = []
    for modname in ("llmt_kv.lazyvis", "exl3xpu.lazyvis", "llmt_lazyvis"):
        m = sys.modules.get(modname)
        towers = (getattr(m, "_STATE", {}) or {}).get("towers", []) if m else []
        found += [t for t in towers if all(t is not f for f in found)]
    if len(found) > 1:
        if not _ST.get("vis_multi_logged"):
            _ST["vis_multi_logged"] = True
            _log("VIS-LEND: %d lazy vision towers in this process (double install): vision lending OFF", len(found),
                 level="error")
        return None
    return found[0] if found else None


def _vis_plan(units, run):
    """The whole tower into one span per KV storage of `run` consecutive pages: block units whole, the head unit
    (patch_embed, pos_embed, merger) tensor by tensor. {unit idx: (storage, offset), 0: "split", ("h", k): (storage,
    offset) of the head's k-th checkpoint tensor} or None."""
    spans = [[s, 0, run * P] for s, (_, P) in enumerate(_KVL["stores"])]
    plan = {}
    items = [(u.n, u.idx) for u in units if u.n and u.idx != 0]
    head = next((u for u in units if u.idx == 0), None)
    if head is not None and head.n:
        items += [(r.nbytes, ("h", k)) for k, r in enumerate(head.entry) if r.nbytes]
        plan[0] = "split"
    for n, key in sorted(items, key=lambda x: -x[0]):
        for sp in spans:
            off = (sp[1] + _ALIGN_B - 1) // _ALIGN_B * _ALIGN_B
            if off + n <= sp[2]:
                plan[key] = (sp[0], off)
                sp[1] = off + n
                break
        else:
            return None
    return plan


def _vis_tick(sched, pool):
    """Scheduler thread: an image request is queued or running, the tower is not resident and has no lend: lend a run
    of consecutive blocks covering its block units, so visual.forward loads it whole into KV pages."""
    if not (VIS_LEND and _KVL["ok"] and _ST["ready"]):
        return
    if PROF and os.path.exists(os.path.join(PROF, "novislend")):   # lab A/B toggle: streamed as before
        return
    lv = _vis_tower()
    if lv is None or lv.resident or _TENANTS.get("vision") or _ST.get("vis_pending"):
        return
    need = False
    for q in _sched_queues(sched):
        try:
            for r in q:
                if getattr(r, "has_encoder_inputs", False) or getattr(r, "mm_features", None):
                    need = True
                    break
        except Exception:  # noqa: BLE001
            pass
        if need:
            break
    if not need:
        return
    if lv._room() >= lv.bytes + getattr(sys.modules.get(type(lv).__module__), "MARGIN", 0):
        return     # the allocator has room: lazyvis loads it resident as before
    tot = sum(u.n for u in lv.units)          # the head unit (merger) is lent too
    per_page = sum(P for _, P in _KVL["stores"])
    run = max(1, -(-tot // per_page))
    while run <= VIS_MAX_RUN and _vis_plan(lv.units, run) is None:
        run += 1
    if run > VIS_MAX_RUN:
        return
    blocks = _lend(pool, "vision", run, "vision tower", contiguous=True)
    if blocks is None:
        now = time.monotonic()
        if now - _ST.get("vis_lend_log", 0) > 30:
            _ST["vis_lend_log"] = now
            _log("vision: no run of %d free KV blocks to lend: the tower streams", run)
        return
    _ST["vis_pending"] = (blocks, _vis_plan(lv.units, run))


def _patch_vision():
    """Tower placement into lent pages: wrap the lazy tower's _unit_on / _room / maybe_release (exl3xpu.lazyvis on the
    Arc, llmt_lazyvis on the 5090); no change to its file. Installed when the tower object appears."""
    if not VIS_LEND:
        return
    import sys
    for modname in ("llmt_kv.lazyvis", "exl3xpu.lazyvis", "llmt_lazyvis"):
        m = sys.modules.get(modname)
        cls = getattr(m, "LazyVision", None) if m else None
        # skip compat classes without the lazy-tower hooks (e.g. the exl3xpu.lazyvis shim): they would raise
        # AttributeError after being marked, and the full llmt_kv.lazyvis class is the one that places the tower
        if cls is None or getattr(cls, "_llmt_vis", False) or not all(
                hasattr(cls, a) for a in ("_unit_on", "_room", "maybe_release", "_enter")):
            continue
        cls._llmt_vis = True
        orig_on, orig_room, orig_rel, orig_enter = cls._unit_on, cls._room, cls.maybe_release, cls._enter

        def _unit_on(self, u, src, _o=orig_on):
            plan = getattr(self, "_llmt_plan", None)
            if not plan or u.idx not in plan:
                return _o(self, u, src)
            if plan[u.idx] == "split":            # head unit: staged once, then each tensor into its lent slot
                tmp = torch.empty(max(u.n, 1), dtype=torch.uint8, device=self.dev)
                src.copy_unit(u, tmp)
                last = None
                for k, r in enumerate(u.entry):
                    si, off = plan[("h", k)]
                    u8, P = _KVL["stores"][si]
                    base = self._llmt_base * P + off
                    dst = u8[base:base + r.nbytes]
                    dst.copy_(tmp[r.off - u.off:r.off - u.off + r.nbytes], non_blocking=True)
                    self._set(r, dst.view(r.dtype).view(r.shape))
                    last = dst
                for r in u.extra:
                    d = torch.empty(r.shape, dtype=r.dtype, device=self.dev)
                    if self.master is not None:
                        d.copy_(self._mview(r), non_blocking=True)
                    else:
                        d.copy_(self.extras[r.xoff:r.xoff + r.nbytes].view(r.dtype).view(r.shape), non_blocking=True)
                    self._set(r, d)
                del tmp                               # stream-ordered: the allocator reuses it after the copies
                _ST["n"]["vis_head_lent"] = _ST["n"].get("vis_head_lent", 0) + 1
                u.dev, u.on_dev = last, True
                return
            si, off = plan[u.idx]
            u8, P = _KVL["stores"][si]
            base = self._llmt_base * P + off
            buf = u8[base:base + u.n]
            src.copy_unit(u, buf)
            for r in u.entry:
                self._set(r, buf[r.off - u.off:r.off - u.off + r.nbytes].view(r.dtype).view(r.shape))
            for r in u.extra:
                d = torch.empty(r.shape, dtype=r.dtype, device=self.dev)
                if self.master is not None:
                    d.copy_(self._mview(r), non_blocking=True)
                else:
                    d.copy_(self.extras[r.xoff:r.xoff + r.nbytes].view(r.dtype).view(r.shape), non_blocking=True)
                self._set(r, d)
            u.dev, u.on_dev = buf, True

        def _room(self, _o=orig_room):
            r = _o(self)
            try:                                       # lazyvis counts device free memory only: apply the torch cap
                free = _acc().mem_get_info(self.dev)[0]
                r -= free - _cap_clamp(free)
            except Exception:  # noqa: BLE001
                pass
            if getattr(self, "_llmt_plan", None) is not None or _ST.get("vis_pending"):
                r += sum(u.n for u in self.units)      # the lent part (every unit's checkpoint bytes) is not allocated
            return r

        def _enter(self, *a, _o=orig_enter, **k):
            with _LOCK:
                pend = _ST.get("vis_pending")
                if pend and not self.resident:
                    blocks, plan = pend
                    self._llmt_plan, self._llmt_base = plan, min(b.block_id for b in blocks)
                    _ST["vis_pending"] = None
            mod = sys.modules.get(type(self).__module__)
            saved = getattr(mod, "MODE", None)
            if getattr(self, "_llmt_plan", None) is not None and not self.resident and mod is not None:
                mod.MODE = "resident"      # the lent pages ARE the room: load whole (head unit from the allocator)
            try:
                r = _o(self, *a, **k)
            finally:
                if mod is not None and saved is not None:
                    mod.MODE = saved
            if not self.resident and getattr(self, "_llmt_plan", None) is not None:   # streamed after all: give back
                self._llmt_plan = None
                _unlend("vision")
            return r

        def maybe_release(self, _o=orig_rel):
            was = self.resident
            _o(self)
            if was and not self.resident and getattr(self, "_llmt_plan", None) is not None:
                self._llmt_plan = None
                _sync()
                _unlend("vision")
        cls._unit_on, cls._room, cls._enter, cls.maybe_release = _unit_on, _room, _enter, maybe_release
        _log("VIS-LEND: %s.LazyVision places the resident tower in lent KV pages when the allocator has no room", modname)


def _grow_to(k):
    """With _LOCK held, slots on the device: k slot rows (k > current) if the room covers the extra rows + margin.
    Reallocates the stacked buffers (contents kept) and drops the LoRA graphs (they hold the old addresses)."""
    cur = _ST["k"]
    extra = (k - cur) * _slot_bytes()
    host = _ST["where"] == "host"
    if _ST.get("lent") or (not host and MODE == "auto" and _ST["ready"] and _room() < extra + MARGIN):
        return _use_lent(k)
    t0 = time.perf_counter()
    _lg_drop("slot buffers grow %d -> %d" % (cur, k))
    try:
        for l in _ST["layers"]:
            _resize(l, k)
        _sync()
    except torch.OutOfMemoryError:
        for l in _ST["layers"]:
            _resize(l, cur)
        _log("growing the LoRA slots to %d hit OOM: staying at %d", k, cur, level="warning")
        return False
    _ST["k"], _ST["bytes"] = k, k * _slot_bytes()
    if not host:
        _lazyvis_base_adjust(extra)
    _ST["n"]["grows"] = _ST["n"].get("grows", 0) + 1
    _log("LoRA slots %d -> %d on the " + ("host" if host else "device") + " (+%.1f MiB, %.1f ms)", cur, k, extra / 2**20, (time.perf_counter() - t0) * 1e3)
    return True


def _shrink_to(k, why):
    """With _LOCK held: drop trailing free slot rows (k >= 1; the caller made sure rows >= k are free)."""
    cur = _ST["k"]
    if _ST["where"] not in ("device", "host") or k >= cur or _ST.get("lent"):
        return
    _lg_drop("slot buffers shrink %d -> %d" % (cur, k))
    for l in _ST["layers"]:
        _resize(l, k)
    _sync()
    freed = (cur - k) * _slot_bytes()
    _ST["k"], _ST["bytes"] = k, k * _slot_bytes()
    if _ST["where"] == "device":
        _lazyvis_base_adjust(-freed)
    try:
        _acc().empty_cache()
    except Exception:  # noqa: BLE001
        pass
    _log("LoRA slots %d -> %d (%s): %.1f MiB back", cur, k, why, freed / 2**20)


def _slot_cap():
    """How many distinct adapters a batch may use now (the scheduler's max_loras): the device slots allocated plus
    those the room can still take, at least 1, at most N. Host slots: N."""
    N = _max_slots()
    if not _ST["ready"] or MODE != "auto" or _ST["where"] == "host":
        return N
    now = time.monotonic()
    c = _ST.get("cap_cache")
    if c and now - c[0] < 0.5:
        return c[1]
    k = _ST["k"] if _ST["where"] == "device" else 0
    try:
        extra = max(0, int((_room() - MARGIN) // max(1, _slot_bytes())))
    except Exception:  # noqa: BLE001
        extra = 0
    pend = _ST.get("lent_pending")
    if _ST.get("lent"):
        extra = 0
    cap = min(N, max(k + extra, pend[1] if pend else 0))
    if HOST_FALLBACK:
        cap = max(1, cap)
    _ST["cap_cache"] = (now, cap)
    return cap


def _ensure_slots(want=1):
    """Called with _LOCK held before an adapter is activated into a slot: device slot rows for the first activation
    (want = rows needed), else host slots (all N rows)."""
    if _ST["where"] != "off":
        return
    dev = _ST["dev"]
    N = _max_slots()
    want = max(1, min(N, want))
    need = want * _slot_bytes()
    where = "device"
    if MODE == "host":
        where = "host"
    elif MODE == "auto" and _ST["ready"] and _ST["act_reserve"] is not None:
        room = _room()
        if room < need + MARGIN and _use_lent(want):
            return
        if room < need + MARGIN:
            if not HOST_FALLBACK and not _ST.get("force_room"):
                raise _NoRoom("no device room for a LoRA slot (%.0f MiB + %d margin, room %.0f MiB)" % (
                    need / 2**20, MARGIN >> 20, room / 2**20))
            if HOST_FALLBACK:
                where = "host"
                _log("no device room for the LoRA slots (%.0f MiB needed + %d margin, room %.0f MiB): host slots, "
                     "streamed per layer", need / 2**20, MARGIN >> 20, room / 2**20, level="warning")
            else:
                _log("LoRA slot allocated beyond the room (%.0f MiB + %d margin, room %.0f MiB): the request was "
                     "already scheduled", need / 2**20, MARGIN >> 20, room / 2**20, level="warning")
    t0 = time.perf_counter()
    try:
        for l in _ST["layers"]:
            _place(l, dev if where == "device" else "cpu", want)
        _sync()
    except torch.OutOfMemoryError:
        for l in _ST["layers"]:
            _place(l, "cpu", want)
        where = "host"
        _log("device allocation of the LoRA slots hit OOM: host slots", level="warning")
    _ST["k"] = want
    need = _ST["k"] * _slot_bytes()
    _ST["bytes"] = need
    _ST["where"] = where
    if where == "device":
        _lazyvis_base_adjust(need)
        _ST["n"]["dev_allocs"] += 1
    else:
        _ST["n"]["host_allocs"] += 1
    _log("LoRA slot buffers allocated on the %s: %.1f MiB over %d layers in %.1f ms", where, need / 2**20,
         len(_ST["layers"]), (time.perf_counter() - t0) * 1e3)


def _release(why):
    """With _LOCK held: deactivate every adapter, free the slot buffers. Registered adapters stay (empty shells)."""
    mgr = _ST["mgr"]
    if _ST["where"] == "off" or mgr is None:
        return
    for lid in list(mgr._active_adapters.cache.keys()):
        try:
            mgr.deactivate_adapter(lid)
        except Exception as e:  # noqa: BLE001
            _log("deactivate %s failed: %r", lid, e, level="warning")
    mgr._last_mapping = None
    mgr._last_slot_layout = None
    _lg_drop("slots released")
    was = _ST["where"]
    N = _max_slots()
    for l in _ST["layers"]:
        _place(l, "meta", N)
        n = len(getattr(l, "adapter_enabled_cpu", []) or [])
        for i in range(n):
            l.adapter_enabled_cpu[i] = False
            if hasattr(l, "adapter_ranks_cpu"):
                l.adapter_ranks_cpu[i] = 0
    lent = _ST.get("lent")
    if lent:
        _sync()
        _unlend("lora", lent)   # the scheduler frees them before its next schedule()
        _ST["lent"] = None
    elif was == "device":
        _lazyvis_base_adjust(-_ST["bytes"])
    _ST["where"] = "off"
    _ST["k"] = 0
    _ST["n"]["releases"] += 1
    if _ST["broker"] is not None:
        _ST["broker"].cold_all()
    _log("LoRA slots released (%s): %.1f MiB %s memory back; counters %s", why, _ST["bytes"] / 2**20,
         "device" if was == "device" else "host", _ST["n"])


def _maybe_promote():
    """With _LOCK held, in the step thread: host slots -> device once there is room."""
    if _ST["where"] != "host" or MODE != "auto" or not _ST["ready"]:
        return
    if _room() < _ST["bytes"] + MARGIN:
        return
    _lg_drop("slots moved")     # all N host rows move
    try:
        for l in _ST["layers"]:
            _move(l, _ST["dev"])
        _sync()
    except torch.OutOfMemoryError:
        for l in _ST["layers"]:
            _move(l, "cpu")
        return
    _ST["where"] = "device"
    _lazyvis_base_adjust(_ST["bytes"])
    _ST["n"]["promotions"] += 1
    _log("LoRA slots promoted host -> device (%.1f MiB)", _ST["bytes"] / 2**20)


def _lg_drop(why):
    """Forget every LoRA graph (has_lora keys) and its pool; base graphs stay."""
    if not _LG["captured"] and _LG["pool"] is None:
        _LG["want"].clear()
        return
    try:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        n = 0
        for w in list(CUDAGraphWrapper._all_instances):
            for k in [k for k in w.concrete_cudagraph_entries if getattr(k, "has_lora", False)]:
                del w.concrete_cudagraph_entries[k]
                n += 1
    except Exception as e:  # noqa: BLE001
        _log("LoRA graph drop: %r", e, level="warning")
        n = -1
    freed = _LG["bytes"]
    if freed:
        _lazyvis_base_adjust(-freed)
    _LG.update(captured=set(), want=set(), seen=set(), pool=None, bytes=0)
    _LG["dropped"] += 1
    import gc
    gc.collect()
    _sync()
    try:
        _acc().empty_cache()
    except Exception:  # noqa: BLE001
        pass
    _log("LoRA graphs dropped (%s): %d entries, %.1f MiB", why, n, freed / 2**20)


def _lg_capture_pending(runner):
    """At the start of execute_model (no step in preparation): capture wanted LoRA decode graphs while the room allows.
    Each capture = one warm-up + one capture dummy run with the REAL active adapter mapped to every token."""
    if not (LGRAPH and _LG["want"] and _ST["ready"] and _ST["where"] == "device"):
        return
    from vllm.config import CUDAGraphMode
    from vllm.compilation import monitor
    done = 0
    for desc in sorted(_LG["want"], key=lambda d: d.num_tokens):
        if done >= LG_MAX_PER_STEP:
            break
        _LG["want"].discard(desc)
        if desc in _LG["captured"]:
            continue
        room = _room()
        if room < LG_EST + LG_MARGIN:
            if ("lowroom", desc) not in _LG["seen"]:
                _LG["seen"].add(("lowroom", desc))
                _log("LoRA graph %s not captured: room %.0f MiB < %.0f MiB", desc, room / 2**20,
                     (LG_EST + LG_MARGIN) / 2**20)
            continue
        if _LG["pool"] is None and LG_OWN_POOL:
            from vllm.platforms import current_platform
            _LG["pool"] = current_platform.graph_pool_handle()
        acc = _acc()
        _sync()
        r0 = acc.memory_reserved()
        t0 = time.perf_counter()
        was = monitor.cudagraph_capturing_enabled
        _LG["capturing"] = desc
        try:
            monitor.set_cudagraph_capturing_enabled(True)
            if getattr(getattr(runner, "device", None), "type", "") == "cuda":
                from vllm.distributed.parallel_state import graph_capture    # CUDA: capture on a side stream
                with graph_capture(device=runner.device):
                    runner._warmup_and_capture(desc, CUDAGraphMode.FULL, num_warmups=1)
            else:
                runner._warmup_and_capture(desc, CUDAGraphMode.FULL, num_warmups=1)
            _sync()
            _LG["captured"].add(desc)
            _LG["n"] += 1
            done += 1
        except Exception as e:  # noqa: BLE001
            _LG["fail"] += 1
            _log("LoRA graph capture %s failed: %r (LoRA steps of this shape stay without a graph)", desc, e,
                 level="warning")
            _LG["seen"].add(("failed", desc))
        finally:
            monitor.set_cudagraph_capturing_enabled(was)
            _LG["capturing"] = None
            pw = _LG.pop("pw", None)
            if pw is not None:
                for n in ("get_active_lora_rank", "has_active_lora"):
                    pw.__dict__.pop(n, None)
            _ST["step_lora"] = False
            mgr = _ST["mgr"]
            if mgr is not None:
                mgr._last_mapping = None       # the next real step re-sends its own mapping
        grew = max(0, acc.memory_reserved() - r0)
        _LG["bytes"] += grew
        _lazyvis_base_adjust(grew)
        _log("LoRA graph captured for %s: %.0f ms, +%.1f MiB (LoRA graphs %d, %.1f MiB)", desc,
             (time.perf_counter() - t0) * 1e3, grew / 2**20, len(_LG["captured"]), _LG["bytes"] / 2**20)


def _patch_lgraph():
    if not LGRAPH:
        return
    import dataclasses
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.config import CUDAGraphMode
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    orig_disp = CudagraphDispatcher.dispatch

    def dispatch(self, num_tokens, uniform_decode=False, has_lora=False, num_active_loras=0, valid_modes=None,
                 invalid_modes=None):
        if has_lora and LGRAPH:
            mode, desc = orig_disp(self, num_tokens, uniform_decode, False, 0, valid_modes, invalid_modes)
            if mode == CUDAGraphMode.FULL:
                ld = dataclasses.replace(desc, has_lora=True, num_active_loras=1)   # one graph for any slot mix
                if _LG["capturing"] == ld or ld in _LG["captured"]:
                    return CUDAGraphMode.FULL, ld
                if _ST["where"] == "device" and ("failed", ld) not in _LG["seen"]:
                    if ld in _LG["seen"]:
                        _LG["want"].add(ld)
                    else:
                        _LG["seen"].add(ld)
        return orig_disp(self, num_tokens, uniform_decode, has_lora, num_active_loras, valid_modes, invalid_modes)
    CudagraphDispatcher.dispatch = dispatch

    orig_call = CUDAGraphWrapper.__call__

    def __call__(self, *a, **k):
        if _LG["capturing"] is not None:
            from vllm.forward_context import get_forward_context, is_forward_context_available
            if is_forward_context_available():
                bd = get_forward_context().batch_descriptor
                if bd is not None and getattr(bd, "has_lora", False) and _LG["pool"] is not None:
                    saved = self.graph_pool
                    self.graph_pool = _LG["pool"]     # LoRA graphs in their own pool: freed as a whole
                    try:
                        return orig_call(self, *a, **k)
                    finally:
                        self.graph_pool = saved
        return orig_call(self, *a, **k)
    CUDAGraphWrapper.__call__ = __call__

    orig_exec = GPUModelRunner.execute_model

    def execute_model(self, *a, **k):
        _ST["runner"] = self
        if _LG["want"]:
            with _LOCK:
                _lg_capture_pending(self)
        return orig_exec(self, *a, **k)
    GPUModelRunner.execute_model = execute_model
    _log("lazy LoRA decode graphs on (estimate %d MiB per graph, at most %d captures per step)", LG_EST >> 20,
         LG_MAX_PER_STEP)


def _reaper():
    while True:
        time.sleep(min(5.0, max(0.5, TTL / 4)))
        try:
            with _LOCK:
                if _ST["where"] == "off":
                    continue
                now = time.time()
                idle = now - _ST["last_use"]
                if idle >= TTL:
                    _release("idle %.0f s" % idle)
                    continue
                mgr = _ST["mgr"]
                if mgr is None:
                    continue
                last = _ST.setdefault("last_by_id", {})
                active = [i for i in mgr.lora_index_to_id if i is not None]
                pressure = (_ST["where"] == "device" and MODE == "auto" and _ST["ready"] and _room() < 0)
                victims = [i for i in active if now - last.get(i, 0) >= TTL]
                if pressure and not victims:
                    idle_ids = sorted((last.get(i, 0), i) for i in active if now - last.get(i, 0) >= PRESSURE_IDLE_S)
                    victims = [idle_ids[0][1]] if idle_ids else []
                for i in victims:
                    mgr.deactivate_adapter(i)
                    _ST["n"]["slot_evictions"] = _ST["n"].get("slot_evictions", 0) + 1
                    _log("adapter id %d left its slot (%s)", i, "memory pressure" if pressure else "idle")
                if victims:
                    mgr._last_mapping = None
                    mgr._last_slot_layout = None
                    used = [j for j, i in enumerate(mgr.lora_index_to_id) if i is not None]
                    if not used:
                        _release("all adapters idle" if not pressure else "memory pressure")
                    elif _ST["where"] in ("device", "host") and max(used) + 1 < _ST["k"]:
                        _shrink_to(max(used) + 1, "trailing slots free")
        except Exception as e:  # noqa: BLE001
            _log("reaper: %r", e, level="warning")


# ------------------------------------------------------------------------------------------ host copy (broker)

def _kv():
    import llmt_kv
    return llmt_kv


def _mv(addr, n):
    return memoryview((ctypes.c_uint8 * n).from_address(addr)).cast("B")


def _segs(ranges, off, n):
    pos, end = 0, off + n
    for po, ln in ranges:
        a, b = max(off, pos), min(end, pos + ln)
        if a < b:
            yield po + (a - pos), b - a, a
        pos += ln
        if pos >= end:
            break


class _Conn:
    def __init__(self, c):
        self.c = c
        self.base = c.pool_address()
        self.size = int(c.pool_bytes)
        self.epoch0 = c.epoch
        self.pool_path = c.pool_path


class _Broker:
    """kind-5 entries in the AllBroker pool (protocol v1); every failure means 'no broker this time'. Same client
    pattern as the lazy vision tower's (lazyvis-broker-20261112)."""

    def __init__(self, sock, engine_id, ns_name):
        self.sock, self.engine_id, self.ns_name = sock, int(engine_id), ns_name
        self.ns = int.from_bytes(hashlib.blake2b(ns_name.encode(), digest_size=4).digest(), "little")
        self.conn, self.next_try, self.keys = None, 0.0, set()
        self.n = {"errors": 0, "connect_fail": 0, "full": 0, "busy": 0}

    def get(self):
        if self.conn is not None:
            return self.conn
        now = time.monotonic()
        if now < self.next_try:
            return None
        self.next_try = now + 30.0
        try:
            k = _kv()
            c = k.Client(socket=self.sock, engine_id=self.engine_id, name="llmt-lora", timeout_ms=TIMEOUT_MS,
                         reconnect_ms=1000.0)
            self.conn = _Conn(c)
            _log("broker %s connected (namespace %r = %d, pool %.1f GiB)", self.sock, self.ns_name, self.ns,
                 self.conn.size / 2**30)
        except Exception as e:  # noqa: BLE001
            self.n["connect_fail"] += 1
            if self.n["connect_fail"] <= 3:
                _log("no broker on %s (%s: %s): adapters load from disk", self.sock, type(e).__name__, e,
                     level="warning")
            return None
        return self.conn

    def _err(self, what, e):
        k = _kv()
        if isinstance(e, getattr(k, "BrokerPoolChanged", ())):
            self.conn, self.next_try = None, 0.0
        self.n["errors"] += 1
        if self.n["errors"] <= 10:
            _log("broker %s: %s: %s", what, type(e).__name__, e, level="warning")

    def lookup(self, key, nbytes, meta):
        conn = self.get()
        if conn is None:
            return None
        try:
            n_hit, lease, ents = conn.c.lookup(self.ns, [key], acquire=True)
        except Exception as e:  # noqa: BLE001
            self._err("lookup", e)
            return None
        if not n_hit:
            return None
        e = ents[0]
        ranges = [tuple(r) for r in e.byte_ranges]
        ok = (int(e.nbytes) == nbytes and int(e.kind) == KIND_WEIGHTS and int(e.format) == FORMAT
              and bytes(e.meta) == meta and sum(l for _, l in ranges) >= nbytes
              and all(0 <= o and o + l <= conn.size for o, l in ranges))
        if not ok:
            _log("broker entry does not match this adapter: ignored", level="warning")
            self.release(conn, lease)
            return None
        self.keys.add(key)
        return _PoolSource(self, conn, ranges, lease, nbytes)

    def release(self, conn, lease):
        try:
            st = conn.c.release([lease])
            return bool(st) and int(st[0]) == int(_kv().OK)
        except Exception as e:  # noqa: BLE001
            self._err("release", e)
            return False

    def store(self, key, nbytes, cost_ms, meta, fill):
        """RESERVE -> fill(views) -> COMMIT. True when stored (or already there)."""
        conn = self.get()
        if conn is None:
            return False
        k = _kv()
        try:
            [(st, ranges)] = conn.c.reserve(self.ns, [{"key": key, "nbytes": int(nbytes), "kind": KIND_WEIGHTS,
                                                         "format": FORMAT, "cost_hint": int(max(1, cost_ms)),
                                                         "meta": meta}])
        except Exception as e:  # noqa: BLE001
            self._err("reserve", e)
            return False
        st = int(st)
        if st == int(k.EXISTS):
            return True
        if st == int(k.BUSY):
            self.n["busy"] += 1
            try:
                conn.c.set_timeout_ms(BUSY_WAIT_MS + 1000.0)
                try:
                    conn.c.wait(self.ns, key, BUSY_WAIT_MS)
                finally:
                    conn.c.set_timeout_ms(TIMEOUT_MS)
            except Exception as e:  # noqa: BLE001
                self._err("wait", e)
            return True
        if st != int(k.OK):
            self.n["full" if st == int(k.FULL) else "errors"] += 1
            _log("broker reserve: %s (%.0f MiB): adapter not stored", k.status_name(st) if hasattr(k, "status_name")
                 else st, nbytes / 2**20, level="warning")
            return False
        ranges = [tuple(r) for r in ranges]
        try:
            fd = os.open(conn.pool_path, os.O_RDWR)   # back the range first: a hole on a full tmpfs would SIGBUS
            try:
                for po, ln in ranges:
                    os.posix_fallocate(fd, po, ln)
            finally:
                os.close(fd)
            fill([(_mv(conn.base + po, ln), eo) for po, ln, eo in _segs(ranges, 0, nbytes)])
            st = conn.c.commit(self.ns, [key])
            if not (st and int(st[0]) == int(k.OK)):
                raise RuntimeError("commit %s" % (st,))
        except Exception as e:  # noqa: BLE001
            self._err("store", e)
            try:
                conn.c.abort(self.ns, [key])
            except Exception:  # noqa: BLE001
                pass
            return False
        self.keys.add(key)
        return True

    def cold_all(self):
        conn = self.conn
        h = getattr(conn.c, "heat", None) if conn is not None else None
        if not callable(h) or not self.keys:
            return
        try:
            h(self.ns, list(self.keys), False)
        except Exception:  # noqa: BLE001
            pass


class _PoolSource:
    """A leased broker entry; bytes read straight from the pool mapping."""
    what = "pool"

    def __init__(self, bk, conn, ranges, lease, nbytes):
        self.bk, self.conn, self.ranges, self.lease, self.nbytes = bk, conn, ranges, lease, nbytes
        self.epoch = conn.epoch0

    def read(self, off, n):
        """uint8 CPU tensor of entry bytes [off, off+n): a zero-copy view when contiguous in the pool."""
        segs = list(_segs(self.ranges, off, n))
        if len(segs) == 1:
            po, ln, _ = segs[0]
            return torch.frombuffer(_mv(self.conn.base + po, ln), dtype=torch.uint8)
        out = torch.empty(n, dtype=torch.uint8)
        for po, ln, eo in segs:
            out[eo - off:eo - off + ln].copy_(torch.frombuffer(_mv(self.conn.base + po, ln), dtype=torch.uint8))
        return out

    def crc(self):
        c = 0
        for po, ln, _ in _segs(self.ranges, 0, self.nbytes):
            c = zlib.crc32(_mv(self.conn.base + po, ln), c)
        return c

    def close(self):
        _sync()
        if self.lease is None:
            return True
        ok = self.bk.release(self.conn, self.lease)
        self.lease = None
        try:
            ok = ok and self.conn.c.epoch == self.epoch
        except Exception:  # noqa: BLE001
            ok = False
        return ok


class _TmpSource:
    """The file read into a temporary host buffer (no broker): dropped on close."""
    what = "disk"

    def __init__(self, buf):
        self.buf = buf

    def read(self, off, n):
        return self.buf[off:off + n]

    def close(self):
        _sync()
        self.buf = None
        return True


def _adapter_key(path):
    """(key, nbytes, meta, st) of <path>/adapter_model.safetensors without reading the whole file."""
    f = os.path.join(path, "adapter_model.safetensors")
    st = os.stat(f)
    h = hashlib.blake2b(digest_size=32)
    h.update(b"llmt-lora|%d|" % FORMAT)
    with open(os.path.join(path, "adapter_config.json"), "rb") as c:
        h.update(c.read())
    h.update(struct.pack("<Q", st.st_size))
    with open(f, "rb") as fh:
        h.update(fh.read(1 << 20))
        if st.st_size > (2 << 20):
            fh.seek(st.st_size - (1 << 20))
            h.update(fh.read(1 << 20))
    return h.digest(), st.st_size, f


def _read_file(f, views):
    """Read file f into [(memoryview, entry_offset)] (entry offset = file offset). Returns crc32 of the file."""
    t0 = time.perf_counter()
    fd = os.open(f, os.O_RDONLY)
    try:
        for mv, eo in views:
            done = 0
            while done < len(mv):
                k = os.preadv(fd, [mv[done:]], eo + done)
                if k <= 0:
                    raise IOError("short read of %s at %d" % (f, eo + done))
                done += k
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:  # noqa: BLE001
            pass
    finally:
        os.close(fd)
    c = 0
    for mv, _ in sorted(views, key=lambda v: v[1]):
        c = zlib.crc32(mv, c)
    return c, (time.perf_counter() - t0) * 1e3


_CRC = {}   # key -> crc32 of the file (this process stored or verified it)


def _open_source(path):
    """A byte source for the adapter's safetensors file: broker hit, broker store + hit, or a temporary host buffer."""
    key, nbytes, f = _adapter_key(path)
    bk = _ST["broker"]
    meta = json.dumps({"v": 1, "n": nbytes}).encode()
    if bk is not None:
        src = bk.lookup(key, nbytes, meta)
        if src is not None:
            if VERIFY_HIT and key not in _CRC:
                c0 = time.perf_counter()
                c = src.crc()
                fc, _ms = _file_crc(f)
                if c != fc:
                    _log("broker entry crc %08x != file %08x: reloading from disk", c, fc, level="warning")
                    src.close()
                    src = None
                else:
                    _CRC[key] = c
                    _log("broker entry verified (crc %08x, %.0f ms)", c, (time.perf_counter() - c0) * 1e3)
            if src is not None:
                _ST["n"]["pool_hits"] += 1
                return src
        if STORE:
            res = {}

            def fill(views):
                res["crc"], res["ms"] = _read_file(f, views)
            t0 = time.perf_counter()
            if bk.store(key, nbytes, nbytes / (DISK_MBPS * 1e3) + PARSE_MS, meta, fill):
                if "crc" in res:
                    _CRC[key] = res["crc"]
                    _ST["n"]["stored"] += 1
                    _log("adapter %s read from disk into the broker (kind 5, %.1f MiB): %.0f ms", os.path.basename(path),
                         nbytes / 2**20, (time.perf_counter() - t0) * 1e3)
                src = bk.lookup(key, nbytes, meta)
                if src is not None:
                    _ST["n"]["disk_loads"] += 1
                    return src
    buf = torch.empty(nbytes, dtype=torch.uint8)
    _crc, ms = _read_file(f, [(_mv(buf.data_ptr(), nbytes), 0)])
    _ST["n"]["tmp_loads"] += 1
    _log("adapter %s read from disk into a temporary host buffer (%.1f MiB, %.0f ms; broker %s)", os.path.basename(path),
         nbytes / 2**20, ms, "off" if bk is None else "not usable")
    return _TmpSource(buf)


def _file_crc(f):
    t0 = time.perf_counter()
    c = 0
    with open(f, "rb") as fh:
        while True:
            b = fh.read(8 << 20)
            if not b:
                break
            c = zlib.crc32(b, c)
    return c, (time.perf_counter() - t0) * 1e3


class _SafeOpen:
    """safetensors.safe_open stand-in over a byte source (header parsed here; tensors are views of the source)."""

    def __init__(self, src):
        self.src = src
        hn = struct.unpack("<Q", bytes(src.read(0, 8).numpy()))[0]
        self.hdr = json.loads(bytes(src.read(8, hn).numpy()))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + hn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def keys(self):
        return list(self.hdr.keys())

    def get_tensor(self, name):
        e = self.hdr[name]
        a, b = e["data_offsets"]
        dt = _ST2T[e["dtype"]]
        return self.src.read(self.base + a, b - a).view(dt).reshape(e["shape"])


class _SafetensorsShim:
    """Replaces the `safetensors` module seen by vllm.lora.lora_model: our sources for adapters being loaded here."""

    def __init__(self, real):
        self._real, self.cur = real, {}

    def safe_open(self, path, framework="pt", **kw):
        src = self.cur.get(os.path.abspath(path))
        if src is not None:
            return _SafeOpen(src)
        return self._real.safe_open(path, framework=framework, **kw)

    def __getattr__(self, n):
        return getattr(self._real, n)


_SHIM = None


# ------------------------------------------------------------------------------------------ patches

def _patch_device_helper():
    """vLLM's _get_lora_device knows weight/qweight/...; EXL3 linears have trellis/suh/svh: any tensor's device."""
    import vllm.lora.layers.base_linear as bl
    import vllm.lora.layers.utils as lu
    orig = lu._get_lora_device

    def _dev(base_layer):
        try:
            return orig(base_layer)
        except ValueError:
            for t in list(base_layer.parameters(recurse=False)) + list(base_layer.buffers(recurse=False)):
                return t.device
            raise
    lu._get_lora_device = _dev
    bl._get_lora_device = _dev


def _patch_lazy_slots():
    import vllm.lora.layers as L
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    classes = {BaseLinearLayerWithLoRA}
    for n in dir(L):
        c = getattr(L, n)
        if isinstance(c, type) and issubclass(c, BaseLinearLayerWithLoRA):
            classes.add(c)
    seen = set()
    for c in classes:
        f = c.__dict__.get("create_lora_weights")
        if f is None or id(f) in seen:
            continue
        seen.add(id(f))

        def wrap(orig):
            def create_lora_weights(self, *a, **k):
                dev = self.device
                if dev.type == "meta":
                    return orig(self, *a, **k)
                self.device = torch.device("meta")
                try:
                    orig(self, *a, **k)
                finally:
                    self.device = dev
                if _ST["dev"] is None:
                    _ST["dev"] = dev
                if not getattr(self, "_llmt_slot", False):
                    self._llmt_slot = True
                    self._llmt_idx = len(_ST["layers"])
                    _ST["layers"].append(self)
            return create_lora_weights
        c.create_lora_weights = wrap(f)

    orig_out = BaseLinearLayerWithLoRA._apply_lora_to_output

    def _eager(self, x, output):
        c = _ST.get("compact")
        if c is not None and x.dim() == 2 and x.shape[0] > c[3]:
            # mixed step without cudagraphs: the LoRA term only for the LoRA rows (punica's token mapping holds their
            # slot ids compacted to the front), not for the thousands of base / prefill rows
            idx = c[0]
            ys = output.index_select(0, idx)
            ys = _eager_rows(self, x.index_select(0, idx), ys)
            output.index_copy_(0, idx, ys)
            return output
        return _eager_rows(self, x, output)

    def _eager_rows(self, x, output):
        a = self.lora_a_stacked
        t0 = a[0] if isinstance(a, (tuple, list)) else a
        if t0.device.type == "meta":
            return output
        hk = _HOOKS["linear"]
        if hk is not None and _ST["where"] == "device" and self.is_lora_active():
            r = hk(self, x, output)
            if r is not None:
                return r
        if t0.device.type == "cpu":   # host slots: this layer's buffers to the device for this call only
            b = self.lora_b_stacked
            self.lora_a_stacked = tuple(t.to(x.device) for t in a)
            self.lora_b_stacked = tuple(t.to(x.device) for t in b)
            try:
                return orig_out(self, x, output)
            finally:
                self.lora_a_stacked, self.lora_b_stacked = a, b
        return orig_out(self, x, output)

    def _apply_lora_to_output(self, x, output):
        if torch.compiler.is_compiling():
            return output        # traced graph: no LoRA terms here (compiled path: inside llmt_lora_linear)
        return _eager(self, x, output)
    BaseLinearLayerWithLoRA._apply_lora_to_output = _apply_lora_to_output

    orig_sync = BaseLinearLayerWithLoRA._apply_sync

    def _apply_sync(self, x, bias=None):
        if (torch.compiler.is_compiling() and PATH == "compiled" and bias is None and x.ndim == 2
                and getattr(self, "_llmt_idx", None) is not None):
            return torch.ops.vllm.llmt_lora_linear(x, self._llmt_idx)
        if bias is None and _ST["step_lora"] and not torch.compiler.is_compiling():
            r = _full_hook(self, x)
            if r is not None:
                return r
        return orig_sync(self, x, bias)
    BaseLinearLayerWithLoRA._apply_sync = _apply_sync

    def _lin(x: torch.Tensor, idx: int) -> torch.Tensor:
        """Opaque functional op = the base linear (the same quant-method call production makes) + the LoRA term,
        decided at run time: only on steps with LoRA tokens and allocated slots. A cudagraph captured without LoRA
        records exactly the base kernel."""
        l = _ST["layers"][idx]
        if _ST["step_lora"]:
            r = _full_hook(l, x)
            if r is not None:
                return r
        out = l._get_quant_method().apply(l.base_layer, x, None)
        if _ST["step_lora"]:
            r = _eager(l, x, out)
            out = r if r is not None else out
        return out

    def _lin_fake(x: torch.Tensor, idx: int) -> torch.Tensor:
        l = _ST["layers"][idx]
        return x.new_empty((x.shape[0], int(l.base_layer.output_size_per_partition)))
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="llmt_lora_linear", op_func=_lin, mutates_args=[], fake_impl=_lin_fake)


def _patch_no_dummy_loras():
    from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin as M
    orig_sel = M.maybe_select_dummy_loras

    @contextlib.contextmanager
    def maybe_setup_dummy_loras(self, lora_config, remove_lora=True):
        yield      # no dummy adapters: profile / capture / warm-up see the production model and its memory

    @contextlib.contextmanager
    def maybe_select_dummy_loras(self, lora_config, num_scheduled_tokens, mapping_type=None, num_sampled_tokens=None,
                                 num_active_loras=0):
        kw = {} if mapping_type is None else {"mapping_type": mapping_type}
        if _LG["capturing"] is not None and num_active_loras > 0 and lora_config is not None:
            # LoRA graph capture: every token mapped to the real adapter in the slot (graphs replay whatever the
            # per-step metadata says; -1 rows = base requests skip the LoRA kernels)
            import numpy as np
            mgr = _ST["mgr"]
            lid = next((i for i in mgr.lora_index_to_id if i is not None), None)
            lm = mgr._registered_adapters.cache.get(lid) if lid is not None else None
            req = getattr(lm, "_llmt_req", None)
            if req is None:
                raise RuntimeError("no active adapter to capture a LoRA graph with")
            if num_sampled_tokens is None:
                num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)
            pm = np.full(len(num_scheduled_tokens), lid, dtype=np.int32)
            self._set_active_loras(tuple(np.repeat(pm, num_sampled_tokens)), tuple(np.repeat(pm, num_scheduled_tokens)),
                                   {req}, **kw)
            pw = getattr(mgr, "punica_wrapper", None) or next(iter(getattr(mgr, "punica_wrapper_mapping", {}).values()), None)
            if pw is not None:
                # the graph must hold for ANY mix of slots and base rows: bgmv kernels for every row count, the
                # max rank (adapters of lower rank are zero-padded in their slot), LoRA terms always recorded
                pw.is_prefill = False
                pw.single_lora_index = None
                mr = int(getattr(mgr.lora_config, "max_lora_rank", 16))
                pw.get_active_lora_rank = lambda *a, **k: mr
                pw.has_active_lora = lambda *a, **k: True
                _LG["pw"] = pw
            yield
            return
        with orig_sel(self, lora_config, num_scheduled_tokens, num_sampled_tokens=num_sampled_tokens,
                      num_active_loras=0, **kw):
            yield
    M.maybe_setup_dummy_loras = maybe_setup_dummy_loras
    M.maybe_select_dummy_loras = maybe_select_dummy_loras

    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    CudagraphDispatcher._get_lora_cases = lambda self: [0]   # graphs only without LoRA; LoRA steps -> NONE


class _LoraCfgView:
    """The scheduler's view of the LoRA config: max_loras = what the device can hold now (_slot_cap), never below
    the adapters of running requests (they own their slots). Requests for another adapter wait in the queue."""

    def __init__(self, cfg):
        self._cfg, self._floor = cfg, 0

    @property
    def max_loras(self):
        return max(self._floor, min(self._cfg.max_loras, _slot_cap()))

    def __getattr__(self, n):
        return getattr(self._cfg, n)


def _patch_scheduler():
    from llmt_kv import vllm_hooks as H

    def init(sched):
        if getattr(sched, "lora_config", None) is not None:
            sched.lora_config = _LoraCfgView(sched.lora_config)
        try:
            _ST["block_tokens"] = int(sched.cache_config.block_size or 0)
        except Exception:  # noqa: BLE001
            pass

    def before(sched):
        v = getattr(sched, "lora_config", None)
        _ST["sched"] = sched
        try:
            _sess_note(sched)
        except Exception:  # noqa: BLE001
            pass
        with _LOCK:          # every engine with this plugin: LoRA and vision tenants (vision also without --enable-lora)
            try:
                _lend_tick(sched)
            except Exception as e:  # noqa: BLE001
                _log("lend tick: %r", e, level="warning")
            _lent_count(getattr(getattr(sched, "kv_cache_manager", None), "block_pool", None))
            try:
                _stats_tick()
            except Exception:  # noqa: BLE001
                pass
        if isinstance(v, _LoraCfgView):
            v._floor = len({r.lora_request.lora_int_id for r in sched.running
                            if r.lora_request and r.lora_request.lora_int_id > 0})

    def after(sched, out):
        v = getattr(sched, "lora_config", None)
        if not isinstance(v, _LoraCfgView):
            return
        try:
            nw = sum(1 for q in _sched_queues(sched)[1:] for r in q if getattr(r, "lora_request", None))
        except Exception:  # noqa: BLE001
            nw = 0
        if nw:
            _ST["n"]["lora_wait_steps"] = _ST["n"].get("lora_wait_steps", 0) + 1
            now = time.monotonic()
            if now - _ST.get("wait_log", 0) >= 30:
                _ST["wait_log"] = now
                try:
                    room = _room() if _ST["ready"] else float("nan")
                except Exception:  # noqa: BLE001
                    room = float("nan")
                _log("%d LoRA request(s) waiting; device slot capacity %d of %d (slot %.0f MiB + margin %d MiB, "
                     "room %.0f MiB); base requests are not held back", nw, v.max_loras, v._cfg.max_loras,
                     _slot_bytes() / 2**20, MARGIN >> 20, room / 2**20)

    try:
        H.register("scheduler.init", init)
        H.register("scheduler.before_schedule", before)
        H.register("scheduler.after_schedule", after)
    except Exception as e:  # noqa: BLE001
        _log("scheduler hooks not installed (%r): batches may ask for more adapters than the device holds", e,
             level="warning")
        return
    if ADMIT is not None:
        H.register("scheduler.admit", ADMIT)


COMPACT = _ENV("EXL3_LORA_COMPACT", "1") == "1"
COMPACT_FRAC = float(_ENV("EXL3_LORA_COMPACT_FRAC", "0.5"))


def _compact_on(k):
    """Before a LoRA step's forward WITHOUT cudagraphs: if LoRA rows are a minority, gather them. The punica token
    mapping (both the bgmv and the fused-linear copy) gets their slot ids at the front; _eager runs the LoRA term on
    those rows only and scatters the result back. Graph steps keep the full mapping (graphs read it as captured)."""
    _ST["compact"] = None
    if not COMPACT or (PROF and os.path.exists(os.path.join(PROF, "nocompact"))):   # lab A/B toggle
        return
    if getattr(_ST.get("dev"), "type", "") == "cuda":
        return          # CUDA punica kernels use LoRAKernelMeta's sorted indices of the full batch: gathering rows breaks them
    from vllm.config import CUDAGraphMode
    mode = k.get("cudagraph_runtime_mode")
    if mode is not None and mode != CUDAGraphMode.NONE:
        return
    lr = _ST.get("lrows")
    mgr = _ST.get("mgr")
    if lr is None or mgr is None:
        return
    rows, ids, total = lr
    n = len(rows)
    if n == 0 or n > COMPACT_FRAC * total:
        return
    pw = getattr(mgr, "punica_wrapper", None) or next(iter(getattr(mgr, "punica_wrapper_mapping", {}).values()), None)
    if pw is None:
        return
    slot = {lid: j for j, lid in enumerate(mgr.lora_index_to_id) if lid is not None}
    try:
        import numpy as np
        sl = np.fromiter((slot[i] for i in ids), dtype=np.int64, count=n)
    except KeyError:
        return
    dev = _ST["dev"]
    idx = torch.from_numpy(rows.astype(np.int64)).to(dev, non_blocking=True)
    s64 = torch.from_numpy(sl).to(dev, non_blocking=True)
    pw._token_lora_indices[:n].copy_(s64)
    tm = getattr(getattr(pw, "token_mapping_meta", None), "token_lora_mapping", None)
    if tm is not None:
        tm[:n].copy_(s64.to(tm.dtype))
    mgr._last_mapping = None          # the next step re-sends its full mapping
    _ST["compact"] = (idx, n, total, int(rows[-1]))
    _ST["n"]["compact_steps"] = _ST["n"].get("compact_steps", 0) + 1


def _patch_punica_meta():
    """The XPU punica path reads only token_lora_mapping from LoRAKernelMeta; the stock prepare_tensors also sorts,
    uniques and tests the mapping on the device (data-dependent sizes = host syncs every LoRA step). XPU: copy only."""
    try:
        from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    except Exception:  # noqa: BLE001
        return
    orig = LoRAKernelMeta.prepare_tensors

    def prepare_tensors(self, token_lora_mapping):
        if token_lora_mapping.device.type != "xpu":
            return orig(self, token_lora_mapping)
        n = token_lora_mapping.size(0)
        self.token_lora_mapping[:n].copy_(token_lora_mapping, non_blocking=True)
        self.no_lora_flag_cpu[0] = False   # update_metadata only gets here with LoRA tokens (CPU check)
    LoRAKernelMeta.prepare_tensors = prepare_tensors


def _patch_steps():
    from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin as M
    from vllm.lora.layers import LoRAMappingType
    orig = M._set_active_loras

    def _set_active_loras(self, prompt_lora_mapping, token_lora_mapping, lora_requests,
                          mapping_type=LoRAMappingType.LANGUAGE):
        if mapping_type != LoRAMappingType.LANGUAGE:
            return orig(self, prompt_lora_mapping, token_lora_mapping, lora_requests, mapping_type)
        with _LOCK:
            has = bool(lora_requests)
            _ST["step_ids"] = {r.lora_int_id for r in lora_requests if r}
            if has:
                _ST["last_use"] = time.time()
                lb = _ST.setdefault("last_by_id", {})
                for i in _ST["step_ids"]:
                    lb[i] = _ST["last_use"]
                _ST["n"]["lora_steps"] += 1
                _maybe_promote()
            _ST["in_step"] = True
            try:
                r = orig(self, prompt_lora_mapping, token_lora_mapping, lora_requests, mapping_type)
            finally:
                _ST["in_step"] = False
            _ST["step_lora"] = has and any(i > 0 for i in token_lora_mapping)
            _ST["lrows"] = None
            if _ST["step_lora"] and COMPACT:
                import numpy as np
                tm = np.asarray(token_lora_mapping, dtype=np.int64)
                rows = np.nonzero(tm > 0)[0]
                _ST["lrows"] = (rows, tm[rows], len(tm))
            return r
    M._set_active_loras = _set_active_loras

    import vllm.v1.worker.gpu_model_runner as gmr
    orig_sfc = gmr.set_forward_context

    @contextlib.contextmanager
    def set_forward_context(*a, **k):
        bd = k.get("batch_descriptor")
        if bd is not None and getattr(bd, "has_lora", False) and not _ST["step_lora"] and _LG["capturing"] is None:
            # a LoRA-graph shape but no LoRA token in this step: the base graph of the same shape is exact (a LoRA
            # graph would apply the adapter by the previous step's metadata)
            import dataclasses
            k["batch_descriptor"] = dataclasses.replace(bd, has_lora=False, num_active_loras=0)
        if _ST["step_lora"]:
            if PATH != "compiled":
                k["skip_compiled"] = True
            with _LOCK, orig_sfc(*a, **k) as c:   # the reaper never frees slots under a running LoRA forward
                _compact_on(k)
                try:
                    yield c
                finally:
                    _ST["compact"] = None
        else:
            with orig_sfc(*a, **k) as c:
                yield c
    gmr.set_forward_context = set_forward_context


def _patch_fused_mlp_gate():
    """The LoRA wrappers hide the exl3 attributes the fused decode MLP (exl3xpu _patch_fused_mlp) checks, which turned
    it off for all traffic. Non-LoRA steps (and every traced graph): fused op on the base layers, as in production.
    LoRA steps (eager): the plain MLP through the LoRA layers."""
    try:
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    except Exception:  # noqa: BLE001
        return
    prev = Qwen2MoeMLP.forward
    import torch.nn.functional as F

    def plain(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        if getattr(self, "expert_gate", None) is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out
        return out

    def _fused_ok(self):
        gu, dn = self.gate_up_proj, self.down_proj
        gb, db = getattr(gu, "base_layer", gu), getattr(dn, "base_layer", dn)
        return (getattr(Qwen2MoeMLP, "_exl3_mlp_patched", False)
                and getattr(gb, "exl3_cpp", False) and getattr(db, "exl3_cpp", False) and gb.exl3_cb == db.exl3_cb
                and len(gb.exl3_bounds) == 3 and len(db.exl3_bounds) == 2 and getattr(gb, "bias", None) is None
                and getattr(self, "expert_gate", None) is None)

    def fused_or_plain(self, x):
        gu, dn = self.gate_up_proj, self.down_proj
        gb, db = getattr(gu, "base_layer", gu), getattr(dn, "base_layer", dn)
        if (getattr(Qwen2MoeMLP, "_exl3_mlp_patched", False) and getattr(gb, "exl3_cpp", False)
                and getattr(db, "exl3_cpp", False) and gb.exl3_cb == db.exl3_cb and len(gb.exl3_bounds) == 3
                and len(db.exl3_bounds) == 2 and getattr(gb, "bias", None) is None
                and getattr(self, "expert_gate", None) is None):
            from exl3xpu import ops as _ops
            return torch.ops.exl3xpu_C.mlp_auto(x, gb.trellis, gb.suh, gb.svh, gb.exl3_shard_of_nb, gb.exl3_bounds,
                                                db.trellis, db.suh, db.svh, db.exl3_shard_of_nb, db.exl3_bounds,
                                                gb.exl3_K, db.exl3_K, gb.exl3_cb, _ops.SMALL_M_MAX, _ops.RECON_SLICE_N)
        return plain(self, x)

    MLPS = _ST.setdefault("mlps", [])

    def lora_mlp(m, x):
        hk = _HOOKS["mlp"]
        if hk is not None:
            r = hk(m, x)
            if r is not None:
                return r
        if LORA_MLP_ROWS > 0 and x.ndim == 2 and x.shape[0] > LORA_MLP_ROWS and _ST.get("lrows") is not None:
            try:
                if _fused_ok(m):
                    plan = _mlp_split_plan(_ST["lrows"], x.shape[0])
                    if plan is not None:
                        return _mlp_split(m, x, plain, fused_or_plain, LORA_MLP_ROWS, plan)
            except torch.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001
                if not _ST.get("mlp_split_err"):
                    _ST["mlp_split_err"] = True
                    _log("LoRA MLP row split failed (%r): all-plain path", e, level="warning")
        return plain(m, x)

    def forward(self, x):
        gu, dn = self.gate_up_proj, self.down_proj
        if getattr(gu, "base_layer", None) is None and getattr(dn, "base_layer", None) is None:
            return prev(self, x)
        if torch.compiler.is_compiling():
            idx = getattr(self, "_llmt_mlp_idx", None)
            if PATH == "compiled" and idx is not None and x.ndim == 2:
                return torch.ops.vllm.llmt_lora_mlp(x, idx)
            return fused_or_plain(self, x)     # eager path: the traced graph is the production model
        if _ST["step_lora"]:
            return lora_mlp(self, x)
        return fused_or_plain(self, x)
    Qwen2MoeMLP.forward = forward

    def _mlp(x: torch.Tensor, idx: int) -> torch.Tensor:
        """Opaque op: the whole MLP; LoRA step -> plain MLP through the LoRA layers, else the fused exl3 decode MLP
        (production's kernels; a graph captured without LoRA records exactly those)."""
        m = MLPS[idx]
        return lora_mlp(m, x) if _ST["step_lora"] else fused_or_plain(m, x)

    def _mlp_fake(x: torch.Tensor, idx: int) -> torch.Tensor:
        return x.new_empty((*x.shape[:-1], _ST["mlp_out"][idx]))
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="llmt_lora_mlp", op_func=_mlp, mutates_args=[], fake_impl=_mlp_fake)

    import vllm.lora.model_manager as mm
    orig_clm = mm.LoRAModelManager._create_lora_modules

    def _create_lora_modules(self):
        orig_clm(self)
        outs = _ST.setdefault("mlp_out", [])
        for mod in self.model.modules():
            if isinstance(mod, Qwen2MoeMLP) and getattr(mod, "_llmt_mlp_idx", None) is None:
                dn = getattr(mod.down_proj, "base_layer", mod.down_proj)
                mod._llmt_mlp_idx = len(MLPS)
                MLPS.append(mod)
                outs.append(int(getattr(dn, "output_size_per_partition", getattr(dn, "output_size", 0))))
        _log("%d MLPs routed through llmt_lora_mlp (path %s)", len(MLPS), PATH)
    mm.LoRAModelManager._create_lora_modules = _create_lora_modules


def _patch_host_cache():
    global _SHIM
    import vllm.lora.lora_model as lm
    import vllm.lora.model_manager as mm
    import vllm.lora.worker_manager as wm
    lm.PIN_MEMORY = False          # no private pinned host copy of any adapter
    mm.PIN_MEMORY = False
    _SHIM = _SafetensorsShim(lm.safetensors)
    lm.safetensors = _SHIM

    orig_load = wm.WorkerLoRAManager._load_adapter

    def _load_adapter(self, lora_request):
        from vllm.lora.utils import get_adapter_absolute_path
        _ST["worker"] = self
        path = get_adapter_absolute_path(lora_request.lora_path)
        f = os.path.abspath(os.path.join(path, "adapter_model.safetensors"))
        if not os.path.isfile(f):
            return orig_load(self, lora_request)   # .bin / tensorizer / missing: vLLM's own path and errors
        t0 = time.perf_counter()
        src = _open_source(path)
        _SHIM.cur[f] = src
        try:
            lora = orig_load(self, lora_request)
        except BaseException:
            src.close()
            raise
        finally:
            _SHIM.cur.pop(f, None)
        lora._llmt_src = src
        lora._llmt_req = lora_request
        _log("adapter %r parsed from %s in %.0f ms", lora_request.lora_name, src.what, (time.perf_counter() - t0) * 1e3)
        return lora
    wm.WorkerLoRAManager._load_adapter = _load_adapter

    orig_act = mm.LoRAModelManager.activate_adapter

    def _strip(lora):
        lora.loras = {}
        lora._llmt_stripped = True
        src = getattr(lora, "_llmt_src", None)
        lora._llmt_src = None
        if src is not None:
            _sync()
            if not src.close():
                _log("broker lease invalid after the copy: the adapter is reloaded on its next activation",
                     level="warning")
                return False
        return True

    def activate_adapter(self, lora_id):
        if lora_id in self._active_adapters:
            return orig_act(self, lora_id)
        with _LOCK:
            _ST["mgr"] = self
            _ST["last_use"] = time.time()
            _ST.setdefault("last_by_id", {})[lora_id] = time.time()
            N = len(self.lora_index_to_id)
            _ST["max_slots"] = N
            try:
                _ensure_slots(1)
            except _NoRoom as e:
                if not _ST.get("in_step"):
                    _ST["n"]["deferred"] = _ST["n"].get("deferred", 0) + 1
                    lz = self._registered_adapters.cache.get(lora_id)
                    if lz is not None and getattr(lz, "_llmt_req", None) is not None and not getattr(lz, "_llmt_stripped", False):
                        _strip(lz)      # no private host copy while it waits: re-read from the broker / disk later
                    _log("adapter id %d registered, not activated: %s; its requests wait for a slot", lora_id, e)
                    return False
                _ST["force_room"] = True
                try:
                    _ensure_slots(1)
                finally:
                    _ST["force_room"] = False
            if _ST["where"] in ("device", "host"):
                free = [j for j, i in enumerate(self.lora_index_to_id) if i is None]
                if not any(j < _ST["k"] for j in free):
                    if free and _grow_to(min(N, _ST["k"] + 1)):
                        pass
                    else:   # no room for another slot row: an adapter not in this step leaves its slot
                        busy = _ST.get("step_ids", set())
                        last = _ST["last_by_id"]
                        cand = sorted((last.get(i, 0), i) for i in self.lora_index_to_id[:_ST["k"]]
                                      if i is not None and i not in busy and i != lora_id)
                        if cand:
                            self.deactivate_adapter(cand[0][1])
                            _ST["n"]["slot_evictions"] = _ST["n"].get("slot_evictions", 0) + 1
                            _log("adapter id %d left its slot for id %d (no room for another slot row)",
                                 cand[0][1], lora_id)
                        elif not _ST.get("in_step"):
                            # API-time load (add_lora): no row without lending more KV; the scheduler lends when the
                            # request is queued. Register only.
                            _ST["n"]["deferred"] = _ST["n"].get("deferred", 0) + 1
                            lz = self._registered_adapters.cache.get(lora_id)
                            if lz is not None and getattr(lz, "_llmt_req", None) is not None and not getattr(lz, "_llmt_stripped", False):
                                _strip(lz)
                            _log("adapter id %d registered, not activated: every slot row busy; its requests wait", lora_id)
                            return False
                        elif not _grow_to(min(N, _ST["k"] + 1)):
                            _log("no LoRA slot row for adapter id %d and every slot is in use by this step: growing "
                                 "beyond the room", lora_id, level="warning")
                            saved_margin = globals()["MARGIN"]
                            try:
                                globals()["MARGIN"] = -(1 << 40)
                                _grow_to(min(N, _ST["k"] + 1))
                            finally:
                                globals()["MARGIN"] = saved_margin
                # vLLM takes the first free index: hide the rows we have not allocated
                hide = [j for j in range(_ST["k"], N) if self.lora_index_to_id[j] is None]
            else:
                hide = []
            for j in hide:
                self.lora_index_to_id[j] = -1
            lora = self._registered_adapters.cache.get(lora_id)
            req = getattr(lora, "_llmt_req", None)
            for attempt in (0, 1):
                if lora is not None and getattr(lora, "_llmt_stripped", False) and req is not None:
                    t0 = time.perf_counter()
                    new = _ST["worker"]._load_adapter(req)
                    self._create_merged_loras_inplace(new)
                    lora.loras, lora._llmt_src, lora._llmt_stripped = new.loras, new._llmt_src, False
                    _log("adapter %r re-materialized for activation in %.0f ms", req.lora_name,
                         (time.perf_counter() - t0) * 1e3)
                t0 = time.perf_counter()
                try:
                    r = orig_act(self, lora_id)     # same slot buffers: the LoRA graphs stay valid
                finally:
                    for j in hide:
                        if self.lora_index_to_id[j] == -1:
                            self.lora_index_to_id[j] = None
                _ST["n"]["activations"] += 1
                ok = _strip(lora) if (lora is not None and req is not None) else True
                if ok:
                    break
                self._deactivate_adapter(lora_id)   # lease lost during the copy: redo from a fresh source
                self._active_adapters.pop(lora_id, None)
            _log("adapter id %d active in %s slot %d of %d (activate %.0f ms)", lora_id, _ST["where"],
                 self.lora_index_to_id.index(lora_id) if lora_id in self.lora_index_to_id else -1, _ST["k"],
                 (time.perf_counter() - t0) * 1e3)
            return r
    mm.LoRAModelManager.activate_adapter = activate_adapter


def _patch_kv_layout():
    if not LEND:
        return
    from llmt_kv import vllm_hooks as H

    def kv_caches_initialized(runner, kv, kv_cache_config):
        try:
            _kv_layout_init(kv, int(kv_cache_config.num_blocks))
        except Exception as e:  # noqa: BLE001
            _KVL.update(ok=False, why=repr(e))
            _log("KV-block lending off: %r", e, level="warning")

    H.register("model_runner.kv_caches_initialized", kv_caches_initialized)


def _patch_memory_hooks():
    from llmt_kv import vllm_hooks as H

    def available_kv_memory(r, worker):
        _ST["act_reserve"] = int(getattr(worker, "peak_activation_memory", 0) or 0)
        _log("activation reserve from the profile run: %.2f GiB; LoRA slot layers %d (%.1f MiB when allocated)",
             _ST["act_reserve"] / 2**30, len(_ST["layers"]), sum(_layer_bytes(l) for l in _ST["layers"]) / 2**20)
        return r

    def warmed_up(worker):
        with _LOCK:
            if _ST["where"] != "off":
                _release("warm-up done")
            _sync()
            _ST["base_alloc"] = _acc().memory_allocated()
            _ST["ready"] = True
        threading.Thread(target=_reaper, name="llmt-lora-reaper", daemon=True).start()
        _log("ready (mode %s, ttl %.0f s, margin %d MiB, broker %s)", MODE, TTL, MARGIN >> 20,
             ("%s ns %r" % (_ST["broker"].sock, NS_NAME)) if _ST["broker"] else "off")

    H.register("worker.available_kv_memory", available_kv_memory)
    H.register("worker.warmed_up", warmed_up)


def _setup_broker():
    if BROKER == "off":
        return
    sock = BROKER
    if sock == "auto":
        sock = "/run/llmt-kv/broker.sock"
    eid = int(_ENV("EXL3_LORA_ENGINE_ID", _ENV("LLMT_KV_ENGINE_ID", "0")) or 0)
    _ST["broker"] = _Broker(sock, eid, NS_NAME)


# ------------------------------------------------------------------------------------------ resolver (API server)

def stable_id(name):
    return 1 + int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "little") % 0x7FFFFFF0


def find_adapter(root, name, max_rank=None):
    """<root>/<name> if it is a usable LoRA adapter dir, else None (also for names that try to leave root)."""
    if not name or "/" in name or "\\" in name or name.startswith(".") or "\0" in name:
        return None
    path = os.path.join(root, name)
    cfg = os.path.join(path, "adapter_config.json")
    try:
        with open(cfg) as f:
            c = json.load(f)
    except (OSError, ValueError):
        return None
    if c.get("peft_type") != "LORA":
        return None
    if max_rank is not None and int(c.get("r", 0) or 0) > max_rank:
        return None
    if not (os.path.isfile(os.path.join(path, "adapter_model.safetensors"))
            or os.path.isfile(os.path.join(path, "adapter_model.bin"))):
        return None
    return path


def _register_resolver():
    try:
        from vllm.lora.request import LoRARequest
        from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
    except Exception:  # noqa: BLE001
        return
    max_rank = None
    try:
        max_rank = int(_ENV("EXL3_LORA_MAX_RANK", "0")) or None
    except ValueError:
        pass

    class DirResolver(LoRAResolver):
        async def resolve_lora(self, base_model_name, lora_name):
            path = find_adapter(LORA_DIR, lora_name, max_rank)
            if path is None:
                return None
            return LoRARequest(lora_name=lora_name, lora_int_id=stable_id(lora_name), lora_path=path)
    if os.path.isdir(LORA_DIR):
        LoRAResolverRegistry.register_resolver("llmt-lora dir", DirResolver())


# ------------------------------------------------------------------------------------------ lab profiling
# EXL3_LORA_PROF=<dir> (lab only): a file <dir>/arm with "N label" makes the next N execute_model calls run under
# torch.profiler (CPU + XPU); tables of op names / counts / times (no request data) go to <dir>/<label>.txt, then the
# arm file is removed. <dir>/nograph present = every step dispatched without cudagraphs (base no-graph speed).
PROF = _ENV("EXL3_LORA_PROF", "")


def _patch_prof():
    if not PROF:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.forward_context import BatchDescriptor
    from vllm.config import CUDAGraphMode
    orig_disp = CudagraphDispatcher.dispatch

    def dispatch(self, num_tokens, *a, **k):
        if os.path.exists(os.path.join(PROF, "nograph")):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
        return orig_disp(self, num_tokens, *a, **k)
    CudagraphDispatcher.dispatch = dispatch
    st = {"p": None, "left": 0, "label": "", "t0": 0.0, "steps": 0, "lora": 0}
    orig_exec = GPUModelRunner.execute_model

    def _finish():
        p = st["p"]
        _sync()
        p.stop()
        wall = time.perf_counter() - st["t0"]
        ka = p.key_averages()
        dev_attr = "self_device_time_total" if hasattr(next(iter(ka), None), "self_device_time_total") else "self_xpu_time_total"
        n_steps = max(1, st["steps"])
        kern = 0
        dev_us = 0.0
        try:
            for e in p.events():
                dt = getattr(e, "device_type", None)
                if dt is not None and "CPU" not in str(dt):
                    kern += 1
                    dev_us += float(getattr(e, "device_time", 0.0) or getattr(e, "self_device_time_total", 0.0) or 0.0)
        except Exception as ex:  # noqa: BLE001
            kern = -1
            _log("prof: event walk failed %r", ex, level="warning")
        out = os.path.join(PROF, st["label"] + ".txt")
        with open(out, "w") as f:
            f.write("label %s: %d execute_model steps (%d with LoRA tokens), wall %.1f ms (%.2f ms/step), device events "
                    "%d (%.1f/step), device time %.1f ms (%.2f ms/step)\n\n" % (
                        st["label"], n_steps, st["lora"], wall * 1e3, wall * 1e3 / n_steps, kern, kern / n_steps,
                        dev_us / 1e3, dev_us / 1e3 / n_steps))
            for key, title in ((dev_attr, "by self device time"), ("count", "by count"),
                               ("self_cpu_time_total", "by self CPU time")):
                try:
                    f.write("=== top 30 %s\n" % title)
                    f.write(ka.table(sort_by=key, row_limit=30, max_name_column_width=90) + "\n\n")
                except Exception as ex:  # noqa: BLE001
                    f.write("(table %s failed: %r)\n" % (key, ex))
        st["p"] = None
        _log("prof: wrote %s (%d steps, %.2f ms/step wall)", out, n_steps, wall * 1e3 / n_steps)

    def execute_model(self, *a, **k):
        if st["p"] is not None:
            if st["left"] <= 0:
                _finish()
            else:
                st["left"] -= 1
                st["steps"] += 1
                st["lora"] += 1 if _ST["step_lora"] else 0
        elif os.path.exists(os.path.join(PROF, "arm")):
            try:
                n, label = open(os.path.join(PROF, "arm")).read().split()[:2]
                os.remove(os.path.join(PROF, "arm"))
                from torch.profiler import ProfilerActivity, profile
                acts = [ProfilerActivity.CPU] + ([ProfilerActivity.XPU] if hasattr(ProfilerActivity, "XPU")
                                                  and _ENV("EXL3_LORA_PROF_CPU_ONLY", "0") != "1" else [])
                _sync()
                st.update(p=profile(activities=acts, record_shapes=False), left=int(n) - 1, label=label, steps=1,
                          lora=1 if _ST["step_lora"] else 0)
                st["p"].start()
                st["t0"] = time.perf_counter()
            except Exception as ex:  # noqa: BLE001
                _log("prof: could not start: %r", ex, level="warning")
                st["p"] = None
        return orig_exec(self, *a, **k)
    GPUModelRunner.execute_model = execute_model
    _log("lab profiling hooks on (%s)", PROF)



ADMIT = None


def set_admission_policy(fn):
    global ADMIT
    ADMIT = fn
    if _ST.get("installed"):
        from llmt_kv import vllm_hooks as H
        return H.register("scheduler.admit", fn)
    return True

LEND_ONLY = _ENV("EXL3_LEND_ONLY", "0") == "1"


def _hook_mode():
    try:
        from llmt_kv import vllm_hooks as H
        return "vLLM hooks: %s" % ("native (echowarp/vllm extension hooks)" if H.native() is not None else
                                   "fallback (one-time method wrappers on stock vLLM)")
    except Exception as e:  # noqa: BLE001
        return "vLLM hooks: unknown (%r)" % (e,)


def register():
    """vllm.general_plugins entry point (every vLLM process)."""
    if _ST.get("installed"):
        return
    if not ON and LEND_ONLY and VIS_LEND:
        # an engine without --enable-lora: the vision tower is the only tenant
        _ST["installed"] = True
        _patch_load_fence()
        _patch_scheduler()
        _patch_block_pool()
        _patch_memory_hooks()
        _patch_kv_layout()
        _log("installed: KV lending for the vision tower only (no LoRA); %s", _hook_mode())
        return
    if not ON:
        return
    _ST["installed"] = True
    _patch_load_fence()
    _register_resolver()
    _patch_device_helper()
    _patch_lazy_slots()
    _patch_no_dummy_loras()
    _patch_steps()
    _patch_punica_meta()
    _load_kernels()
    _patch_lgraph()
    _patch_scheduler()
    _patch_block_pool()
    _patch_fused_mlp_gate()
    _patch_host_cache()
    _patch_memory_hooks()
    _patch_kv_layout()
    _patch_prof()
    _setup_broker()
    _log("installed (mode %s, ttl %.0f s, dir %s); %s", MODE, TTL, LORA_DIR, _hook_mode())
