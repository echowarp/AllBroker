"""Weight keeper loader for vLLM (Intel Arc / XPU and NVIDIA CUDA): processed model weights are kept resident in device
memory by a holder process (llmt-keeper) across engine restarts, so a restart attaches to them instead of reloading
from disk, and two engines on one GPU can share a single weight copy.

EXL3_WHX=<keeper socket> turns it on. Each BaseModelLoader.load_model call (target model, MTP drafter) gets a key from
the model path, the call index, the image and every EXL3_* / VLLM_* setting that changes weight processing.

  export  (keeper has no entry for the key): normal load (disk -> GPU -> process_weights_after_loading). Then every
          device tensor reachable from the model (parameters, buffers, module attributes, nested lists/dicts/objects,
          plus the backend's extra roots) is moved onto fd-exportable device memory (chunks of EXL3_WHX_CHUNK_MB) with
          Tensor.set_, so the model keeps working on it unchanged. The fds + a manifest go to the keeper.
  attach  (keeper has a matching entry): the model is built on the meta device and "loaded" from meta tensors of the
          checkpoint's shapes (no disk read), processed, then every reachable tensor is replaced by a view of the
          imported memory, matched by path and checked against the manifest (shape, stride, dtype, checksum).
          Any mismatch or error drops the keeper's entries and stops the engine; the next start loads and exports.
The memory is shared, not copied: with EXL3_WHX_SHARE=1 a second live engine attaches to the first engine's weights.

Device backends (one per vendor; the protocol, key, manifest and checksums are shared):
  xpu   Level Zero dma-buf export / import (whx.so from weights/xpu/whx_ops.sycl, EXL3_WHX_LIB).
  cuda  CUDA VMM allocations with POSIX-fd shareable handles (libcuda over ctypes, no build step). The exl3cuda plugin's
        keeper hooks (whx_registry / whx_finish in exl3cuda/vllm_plugin.py) expose the EXL3 linears, which live in a
        module-level table, and rebuild what holds raw pointers after the move; without those hooks the loader stays off.
The backend follows the package this file is installed in (exl3xpu / exl3cuda); EXL3_WHX_BACKEND=xpu|cuda overrides.
EXL3_WHX_EXCLUSIVE=1 (production: one keeper per GPU): a keeper miss on the target model drops every held entry before
the normal load, so a changed setting never loads next to a stale 15 GiB copy.
EXL3_WHX_READONLY=1 (lab engines next to a production keeper): attach only; a miss stops the engine, and nothing is
ever put or dropped.
EXL3_WHX_DIR: where the side files go (env record, CPU / private tensors, DRAFT-DYN host table); default: the socket's
directory. EXL3_WHX_* settings are never part of the key.
Tensor paths are the BASE model's canonical module names: a module reached through any other route
(vLLM LoRA's lora_manager.modules[<name>], a wrapper's base_layer) gets the name named_modules() gives it with every
".base_layer" removed, and the walk depth follows that name, so the same weights get the same paths whether or not
LoRA wrapped the layers (export runs after the wrap, attach before it). Entries recorded with the manager's alias
paths (".lora_manager.modules[...]", older versions of this loader) are matched through the same normalisation.
EXL3_WHX_STALL_S (default 240, 0 = off): load watchdog. A load / attach / export that makes no progress (weight
tensors read, chunks moved) for that long logs "WHX: LOAD STALL ..." with every thread's stack and exits the engine
(exit code 70) instead of hanging until a supervisor's health timeout (seen on Intel Arc: a copy-engine page
fault and exec-queue reset during a cold load left EngineCore waiting forever).
"""
import ctypes
import faulthandler
import hashlib
import importlib
import json
import os
import sys
import socket
import threading
import time

import torch

_LIB = os.environ.get("EXL3_WHX_LIB", "/opt/exl3xpu/whx.so")   # XPU backend: build weights/xpu/whx_ops.sycl
_CHUNK = int(os.environ.get("EXL3_WHX_CHUNK_MB", "1024")) << 20
_ALIGN = 1 << 16
_calls = [0]
_installed = [False]
_CFG_SUB = (".vllm_config", ".compilation_config", "static_forward_context", ".quant_config")
_SKIP_SUB = ("[dyn]",) + _CFG_SUB      # [dyn]: DRAFT-DYN per-row runtime buffers, private per engine (saved by value)
# vLLM LoRA (--enable-lora): the runner wraps the layers after the loader returns, so an attach never sees the wrapper's
# adapter slots and index buffers (runtime state, refilled per adapter): never held, and not part of the path set
_LORA_SUB = (".lora_a_stacked", ".lora_b_stacked", ".lora_bias_stacked", ".punica_wrapper", ".lora_config",
             ".lora_manager")   # the manager (model.lora_manager) only aliases the model's modules
_LM_RE = None
_PENDING = []   # (model, key, need): exports deferred until the model runner has loaded and tied every model
_SKIP = (".visual",)   # not held: the vision tower is built and loaded normally (lazyvis moves it to the host)
_PATHS = []   # path of each tensor _walk yielded, same order
_REG = []    # this process's held buffers: (key, chunk index, data_ptr, nbytes, buffer) for cross-model references


class _NeedCPU(RuntimeError):
    """attach found weight-derived CPU tensors the export did not save: reload once and export them too."""
    def __init__(self, idx):
        super().__init__(f"{len(idx)} weight-derived CPU tensors not saved"); self.idx = idx


def _log(msg):
    print(f"WHX: {msg}", flush=True)


def _dir():
    return os.environ.get("EXL3_WHX_DIR") or os.path.dirname(os.environ.get("EXL3_WHX", "/tmp/x"))


def _plugin():
    """The engine plugin whose keeper hooks this loader uses: the package this file is installed in when that is
    exl3xpu / exl3cuda, else (installed as llmt_kv.whx_loader) the plugin of the device backend."""
    pkg = (__package__ or "").split(".")[0]
    if pkg not in ("exl3xpu", "exl3cuda"):
        pkg = {"xpu": "exl3xpu", "cuda": "exl3cuda"}[_bk().name]
    return importlib.import_module(pkg + ".vllm_plugin")


# ------------------------------------------------------------------ device backends
class XPUBackend:
    """Intel Arc: Level Zero dma-buf fds from whx.so (torch.ops.whx)."""
    name = dev = "xpu"
    mods = ("exl3xpu", "vllm", "__main__")     # object types whose attributes the walk follows
    skip = ()                                  # extra path substrings that are not held
    worker = "vllm.v1.worker.xpu_worker"
    margin_mb = 512                            # attach KV margin (measured: attached total = normal + 0.49 GiB)

    def check(self):
        pass

    def init(self):
        torch.ops.load_library(_LIB)

    def acc(self):
        return torch.xpu

    def export_alloc(self, n):
        return torch.ops.whx.export_alloc(n)

    def import_fd(self, fd, n):
        return torch.ops.whx.import_fd(fd, n)

    def roots(self, model):
        return None

    def finish(self, model):
        pass

    def dyn_dump(self, host):
        return [(list(k), {"svh": v["svh"], "table": [torch.from_numpy(x) for x in v["table"]]}) for k, v in host.items()]

    def dyn_load(self, items, host):
        for k, v in items:
            kk = tuple(tuple(x) if isinstance(x, list) else x for x in k)
            host.setdefault(kk, {"svh": v["svh"], "table": tuple(x.numpy() for x in v["table"])})


class _CuLoc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CuFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _CuProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", _CuLoc),
                ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", _CuFlags)]


class _CuAccess(ctypes.Structure):
    _fields_ = [("location", _CuLoc), ("flags", ctypes.c_int)]


class _CuArr:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 3}


class CUDABackend:
    """NVIDIA: CUDA VMM (cuMemCreate with POSIX-fd shareable handles, mapped into torch's primary context). The mapping
    lives as long as the process (weights are never freed); the keeper's fd keeps the physical memory alive."""
    name = dev = "cuda"
    mods = ("exl3cuda", "exllamav3", "vllm", "__main__")
    # draft heads (+ TAIL-FUSE maps, DRAFT-DYN buffers): rebuilt by the plugin's whx_finish on every attach, not held
    skip = (".exl3_draft", "._exl3_dw_head")
    worker = "vllm.v1.worker.gpu_worker"
    margin_mb = 0                              # measured: attached total = normal + 0.03 GiB (512 cost 6.6% KV)
    POSIX_FD = 1
    _cu = None
    _keep = []   # (va, size, handle)

    def init(self):
        if CUDABackend._cu is None:
            CUDABackend._cu = ctypes.CDLL("libcuda.so.1")

    def check(self):
        p = _plugin()
        if not (hasattr(p, "whx_registry") and hasattr(p, "whx_finish")):
            raise RuntimeError(f"{p.__name__} has no keeper hooks (whx_registry / whx_finish): the EXL3 linears "
                               "would not be held")

    def acc(self):
        return torch.cuda

    @staticmethod
    def _ck(r, what):
        if r != 0:
            raise RuntimeError(f"{what} failed: CUresult {r}")

    def _device(self):
        torch.cuda.init(); torch.empty(1, device="cuda")      # primary context current
        return torch.cuda.current_device()

    def _prop(self, dev):
        p = _CuProp(); p.type = 1; p.requestedHandleTypes = self.POSIX_FD; p.location = _CuLoc(1, dev)
        return p

    def _round(self, n, dev):
        g = ctypes.c_size_t()
        self._ck(self._cu.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(self._prop(dev)), 0), "granularity")
        return (n + g.value - 1) // g.value * g.value

    def _map(self, handle, n, dev):
        cu = self._cu; va = ctypes.c_uint64()
        self._ck(cu.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(n), ctypes.c_size_t(0), ctypes.c_uint64(0),
                                        ctypes.c_uint64(0)), "reserve")
        self._ck(cu.cuMemMap(va, ctypes.c_size_t(n), ctypes.c_size_t(0), handle, ctypes.c_uint64(0)), "map")
        acc = _CuAccess(_CuLoc(1, dev), 3)
        self._ck(cu.cuMemSetAccess(va, ctypes.c_size_t(n), ctypes.byref(acc), ctypes.c_size_t(1)), "access")
        self._keep.append((va.value, n, handle.value))
        return torch.as_tensor(_CuArr(va.value, n), device=f"cuda:{dev}")

    def export_alloc(self, n):
        dev = self._device(); n = self._round(n, dev); h = ctypes.c_uint64()
        self._ck(self._cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(n), ctypes.byref(self._prop(dev)),
                                      ctypes.c_uint64(0)), "create")
        fd = ctypes.c_int()
        self._ck(self._cu.cuMemExportToShareableHandle(ctypes.byref(fd), h, self.POSIX_FD, ctypes.c_uint64(0)), "export")
        return self._map(h, n, dev), fd.value

    def import_fd(self, fd, n):
        dev = self._device(); n = self._round(n, dev); h = ctypes.c_uint64()
        self._ck(self._cu.cuMemImportFromShareableHandle(ctypes.byref(h), ctypes.c_void_p(fd), self.POSIX_FD), "import")
        return self._map(h, n, dev)

    def roots(self, model):
        reg = _plugin().whx_registry(model)
        return {k: v for k, v in reg.items() if not ("." + k).startswith(_SKIP)} or None

    def finish(self, model):
        _plugin().whx_finish(model)

    def dyn_dump(self, host):
        return [(list(k), [torch.from_numpy(x) for x in v]) for k, v in host.items()]   # (V, K, gs, svh sum) -> (hq, hs, hz)

    def dyn_load(self, items, host):
        for k, v in items:
            host.setdefault(tuple(k), tuple(x.numpy() for x in v))


_BACKENDS = {"xpu": XPUBackend, "cuda": CUDABackend}
_BK = [None]


def _bk():
    if _BK[0] is None:
        name = os.environ.get("EXL3_WHX_BACKEND") or {"exl3xpu": "xpu", "exl3cuda": "cuda"}.get(
            (__package__ or "").split(".")[0]) or ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cuda")
        _BK[0] = _BACKENDS[name]()
    return _BK[0]


# ------------------------------------------------------------------ keeper protocol (unix socket, SCM_RIGHTS)
def _rpc(req, fds=()):
    s = socket.socket(socket.AF_UNIX)
    s.connect(os.environ["EXL3_WHX"])
    data = json.dumps(req).encode()
    socket.send_fds(s, [len(data).to_bytes(8, "little")], list(fds)); s.sendall(data)
    hdr, rfds, _, _ = socket.recv_fds(s, 8, 256)
    n = int.from_bytes(hdr, "little"); body = b""
    while len(body) < n:
        body += s.recv(n - len(body))
    s.close()
    return json.loads(body), rfds


# ------------------------------------------------------------------ traversal
def _canon_path(p):
    """Base-model name of a recorded path: '.lora_manager.modules[<name>]<rest>' -> '.<name><rest>', no '.base_layer'
    (older entries recorded LoRA-wrapped modules under the manager's alias)."""
    global _LM_RE
    if _LM_RE is None:
        import re
        _LM_RE = re.compile(r"^((?:<roots>)?)\.lora_manager\.modules\[([^\]]+)\]")
    p = _LM_RE.sub(lambda m: m.group(1) + "." + m.group(2), p)
    while ".base_layer." in p or p.endswith(".base_layer"):
        p = p.replace(".base_layer.", ".", 1) if ".base_layer." in p else p[:-len(".base_layer")]
    return p


def _canon_names(model):
    """id(module) -> canonical name (named_modules, '.base_layer' removed); the root is ''."""
    names = {}
    try:
        for n, m in model.named_modules():
            names.setdefault(id(m), _canon_path("." + n)[1:] if n else "")
    except Exception:  # noqa
        pass
    return names


def _walk(model, private=False):
    """Yield (setter, tensor) for every tensor reachable from the model (and the backend's extra roots, under the path
    prefix <roots>), in a deterministic order; setter(v) puts v in the tensor's place (dict/list item, module
    parameter/buffer/attribute, or an element of a tuple, rebuilt)."""
    seen = set()
    _PATHS.clear()
    bk = _bk()
    sub = (_CFG_SUB if private else _SKIP_SUB) + _LORA_SUB + bk.skip

    canon = _canon_names(model) if isinstance(model, torch.nn.Module) else {}

    def item(c, k):
        return lambda v: c.__setitem__(k, v)

    def visit(obj, depth, get, put, path=''):
        if isinstance(obj, torch.nn.Module) and id(obj) in canon and id(obj) not in seen and path != "<roots>":
            cn = canon[id(obj)]              # same module, same name and depth, whatever route reached it
            path, depth = ("." + cn if cn else ""), (cn.count(".") + 1 if cn else 0)
        if depth > 6 or id(obj) in seen or path.startswith(_SKIP) or any(x in path for x in sub):
            return
        if isinstance(obj, torch.nn.Module):
            seen.add(id(obj))
            for name in list(obj._parameters):
                if obj._parameters[name] is not None:
                    _PATHS.append(path + '.' + name)
                    yield item(obj._parameters, name), obj._parameters[name]
            for name in list(obj._buffers):
                if obj._buffers[name] is not None:
                    _PATHS.append(path + '.' + name)
                    yield item(obj._buffers, name), obj._buffers[name]
            for name, v in list(vars(obj).items()):
                if name in ("_parameters", "_buffers", "_modules"):
                    continue
                if isinstance(v, torch.Tensor):
                    _PATHS.append(path + '.' + name); yield item(obj.__dict__, name), v
                elif not isinstance(v, torch.nn.Module):
                    yield from visit(v, depth + 1, (lambda d=obj.__dict__, n=name: d[n]), item(obj.__dict__, name), path + '.' + name)
            for name, m in obj._modules.items():
                if m is not None:     # a LoRA wrapper's base_layer is the loader's module: same path and depth as without LoRA
                    yield from visit(m, depth if name == "base_layer" else depth + 1, None, None,
                                     path if name == "base_layer" else path + '.' + name)
        elif isinstance(obj, (dict, list)):
            seen.add(id(obj))
            for k in (list(obj) if isinstance(obj, dict) else range(len(obj))):
                v = obj[k]
                if isinstance(v, torch.Tensor):
                    _PATHS.append(f"{path}[{k}]"); yield item(obj, k), v
                else:
                    yield from visit(v, depth + 1, (lambda c=obj, kk=k: c[kk]), item(obj, k), f"{path}[{k}]")
        elif isinstance(obj, tuple):
            for i, v in enumerate(obj):
                def tput(x, i=i):
                    t = get(); put(type(t)(*(x if j == i else e for j, e in enumerate(t)))
                                    if hasattr(t, "_fields") else tuple(x if j == i else e for j, e in enumerate(t)))
                if isinstance(v, torch.Tensor):
                    _PATHS.append(f"{path}({i})"); yield (tput if put is not None else None), v
                else:
                    yield from visit(v, depth + 1, (lambda i=i: get()[i]) if get else None, None, f"{path}({i})")
        elif hasattr(obj, "__dict__") and type(obj).__module__.split(".")[0] in bk.mods:
            seen.add(id(obj))
            for k, v in list(vars(obj).items()):
                if isinstance(v, torch.Tensor):
                    _PATHS.append(path + '.' + k); yield item(obj.__dict__, k), v
                else:
                    yield from visit(v, depth + 1, (lambda d=obj.__dict__, kk=k: d[kk]), item(obj.__dict__, k), path + '.' + k)

    yield from visit(model, 0, None, None)
    roots = bk.roots(model)
    if roots:
        yield from visit(roots, 0, None, None, "<roots>")


def _files_identity(model_config):
    """staleness check: the source files' identity (inode, size, mtime ns)."""
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(str(model_config.model), "*.safetensors"))
                    + glob.glob(os.path.join(str(model_config.model), "*.json"))):
        st = os.stat(f); out.append([os.path.basename(f), st.st_ino, st.st_size, st.st_mtime_ns])
    return out


def _tsum(t):
    b = t.detach().reshape(-1).view(torch.uint8) if t.is_contiguous() else t.detach().contiguous().reshape(-1).view(torch.uint8)
    n = b.numel() // 4 * 4
    return int(b[:n].view(torch.int32).to(torch.int64).sum().item()) if n else 0


def _csum(bufs):
    """per-chunk checksum (int32 sum on the device)."""
    return [int(b[: b.numel() // 4 * 4].view(torch.int32).to(torch.int64).sum().item()) for b in bufs]


_KEY_ENV = [None]


def _env_file(key):
    return os.path.join(_dir(), f"{key}-env.json")


def _miss_report(key):
    """On a keeper miss, log which key settings differ from each held copy (a miss reloads
    15 GiB from disk, or starves this engine's KV next to the held copy; every miss is a key bug to fix)."""
    try:
        held, _ = _rpc({"op": "list"})
        if not held:
            _log(f"keeper empty: first load for key {key}"); return
        mine = _KEY_ENV[0] or {}
        for k2 in held:
            try:
                other = json.load(open(_env_file(k2)))
            except Exception:  # noqa
                _log(f"MISS key {key} vs held {k2}: held copy has no env record"); continue
            diff = sorted(set(mine) | set(other))
            diff = [f"{k}: held={other.get(k)!r} now={mine.get(k)!r}" for k in diff if mine.get(k) != other.get(k)]
            _log(f"MISS key {key} vs held {k2}: " + ("; ".join(diff) if diff else "same env (model / image / files differ)"))
    except Exception as e:  # noqa
        _log(f"miss report failed ({e!r})")


def _key_env(environ):
    """The settings that are part of the key: every EXL3_* / VLLM_* variable except runtime-only ones."""
    return {k: v for k, v in sorted(environ.items())
            if (k.startswith("EXL3_") or k.startswith("VLLM_")) and not k.startswith(("EXL3_WHX", "EXL3_SCHED_", "VLLM_SERVER_DEV_MODE", "EXL3_DRAFT_WIN", "EXL3_DRAFT_GRAPH", "EXL3_SPEC_",
                # runtime-only settings that never change a weight tensor (each K-smoothing table
                # otherwise exported another 15 GiB copy into the keeper)
                "EXL3_KSMOOTH", "EXL3_I8_SPLITS", "EXL3_I4_", "EXL3_KV_DTYPE_TAG", "EXL3_STATS_DIR", "EXL3_OFFLOAD_DEBUG",
                "EXL3_SAMPLE_REQUESTS", "EXL3_TAIL_", "EXL3_LOOKUP_STATS", "EXL3_STEP_STATS", "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
                # tier / layout / KSV settings
                "EXL3_KV_COPY_PACE", "EXL3_KV_RING", "VLLM_KV_CACHE_LAYOUT", "EXL3_KSV",
                # tier serving-policy settings
                "EXL3_BEGINNINGS_", "EXL3_BRANCH_", "EXL3_CACHE_STATS", "EXL3_JUNCTION_", "EXL3_KV_LAZY_", "EXL3_KV_RAM_GUARD",
                "EXL3_SESSION_", "EXL3_SHARED_", "EXL3_SUBBLOCK_", "EXL3_STATS_UID",
                # state handling, not weights (keeps one held copy instead of evicting it on a change; a changed
                # tensor set fails the attach loudly)
                "EXL3_GDN_RECOMPUTE", "EXL3_GDN_RC_",
                # LoRA hot loading / KV lending / lazy-vision runtime settings: adapters, lend policy
                # and tower residency never change a base weight; a changed tensor set still fails the attach loudly
                "EXL3_LORA_", "EXL3_LEND_", "EXL3_LAZY_VISION_TTL", "EXL3_LAZY_VISION_MARGIN", "EXL3_LAZY_VISION_BROKER",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
                # compile-cache location / switch: a launcher may hash its mounted files into VLLM_CACHE_ROOT, which
                # would force a full load on every mount change; no weight byte depends on it
                "VLLM_CACHE_ROOT", "VLLM_DISABLE_COMPILE_CACHE"))}


def _key(model_config, idx):
    env = _key_env(os.environ)
    _KEY_ENV[0] = env
    s = json.dumps([3, str(model_config.model), idx, str(model_config.dtype), os.environ.get("LLMT_IMAGE", ""), env,
                    _files_identity(model_config)])
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ export
def _export(model, key, need_cpu=()):
    import torch.nn as nn  # noqa
    bk = _bk(); acc = bk.acc(); dev = bk.dev
    bk.init()
    allents = list(_walk(model))
    bad = sorted({t.device.type for _, t in allents} - {dev, "cpu"})
    if bad:
        raise RuntimeError(f"tensors on {bad}")
    ents = [(f, t) for f, t in allents if t.device.type == dev]
    kinds = [t.device.type for _, t in allents]
    paths = list(_PATHS)
    priv = [(pp, t) for (f, t), pp in zip(list(_walk(model, private=True)), list(_PATHS)) if "[dyn]" in pp]
    priv_path = os.path.join(_dir(), f"{key}-priv.pt")
    torch.save({pp: t.detach().cpu() for pp, t in priv}, priv_path)
    _log(f"{len(priv)} private runtime buffers saved by value ({sum(t.numel() * t.element_size() for _, t in priv) / 2**20:.1f} MiB)")
    cpus = [t for _, t in allents if t.device.type == "cpu"]
    need = sorted(set(need_cpu))
    cpu_path = os.path.join(_dir(), f"{key}-cpu.pt")
    if need:
        torch.save({i: cpus[i].detach() for i in need}, cpu_path)
    _log(f"{len(cpus)} CPU tensors in the model ({sum(t.numel() * t.element_size() for t in cpus) / 2**20:.0f} MiB); "
         f"{len(need)} weight-derived saved ({sum(cpus[i].numel() * cpus[i].element_size() for i in need) / 2**20:.0f} MiB)")
    # group by storage, in address order (so the allocator's segments empty out as chunks move)
    stor, users = {}, {}
    for f, t in ents:                     # one entry per tensor object (aliases share it), offset taken before set_
        st = t.untyped_storage(); p = st.data_ptr()
        stor.setdefault(p, st); users.setdefault(p, {})[id(t)] = (t, t.storage_offset())
    order = sorted(stor)
    chunks, bufs, fds, where = [], [], [], {}
    refs = {}                                    # storage ptr -> (key, ci, byte offset) inside an earlier export

    def held(p):
        for k2, c2, base, nb, _ in _REG:
            if base <= p < base + nb:
                return k2, c2, p - base
        return None
    for p in list(order):
        h = held(p)
        if h:
            refs[p] = h; order.remove(p)
    groups, cur, off = [], [], 0
    for p in order:
        nb = stor[p].nbytes(); sz = (nb + _ALIGN - 1) // _ALIGN * _ALIGN
        if cur and off + sz > _CHUNK:
            groups.append((cur, off)); cur, off = [], 0
        cur.append((p, off)); off += sz
    if cur:
        groups.append((cur, off))
    with torch.no_grad():
        for ci, (members, nbytes) in enumerate(groups):   # one chunk at a time: peak = model + one chunk
            _tick()
            b, fd = bk.export_alloc(max(nbytes, _ALIGN)); bufs.append(b); fds.append(fd); chunks.append(nbytes)
            for p, o in members:
                nb = stor[p].nbytes()
                b[o:o + nb].copy_(torch.empty(0, dtype=torch.uint8, device=dev).set_(stor[p], 0, (nb,), (1,)))
            acc.synchronize()
            for p, o in members:
                for t, so in users[p].values():
                    es = t.element_size(); assert o % es == 0
                    where[id(t)] = (ci, o // es + so)
                    t.set_(b.untyped_storage(), o // es + so, t.shape, t.stride())
                del stor[p]
            acc.empty_cache()
    for p, (k2, c2, boff) in refs.items():      # shared with an earlier model (MTP drafter <- target): reference it
        for t, so in users[p].values():
            es = t.element_size()
            where[id(t)] = ("ref", k2, c2, boff // es + so)
    man = [[*where[id(t)], list(t.shape), list(t.stride()), str(t.dtype)] for f, t in ents]
    tsums = [_tsum(t) for f, t in ents]
    for ci, b in enumerate(bufs):
        _REG.append((key, ci, b.data_ptr(), b.numel(), b))
    acc.synchronize(); acc.empty_cache()
    model._whx_bufs = bufs              # keep this process's mapping alive
    tot = sum(chunks)
    dyn = ""
    try:                                    # DRAFT-DYN host int4 lm_head (numpy, built from the weights): keep it too
        vp = _plugin()
        if getattr(vp, "_DYN_HOST", None):
            dyn = os.path.join(_dir(), f"{key}-dyn.pt")
            torch.save(bk.dyn_dump(vp._DYN_HOST), dyn)
    except Exception as e:  # noqa
        _log(f"DRAFT-DYN table not saved ({e!r})"); dyn = ""
    try:
        json.dump(_KEY_ENV[0] or {}, open(_env_file(key), "w"))
    except Exception as e:  # noqa
        _log(f"env record not saved ({e!r})")
    _rpc({"op": "put", "key": key, "chunks": chunks, "manifest": man, "csum": _csum(bufs), "kinds": kinds,
          "cpu_need": need, "dyn": dyn, "paths": paths,
          "tsums": tsums}, fds)
    for fd in fds:
        os.close(fd)
    _log(f"exported {len(ents)} tensors / {len(order)} storages, {tot / 2**30:.2f} GiB in {len(chunks)} chunks, "
         f"{len(refs)} storages shared with an earlier model (key {key})")


# ------------------------------------------------------------------ attach
_LAZY_MODS = ("llmt_kv.lazyvis", "exl3xpu.lazyvis", "llmt_lazyvis")


def _lazy_installed() -> bool:
    return any(getattr(sys.modules.get(n), "_STATE", {}).get("installed") for n in _LAZY_MODS)


def _lazy_tower(model, early):
    """LAZY TOWER UNDER ATTACH. The tower is not held: _attach builds it on the device and loads it
    from the checkpoint, but the lazy vision module (llmt_kv.lazyvis; or exl3xpu.lazyvis / llmt_lazyvis) offloads it
    from base_loader hooks that the attach path never calls (it processes on the meta device; the hook's pwal would
    move the host tower to "meta"). So run its offload here at the export path's two points: early=True right after
    the model is built (tower -> host copy before load_weights, so the checkpoint is read straight into host memory
    and no device page of the tower is ever touched (on the Arc, offloading after the load left the freed
    0.86 GiB counted as non-torch, KV unchanged) and early=False after the attach (finalize: checkpoint digest,
    broker kind-5 entry). The meta post-processing pass skips the tower; it is processed on the real device as on
    an export start. No lazy module: nothing to do (the tower stays resident as before)."""
    for name in _LAZY_MODS:
        m = sys.modules.get(name)
        off = getattr(m, "_offload", None) if m is not None else None
        if callable(off) and getattr(m, "_STATE", {}).get("installed") and hasattr(model, "visual"):
            off(model, early=early)
            if not early:
                _log(f"vision tower lazy ({name}): offloaded before the weight read, finalized after the attach")
            return


def _attach(self, vllm_config, model_config, prefix, key, entry, fds):
    from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
    from vllm.utils.torch_utils import set_default_torch_dtype
    try:
        from . import nommap_loader as nl
    except ImportError:
        from llmt_kv import nommap_loader as nl
    bk = _bk(); dev = bk.dev
    bk.init()
    t0 = time.time()
    real_iter = nl.safetensors_weights_iterator_nommap

    def meta_iter(files, *a, **kw):           # checkpoint shapes, no data (real data for the vision tower)
        import struct
        for f in sorted(files):
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8)); hdr = json.loads(fh.read(n))
            base = 8 + n
            fd = os.open(f, os.O_RDONLY)
            try:
                for name, info in sorted(((k, v) for k, v in hdr.items() if k != "__metadata__"),
                                         key=lambda kv: kv[1]["data_offsets"][0]):
                    dt = nl._DT[info["dtype"]]
                    if ".visual." in name or name.startswith("visual."):
                        b0, b1 = info["data_offsets"]
                        buf = torch.empty(b1 - b0, dtype=torch.uint8); mv = memoryview(buf.numpy()); got = 0
                        while got < b1 - b0:
                            got += os.preadv(fd, [mv[got:]], base + b0 + got)
                        yield name, buf.view(dt).reshape(info["shape"])
                    else:
                        yield name, torch.empty(info["shape"], dtype=dt, device="meta")
            finally:
                os.close(fd)
    import vllm.model_executor.models.qwen3_vl as qvl  # noqa
    import vllm.model_executor.models.qwen3_5 as q35
    VT = q35.Qwen3_VisionTransformer
    vt_init = VT.__init__

    def vt_real_init(self_, *a, **k):         # the vision tower is not held: build it on the real device
        if _lazy_installed():                 # lazy tower: built on the HOST, so the device never holds (or pools) it
            with torch.device("cpu"):
                vt_init(self_, *a, **k)
            self_._lazyvis_dev = torch.device(vllm_config.device_config.device)
            return
        with torch.device(vllm_config.device_config.device):
            vt_init(self_, *a, **k)
    import vllm.model_executor.model_loader.default_loader as dl
    import vllm.model_executor.model_loader.weight_utils as wu
    wu.safetensors_weights_iterator = dl.safetensors_weights_iterator = meta_iter
    T = torch.Tensor
    saved_m = {n: getattr(T, n) for n in ("item", "tolist", "cpu", "to", "numpy", "copy_")}
    tainted = set()

    def copy_(self, src, *a, **k):     # meta data into a real tensor (CPU-resident params): no data -> remember it
        if isinstance(src, torch.Tensor) and src.is_meta and not self.is_meta:
            tainted.add(self.untyped_storage().data_ptr()); return self   # by storage: param.data is a new object
        return saved_m["copy_"](self, src, *a, **k)

    def item(self):
        return 0 if self.is_meta and not self.is_floating_point() else (0.0 if self.is_meta else saved_m["item"](self))

    def tolist(self):
        if not self.is_meta:
            return saved_m["tolist"](self)
        z = 0 if not self.is_floating_point() else 0.0
        def nest(sh):
            return [nest(sh[1:]) for _ in range(sh[0])] if sh else z
        return nest(list(self.shape))

    def cpu(self, *a, **k):
        return self if self.is_meta else saved_m["cpu"](self, *a, **k)

    def to(self, *a, **k):
        if self.is_meta:
            dv = k.get("device", a[0] if a and isinstance(a[0], (str, torch.device)) else None)
            if dv is not None and torch.device(dv).type == "cpu":
                k = dict(k); k.pop("device", None)
                a = tuple(x for x in a if not isinstance(x, (str, torch.device)))
                return saved_m["to"](self, *a, **k) if (a or k) else self
        return saved_m["to"](self, *a, **k)

    def numpy(self, *a, **k):
        if self.is_meta:
            raise RuntimeError("numpy() of a meta tensor during the WHX meta build")
        return saved_m["numpy"](self, *a, **k)
    if entry.get("dyn") and os.path.exists(entry["dyn"]):
        vp = _plugin()
        bk.dyn_load(torch.load(entry["dyn"], mmap=True), vp._DYN_HOST)
    eq0 = torch.equal

    def equal(a, b):                   # the DRAFT-DYN cache check compares weights: the held-memory checksum covers it
        if (isinstance(a, torch.Tensor) and a.is_meta) or (isinstance(b, torch.Tensor) and b.is_meta):
            return True
        return eq0(a, b)
    try:
        torch.equal = equal
        VT.__init__ = vt_real_init
        T.item, T.tolist, T.cpu, T.to, T.numpy, T.copy_ = item, tolist, cpu, to, numpy, copy_
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
            _lazy_tower(model, early=True)       # as on an export start: tower -> host copy before any weight is read
            self.load_weights(model, model_config)
            vis = model._modules.pop("visual", None) if "visual" in getattr(model, "_modules", {}) else None
            try:                                 # the meta pass cannot move a host-resident tower to "meta"
                process_weights_after_loading(model, model_config, torch.device("meta"))
            finally:
                if vis is not None:
                    model._modules["visual"] = vis
            if vis is not None:                  # the tower is processed as on an export start (real target device)
                process_weights_after_loading(vis, model_config, torch.device(vllm_config.device_config.device))
                try:                         # the per-module device round trips of that pass leave pooled blocks: hand them back
                    acc = _bk().acc(); acc.synchronize(); acc.empty_cache()
                except Exception:
                    pass
    finally:
        wu.safetensors_weights_iterator = dl.safetensors_weights_iterator = real_iter
        for n, f in saved_m.items():
            setattr(T, n, f)
        torch.equal = eq0
        VT.__init__ = vt_init
    _lazy_tower(model, early=False)
    bufs = _attach_tensors(model, key, entry, fds, tainted, saved_m["copy_"])
    bk.finish(model)                   # CUDA: EXL3 linears from the held views, deferred draft heads
    left = [pp for (_, t), pp in zip(list(_walk(model, private=True)), list(_PATHS)) if t.device.type == "meta"]
    if left:
        raise RuntimeError(f"{len(left)} meta tensors left, e.g. {left[:3]}")
    model._whx_bufs = bufs
    _log(f"attached {len(entry['manifest'])} tensors, {sum(entry['chunks']) / 2**30:.2f} GiB in {time.time() - t0:.1f} s (key {key})")
    return model.eval()


def _attach_tensors(model, key, entry, fds, tainted=(), copy_=None):
    """Puts views of the imported chunks in place of the (meta) model's tensors, by manifest path; returns the chunks.
    Device-independent apart from the backend's import; the CPU test drives it with a fake backend."""
    bk = _bk(); dev = bk.dev
    copy_ = copy_ or torch.Tensor.copy_
    allents = list(_walk(model))
    kinds = entry["kinds"]
    ex, at = [_canon_path(p) for p in (entry.get("paths") or [])], list(_PATHS)
    if len(allents) != len(kinds) or set(ex) != set(at) or len(set(at)) != len(at):
        miss = [x for x in ex if x not in set(at)][:6]; extra = [x for x in at if x not in set(ex)][:6]
        raise RuntimeError(f"tensor count {len(allents)} vs export {len(kinds)}; missing {miss}; extra {extra}")
    if at != ex:                       # same tensors, other traversal order (weights tied after load): export order
        pos = {p: i for i, p in enumerate(at)}
        allents = [allents[pos[p]] for p in ex]
    need = entry.get("cpu_need") or []
    saved = torch.load(os.path.join(_dir(), f"{key}-cpu.pt"), mmap=True) if need else {}
    ents, ci_, missing = [], 0, []
    for (put, t), kind in zip(allents, kinds):
        if kind != dev:                                 # CPU tensor
            if t.device.type == "cpu" and t.untyped_storage().data_ptr() not in tainted:
                pass                                    # rebuilt for real by the meta-device build (from files/config)
            elif ci_ in saved and (put is not None or t.device.type == "cpu"):
                c = saved[ci_]
                if list(t.shape) != list(c.shape) or t.dtype != c.dtype:
                    raise RuntimeError(f"CPU tensor mismatch {tuple(t.shape)} vs {tuple(c.shape)}")
                if t.device.type == "cpu":
                    with torch.no_grad():
                        copy_(t.data if isinstance(t, torch.nn.Parameter) else t, c)   # in place: aliases see it
                else:
                    put(torch.nn.Parameter(c, requires_grad=False) if isinstance(t, torch.nn.Parameter) else c)
            else:
                missing.append(ci_)
            ci_ += 1
        else:
            ents.append((put, t))
    if missing:
        raise _NeedCPU(missing)
    man = entry["manifest"]
    if len(ents) != len(man):
        raise RuntimeError(f"{dev} tensor count {len(ents)} != manifest {len(man)}")
    bufs = []
    for fd, nb in zip(fds, entry["chunks"]):
        _tick(); bufs.append(bk.import_fd(fd, max(nb, _ALIGN)))
    for ci, b in enumerate(bufs):
        _REG.append((key, ci, b.data_ptr(), b.numel(), b))
    regbuf = {(k2, c2): b for k2, c2, _, _, b in _REG}
    for fd in fds:                    # the L0 import may already own/close the fd
        try:
            os.close(fd)
        except OSError:
            pass
    new = {}
    if entry.get("tsums"):
        xp = [_canon_path(p) for p, k in zip(entry.get("paths") or [], entry["kinds"]) if k == dev]
        bad = []
        for i, (m, ts) in enumerate(zip(man, entry["tsums"])):
            src = regbuf.get((m[1], m[2])) if m[0] == "ref" else bufs[m[0]]
            off, shape, stride, dt = (m[3], m[4], m[5], m[6]) if m[0] == "ref" else (m[1], m[2], m[3], m[4])
            v = torch.empty(0, dtype=getattr(torch, dt.split(".")[1]), device=dev).set_(src.untyped_storage(), off, shape, stride)
            if _tsum(v) != ts:
                bad.append(xp[i] if i < len(xp) else i)
        if bad:
            raise RuntimeError(f"{len(bad)} held tensors changed since the export (runtime-written?): {bad[:8]}")
    for (put, t), m in zip(ents, man):
        if m[0] == "ref":
            _, k2, c2, off, shape, stride, dt = m; src = regbuf.get((k2, c2))
            if src is None:
                raise RuntimeError(f"reference to {k2}/{c2}, not attached in this process")
        else:
            ci, off, shape, stride, dt = m; src = bufs[ci]
        if put is None:
            raise RuntimeError("tensor in a tuple without a settable parent")
        if list(t.shape) != shape or str(t.dtype) != dt:
            raise RuntimeError(f"manifest mismatch: {tuple(t.shape)} {t.dtype} vs {shape} {dt}")
        if id(t) not in new:
            v = torch.empty(0, dtype=t.dtype, device=dev).set_(src.untyped_storage(), off, shape, stride)
            if isinstance(t, torch.nn.Parameter):
                v = torch.nn.Parameter(v, requires_grad=False)
                for a in ("weight_loader",):
                    if hasattr(t, a):
                        setattr(v, a, getattr(t, a))
            new[id(t)] = v
        put(new[id(t)])
    pf = os.path.join(_dir(), f"{key}-priv.pt")
    pv = {_canon_path(k): v for k, v in (torch.load(pf) if os.path.exists(pf) else {}).items()}
    pents = list(_walk(model, private=True)); ppaths = list(_PATHS)
    for (put, t), pp in zip(pents, ppaths):
        if "[dyn]" in pp and t.is_meta:
            if pp not in pv or put is None:
                raise RuntimeError(f"private buffer {pp} not saved")
            put(pv[pp].to(dev))
    return bufs


def _finish_export(m, key, need):
    try:
        _export(m, key, need)
    except Exception as e:  # noqa
        _log(f"export failed ({e!r}); continuing without the keeper")
    try:
        _bk().finish(m)             # CUDA: the fused-decode pointer tables follow the moved tensors
    except Exception as e:
        raise RuntimeError(f"WHX: post-export rebuild failed ({e!r}); the engine cannot run on moved weights") from e


# ------------------------------------------------------------------ load watchdog
_WD = {"t": 0.0, "phase": "", "on": False, "armed": 0.0, "lim": 0.0}


def _tick(phase=None):
    """Progress during the load (a weight tensor read, an attach / export step); re-arms the native backstop."""
    now = time.monotonic()
    _WD["t"] = now
    if phase:
        _WD["phase"] = phase
    if _WD["on"] and now - _WD["armed"] > 5:
        _WD["armed"] = now
        try:   # C-level timer: fires even if a native call holds the GIL
            faulthandler.dump_traceback_later(_WD["lim"] + 60, exit=True)
        except Exception:  # noqa
            pass


def _wd_start(phase):
    lim = float(os.environ.get("EXL3_WHX_STALL_S", "240") or 0)
    if lim <= 0:
        return
    _WD["lim"] = lim
    if not _WD["on"]:
        _WD["on"] = True

        def run():
            while _WD["on"]:
                time.sleep(2)
                idle = time.monotonic() - _WD["t"]
                if _WD["on"] and idle > lim:
                    _log(f"LOAD STALL: no load progress for {idle:.0f} s (limit EXL3_WHX_STALL_S={lim:.0f}) during "
                         f"{_WD['phase']}; a GPU fault / exec-queue reset (host dmesg: 'Faulted Address', 'exec queue "
                         "reset') leaves the engine waiting forever. Stacks follow; exiting the engine (code 70) so it "
                         "restarts now instead of at the supervisor's health timeout")
                    try:
                        faulthandler.dump_traceback(all_threads=True)
                        sys.stderr.flush(); sys.stdout.flush()
                    finally:
                        os._exit(70)
        threading.Thread(target=run, name="whx-load-watchdog", daemon=True).start()
    _WD["armed"] = 0.0
    _tick(phase)


def _wd_stop():
    if _WD["on"]:
        _WD["on"] = False
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:  # noqa
            pass


def _wd_wrap_weights():
    """Tick for every weight tensor the loader reads (normal load) or 'reads' (attach: meta shapes)."""
    try:
        from vllm.model_executor.model_loader import default_loader as dl
    except Exception as e:  # noqa
        _log(f"load watchdog: weight progress hook not installed ({e!r}); phase ticks only"); return
    orig = dl.DefaultModelLoader.get_all_weights
    if getattr(orig, "_whx_tick", False):
        return

    def get_all_weights(self, *a, **k):
        for x in orig(self, *a, **k):
            _tick()
            yield x
    get_all_weights._whx_tick = True
    dl.DefaultModelLoader.get_all_weights = get_all_weights


def _drop_all(why):
    """EXL3_WHX_EXCLUSIVE=1 (one keeper per GPU, one model): a held copy this engine cannot attach only takes VRAM from
    its load and KV cache, so it goes before the load (its memory is freed once no other engine maps it)."""
    try:
        held, _ = _rpc({"op": "list"})
        for k2 in held:
            _rpc({"op": "drop", "key": k2})
        if held:
            _log(f"dropped {len(held)} held entries ({sum(v.get('bytes', 0) for v in held.values()) / 2**30:.2f} GiB): {why}")
    except Exception as e:  # noqa
        _log(f"drop of the held entries failed ({e!r})")


def _torch_cap():
    """TORCH_CAP_MIB (Arc OOR guard, oorcap/llmt_memprobe.py: caps torch's allocator so the Level Zero driver keeps its
    last device memory) was sized for weights inside torch. Held weights are outside torch, so the cap that bounds the
    device total is TORCH_CAP_MIB - held MiB (this process's mapped chunks). Applied after export / attach, and the
    memprobe's own one-time cap is marked done so it cannot raise it back."""
    import sys
    cap = int(os.environ.get("TORCH_CAP_MIB", "0") or 0)
    if not cap or not _REG:
        return
    held = sum(nb for _, _, _, nb, _ in _REG) >> 20
    eff = max(cap - held, 1024)
    try:
        acc = _bk().acc(); tot = acc.get_device_properties(0).total_memory
        mp = sys.modules.get("llmt_memprobe")
        if mp is not None:
            mp._capped = True
        acc.set_per_process_memory_fraction(min(1.0, (eff << 20) / tot), 0)
        os.environ["LLMT_WHX_HELD_MIB"] = str(held)
        _log(f"torch allocator cap {eff} MiB = TORCH_CAP_MIB {cap} - {held} MiB held (fraction {(eff << 20) / tot:.4f})")
    except Exception as e:  # noqa
        _log(f"torch cap not applied ({e!r})")


def install():
    """Wrap vLLM's model loader (once per process; the engine plugin and llmt_kv.plugin may both call it)."""
    if not os.environ.get("EXL3_WHX") or _installed[0]:
        return
    _installed[0] = True
    try:
        _bk().check()
    except Exception as e:  # noqa
        _log(f"not installed ({e!r}); weights load normally"); return
    acc = _bk().acc()
    from vllm.model_executor.model_loader import base_loader as bl
    orig = bl.BaseModelLoader.load_model

    def load_model(self, vllm_config, model_config, prefix=""):
        idx = _calls[0]; _calls[0] += 1
        key = _key(model_config, idx)
        _wd_start(f"model {idx} load (key {key})")
        try:
            entry, fds = _rpc({"op": "get", "key": key})
        except Exception as e:  # noqa
            _log(f"keeper not reachable ({e!r}); normal load"); return orig(self, vllm_config, model_config, prefix)
        need = []
        ctx = vllm_config.compilation_config.static_forward_context
        ctx_keys = set(ctx)
        ro = os.environ.get("EXL3_WHX_READONLY") == "1"
        if not entry.get("found"):
            _miss_report(key)
            if ro:
                raise RuntimeError(f"WHX: EXL3_WHX_READONLY: no held copy for key {key} (see the MISS line); this engine "
                                   "only attaches to another engine's weights. Drop them or start without the keeper")
            if idx == 0 and os.environ.get("EXL3_WHX_EXCLUSIVE") == "1":
                _drop_all("this engine's weights differ from every held copy")
        if entry.get("found"):
            try:
                _tick(f"model {idx} attach (key {key})")
                return _attach(self, vllm_config, model_config, prefix, key, entry, fds)
            except Exception as e:  # noqa
                import traceback
                tb = traceback.extract_tb(e.__traceback__)[-4:]
                where = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(tb))
                held, _ = _rpc({"op": "list"}) if not ro else ({}, [])
                for k2 in held:                     # all entries: the next start loads and exports cleanly
                    _rpc({"op": "drop", "key": k2})
                if isinstance(e, _NeedCPU):        # remember which CPU tensors to save next time
                    open(os.path.join(_dir(), f"{key}-need.json"), "w").write(
                        json.dumps(sorted(set(entry.get("cpu_need") or []) | set(e.idx))))
                for fd in fds:
                    try: os.close(fd)
                    except OSError: pass
                if ro:                              # EXL3_WHX_READONLY: nothing was dropped
                    raise RuntimeError(f"WHX: attach failed ({e!r}) at {where}; read-only, keeper entries kept") from e
                raise RuntimeError(f"WHX: attach failed ({e!r}) at {where}; keeper entries dropped, restart the engine "
                                   "(it loads normally and re-exports)") from e
            for k in set(ctx) - ctx_keys:          # the meta model's layers registered themselves: forget them
                del ctx[k]
            for fd in fds:
                try: os.close(fd)
                except OSError: pass
            try:                                   # free the held copy first: no room for two on the B70
                _rpc({"op": "drop", "key": key})
            except Exception:  # noqa
                pass
            import gc; gc.collect(); acc.empty_cache()
        nf = os.path.join(_dir(), f"{key}-need.json")
        if os.path.exists(nf):
            need = json.load(open(nf))
        _tick(f"model {idx} normal load from disk (key {key})")
        model = orig(self, vllm_config, model_config, prefix)
        _tick(f"model {idx} loaded, runner setup (key {key})")
        _PENDING.append((model, key, need))   # exported after the runner has tied shared weights (see below)
        return model
    bl.BaseModelLoader.load_model = load_model
    try:                                     # deferred exports: after target + drafter load and weight sharing
        import vllm.v1.worker.gpu_model_runner as gmr
        orig_rl = gmr.GPUModelRunner.load_model

        def runner_load_model(self_, *a, **k):
            _wd_start("runner load_model")
            try:
                r = orig_rl(self_, *a, **k)
                done = bool(_PENDING)
                while _PENDING:
                    m, key, need = _PENDING.pop(0)
                    _tick(f"export (key {key})")
                    _finish_export(m, key, need)
                if done:   # the weights now live outside torch's allocator: count them once, as non-torch memory
                    _tick("post-export synchronize")
                    acc.synchronize(); acc.empty_cache(); acc.reset_peak_memory_stats()
                    if hasattr(self_, "model_memory_usage"):     # what torch still holds (vision tower)
                        self_.model_memory_usage = acc.memory_allocated()
                        _log(f"model memory after export: {self_.model_memory_usage / 2**30:.2f} GiB in torch, rest held")
            finally:
                _wd_stop()
            _torch_cap()
            return r
        gmr.GPUModelRunner.load_model = runner_load_model
    except Exception as e:  # noqa
        _log(f"runner hook not installed ({e!r}); no exports")
    try:                                     # vLLM's startup check counts held weights as "used by other processes"
        xw = importlib.import_module(_bk().worker)
        orig_rm = xw.request_memory

        def request_memory(snap, cache_config):
            try:
                held, _ = _rpc({"op": "list"})
                h = sum(v.get("bytes", 0) for v in held.values())
            except Exception:  # noqa
                h = 0
            if h:   # as if the weights were not loaded yet: their bytes show up later as non-torch use, like a load
                # measured: with the full adjustment an attached engine used 0.49 GiB more VRAM in total than a normal
                # start on the Arc (30.70 vs 30.21 GiB), 0.03 GiB more on the 5090; EXL3_WHX_MARGIN_MB (default: the
                # backend's margin_mb) keeps it at the normal engine's level
                snap.free_memory += h; snap.cuda_memory -= h; snap.non_torch_memory -= h
                # a second engine on the same GPU (dual-engine mode): free memory for the startup check only, KV unaffected
                snap.free_memory += int(os.environ.get("EXL3_WHX_EXTRA_FREE_MB", "0")) << 20
                _log(f"startup memory check: {h / 2**30:.2f} GiB held by the keeper counted as free")
                return orig_rm(snap, cache_config) - (int(os.environ.get("EXL3_WHX_MARGIN_MB", _bk().margin_mb)) << 20)
            return orig_rm(snap, cache_config)
        xw.request_memory = request_memory
    except Exception as e:  # noqa
        _log(f"memory-check hook not installed ({e!r})")
    _install_pacing()
    _wd_wrap_weights()
    _log(f"installed ({_bk().name}, keeper {os.environ['EXL3_WHX']}; load watchdog "
         f"{os.environ.get('EXL3_WHX_STALL_S', '240')} s)")


def _install_pacing():
    """Dual-engine mode: EXL3_WHX_PACE_SIGNAL=<file> (prefill engine) keeps <file> present while it has
    requests; EXL3_WHX_PACE_FLAG=<file> + EXL3_WHX_PACE_MS=N (decode engine on rcs0) sleeps N ms after every engine step
    while <file> exists, handing execution units to the prefill engine. EXL3_WHX_* is not part of the weight key."""
    sig = os.environ.get("EXL3_WHX_PACE_SIGNAL"); flag = os.environ.get("EXL3_WHX_PACE_FLAG")
    ms = float(os.environ.get("EXL3_WHX_PACE_MS", "0") or 0)
    if not sig and not (flag and ms > 0):
        return
    try:
        import vllm.v1.engine.core as core
    except Exception as e:  # noqa
        _log(f"pacing not installed ({e!r})"); return
    orig = core.EngineCoreProc._process_engine_step
    state = {"on": False}

    def step(self, *a, **k):
        r = orig(self, *a, **k)
        if sig:
            busy = bool(self.scheduler.has_requests())
            if busy != state["on"]:
                state["on"] = busy
                try:
                    if busy:
                        open(sig, "w").close()
                    else:
                        os.unlink(sig)
                except OSError:
                    pass
        elif os.path.exists(flag):
            time.sleep(ms / 1000.0)
        return r
    core.EngineCoreProc._process_engine_step = step
    _log(f"pacing: signal={sig} flag={flag} ms={ms}")
