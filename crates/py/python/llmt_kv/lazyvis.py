"""llmt_kv.lazyvis: lazy vision encoder for vLLM Qwen3-VL-style towers, with a broker-managed host copy. The tower is
not kept in VRAM, and not even in host RAM, while nobody sends images: it follows the same rules as KV.

EXL3_LAZY_VISION=1:
  - KV sizing as before: the tower leaves the device right after it is built (before the checkpoint is read), so vLLM
    sizes the KV cache without it; the startup profile run streams it (activation reserve = its peak).
  - Startup (after process_weights_after_loading): a manifest of the tower's final tensors (module path, name, shape,
    dtype, byte offset in the entry layout), per-tensor crc32 + blake2b, and the content digest. Every tensor is
    matched against the checkpoint on disk (raw bytes, or the checkpoint dtype converted to the model dtype as the
    loader does; the 5090 and the Arc both load bf16 -> fp16): matching tensors can be reloaded from disk exactly,
    the rest ("extras": the rotary cos/sin cache buffer, ~0.6 MB) stay in a small private host copy. The startup host
    copy serves the profile run and the warm-up and is FREED when the engine is ready. Nothing goes to the broker at
    startup.
  - An image request (visual.forward) with the tower not resident:
      source = broker entry (LOOKUP acquire=1 in the shared namespace EXL3_LAZY_VISION_NS; key = blake2b(model id |
               manifest | content digest), so engines with identical bytes share one entry), copied from the pool
               mapping, lease released right after the copy;
      miss   = reload from the checkpoint, verified against the startup crc32s, stored as kind 5 (weights) with
               cost_hint = reload ms (RESERVE -> read straight into the reserved slots -> verify -> COMMIT, then leased
               like a hit); FULL / BUSY / any broker error: read into a temporary host buffer instead, dropped after
               the copy. BUSY (another engine is storing it right now) first WAITs up to EXL3_LAZY_VISION_BUSY_WAIT_MS.
    and then, as before, either
      resident: copied in whole, kept while images keep coming, released after EXL3_LAZY_VISION_TTL s idle (120) or
                under memory pressure; nothing is kept on the host;
      streamed: block by block when there is no room (free + cached-unused < tower + activation reserve + margin);
                the source (lease or temporary buffer) is held for that one forward only.
    A lease that turns out invalid after the copy (release not OK, broker restarted, epoch changed) means the bytes
    are not trusted: resident loads are redone from disk, a streamed forward is run again from disk.
  - A checkpoint that no longer matches the startup checksums: logged as an ERROR, the broker is not used again by
    this engine and the reloaded bytes are kept as a private host copy (the old behaviour) for the rest of its life.
  - Broker down / tier off / client missing: disk path every time; the engine never fails because of the broker.
    Client calls time out after EXL3_LAZY_VISION_BROKER_TIMEOUT_MS (100); a failed connect is retried at most every
    EXL3_LAZY_VISION_BROKER_RETRY_S (30).
Knobs: EXL3_LAZY_VISION_MODE=auto|stream|resident, _TTL, _MARGIN_MB (as before); _BROKER=auto (socket from vLLM's
kv_transfer_config, else /run/llmt-kv/broker.sock) | off | <socket>; _NS (default weights:visual); _ENGINE_ID
(default LLMT_KV_ENGINE_ID, else broker_engine_id, else 0); _STORE=1; _VERIFY_HIT=0 (crc the pool bytes on a hit);
_DROP_CACHE=1 (fadvise DONTNEED the checkpoint range after a reload); _COLD_ON_RELEASE=1 (HEAT cold for the entry when
the tower leaves VRAM after the TTL; needs a client with heat()); _KEEP_HOST=0 (1 = keep a private host copy, the
old behaviour without pinning; no broker); _MODEL_DIR (default: vLLM's model path); _THREADS=4; _DISK_MBPS=500 (floor
of cost_hint = bytes / this).
Logs counts/timings only (no request data)."""
import ctypes
import hashlib
import json
import os
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import torch


def _pick_acc():
    try:
        if torch.cuda.is_available():
            return torch.cuda
    except Exception:  # noqa: BLE001
        pass
    x = getattr(torch, "xpu", None)
    try:
        if x is not None and x.is_available():
            return x
    except Exception:  # noqa: BLE001
        pass
    return torch.cuda   # neither: unit tests replace _ACC


_ACC = _pick_acc()

_ENV = os.environ.get
ON = _ENV("EXL3_LAZY_VISION", "0") == "1"
TTL = float(_ENV("EXL3_LAZY_VISION_TTL", "120"))
MARGIN = int(float(_ENV("EXL3_LAZY_VISION_MARGIN_MB", "256")) * 2**20)
MODE = _ENV("EXL3_LAZY_VISION_MODE", "auto")
BROKER = _ENV("EXL3_LAZY_VISION_BROKER", "auto")
NS_NAME = _ENV("EXL3_LAZY_VISION_NS", "weights:visual")
TIMEOUT_MS = float(_ENV("EXL3_LAZY_VISION_BROKER_TIMEOUT_MS", "100"))
RETRY_S = float(_ENV("EXL3_LAZY_VISION_BROKER_RETRY_S", "30"))
BUSY_WAIT_MS = int(_ENV("EXL3_LAZY_VISION_BUSY_WAIT_MS", "5000"))
STORE = _ENV("EXL3_LAZY_VISION_STORE", "1") == "1"
VERIFY_HIT = _ENV("EXL3_LAZY_VISION_VERIFY_HIT", "0") == "1"
DROP_CACHE = _ENV("EXL3_LAZY_VISION_DROP_CACHE", "1") == "1"
KEEP_HOST = _ENV("EXL3_LAZY_VISION_KEEP_HOST", "0") == "1"
THREADS = max(1, int(_ENV("EXL3_LAZY_VISION_THREADS", "4")))
DISK_MBPS = float(_ENV("EXL3_LAZY_VISION_DISK_MBPS", "500"))
COLD_ON_RELEASE = _ENV("EXL3_LAZY_VISION_COLD_ON_RELEASE", "1") == "1"

ALIGN = 512            # byte alignment of every tensor in the entry layout (dtype views, kernel alignment)
FORMAT = 1             # entry layout id (RESERVE format); a new layout needs a new id (it is in the key too)
KIND_WEIGHTS = 5
_STATE = {"ready": False, "act_reserve": None, "towers": []}
_SKIP_HOST_BUFFERS = True   # buffers already on the host are not managed (unit tests on CPU set False)

_ST2T = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
         "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
         "BOOL": torch.bool}


def _log(msg, *a, level="info"):
    try:
        from vllm.logger import init_logger
        getattr(init_logger("vllm.lazyvis"), level)("LAZYVIS: " + msg, *a)
    except Exception:  # noqa: BLE001
        print("LAZYVIS: " + (msg % a), flush=True)


def _sync():
    try:
        torch.accelerator.synchronize()
    except Exception:  # noqa: BLE001
        try:
            _ACC.synchronize()
        except Exception:  # noqa: BLE001
            pass


def _kv():
    import llmt_kv
    return llmt_kv


def _al(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _mv(addr, n):
    """Writable byte memoryview of host memory at addr (torch CPU storage or the broker pool mapping)."""
    return memoryview((ctypes.c_uint8 * n).from_address(addr)).cast("B")


def _tmv(t, off=0, n=None):
    """Byte memoryview of a contiguous CPU tensor's storage, from byte `off`."""
    nb = t.numel() * t.element_size()
    n = nb - off if n is None else n
    return _mv(t.data_ptr() + off, n) if n else memoryview(b"")


def _crc(views):
    c = 0
    for v in views:
        c = zlib.crc32(v, c)
    return c


def _segs(ranges, off, n):
    """Pieces (pool_off, length, entry_off) of entry bytes [off, off+n) over the entry's pool ranges (entry order)."""
    pos, end = 0, off + n
    for po, ln in ranges:
        a, b = max(off, pos), min(end, pos + ln)
        if a < b:
            yield po + (a - pos), b - a, a
        pos += ln
        if pos >= end:
            break


def _placeholder(shape, dtype):
    """Shape/dtype-correct CPU stand-in (one element, stride 0) while the tower is not on the device: code outside the
    forward that reads .dtype / .shape keeps working, and nothing holds the tower's bytes."""
    return torch.zeros((), dtype=dtype).expand(tuple(shape))


class _Rec:
    __slots__ = ("m", "name", "kind", "qname", "shape", "dtype", "nbytes", "unit", "moff", "off", "xoff", "ck",
                 "crc", "dig")

    def __init__(self, m, name, kind, qname, t, unit):
        self.m, self.name, self.kind, self.qname, self.unit = m, name, kind, qname, unit
        self.shape, self.dtype = tuple(t.shape), t.dtype
        self.nbytes = t.numel() * t.element_size()
        self.moff = self.off = self.xoff = None
        self.ck = None          # (path, file offset, nbytes, checkpoint dtype) if it can be reloaded from disk
        self.crc = self.dig = None


class _Unit:
    def __init__(self, idx):
        self.idx, self.recs, self.entry, self.extra = idx, [], [], []
        self.off = self.n = 0
        self.dev = None
        self.on_dev = False


# ---------------------------------------------------------------- sources: where the bytes of one load come from

class _MasterSource:
    """Startup host copy (startup layout, one tensor at a time); also the private copy with KEEP_HOST."""
    what = "host"

    def __init__(self, lv):
        self.lv = lv

    def copy_unit(self, u, buf):
        for r in u.entry:
            buf[r.off - u.off:r.off - u.off + r.nbytes].copy_(self.lv.master[r.moff:r.moff + r.nbytes],
                                                               non_blocking=True)

    def close(self):
        _sync()
        return True


class _HostSource:
    """A host buffer in the entry layout: a temporary reload (dropped on close) or the private copy (keep=True)."""
    what = "disk"

    def __init__(self, buf, keep=False):
        self.buf, self.keep = buf, keep

    def copy_unit(self, u, buf):
        if u.n:
            buf.copy_(self.buf[u.off:u.off + u.n], non_blocking=True)

    def close(self):
        _sync()
        if not self.keep:
            self.buf = None
        return True


class _PoolSource:
    """A leased broker entry: copies straight from the pool mapping; close() releases the lease and says whether the
    bytes were valid for the whole copy (lease still held, same broker epoch)."""
    what = "pool"

    def __init__(self, bk, conn, ranges, lease):
        self.bk, self.conn, self.ranges, self.lease = bk, conn, ranges, lease
        self.epoch = conn.epoch0

    def copy_unit(self, u, buf):
        pool = self.conn.pool_t
        for po, ln, eo in _segs(self.ranges, u.off, u.n):
            buf[eo - u.off:eo - u.off + ln].copy_(pool[po:po + ln], non_blocking=True)

    def views(self, off, n):
        return [_mv(self.conn.base + po, ln) for po, ln, _ in _segs(self.ranges, off, n)]

    def close(self):
        _sync()
        if self.lease is None:
            return True
        ok = self.bk.release(self.conn, self.lease)
        self.lease = None
        try:
            if self.conn.c.epoch != self.epoch:
                ok = False
        except Exception:  # noqa: BLE001
            ok = False
        return ok


# ---------------------------------------------------------------- broker access

class _Conn:
    """One connected client + its pool mapping. Sources keep a reference, so a dropped connection's mapping stays
    valid until the last copy from it is done."""

    def __init__(self, c):
        self.c = c
        self.base = c.pool_address()
        self.size = int(c.pool_bytes)
        self.pool_t = torch.frombuffer((ctypes.c_uint8 * self.size).from_address(self.base), dtype=torch.uint8)
        self.epoch0 = c.epoch
        self.pool_path = c.pool_path


class _Broker:
    """Weights entries in the broker (AllBroker protocol v1, kind 5). Every failure means 'no broker this time'."""

    def __init__(self, sock, engine_id, name, ns_name, timeout_ms=TIMEOUT_MS, retry_s=RETRY_S):
        self.sock, self.engine_id, self.name, self.ns_name = sock, int(engine_id), name, ns_name
        self.ns = int.from_bytes(hashlib.blake2b(ns_name.encode(), digest_size=4).digest(), "little")
        self.timeout_ms, self.retry_s = timeout_ms, retry_s
        self.conn, self.next_try, self.disabled = None, 0.0, False
        self.n = dict.fromkeys(("connects", "connect_fail", "errors", "resets", "pool_changed", "hits", "misses",
                                "stored", "store_full", "store_busy", "store_err", "busy_waited", "bad_lease",
                                "meta_mismatch"), 0)

    def _err(self, what, e):
        k = _kv()
        if isinstance(e, k.BrokerPoolChanged):
            self.n["pool_changed"] += 1
            self.conn = None                    # remap on the next use (we never pinned the pool)
            self.next_try = 0.0
        elif isinstance(e, k.BrokerReset):
            self.n["resets"] += 1
        else:
            self.n["errors"] += 1
        if self.n["errors"] + self.n["resets"] + self.n["pool_changed"] <= 10:
            _log("broker %s: %s: %s", what, type(e).__name__, e, level="warning")

    def get(self):
        if self.disabled:
            return None
        if self.conn is not None:
            return self.conn
        now = time.monotonic()
        if now < self.next_try:
            return None
        self.next_try = now + self.retry_s
        try:
            k = _kv()
            c = k.Client(socket=self.sock, engine_id=self.engine_id, name=self.name, timeout_ms=self.timeout_ms,
                         reconnect_ms=1000.0)
            self.conn = _Conn(c)
            self.n["connects"] += 1
            _log("broker %s connected (namespace %r = %d, pool %.1f GiB, slot %.1f MB)", self.sock, self.ns_name,
                 self.ns, self.conn.size / 2**30, c.slot_bytes / 1e6)
        except Exception as e:  # noqa: BLE001 - ImportError, BrokerError, mmap failure: no broker this time
            self.n["connect_fail"] += 1
            if self.n["connect_fail"] <= 3 or self.n["connect_fail"] % 100 == 0:
                _log("no broker on %s (%s: %s): reloads come from disk (retry in %.0f s)", self.sock,
                     type(e).__name__, e, self.retry_s, level="warning")
            return None
        return self.conn

    def _ok_ranges(self, conn, ranges, nbytes):
        tot = 0
        for po, ln in ranges:
            if po < 0 or po + ln > conn.size:
                return False
            tot += ln
        return tot >= nbytes

    def lookup(self, key, nbytes, meta):
        """A _PoolSource holding a lease, or None (miss or broker trouble)."""
        conn = self.get()
        if conn is None:
            return None
        try:
            n_hit, lease, ents = conn.c.lookup(self.ns, [key], acquire=True)
        except Exception as e:  # noqa: BLE001
            self._err("lookup", e)
            return None
        if not n_hit:
            self.n["misses"] += 1
            return None
        e = ents[0]
        ranges = [tuple(r) for r in e.byte_ranges]
        if (int(e.nbytes) != nbytes or int(e.kind) != KIND_WEIGHTS or int(e.format) != FORMAT
                or bytes(e.meta) != meta or not self._ok_ranges(conn, ranges, nbytes)):
            self.n["meta_mismatch"] += 1
            _log("broker entry does not match this tower (bytes %d kind %d format %d): ignored", int(e.nbytes),
                 int(e.kind), int(e.format), level="warning")
            self.release(conn, lease)
            return None
        self.n["hits"] += 1
        return _PoolSource(self, conn, ranges, lease)

    def release(self, conn, lease):
        k = _kv()
        for attempt in (0, 1):
            try:
                st = conn.c.release([lease])
                ok = bool(st) and int(st[0]) == int(k.OK)
                if not ok:
                    self.n["bad_lease"] += 1
                return ok
            except Exception as e:  # noqa: BLE001
                if attempt == 0 and isinstance(e, k.BrokerTimeout):
                    continue
                self._err("release", e)
                self.n["bad_lease"] += 1
                return False
        return False

    def reserve(self, key, nbytes, cost_ms, meta):
        """(conn, ranges) of a fresh reservation, 'busy', or None (FULL / EXISTS race / error)."""
        conn = self.get()
        if conn is None:
            return None
        k = _kv()
        try:
            [(st, ranges)] = conn.c.reserve(self.ns, [{"key": key, "nbytes": int(nbytes), "kind": KIND_WEIGHTS,
                                                         "format": FORMAT, "cost_hint": int(max(1, cost_ms)),
                                                         "meta": meta}])
        except Exception as e:  # noqa: BLE001
            self._err("reserve", e)
            self.n["store_err"] += 1
            return None
        st = int(st)
        if st == int(k.OK):
            ranges = [tuple(r) for r in ranges]
            if not self._ok_ranges(conn, ranges, nbytes):
                self.abort(conn, key)
                self.n["store_err"] += 1
                return None
            return conn, ranges
        if st == int(k.BUSY):
            self.n["store_busy"] += 1
            return "busy"
        if st == int(k.EXISTS):
            return "exists"
        if st == int(k.FULL):
            self.n["store_full"] += 1
        else:
            self.n["store_err"] += 1
        return None

    def commit(self, conn, key):
        k = _kv()
        try:
            st = conn.c.commit(self.ns, [key])
            ok = bool(st) and int(st[0]) == int(k.OK)
        except Exception as e:  # noqa: BLE001
            self._err("commit", e)
            ok = False
        if ok:
            self.n["stored"] += 1
        else:
            self.n["store_err"] += 1
        return ok

    def abort(self, conn, key):
        try:
            conn.c.abort(self.ns, [key])
        except Exception as e:  # noqa: BLE001
            self._err("abort", e)

    def wait(self, key, timeout_ms):
        conn = self.get()
        if conn is None:
            return False
        k = _kv()
        try:
            conn.c.set_timeout_ms(float(timeout_ms) + 1000.0)
            try:
                st = conn.c.wait(self.ns, key, int(timeout_ms))
            finally:
                conn.c.set_timeout_ms(self.timeout_ms)
            self.n["busy_waited"] += 1
            return st is None or int(st) == int(k.OK)
        except Exception as e:  # noqa: BLE001
            self._err("wait", e)
            return False

    def cold(self, key):
        """HEAT cold for the entry once the burst is over (each hit adds a heat bonus; the host copy should
        live only while images are asked for). Only with a client that has heat() and a broker that knows HEAT; any
        error is ignored."""
        conn = self.conn
        if conn is None or not hasattr(conn.c, "heat"):
            return False
        try:
            conn.c.heat(self.ns, [key], False)
            self.n["cold"] = self.n.get("cold", 0) + 1
            return True
        except Exception as e:  # noqa: BLE001
            self.n["cold_err"] = self.n.get("cold_err", 0) + 1
            if self.n["cold_err"] <= 3:
                _log("broker heat cold: %s: %s", type(e).__name__, e, level="warning")
            return False

    def fallocate(self, conn, ranges):
        """Back the reserved byte ranges before writing them (a hole on a full tmpfs would SIGBUS the engine)."""
        try:
            fd = os.open(conn.pool_path, os.O_RDWR)
            try:
                for po, ln in ranges:
                    os.posix_fallocate(fd, po, ln)
            finally:
                os.close(fd)
            return True
        except OSError as e:
            _log("cannot back the reserved pool range (%s): not storing", e, level="warning")
            return False


# ---------------------------------------------------------------- checkpoint access

def _ckpt_index(model_dir):
    """{tensor name: (path, file offset, nbytes, dtype str, shape)} over the model's safetensors files."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    out = {}
    for fn in files:
        p = os.path.join(model_dir, fn)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, m in h.items():
            if k == "__metadata__":
                continue
            s, e = m["data_offsets"]
            out[k] = (p, 8 + n + s, e - s, m["dtype"], tuple(m["shape"]))
    return out


def _preadv_all(fd, views, off):
    """Read exactly sum(len(views)) bytes at `off` into `views`."""
    views = [v for v in views if len(v)]
    while views:
        got = os.preadv(fd, views, off)
        if got <= 0:
            raise OSError("short read from the checkpoint")
        off += got
        while views and got >= len(views[0]):
            got -= len(views[0])
            views.pop(0)
        if views and got:
            views[0] = views[0][got:]


def _model_id(model_dir):
    """Checkpoint vision config (canonical JSON): the same file on both GPUs, independent of local paths/images."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        vc = c.get("vision_config") or c
        return hashlib.sha256(json.dumps(vc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- the tower

class LazyVision:
    def __init__(self, vis, copy=True):
        self.vis, self.lock = vis, threading.RLock()
        # a tower built on the host (the keeper attach path) names its target device in _lazyvis_dev; its
        # host-resident buffers (rotary tables) are managed too, and no device page is allocated at load time
        _hb = getattr(vis, "_lazyvis_dev", None)
        self.dev = _hb if _hb is not None else next(vis.parameters()).device
        blocks = getattr(vis, "blocks", None)
        bunit = {}
        for i, blk in enumerate(blocks or []):
            for m in blk.modules():
                bunit[id(m)] = i + 1
        self.units = [_Unit(0)] + [_Unit(i + 1) for i in range(len(blocks or []))]
        self.recs = []
        for path, m in vis.named_modules():
            u = self.units[bunit.get(id(m), 0)]
            for kind, d in (("p", m._parameters), ("b", m._buffers)):
                for n, t in d.items():
                    if t is None or (kind == "b" and _SKIP_HOST_BUFFERS and _hb is None and t.device.type == "cpu"):
                        continue
                    r = _Rec(m, n, kind, f"{path}.{n}" if path else n, t, u)
                    u.recs.append(r)
                    self.recs.append(r)
        stray = [(type(m).__name__, k) for m in vis.modules() for k, v in vars(m).items()
                 if isinstance(v, torch.Tensor) and v.device.type != "cpu" and k not in ("_parameters", "_buffers")]
        # startup host copy (pageable, one allocation: freed in one piece when the engine is ready)
        off = 0
        for r in self.recs:
            r.moff, off = off, _al(off + r.nbytes)
        self.master = torch.empty(max(off, 1), dtype=torch.uint8)
        for r in self.recs:
            mv = self._mview(r)
            if copy:
                mv.copy_(self._cur(r).detach().reshape(r.shape))
            self._set(r, mv)
        self.bytes = sum(r.nbytes for r in self.recs)
        self.head_bytes = sum(r.nbytes for r in self.units[0].recs)
        self.resident, self.streaming, self.last_use, self.n_img = False, False, 0.0, 0
        self.src = None
        self.pool_used = False       # used the broker entry since the last cold hint
        self.copy_s, self.copy_b = 0.0, 0   # host-side time / bytes of the H2D copies of the current load
        self.final = False
        self.keep_host = KEEP_HOST
        self.private = None          # _HostSource kept for life after a checkpoint mismatch
        self.extras = None
        self.broker = None
        self.reload_ms = None
        self.stats = {"resident_loads": 0, "streamed": 0, "releases": 0, "disk_loads": 0, "pool_loads": 0,
                      "host_loads": 0, "redo": 0, "mismatch": 0}
        if blocks is not None:
            for blk, u in zip(blocks, self.units[1:]):
                blk.register_forward_pre_hook(self._blk_pre(u))
                blk.register_forward_hook(self._blk_post(u))
        orig = vis.forward

        def forward(*a, **k):
            with self.lock:
                self._enter()
                try:
                    out = orig(*a, **k)
                finally:
                    ok = self._exit()
                if not ok:      # streamed from a lease that did not hold: the bytes are not trusted, run again
                    self.stats["redo"] += 1
                    _log("streamed source invalid after the forward (lease lost): recomputing from disk",
                         level="warning")
                    self._enter(force_disk=True)
                    try:
                        out = orig(*a, **k)
                    finally:
                        self._exit()
                return out
        vis.forward = forward
        vis._lazyvis = self
        _log("tower placed in host memory: %.3f GiB (%d blocks, head %.1f MiB)%s; mode %s, ttl %.0f s, margin %d MiB",
             self.bytes / 2**30, len(self.units) - 1, self.head_bytes / 2**20,
             (" stray device tensors %s" % stray) if stray else "", MODE, TTL, MARGIN >> 20)

    # --- tensor plumbing
    def _mview(self, r):
        return self.master[r.moff:r.moff + r.nbytes].view(r.dtype).view(r.shape)

    @staticmethod
    def _cur(r):
        return r.m._parameters[r.name] if r.kind == "p" else r.m._buffers[r.name]

    @staticmethod
    def _set(r, t):
        if r.kind == "p":
            r.m._parameters[r.name].data = t
        else:
            r.m._buffers[r.name] = t

    def repoint_master(self):
        """process_weights_after_loading's device_loading_context may have swapped in fresh CPU copies (or device
        tensors): copy the current bytes into the startup copy and point every tensor back at it."""
        for r in self.recs:
            t, mv = self._cur(r), self._mview(r)
            if t.data_ptr() != mv.data_ptr():
                mv.copy_(t.detach().reshape(r.shape))
                self._set(r, mv)

    # --- startup: manifest, checksums, checkpoint match
    def finalize(self, model_dir, broker=None):
        with self.lock:
            t0 = time.perf_counter()
            self.repoint_master()
            self.model_dir = model_dir
            ck = {}
            if model_dir and os.path.isdir(model_dir):
                try:
                    ck = _ckpt_index(model_dir)
                except Exception as e:  # noqa: BLE001
                    _log("checkpoint index unreadable (%r)", e, level="warning")
            prefix = self._prefix(ck)
            nconv = 0
            treads = 0.0
            fds = {}
            try:
                for r in self.recs:
                    if prefix is None:
                        break
                    c = ck.get(prefix + r.qname)
                    cdt = _ST2T.get(c[3]) if c else None
                    if c is None or cdt is None or tuple(c[4]) != r.shape:
                        continue
                    if c[2] != _numel(r.shape) * _esize(cdt):
                        continue
                    fd = fds.get(c[0])
                    if fd is None:
                        fd = fds[c[0]] = os.open(c[0], os.O_RDONLY)
                    raw = torch.empty(c[2], dtype=torch.uint8)
                    t1 = time.perf_counter()
                    _preadv_all(fd, [_tmv(raw)], c[1])
                    treads += time.perf_counter() - t1
                    final = self.master[r.moff:r.moff + r.nbytes]
                    if cdt == r.dtype:
                        same = torch.equal(raw, final)
                    else:
                        conv = raw.view(cdt).view(r.shape).to(r.dtype).reshape(-1).view(torch.uint8)
                        same = torch.equal(conv, final)
                        nconv += same
                    if same:
                        r.ck = (c[0], c[1], c[2], cdt)
            finally:
                for fd in fds.values():
                    os.close(fd)
            # entry layout: unit-major, checkpoint-backed tensors only; extras in a small private host copy
            off, xoff = 0, 0
            for u in self.units:
                u.entry = [r for r in u.recs if r.ck is not None]
                u.extra = [r for r in u.recs if r.ck is None]
                u.off = off
                for r in u.entry:
                    r.off, off = off, _al(off + r.nbytes)
                u.n = off - u.off
                for r in u.extra:
                    r.xoff, xoff = xoff, _al(xoff + r.nbytes)
            self.entry_bytes = off
            self.entry_recs = [r for u in self.units for r in u.entry]
            self.extra_recs = [r for u in self.units for r in u.extra]
            self.extras = torch.empty(max(xoff, 1), dtype=torch.uint8)
            for r in self.extra_recs:
                self.extras[r.xoff:r.xoff + r.nbytes].copy_(self.master[r.moff:r.moff + r.nbytes])
            # checksums of the final bytes (threads: zlib / hashlib release the GIL on large buffers)
            with ThreadPoolExecutor(THREADS) as ex:
                def one(r):
                    v = _tmv(self.master, r.moff, r.nbytes)
                    return zlib.crc32(v), hashlib.blake2b(v, digest_size=32).digest()
                for r, (c, d) in zip(self.entry_recs, ex.map(one, self.entry_recs)):
                    r.crc, r.dig = c, d
            self.digest = hashlib.blake2b(b"".join(r.dig for r in self.entry_recs), digest_size=32).digest()
            man = {"v": FORMAT, "align": ALIGN, "bytes": self.entry_bytes,
                   "t": [[r.qname, r.kind, list(r.shape), str(r.dtype).replace("torch.", ""), r.off, r.nbytes]
                         for r in self.entry_recs]}
            self.manifest = json.dumps(man, separators=(",", ":"), sort_keys=True).encode()
            self.model_id = _model_id(model_dir) if model_dir else "unknown"
            self.key = hashlib.blake2b(b"llmt-lazyvis|%d|" % FORMAT + self.model_id.encode() + b"|" + self.manifest
                                       + b"|" + self.digest, digest_size=32).digest()
            self.meta = b"LZV1" + self.digest + struct.pack("<QI", self.entry_bytes, len(self.entry_recs))
            ckb = sum(r.nbytes for r in self.entry_recs)
            self.reload_ms = treads * 1e3 if ckb else None
            if not self.entry_recs:
                self.keep_host = True
                _log("no tensor of the tower matches the checkpoint in %s (prefix %r): keeping a private host copy "
                     "(old behaviour)", model_dir, prefix, level="warning")
            self.broker = broker if (self.entry_recs and not self.keep_host) else None
            self.final = True
            _log("manifest: %d tensors, %.3f GiB reloadable from %s (prefix %r, %d converted %s), extras %d tensors "
                 "%.2f MiB kept on the host; digest %s key %s; checkpoint read %.0f ms, finalize %.0f ms; broker %s",
                 len(self.entry_recs), ckb / 2**30, model_dir, prefix, nconv,
                 "from the checkpoint dtype" if nconv else "", len(self.extra_recs),
                 sum(r.nbytes for r in self.extra_recs) / 2**20, self.digest.hex()[:16], self.key.hex()[:16],
                 treads * 1e3, (time.perf_counter() - t0) * 1e3,
                 ("%s ns %r" % (broker.sock, broker.ns_name)) if self.broker else "off")

    def _prefix(self, ck):
        names = [r.qname for r in self.recs if r.kind == "p"]
        if not names or not ck:
            return None
        best, score = None, 0
        first = names[0]
        for k in ck:
            if k == first or k.endswith("." + first):
                p = k[:len(k) - len(first)]
                s = sum(1 for n in names if p + n in ck)
                if s > score:
                    best, score = p, s
        return best

    def mark_ready(self):
        """Engine warmed up: free the startup host copy (nothing of the tower stays in host RAM but the extras)."""
        with self.lock:
            if self.resident:
                self._all_off()
                self.resident = False
            if not self.final or self.keep_host:
                _log("startup host copy kept (%s)", "KEEP_HOST" if self.keep_host else "not finalized")
                return
            for r in self.recs:
                self._set(r, _placeholder(r.shape, r.dtype))
            self.master = None
            _log("startup host copy freed: %.3f GiB (extras kept: %.2f MiB)", self.bytes / 2**30,
                 sum(r.nbytes for r in self.extra_recs) / 2**20)

    # --- sources
    def _cost_ms(self):
        floor = self.entry_bytes / (DISK_MBPS * 1e6) * 1e3
        return max(floor, self.reload_ms or 0.0)

    def _open(self, force_disk=False):
        if self.master is not None:
            return _MasterSource(self)
        if self.private is not None:
            return self.private
        if force_disk or self.broker is None:
            return self._disk(None)
        src = self.broker.lookup(self.key, self.entry_bytes, self.meta)
        if src is not None and VERIFY_HIT and not self._verify(src.views):
            self.broker.release(src.conn, src.lease)
            _log("broker entry failed its crc check: reloading from disk", level="warning")
            src = None
        if src is not None:
            return src
        return self._disk(self.broker if (STORE and _STATE["ready"]) else None)

    def _verify(self, views_of):
        with ThreadPoolExecutor(THREADS) as ex:
            res = list(ex.map(lambda r: _crc(views_of(r.off, r.nbytes)) == r.crc, self.entry_recs))
        return all(res)

    def _read(self, views_of):
        """Checkpoint -> destination (entry layout), converting dtypes as the loader did; returns the tensors whose
        crc32 differs from startup."""
        fds = {}
        bad = []
        spans = {}
        try:
            with ThreadPoolExecutor(THREADS) as ex:
                futs = []
                for r in sorted(self.entry_recs, key=lambda r: (r.ck[0], r.ck[1])):
                    path, foff, n, cdt = r.ck
                    fd = fds.get(path)
                    if fd is None:
                        fd = fds[path] = os.open(path, os.O_RDONLY)
                    a, b = spans.get(path, (foff, foff + n))
                    spans[path] = (min(a, foff), max(b, foff + n))
                    dst = views_of(r.off, r.nbytes)
                    if cdt == r.dtype:
                        _preadv_all(fd, list(dst), foff)
                    else:
                        raw = torch.empty(n, dtype=torch.uint8)
                        _preadv_all(fd, [_tmv(raw)], foff)
                        conv = raw.view(cdt).view(r.shape).to(r.dtype).reshape(-1).view(torch.uint8)
                        src, pos = _tmv(conv), 0
                        for v in dst:
                            v[:] = src[pos:pos + len(v)]
                            pos += len(v)
                    futs.append((r, ex.submit(_crc, dst)))
                bad = [r for r, f in futs if f.result() != r.crc]
        finally:
            for path, fd in fds.items():
                if DROP_CACHE and path in spans:
                    try:
                        a, b = spans[path]
                        os.posix_fadvise(fd, a, b - a, os.POSIX_FADV_DONTNEED)
                    except (OSError, AttributeError):
                        pass
                os.close(fd)
        return bad

    def _disk(self, broker):
        """Reload from the checkpoint. With a broker: store it (kind 5) and hand out a leased pool source; otherwise a
        temporary host buffer."""
        t0 = time.perf_counter()
        self.stats["disk_loads"] += 1
        if broker is not None:
            res = broker.reserve(self.key, self.entry_bytes, self._cost_ms(), self.meta)
            if res == "busy" or res == "exists":
                if res == "exists" or broker.wait(self.key, BUSY_WAIT_MS):
                    src = broker.lookup(self.key, self.entry_bytes, self.meta)
                    if src is not None:
                        self.stats["disk_loads"] -= 1
                        return src
                res = None
            if res is not None:
                conn, ranges = res
                stored = False
                if broker.fallocate(conn, ranges):
                    views_of = lambda off, n: [_mv(conn.base + po, ln) for po, ln, _ in _segs(ranges, off, n)]  # noqa: E731
                    try:
                        bad = self._read(views_of)
                    except OSError as e:
                        broker.abort(conn, self.key)
                        raise RuntimeError(f"lazyvis: checkpoint reload failed: {e}") from e
                    if bad:
                        broker.abort(conn, self.key)
                        return self._mismatch(bad)
                    stored = broker.commit(conn, self.key)
                else:
                    broker.abort(conn, self.key)
                if stored:
                    self.reload_ms = (time.perf_counter() - t0) * 1e3
                    _log("tower reloaded from disk into the broker (kind 5, %.3f GiB, cost_hint %.0f ms): %.0f ms",
                         self.entry_bytes / 2**30, self._cost_ms(), self.reload_ms)
                    src = broker.lookup(self.key, self.entry_bytes, self.meta)
                    if src is not None:
                        return src
                    _log("stored entry gone before it could be leased: temporary host copy", level="warning")
        buf = torch.empty(max(self.entry_bytes, 1), dtype=torch.uint8)
        try:
            bad = self._read(lambda off, n: [_tmv(buf, off, n)])
        except OSError as e:
            raise RuntimeError(f"lazyvis: checkpoint reload failed: {e}") from e
        if bad:
            return self._mismatch(bad, buf)
        self.reload_ms = (time.perf_counter() - t0) * 1e3
        if self.stats["disk_loads"] <= 3 or self.stats["disk_loads"] % 100 == 0:
            _log("tower reloaded from disk into a temporary host buffer: %.0f ms (disk load %d)", self.reload_ms,
                 self.stats["disk_loads"])
        return _HostSource(buf)

    def _mismatch(self, bad, buf=None):
        self.stats["mismatch"] += 1
        _log("CHECKPOINT CHANGED: %d of %d vision tensors differ from the startup checksums (first: %s). The broker is "
             "not used again by this engine; the bytes now on disk are kept as a private host copy. Restart the "
             "engine to pick up the new checkpoint.", len(bad), len(self.entry_recs), bad[0].qname, level="error")
        self.broker = None
        if buf is None:
            buf = torch.empty(max(self.entry_bytes, 1), dtype=torch.uint8)
            self._read(lambda off, n: [_tmv(buf, off, n)])
        self.private = _HostSource(buf, keep=True)
        return self.private

    # --- device side
    def _unit_on(self, u, src):
        buf = torch.empty(u.n, dtype=torch.uint8, device=self.dev) if u.n else None
        if buf is not None:
            t0 = time.perf_counter()
            src.copy_unit(u, buf)
            self.copy_s += time.perf_counter() - t0
            self.copy_b += u.n
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

    def _unit_off(self, u):
        for r in u.recs:
            self._set(r, self._mview(r) if self.master is not None else _placeholder(r.shape, r.dtype))
        u.dev, u.on_dev = None, False

    def _all_off(self):
        for u in self.units:
            if u.on_dev:
                self._unit_off(u)

    def _count(self, src):
        if src.what == "pool":
            self.stats["pool_loads"] += 1
            self.pool_used = True
        elif src.what == "host":
            self.stats["host_loads"] += 1

    # --- block streaming hooks (active only while self.streaming)
    def _blk_pre(self, u):
        def pre(_m, _a):
            if self.streaming:
                self._unit_on(u, self.src)
        return pre

    def _blk_post(self, u):
        def post(_m, _a, _o):
            if self.streaming:
                self._unit_off(u)
        return post

    def _room(self):
        """Device memory left after the engine's activation reserve (profiled peak), counting what is free on the
        device, cached unused in torch, and activations in flight right now (they come out of that reserve)."""
        free, _total = _ACC.mem_get_info(self.dev)
        alloc = _ACC.memory_allocated(self.dev)
        cached = _ACC.memory_reserved(self.dev) - alloc
        transient = 0
        if _STATE.get("base_alloc") is not None:
            transient = max(0, alloc - _STATE["base_alloc"] - (self.bytes if self.resident else 0))
        return free + cached + transient - (_STATE["act_reserve"] or 0)

    def _rate(self):
        r = "%.3f GiB copied in %.0f ms host-side (%.2f GiB/s)" % (self.copy_b / 2**30, self.copy_s * 1e3,
                                                                     self.copy_b / 2**30 / max(self.copy_s, 1e-9))
        self.copy_s, self.copy_b = 0.0, 0
        return r

    def _load_all(self, force_disk=False):
        t0 = time.perf_counter()
        self.copy_s, self.copy_b = 0.0, 0
        src = self._open(force_disk)
        ok = False
        try:
            for u in self.units:
                self._unit_on(u, src)
        except BaseException:
            src.close()
            self._all_off()
            raise
        ok = src.close()
        if not ok:
            self.stats["redo"] += 1
            self._all_off()
            _log("resident copy from the broker invalid after the copy (lease lost): reloading from disk",
                 level="warning")
            return self._load_all(force_disk=True)
        self._count(src)
        return src.what, (time.perf_counter() - t0) * 1e3

    def _enter(self, force_disk=False):
        now = time.time()
        self.last_use, self.n_img = now, self.n_img + 1
        if self.resident:
            return
        want = MODE == "resident" or (MODE == "auto" and _STATE["ready"] and _STATE["act_reserve"] is not None
                                      and self._room() >= self.bytes + MARGIN)
        if want and not force_disk:
            try:
                what, ms = self._load_all()
                self.resident = True
                self.stats["resident_loads"] += 1
                _log("tower loaded resident from %s: %.3f GiB in %.1f ms (load %d; pool %d disk %d host %d; %s)", what,
                     self.bytes / 2**30, ms, self.stats["resident_loads"], self.stats["pool_loads"],
                     self.stats["disk_loads"], self.stats["host_loads"], self._rate())
                return
            except torch.OutOfMemoryError:
                _log("resident load hit OOM: streaming instead")
        self.streaming = True
        self.copy_s, self.copy_b = 0.0, 0
        t0 = time.perf_counter()
        self.src = self._open(force_disk)
        try:
            self._unit_on(self.units[0], self.src)
        except BaseException:
            self._exit()
            raise
        self.stats["streamed"] += 1
        if self.stats["streamed"] <= 3 or self.stats["streamed"] % 100 == 0:
            _log("streaming the tower block by block from %s (call %d; ready %s, room %s; source %.0f ms)",
                 self.src.what, self.stats["streamed"], _STATE["ready"],
                 None if _STATE["act_reserve"] is None else "%.2f GiB" % (self._room() / 2**30),
                 (time.perf_counter() - t0) * 1e3)

    def _exit(self):
        if not self.streaming:
            return True
        self._all_off()
        self.streaming = False
        src, self.src = self.src, None
        if src is None:
            return True
        ok = src.close()
        if ok:
            self._count(src)
        if self.stats["streamed"] <= 5 or self.stats["streamed"] % 100 == 0:
            _log("streamed forward %d from %s: %s", self.stats["streamed"], src.what, self._rate())
        return ok

    def maybe_release(self):
        with self.lock:
            idle = time.time() - self.last_use
            if not self.resident:
                # streaming engines (the 5090: no room for a resident tower) never "release": the burst ends here
                if COLD_ON_RELEASE and self.pool_used and self.broker is not None and idle >= TTL:
                    self.pool_used = False
                    self.broker.cold(self.key)
                return
            why = None
            if idle >= TTL:
                why = "idle %.0f s" % idle
            elif MODE == "auto" and self._room() < 0:   # the activation reserve is no longer guaranteed
                why = "memory pressure"
            if why:
                self._all_off()
                self.resident = False
                self.stats["releases"] += 1
                if COLD_ON_RELEASE and self.broker is not None and why.startswith("idle"):
                    self.pool_used = False
                    self.broker.cold(self.key)
                b = self.broker.n if self.broker else {}
                _log("tower released (%s): %.3f GiB back to the device pool; nothing kept on the host (stats %s; "
                     "broker %s)", why, self.bytes / 2**30, self.stats, {k: v for k, v in b.items() if v})


def _numel(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _esize(dt):
    return torch.empty((), dtype=dt).element_size()


def _reaper():
    while True:
        time.sleep(min(5.0, max(0.5, TTL / 4)))
        for lv in list(_STATE["towers"]):
            try:
                lv.maybe_release()
            except Exception as e:  # noqa: BLE001
                _log("release check failed: %r", e)


def _vllm_ctx():
    """(model dir, broker socket, engine id) from vLLM's current config and the environment."""
    model_dir, sock, eid = _ENV("EXL3_LAZY_VISION_MODEL_DIR"), None, None
    try:
        from vllm.config import get_current_vllm_config
        vc = get_current_vllm_config()
        model_dir = model_dir or getattr(vc.model_config, "model", None)
        kt = getattr(vc, "kv_transfer_config", None)
        extra = (getattr(kt, "kv_connector_extra_config", None) or {}) if kt is not None else {}
        sock = extra.get("broker_socket")
        eid = extra.get("broker_engine_id")
    except Exception:  # noqa: BLE001
        pass
    if BROKER == "off":
        sock = None
    elif BROKER != "auto":
        sock = BROKER
    elif not sock:
        sock = "/run/llmt-kv/broker.sock"
    eid = _ENV("EXL3_LAZY_VISION_ENGINE_ID") or _ENV("LLMT_KV_ENGINE_ID") or eid or 0
    return model_dir, sock, int(eid)


def _make_broker(sock, eid):
    if not sock or KEEP_HOST:
        return None
    return _Broker(sock, eid, _ENV("EXL3_LAZY_VISION_NAME", "lazyvis-%d" % eid), NS_NAME)


def _finalize(lv):
    model_dir, sock, eid = _vllm_ctx()
    try:
        lv.finalize(model_dir, _make_broker(sock, eid))
    except Exception as e:  # noqa: BLE001 - never fail the engine start: keep the startup copy for life
        lv.keep_host = True
        _log("finalize failed (%r): keeping a private host copy (old behaviour)", e, level="warning")


def _offload(model, early):
    vis = getattr(model, "visual", None)
    if isinstance(vis, torch.nn.Module) and hasattr(vis, "_lazyvis") and not early:
        _finalize(vis._lazyvis)
        return
    if isinstance(vis, torch.nn.Module) and not hasattr(vis, "_lazyvis") and any(True for _ in vis.parameters()):
        try:
            lv = LazyVision(vis, copy=True)   # always copy: buffers (rotary cos/sin cache, ...) are set at construction
        except Exception as e:  # noqa: BLE001
            _log("not enabled (%r): tower stays resident", e)
            return
        # hand the freed blocks back to the device: vLLM's profiler counts torch-cached-but-unallocated memory as
        # "non-torch" usage, which would take the tower's bytes away from the KV cache again
        _sync()
        _ACC.empty_cache()
        _STATE["towers"].append(lv)
        if len(_STATE["towers"]) == 1:
            threading.Thread(target=_reaper, name="lazyvis-reaper", daemon=True).start()
        if not early:
            _finalize(lv)


def install():
    """Install the lazy tower (every vLLM process; the plugin entry point calls it). Model-loader points are wrapped
    (vLLM has no loader hook yet); KV sizing and warm-up use llmt_kv.vllm_hooks."""
    if not ON or _STATE.get("installed"):
        return
    _STATE["installed"] = True
    import vllm.model_executor.model_loader.base_loader as bl
    from llmt_kv import vllm_hooks as H
    orig_pwal = bl.process_weights_after_loading

    def pwal(model, *a, **k):
        r = orig_pwal(model, *a, **k)
        _offload(model, early=False)
        return r
    bl.process_weights_after_loading = pwal

    # earliest point: right after the model is built, before load_weights (base_loader calls log_model_inspection in
    # between). The tower's (still uninitialized) params move to the host copy and the checkpoint is read straight
    # into it, so the tower never holds device memory that the driver might keep pooled afterwards.
    orig_inspect = bl.log_model_inspection

    def inspect(model, *a, **k):
        r = orig_inspect(model, *a, **k)
        _offload(model, early=True)
        return r
    bl.log_model_inspection = inspect

    def available_kv_memory(r, worker):
        _STATE["act_reserve"] = int(getattr(worker, "peak_activation_memory", 0) or 0)
        _log("activation reserve from the profile run: %.2f GiB", _STATE["act_reserve"] / 2**30)
        return r

    def warmed_up(worker):
        _STATE["ready"] = True
        for lv in _STATE["towers"]:   # whatever warm-up loaded goes back out; the startup host copy is freed
            lv.mark_ready()
        _sync()
        _STATE["base_alloc"] = _ACC.memory_allocated()   # idle engine: weights + KV + persistent buffers

    H.register("worker.available_kv_memory", available_kv_memory)
    H.register("worker.warmed_up", warmed_up)
    _log("installed (mode %s, ttl %.0f s, broker %s, namespace %r)", MODE, TTL, BROKER, NS_NAME)
