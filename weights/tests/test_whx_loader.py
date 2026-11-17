#!/usr/bin/env python3
"""CPU test of the weight keeper loader (llmt_kv/whx_loader.py): key, walk, export, attach, checksums and fd
passing, with memfds standing in for GPU memory and a fake keeper thread speaking the llmt-keeper wire protocol.
No GPU and no vLLM needed (torch only).  python3 test_whx_loader.py   (or pytest)"""
import json
import mmap
import os
import socket
import sys
import tempfile
import threading
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "crates", "py", "python", "llmt_kv"))
import whx_loader as L  # noqa: E402


# ------------------------------------------------------------------ fake keeper (same framing as llmt-keeper)
class FakeKeeper:
    def __init__(self, path):
        self.path, self.entries = path, {}
        self.s = socket.socket(socket.AF_UNIX); self.s.bind(path); self.s.listen(8)
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            c, _ = self.s.accept()
            hdr, fds, _, _ = socket.recv_fds(c, 8, 256)
            while len(hdr) < 8:
                hdr += c.recv(8 - len(hdr))
            n = int.from_bytes(hdr, "little"); body = b""
            while len(body) < n:
                body += c.recv(n - len(body))
            req = json.loads(body); op = req.pop("op"); key = req.pop("key", None); out, ofds = {}, []
            if op == "put":
                assert len(fds) == len(req["chunks"]), (len(fds), req["chunks"])
                self.entries[key] = (req, fds); out = {"ok": True}
            elif op == "get":
                if key in self.entries:
                    out = dict(self.entries[key][0], found=True); ofds = self.entries[key][1]
                else:
                    out = {"found": False}
            elif op == "list":
                out = {k: {"chunks": len(f), "bytes": sum(m["chunks"])} for k, (m, f) in self.entries.items()}
            elif op == "drop":
                e = self.entries.pop(key, None)
                for fd in (e[1] if e else []):
                    os.close(fd)
                out = {"dropped": e is not None}
            d = json.dumps(out).encode()
            socket.send_fds(c, [len(d).to_bytes(8, "little")], ofds); c.sendall(d); c.close()


# ------------------------------------------------------------------ fake device backend: memfds on the CPU
class FakeBackend(L.XPUBackend):
    name = dev = "cpu"
    mods = ("__main__", "test_whx_loader")
    skip = (".scratch",)
    maps = []
    registry = {}           # id(model) -> extra root (stands in for the CUDA plugin's _SEGS table)
    finished = []

    def check(self):
        pass

    def init(self):
        pass

    def acc(self):
        return types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)

    def _map(self, fd, n):
        m = mmap.mmap(fd, n); self.maps.append(m)
        return torch.frombuffer(m, dtype=torch.uint8)

    def export_alloc(self, n):
        fd = os.memfd_create("whx-test"); os.ftruncate(fd, n)
        return self._map(fd, n), fd

    def import_fd(self, fd, n):
        return self._map(fd, n)

    def roots(self, model):
        return self.registry.get(id(model))

    def finish(self, model):
        self.finished.append(id(model))


class Lin(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, k, dtype=torch.float16), requires_grad=False)
        self.register_buffer("scale", torch.empty(n))
        self.extra = (torch.empty(4, dtype=torch.int32), 3)          # tensor inside a tuple attribute
        self.d = {"idx": torch.empty(8, dtype=torch.int64)}          # tensor inside a dict attribute
        self.scratch = {"buf": torch.empty(16)}                       # skipped by the backend (runtime buffers;
        # the skip list matches container paths, like the CUDA draft-head dicts)


class Model(nn.Module):
    def __init__(self, n_layers=3, extra=False):
        super().__init__()
        self.embed = nn.Parameter(torch.empty(32, 64, dtype=torch.bfloat16), requires_grad=False)
        self.layers = nn.ModuleList([Lin(64, 128) for _ in range(n_layers)])
        self.view = self.layers[0].weight[:4]                         # a view (shares storage with a parameter)
        if extra:
            self.more = nn.Parameter(torch.empty(3))


def fill(m, seed):
    g = torch.Generator().manual_seed(seed)
    for _, t in L._walk(m):
        if t.is_floating_point():
            t.data.copy_(torch.randn(t.shape, generator=g).to(t.dtype))
        else:
            t.data.copy_(torch.randint(0, 1000, t.shape, generator=g).to(t.dtype))


def registry(dev="cpu"):
    with torch.device(dev):
        return {"layers.0": [{"tr": torch.empty(16, 8, dtype=torch.int16), "suh": torch.empty(64, dtype=torch.float16)}],
                "layers.1": [{"tr": torch.empty(16, 8, dtype=torch.int16)}, {"tr": torch.empty(8, 8, dtype=torch.int16)}]}


def snapshot(m):
    return {p: t.detach().clone() for (_, t), p in zip(list(L._walk(m)), list(L._PATHS))
            if not p.startswith("._whx_bufs")}                     # the chunks themselves (kept alive by the model)


def setup():
    d = tempfile.mkdtemp(prefix="whx-test-")
    os.environ["EXL3_WHX"] = os.path.join(d, "keeper.sock")
    os.environ["EXL3_WHX_DIR"] = d
    os.environ["EXL3_WHX_CHUNK_MB"] = "1"
    L._CHUNK = 1 << 20
    L._BK[0] = FakeBackend()
    L._REG.clear()
    FakeBackend.registry.clear()
    return FakeKeeper(os.environ["EXL3_WHX"]), d


def export_one(seed=1, key="k1", n_layers=3):
    m = Model(n_layers)
    FakeBackend.registry[id(m)] = registry()
    fill(m, seed)
    ref = snapshot(m)
    L._export(m, key)
    return m, ref


def attach_meta(key, n_layers=3, extra=False):
    with torch.device("meta"):
        mm = Model(n_layers, extra)
    FakeBackend.registry[id(mm)] = registry("meta")
    entry, fds = L._rpc({"op": "get", "key": key})
    assert entry["found"], entry
    L._attach_tensors(mm, key, entry, fds)
    return mm


# ------------------------------------------------------------------ tests
def test_key_env():
    env = {"EXL3_INT8_PREFILL": "1", "EXL3_WHX": "/s", "EXL3_WHX_DIR": "/d", "EXL3_KSMOOTH": "/t", "LLMT_FOO": "1",
           "VLLM_USE_X": "1", "VLLM_KV_CACHE_LAYOUT": "NHD", "EXL3_TAIL_HOLD": "1", "EXL3_GDN_RECOMPUTE": "1", "HOME": "/"}
    assert L._key_env(env) == {"EXL3_INT8_PREFILL": "1", "VLLM_USE_X": "1"}, L._key_env(env)
    cfg = types.SimpleNamespace(model=tempfile.mkdtemp(), dtype=torch.float16)
    open(os.path.join(cfg.model, "config.json"), "w").write("{}")
    saved = dict(os.environ)
    try:
        k0 = L._key(cfg, 0)
        assert L._key(cfg, 0) == k0 and L._key(cfg, 1) != k0
        os.environ["LLMT_TEST_FLAG"] = "1"; assert L._key(cfg, 0) == k0      # test flags: non-EXL3 names
        os.environ["EXL3_WHX_CHUNK_MB"] = "64"; assert L._key(cfg, 0) == k0
        os.environ["EXL3_NEW_WEIGHT_SETTING"] = "1"; assert L._key(cfg, 0) != k0
        del os.environ["EXL3_NEW_WEIGHT_SETTING"]
        os.utime(os.path.join(cfg.model, "config.json"), ns=(1, 1)); assert L._key(cfg, 0) != k0  # files changed
    finally:
        os.environ.clear(); os.environ.update(saved)


def test_walk_paths():
    setup()
    m = Model(2)
    FakeBackend.registry[id(m)] = registry()
    list(L._walk(m)); p = list(L._PATHS)
    assert ".layers.0.weight" in p and ".layers.0.extra(0)" in p and ".layers.0.d[idx]" in p and ".view" in p, p
    assert not any("scratch" in x for x in p), p                       # backend skip list
    assert "<roots>[layers.1][1][tr]" in p, p
    assert len(p) == len(set(p))


def test_export_attach_roundtrip():
    k, d = setup()
    m, ref = export_one()
    assert snapshot(m).keys() == ref.keys()
    for p, t in snapshot(m).items():                                   # the model keeps working on the moved memory
        assert torch.equal(t, ref[p]), p
    bases = [(b, n) for _, _, b, n, _ in L._REG]
    for (_, t), p in zip(list(L._walk(m)), list(L._PATHS)):          # every tensor now lives in an exported chunk
        if p.startswith("._whx_bufs"):
            continue
        assert any(b <= t.untyped_storage().data_ptr() < b + n for b, n in bases), p
    held = L._rpc({"op": "list"})[0]["k1"]
    assert held["chunks"] >= 2, held                                   # 1 MiB chunks: the model spans several
    mm = attach_meta("k1")
    got = snapshot(mm)
    assert got.keys() == ref.keys()
    for p in ref:
        assert torch.equal(got[p], ref[p]), p
    assert isinstance(mm.layers[1].weight, nn.Parameter)
    assert mm.view.untyped_storage().data_ptr() == mm.layers[0].weight.untyped_storage().data_ptr()  # alias kept
    reg = FakeBackend.registry[id(mm)]
    assert not reg["layers.1"][1]["tr"].is_meta                        # the extra root got its views too


def test_checksum_mismatch_fails():
    k, d = setup()
    m, ref = export_one()
    with torch.no_grad():
        m.layers[2].weight.add_(1)                                     # runtime write into held memory
    try:
        attach_meta("k1")
    except RuntimeError as e:
        assert "changed since the export" in str(e) and ".layers.2.weight" in str(e), e
    else:
        raise AssertionError("tampered memory attached")


def test_tensor_set_mismatch_fails():
    setup()
    export_one()
    try:
        attach_meta("k1", extra=True)
    except RuntimeError as e:
        assert "extra ['.more']" in str(e), e
    else:
        raise AssertionError("different tensor set attached")


def test_shared_storage_reference():
    setup()
    a, ref = export_one(key="target")
    b = Model(1)
    FakeBackend.registry[id(b)] = None
    fill(b, 7)
    b.embed = a.embed                                                  # drafter shares the target's embedding
    rb = snapshot(b)
    L._export(b, "drafter")
    man = L._rpc({"op": "get", "key": "drafter"})[0]["manifest"]
    assert sum(1 for x in man if x[0] == "ref") == 1, [x[:3] for x in man]
    L._REG.clear()                                                     # new process: attach target, then drafter
    at = attach_meta("target")
    with torch.device("meta"):
        bm = Model(1)
    entry, fds = L._rpc({"op": "get", "key": "drafter"})
    L._attach_tensors(bm, "drafter", entry, fds)
    assert torch.equal(bm.embed, rb[".embed"])
    assert bm.embed.untyped_storage().data_ptr() == at.embed.untyped_storage().data_ptr()   # same memory, not a copy


def test_fds_closed():
    k, _ = setup()
    before, m0 = len(os.listdir("/proc/self/fd")), len(FakeBackend.maps)
    export_one()
    attach_meta("k1")
    held = sum(len(f) for _, f in k.entries.values())
    maps = len(FakeBackend.maps) - m0                                  # Python's mmap keeps a dup of its fd
    after = len(os.listdir("/proc/self/fd"))
    assert after == before + held + maps, (before, held, maps, after)  # the loader closed every fd it got or made


class LoRAWrap(nn.Module):        # vLLM's *WithLoRA shape: base_layer + adapter slots + a shared punica wrapper
    def __init__(self, base, punica):
        super().__init__()
        self.base_layer = base
        self.lora_a_stacked = (torch.zeros(1, 1, 8, 64),)
        self.lora_b_stacked = (torch.zeros(1, 1, 128, 8),)
        self.punica_wrapper = punica


class Punica:
    def __init__(self):
        self._token_lora_indices = torch.zeros(32, dtype=torch.long)


def test_lora_wrapped_export_attaches_unwrapped():
    setup()
    m = Model(2)
    FakeBackend.registry[id(m)] = None
    fill(m, 3)
    ref = snapshot(m)
    pw = Punica()
    for i in range(2):                                                 # the runner wraps after the loader returned
        m.layers[i] = LoRAWrap(m.layers[i], pw)
    assert snapshot(m).keys() == ref.keys()                            # same path set as the unwrapped model
    L._export(m, "lora")
    with torch.device("meta"):
        mm = Model(2)
    FakeBackend.registry[id(mm)] = None
    entry, fds = L._rpc({"op": "get", "key": "lora"})
    L._attach_tensors(mm, "lora", entry, fds)                          # attach builds the unwrapped model
    got = snapshot(mm)
    for p in ref:
        assert torch.equal(got[p], ref[p]), p


def test_plugin_module_follows_backend():
    """Installed as llmt_kv.whx_loader the keeper hooks come from the device backend's engine plugin (exl3xpu /
    exl3cuda), not from a llmt_kv.vllm_plugin that does not exist."""
    saved_pkg, saved_bk = L.__package__, L._BK[0]
    try:
        L.__package__ = "llmt_kv"
        for name, pkg in (("xpu", "exl3xpu"), ("cuda", "exl3cuda")):
            mod = types.ModuleType(pkg + ".vllm_plugin")
            sys.modules[pkg], sys.modules[pkg + ".vllm_plugin"] = types.ModuleType(pkg), mod
            L._BK[0] = types.SimpleNamespace(name=name)
            assert L._plugin() is mod
        L.__package__ = "exl3cuda"                                     # vendored into a plugin package: that package
        L._BK[0] = types.SimpleNamespace(name="xpu")
        assert L._plugin() is sys.modules["exl3cuda.vllm_plugin"]
    finally:
        L.__package__, L._BK[0] = saved_pkg, saved_bk
        for pkg in ("exl3xpu", "exl3cuda"):
            sys.modules.pop(pkg + ".vllm_plugin", None); sys.modules.pop(pkg, None)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("test_whx_loader OK")
