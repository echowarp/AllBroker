"""CPU test of the weight keeper loader's tensor names and load watchdog (torch only, no GPU, no vLLM).

1. export with LoRA on, attach without: vLLM's LoRA support wraps the layers after the loader returns
   (model.lora_manager.modules[<name>] = wrapper, wrapper.base_layer = the loader's module), so an export (which runs
   after the wrap) and an attach (before it) must still give the same weights the same paths. A toy model with the
   shape of a real tree (linears at depth 6: language_model.model.layers.N.linear_attn.in_proj_qkvz) is exported
   through the real _export (fake CPU backend and keeper) with LoRA wrappers + a lora_manager alias, then a fresh
   unwrapped meta model is attached through the real _attach_tensors: every tensor must be found by path and carry the
   exported bytes. Entries recorded with the manager's alias paths (older loader versions) are matched through
   _canon_path.
2. the load watchdog: a load that stops making progress exits the process (code 70) with a 'WHX: LOAD STALL' line and
   the stacks instead of hanging (subprocess, EXL3_WHX_STALL_S=3).

python3 test_whx_lorapaths.py [<whx_loader.py>]   (default: this repo's llmt_kv/whx_loader.py; or pytest)"""
import importlib.machinery as im
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import types

import torch


def load(path, name):
    pkg = types.ModuleType(name); pkg.__path__ = []; sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location(name + ".whx", path, loader=im.SourceFileLoader(name + ".whx", path))
    W = importlib.util.module_from_spec(spec); sys.modules[name + ".whx"] = W; spec.loader.exec_module(W)
    return W


class Exl3Linear(torch.nn.Module):        # stands in for the EXL3 linear: trellis/suh/svh/mul1 + a CPU attribute
    def __init__(self, n):
        super().__init__()
        self.trellis = torch.nn.Parameter(torch.zeros(n, 16, dtype=torch.int16), requires_grad=False)
        self.suh = torch.nn.Parameter(torch.zeros(n), requires_grad=False)
        self.svh = torch.nn.Parameter(torch.zeros(n), requires_grad=False)
        self.register_buffer("mul1", torch.zeros(1))
        self.exl3_shard_of_nb = torch.zeros(2, dtype=torch.int32)


class LinAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkvz = Exl3Linear(8); self.in_proj_ba = torch.nn.Linear(4, 4, bias=False); self.out_proj = Exl3Linear(4)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear_attn = LinAttn(); self.input_layernorm = torch.nn.LayerNorm(4)


class Inner(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.layers = torch.nn.ModuleList(Layer() for _ in range(n)); self.norm = torch.nn.LayerNorm(4)


class LM(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.model = Inner(n); self.lm_head = torch.nn.Linear(4, 6, bias=False)


class Top(torch.nn.Module):
    def __init__(self, n=3):
        super().__init__(); self.language_model = LM(n)


class LoRAWrapper(torch.nn.Module):       # vLLM BaseLayerWithLoRA shape: base_layer + adapter slots (runtime state)
    def __init__(self, base):
        super().__init__(); self.base_layer = base
        self.lora_a_stacked = (torch.zeros(4, 1, 16, 8),); self.lora_b_stacked = (torch.zeros(4, 1, 8, 16),)
        self.output_slices = (8,)


class LoRAModelManager:                   # vLLM's manager: model.lora_manager = self; .modules = {name: wrapper}
    pass


LoRAModelManager.__module__ = "vllm.lora.model_manager"


def lora_wrap(model, adapters=True):
    mgr = LoRAModelManager(); mgr.model = model; mgr.modules = {}
    if adapters:                                              # adapter weights: never part of the held set
        mgr._registered_adapters = {7: {"w": torch.ones(3)}}
    for name, m in list(model.named_modules()):
        if isinstance(m, (Exl3Linear, torch.nn.Linear)) and "lm_head" not in name:
            parent, attr = name.rsplit(".", 1)
            w = LoRAWrapper(m); setattr(model.get_submodule(parent), attr, w); mgr.modules[name] = w
    model.lora_manager = mgr
    return model


def fill(model):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, t in list(model.named_parameters()) + list(model.named_buffers()):
            if t.dtype.is_floating_point:
                t.copy_(torch.randn(t.shape, generator=g))
            else:
                t.copy_(torch.randint(0, 100, t.shape, generator=g).to(t.dtype))
    return model


def fake_backend(W, store):
    class BK:
        name, dev, skip, mods, margin_mb, worker = "fake", "cpu", (), ("vllm", "__main__"), 0, "x"

        def init(self): pass
        def check(self): pass
        def acc(self): return types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
        def roots(self, model): return None
        def finish(self, model): pass
        def dyn_dump(self, host): return {}

        def export_alloc(self, n):
            b = torch.zeros(n, dtype=torch.uint8); fd = os.open("/dev/null", os.O_RDONLY); store["bufs"][fd] = b
            return b, fd

        def import_fd(self, fd, n):
            return store["by_ci"][store["fds"].index(fd)]
    W._BK[0] = BK()


def roundtrip(W, tmp):
    """Real _export of a LoRA-wrapped model, real _attach_tensors into a fresh unwrapped meta model."""
    store = {"bufs": {}}
    fake_backend(W, store)
    os.environ["EXL3_WHX_DIR"] = tmp
    W._REG.clear()
    put = {}

    def rpc(req, fds=()):
        if req["op"] == "put":
            put.update(req); store["by_ci"] = [store["bufs"][fd] for fd in fds]
        return {}, []
    W._rpc = rpc
    src = lora_wrap(fill(Top()))
    ref = {n: t.detach().clone() for n, t in fill(Top()).state_dict().items()}   # same seed: the exported bytes
    W._export(src, "k1")
    assert not any("lora_manager" in p or "base_layer" in p or "lora_a" in p for p in put["paths"]), put["paths"][:5]
    W._REG.clear()
    with torch.device("meta"):
        dst = Top()
    store["fds"] = list(range(len(store["by_ci"])))
    entry = {k: put[k] for k in ("chunks", "manifest", "kinds", "cpu_need", "paths", "tsums")}
    entry["csum"] = put.get("csum")
    # CPU attributes of the meta model are rebuilt "from files" in production: give them real storage here
    for m in dst.modules():
        if isinstance(m, Exl3Linear):
            m.exl3_shard_of_nb = torch.zeros(2, dtype=torch.int32)
    W._attach_tensors(dst, "k1", entry, store["fds"])
    sd = dst.state_dict()
    for n, t in ref.items():
        assert sd[n].device.type == "cpu" and torch.equal(sd[n], t), n
    return put["paths"]


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "..", "crates", "py", "python", "llmt_kv", "whx_loader.py")


def test_lorapaths_and_watchdog():
    main([DEFAULT])


def main(argv=None):
    argv = argv if argv is not None else (sys.argv[1:] or [DEFAULT])
    new = os.path.abspath(argv[0])
    W = load(new, "pkgnew")
    with tempfile.TemporaryDirectory() as tmp:
        ex = roundtrip(W, tmp)
    fake_backend(W, {"bufs": {}})
    plain = list(W._walk(fill(Top()))); pp = list(W._PATHS)
    wrapped = list(W._walk(lora_wrap(fill(Top())))); wp = list(W._PATHS)
    assert set(pp) == set(wp) == set(ex) and len(pp) == len(wp), (set(pp) ^ set(wp))
    assert any(p.endswith("layers.0.linear_attn.in_proj_qkvz.trellis") for p in pp)
    print(f"EXPORT-WITH-LORA -> ATTACH OK: {len(ex)} tensors, same {len(pp)} paths with and without LoRA "
          f"(e.g. {sorted(pp)[0]})")
    alias = {f".{n}": f".lora_manager.modules[{n}]" for n, m in lora_wrap(fill(Top())).lora_manager.modules.items()}
    op = []
    for p in pp:                                   # what an older loader recorded: wrapped linears under the alias
        mod = p.rsplit(".", 1)[0]
        op.append(alias[mod] + p[len(mod):] if mod in alias else p)
    assert sum(".lora_manager." in p for p in op) > 0
    assert sorted(map(W._canon_path, op)) == sorted(pp)
    print(f"OLD ALIAS PATHS NORMALISED OK: {sum('.lora_manager.' in p for p in op)} of {len(op)} paths "
          f"(e.g. {next(p for p in op if '.lora_manager.' in p)})")
    # watchdog: a stalled load exits (code 70) with a clear line, also when the stall is in a blocking wait
    code = (f"import sys, threading; sys.argv=['x']; __file__={__file__!r}; exec(open({__file__!r}).read().split('# --' + '-main')[0]); "
            f"W = load({new!r}, 'pkgwd'); W._wd_start('test load'); W._tick(); threading.Event().wait(60)")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "EXL3_WHX_STALL_S": "3"},
                       capture_output=True, text=True, timeout=90)
    dt = time.time() - t0
    out = r.stdout + r.stderr
    assert r.returncode == 70 and "WHX: LOAD STALL" in out and "test load" in out and "Thread" in out, (r.returncode, out[-2000:])
    print(f"LOAD WATCHDOG OK: stalled load exited with code 70 after {dt:.1f} s ('{[l for l in out.splitlines() if 'LOAD STALL' in l][0][:90]}...')")
    # and a progressing load is left alone
    code2 = code.replace("threading.Event().wait(60)", "import time\nfor _ in range(30): time.sleep(0.25); W._tick()\nW._wd_stop(); time.sleep(5); print('done')")
    r = subprocess.run([sys.executable, "-c", code2], env={**os.environ, "EXL3_WHX_STALL_S": "3"},
                       capture_output=True, text=True, timeout=90)
    assert r.returncode == 0 and "done" in r.stdout, (r.returncode, r.stdout[-500:], r.stderr[-1500:])
    print("LOAD WATCHDOG OK: a load that keeps ticking (7.5 s > limit 3 s) and then stops the watchdog runs to the end")


# ---main
if __name__ == "__main__":
    main()
