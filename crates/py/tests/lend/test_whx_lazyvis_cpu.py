"""CPU check: the keeper ATTACH path fires base_loader's model hooks (bl.log_model_inspection before load_weights,
bl.process_weights_after_loading after), which is where the lazy vision tower (exl3xpu.lazyvis / llmt_lazyvis) does
its early offload and late finalize -- so after an attach the tower is offloaded exactly as after an export start.
Runs both deploy copies (Arc, 5090) with stubbed vLLM model building and keeper backend."""
import importlib.util, sys, types, torch

import vllm.model_executor.model_loader.base_loader as bl
import vllm.model_executor.model_loader.utils as mu

class Vis(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.zeros(4))
class M(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.visual = Vis(); self.lm = torch.nn.Linear(2, 2)

for path in sys.argv[1:]:
    pkg = types.ModuleType("fakepkg"); pkg.__path__ = []; sys.modules["fakepkg"] = pkg
    nl = types.ModuleType("fakepkg.nommap_loader"); nl.safetensors_weights_iterator_nommap = lambda *a, **k: iter(()); nl._DT = {}
    sys.modules["fakepkg.nommap_loader"] = nl
    import importlib.machinery as _im
    spec = importlib.util.spec_from_file_location("fakepkg.whx", path, loader=_im.SourceFileLoader("fakepkg.whx", path))
    W = importlib.util.module_from_spec(spec)
    sys.modules["fakepkg.whx"] = W; spec.loader.exec_module(W)
    calls = []
    accs = []
    _acc = types.SimpleNamespace(synchronize=lambda: accs.append("sync"), empty_cache=lambda: accs.append("empty"))
    W._bk = lambda: types.SimpleNamespace(dev="cpu", init=lambda: None, finish=lambda m: None, acc=lambda: _acc)
    W._attach_tensors = lambda model, *a, **k: []
    W._walk = lambda model, private=False: iter(())
    built = {}
    def init_model(**k):
        m = M(); built["m"] = m; return m
    mu.initialize_model = init_model
    def pwal(model, cfg, dev):
        calls.append(("pwal-%s%s" % ("tower" if isinstance(model, Vis) else "model", "" if isinstance(model, Vis) or
                      "visual" not in model._modules else "+tower"), str(dev)))
    mu.process_weights_after_loading = pwal
    o_ins, o_pw = bl.log_model_inspection, bl.process_weights_after_loading
    lz = types.ModuleType("exl3xpu.lazyvis"); lz._STATE = {"installed": True}
    def offload(model, early):
        assert "visual" in model._modules, "the tower is attached when the lazy module sees it"
        calls.append(("early" if early else "late", model))
    lz._offload = offload
    sys.modules["exl3xpu.lazyvis"] = lz
    loader = types.SimpleNamespace(load_weights=lambda model, cfg: calls.append(("load", model)))
    vc = types.SimpleNamespace(device_config=types.SimpleNamespace(device="cpu"))
    mc = types.SimpleNamespace(dtype=torch.float32)
    try:
        model = W._attach(loader, vc, mc, "", "k", {"manifest": [], "chunks": []}, [])
    finally:
        bl.log_model_inspection, bl.process_weights_after_loading = o_ins, o_pw
    seq = [c[0] for c in calls]
    assert seq == ["early", "load", "pwal-model", "pwal-tower", "late"], seq
    assert dict(calls)["pwal-model"] == "meta" and dict(calls)["pwal-tower"] == "cpu", calls   # tower: real device
    assert accs == ["sync", "empty"], accs     # the tower's post-load pass hands its pooled blocks back
    assert W._lazy_installed(), "lazy module detected"
    assert "visual" in model._modules and model is built["m"]
    print("WHX ATTACH LAZYVIS OK:", path.rsplit("/", 1)[-1], [c[0] for c in calls])
    sys.modules.pop("exl3xpu.lazyvis", None)
