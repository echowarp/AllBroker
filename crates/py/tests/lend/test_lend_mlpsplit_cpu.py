"""LoRA-step MLP row split: base rows fused, adapter rows plain+LoRA in chunks of EXL3_LORA_MLP_ROWS; the result must equal
the all-plain path with a bounded plain intermediate. Fake MLP/LoRA: the plain path reads the punica mapping front entries
exactly as the real kernels do."""
import os
os.environ["EXL3_LORA_HOT"] = "1"
import numpy as np, torch
from llmt_kv import lend as L
torch.manual_seed(0)
H, I, NS, R = 32, 96, 3, 4
Wg, Wd = torch.randn(H, I) * .1, torch.randn(I, H) * .1
A = torch.randn(NS, H, R) * .1; B = torch.randn(NS, R, H) * .1
class PW: pass
pw = PW(); pw._token_lora_indices = torch.zeros(5000, dtype=torch.int64)
pw.token_mapping_meta = type("M", (), {"token_lora_mapping": torch.zeros(5000, dtype=torch.int32)})()
mgr = type("G", (), {})(); mgr.punica_wrapper = pw; mgr.lora_index_to_id = [11, 22, None]
L._ST.update(mgr=mgr, dev=torch.device("cpu"))
peak = {"rows": 0, "calls": 0}
def base_mlp(x): return torch.nn.functional.silu(x @ Wg) @ Wd
def plain(m, x):   # base + LoRA delta by the mapping front (slot -1 = none)
    peak["rows"] = max(peak["rows"], x.shape[0]); peak["calls"] += 1
    sl = pw._token_lora_indices[:x.shape[0]]
    y = base_mlp(x)
    for j in range(NS):
        mk = (sl == j)
        if mk.any(): y[mk] = y[mk] + (x[mk] @ A[j]) @ B[j]
    return y
def fused(m, x): return base_mlp(x)
def run(T, nad, ids_cycle):
    x = torch.randn(T, H)
    tm = np.zeros(T, dtype=np.int64)
    if nad:
        sel = np.random.RandomState(T).rand(T) < 0.3
        tm[sel] = np.array(ids_cycle[:nad])[np.arange(sel.sum()) % nad]
    rows = np.nonzero(tm > 0)[0]; lr = (rows, tm[rows], T)
    slot = {11: 0, 22: 1}
    full = torch.tensor([slot.get(int(v), -1) for v in tm])
    pw._token_lora_indices[:T] = full; pw.token_mapping_meta.token_lora_mapping[:T] = full.int()
    ref = plain(None, x)                               # old path: all rows plain, full mapping
    peak["rows"] = 0; L._ST["msplit"] = None; L._ST["lrows"] = lr
    plan = L._mlp_split_plan(lr, T)
    if len(rows) == 0: assert plan is None; return 0.0, 0
    before = pw._token_lora_indices[:T].clone()
    out = L._mlp_split(None, x, plain, fused, 64, plan)
    assert torch.equal(pw._token_lora_indices[:T], before), "mapping restored"
    assert peak["rows"] <= 64, "plain intermediate bounded by the chunk"
    return (out - ref).abs().max().item(), peak["rows"]
for T, nad in ((300, 0), (300, 1), (300, 2), (1000, 2), (200, 2)):
    d, pk = run(T, nad, [11, 22])
    print(f"T={T} adapters={nad}: max|diff|={d:.2e} plain rows<={pk}")
    assert d < 1e-5
# all rows adapters (a LoRA request's own long prefill)
x = torch.randn(500, H); tm = np.array([11, 22] * 250); rows = np.arange(500)
full = torch.tensor([0, 1] * 250); pw._token_lora_indices[:500] = full; pw.token_mapping_meta.token_lora_mapping[:500] = full.int()
ref = plain(None, x); peak["rows"] = 0; L._ST["msplit"] = None
lr = (rows, tm, 500); out = L._mlp_split(None, x, plain, fused, 64, L._mlp_split_plan(lr, 500))
assert (out - ref).abs().max() < 1e-5 and peak["rows"] <= 64
print("ALL-ADAPTER rows chunked OK, exact")
print("MLP-SPLIT OK: split + chunked == all-plain (fp32 rounding level), plain rows bounded, mapping restored")
print("ALL OK")
