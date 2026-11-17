"""llmt_kv.nommap_loader reads every tensor of a safetensors file exactly (dtypes, shapes, zero-size, file order).
  python test_nommap_loader.py   (needs torch; not vLLM)"""
import importlib.util, json, os, struct, sys, tempfile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("nommap_loader", os.path.join(HERE, "..", "python", "llmt_kv", "nommap_loader.py"))
nl = importlib.util.module_from_spec(spec); spec.loader.exec_module(nl)

g = torch.Generator().manual_seed(0)
tensors = {
    "a.weight": torch.randn(64, 33, generator=g).to(torch.bfloat16),
    "b.bias": torch.randn(7, generator=g).to(torch.float16),
    "c.idx": torch.randint(-2**31, 2**31 - 1, (5, 3), generator=g, dtype=torch.int64).to(torch.int32),
    "d.empty": torch.empty(0, 4, dtype=torch.float32),
    "e.big": torch.randn(1 << 20, generator=g),             # 4 MiB, several preadv calls on some kernels
}
names = {torch.bfloat16: "BF16", torch.float16: "F16", torch.int32: "I32", torch.float32: "F32"}
hdr, blobs, off = {"__metadata__": {"format": "pt"}}, [], 0
for k in ["e.big", "a.weight", "d.empty", "b.bias", "c.idx"]:     # file order differs from insertion order below
    b = tensors[k].contiguous().view(torch.uint8).numpy().tobytes()
    hdr[k] = {"dtype": names[tensors[k].dtype], "shape": list(tensors[k].shape), "data_offsets": [off, off + len(b)]}
    blobs.append(b); off += len(b)
h = json.dumps(hdr).encode()
with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
    f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs)); path = f.name
try:
    got = list(nl.iter_file(path))
    assert [k for k, _ in got] == ["e.big", "a.weight", "d.empty", "b.bias", "c.idx"], [k for k, _ in got]
    for k, t in got:
        assert t.dtype == tensors[k].dtype and t.shape == tensors[k].shape, k
        assert torch.equal(t, tensors[k]), k
    print("test_nommap_loader OK")
finally:
    os.unlink(path)
