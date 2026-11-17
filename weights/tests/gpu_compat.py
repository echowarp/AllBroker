#!/usr/bin/env python3
"""keeper_compat.py with real GPU memory: the weight loader's own device backend (llmt_kv/whx_loader.py) exports
three chunks of device memory with a known pattern and puts them into a running keeper; later phases (normally in a
new process, after the keeper was restarted) import the fds the keeper hands back and check every byte on the device.
  gpu_compat.py <socket> put | check | drop      EXL3_WHX_BACKEND=cuda|xpu (default cuda), GPU_COMPAT_MIB (64)
Phase names and the key match keeper_compat.py, so crates/keeper/tests/fdstore_sim.py can drive it (COMPAT=...)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "crates", "py", "python", "llmt_kv"))
os.environ.setdefault("EXL3_WHX_BACKEND", "cuda")
os.environ["EXL3_WHX"] = sys.argv[1]
import whx_loader as L  # noqa: E402

PHASE = sys.argv[2]
N = int(os.environ.get("GPU_COMPAT_MIB", "64")) << 20
KEY = "model-a"
bk = L._bk(); bk.init()


def pattern(i):
    g = torch.Generator(device="cpu").manual_seed(1000 + i)
    return torch.randint(0, 256, (N,), dtype=torch.uint8, generator=g).to(bk.dev)


if PHASE == "put":
    bufs, fds = zip(*(bk.export_alloc(N) for _ in range(3)))
    for i, b in enumerate(bufs):
        b[:N].copy_(pattern(i))
    bk.acc().synchronize()
    r, _ = L._rpc({"op": "put", "key": KEY, "chunks": [N] * 3, "csum": L._csum([b[:N] for b in bufs]),
                   "manifest": [[0, 0, [N], [1], "torch.uint8"]]}, fds)
    assert r == {"ok": True}, r
    for fd in fds:
        os.close(fd)
elif PHASE == "check":
    e, fds = L._rpc({"op": "get", "key": KEY})
    assert e.get("found") and len(fds) == 3, (e.get("found"), len(fds))
    bufs = [bk.import_fd(fd, nb) for fd, nb in zip(fds, e["chunks"])]
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass
    assert L._csum([b[:N] for b in bufs]) == e["csum"], "chunk checksums differ"
    for i, b in enumerate(bufs):
        assert torch.equal(b[:N], pattern(i)), f"chunk {i} differs"
elif PHASE == "drop":
    assert L._rpc({"op": "drop", "key": KEY})[0] == {"dropped": True}
    assert L._rpc({"op": "list"})[0] == {}
print("gpu_compat", PHASE, "OK", bk.name, f"{3 * N >> 20} MiB")
