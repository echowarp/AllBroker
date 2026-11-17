#!/usr/bin/env python3
"""Wire test against a running keeper, with the engine loader's own RPC (whx_loader._rpc) and memfds standing in for
GPU memory fds.  keeper_compat.py <socket> [phase: all | put | check | drop]"""
import json, mmap, os, socket, sys

SOCK = sys.argv[1]
PHASE = sys.argv[2] if len(sys.argv) > 2 else "all"


def _rpc(req, fds=()):                     # verbatim from whx_loader.py
    s = socket.socket(socket.AF_UNIX)
    s.connect(SOCK)
    data = json.dumps(req).encode()
    socket.send_fds(s, [len(data).to_bytes(8, "little")], list(fds)); s.sendall(data)
    hdr, rfds, _, _ = socket.recv_fds(s, 8, 256)
    n = int.from_bytes(hdr, "little"); body = b""
    while len(body) < n:
        body += s.recv(n - len(body))
    s.close()
    return json.loads(body), rfds


def chunk(i, n=1 << 20):
    fd = os.memfd_create(f"w{i}")
    os.ftruncate(fd, n)
    with mmap.mmap(fd, n) as m:
        m[:16] = f"chunk{i:02d}-marker!!".encode()
    return fd, n


def check(key, nchunks):
    e, fds = _rpc({"op": "get", "key": key})
    assert e["found"], e
    assert len(fds) == nchunks == len(e["chunks"]), (len(fds), e["chunks"])
    assert e["manifest"][0] == ["model.embed", [4, 4], "torch.bfloat16"], e["manifest"][0]
    for i, fd in enumerate(fds):
        with mmap.mmap(fd, e["chunks"][i], prot=mmap.PROT_READ) as m:
            assert m[:16] == f"chunk{i:02d}-marker!!".encode(), (i, m[:16])
        os.close(fd)


if PHASE in ("all", "put"):
    fds, sizes = zip(*(chunk(i) for i in range(3)))
    big = {"op": "put", "key": "model-a", "chunks": list(sizes), "csum": "x" * 64, "dyn": "",
           "manifest": [["model.embed", [4, 4], "torch.bfloat16"]] + [[f"t{i}", [i], "torch.uint8"] for i in range(20000)]}
    assert _rpc(big, fds)[0] == {"ok": True}
    for fd in fds:
        os.close(fd)                           # the keeper's copies keep the memory now
    r, _ = _rpc({"op": "put", "key": "bad", "chunks": [1, 2]}, [chunk(9)[0]])
    assert r.get("ok") is False and "fds" in r["error"], r
if PHASE in ("all", "check"):
    check("model-a", 3)
    lst, _ = _rpc({"op": "list"})
    assert lst == {"model-a": {"chunks": 3, "bytes": 3 << 20, "gib": 0.0}}, lst
    assert _rpc({"op": "get", "key": "nope"})[0] == {"found": False}
    assert _rpc({"op": "bogus"})[0] == {"error": "unknown op"}
if PHASE in ("all", "drop"):
    assert _rpc({"op": "drop", "key": "model-a"})[0] == {"dropped": True}
    assert _rpc({"op": "drop", "key": "model-a"})[0] == {"dropped": False}
    assert _rpc({"op": "list"})[0] == {}
print("keeper_compat", PHASE, "OK")
