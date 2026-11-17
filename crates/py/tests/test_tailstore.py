"""TailStore (vllm_tier): partial-block spills as ephemeral broker entries, against a private broker started here.
Needs vllm importable (engine image); no GPU (the pool is mapped unpinned).
    python test_tailstore.py BROKER_BINARY SCRATCH_DIR
"""
import os, signal, subprocess, sys, time

import torch

import llmt_kv
from llmt_kv import vllm_tier as T

BIN, S = sys.argv[1], sys.argv[2]
SOCK, POOL = f"{S}/broker.sock", f"{S}/pool"
SLOT, N_SLOTS = 256 << 10, 6
EXTRA = {"broker_socket": SOCK, "broker_wait_s": 1, "broker_reconnect_ms": 100, "broker_timeout_ms": 500}


def check(cond, what):
    assert cond, what
    print("ok -", what, flush=True)


def start():
    os.makedirs(S, exist_ok=True)
    p = subprocess.Popen([BIN, "--pool-gib", str(N_SLOTS * SLOT / 2**30), "--slot-kib", str(SLOT >> 10), "--pool-dir",
                          POOL, "--socket", SOCK, "--metrics", "off", "--low-water-pct", "0", "--reserve-timeout-ms",
                          "300", "--reuse-grace-ms", "0", "--hotset-dir", "off"], stderr=open(f"{S}/broker.log", "a"))
    for _ in range(100):
        if os.path.exists(SOCK):
            try:
                llmt_kv.Client(SOCK)
                return p
            except llmt_kv.BrokerError:
                pass
        time.sleep(0.05)
    raise SystemExit("broker did not start")


def mgr(name):
    return T.BrokerOffloadingManager(T.connect(EXTRA, 1, name), "own-ns", SLOT, 1024, 0, ns_allow=["shared"])


b = start()
try:
    m1, m2 = mgr("eng-a"), mgr("eng-b")
    region = T.BrokerPoolRegion(m1.c.pool_path, m1.c.n_slots, m1.c.slot_bytes)
    t1, t2 = T.TailStore(m1, region), T.TailStore(m2, region)
    check(t1.namespace_for("shared") == "shared" and t1.namespace_for("other") == "own-ns"
          and t1.namespace_for(None) == "own-ns" and t1.ns_for({"llmt_ns": "shared"}) == "shared"
          and t1.ns_for(None) == "own-ns", "namespace_for / ns_for: allowlisted name, else the engine's own")
    ns = t1.namespace_for("shared")
    k = t1.key(ns, b"\x11" * 32)
    check(len(k) == 32 and k == t2.key(ns, b"\x11" * 32) and k != t1.key("own-ns", b"\x11" * 32),
          "key: 32 bytes, same on both engines, differs per namespace")

    nbytes, meta = 2 * SLOT + 1000, b"TAIL1" + bytes(range(40))
    ranges = t1.reserve(ns, k, nbytes, 60_000, meta)
    check(isinstance(ranges, list) and sum(n for _, n in ranges) >= nbytes, f"reserve: {len(ranges)} range(s), "
          f"{sum(n for _, n in ranges)} B >= {nbytes}")
    host, base = t1.host_tensor(), t1.host_base()
    pat = torch.randint(-128, 127, (nbytes,), dtype=torch.int8)
    off = 0
    for ptr, n in ranges:
        take = min(n, nbytes - off)
        host[ptr - base:ptr - base + take].copy_(pat[off:off + take])
        off += take
    check(t1.commit(ns, k), "commit")
    check(t1.reserve(ns, k, nbytes, 60_000, meta) == "exists", "same key again -> 'exists'")

    hit = t2.lookup(ns, k)
    check(hit is not None and hit[2] == meta and hit[3] == nbytes, "other engine: lookup finds it with meta and nbytes")
    lease, r2, _, _ = hit
    got = torch.cat([host[p - base:p - base + n] for p, n in r2])[:nbytes]
    check(torch.equal(got, pat), "other engine reads the same bytes")
    check(t2.lookup("own-ns", t2.key("own-ns", b"\x11" * 32)) is None, "other namespace: miss")
    t2.release(lease)

    # read once: after its lease ends the tail goes first when room is needed (pool 6 slots; tail holds 3)
    k2 = t1.key(ns, b"\x33" * 32)
    r = t1.reserve(ns, k2, 4 * SLOT, 60_000, b"x")
    check(isinstance(r, list), "4-slot reserve succeeds by evicting the read tail")
    t1.commit(ns, k2)
    check(t1.lookup(ns, k) is None, "the read tail is gone (ephemeral, read once)")

    try:
        t1.reserve(ns, k, 10, 1000, b"m" * 1025)
        check(False, "meta limit")
    except ValueError:
        check(True, "meta > 1024 B rejected")

    # broker down: a miss, never an exception
    b.send_signal(signal.SIGKILL)
    b.wait()
    os.unlink(SOCK)
    k3 = t1.key(ns, b"\x44" * 32)
    check(t1.reserve(ns, k3, SLOT, 1000, b"m") is None and t1.lookup(ns, k3) is None, "broker down: reserve/lookup -> None")
    print("counters", t1.n, t2.n)
    print("PASS tailstore")
finally:
    if b.poll() is None:
        b.send_signal(signal.SIGKILL)
        b.wait()
