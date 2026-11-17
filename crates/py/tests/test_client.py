"""End-to-end test of the llmt_kv wheel against a running test broker (synthetic keys and bytes only).
    python test_client.py SOCKET"""
import os, sys, threading, time
import llmt_kv as kv

sock = sys.argv[1]
NS = 11
run = os.getpid().to_bytes(4, "little")
K = lambda i: run + i.to_bytes(4, "little") + bytes(24)

a = kv.Client(sock, engine_id=1, name="py-a", timeout_ms=200)
b = kv.Client(sock, engine_id=2, name="py-b", timeout_ms=200)
print("broker", a.broker_version, "epoch %016x" % a.epoch, a.n_slots, "slots of", a.slot_bytes)
assert a.epoch == b.epoch and kv.PROTOCOL_VERSION == 1
base_a, base_b = a.pool_address(), b.pool_address()
assert base_a and base_b

# store a two-entry chain
data = bytes((i * 7) % 251 for i in range(a.slot_bytes + 12345))
[(st, rng)] = a.reserve(NS, [{"key": K(1), "nbytes": len(data), "kind": 2, "format": 3, "meta": b"layout-v1"}])
assert st == kv.OK and sum(l for _, l in rng) >= len(data), (st, rng)
a.write(rng, data)
assert b.reserve(NS, [{"key": K(1), "nbytes": 10}])[0][0] == kv.BUSY
assert a.commit(NS, [K(1)]) == [kv.OK]
[(st2, rng2)] = a.reserve(NS, [{"key": K(2), "parent": K(1), "nbytes": 100}])
a.write(rng2, b"x" * 100)
a.commit(NS, [K(2)])
assert a.reserve(NS, [{"key": K(9), "parent": K(8), "nbytes": 1}])[0][0] == kv.NO_PARENT

# read through the other client's own mapping
n, lease, es = b.lookup(NS, [K(1), K(2), K(3)], acquire=True)
assert n == 2 and lease > 0 and len(es) == 2, (n, lease, es)
e = es[0]
assert (e.nbytes, e.kind, e.format, e.producer, e.meta) == (len(data), 2, 3, 1, b"layout-v1"), e
assert b.read(e.byte_ranges, e.nbytes) == data
assert b.lookup(NS, [K(1)], acquire=False)[:2] == (1, 0)
assert b.release([lease]) == [kv.OK]

# WAIT in a thread while the main thread commits (needs the GIL released inside wait)
c = kv.Client(sock, engine_id=3, name="py-c", timeout_ms=200)
out = {}
def waiter():
    t = time.time(); out["st"] = c.wait(NS, K(4), 3000); out["t"] = time.time() - t
th = threading.Thread(target=waiter); th.start(); time.sleep(0.2)
a.reserve(NS, [{"key": K(4), "nbytes": 1, "ephemeral": True, "ttl_ms": 1000}]); a.commit(NS, [K(4)])
th.join()
assert out["st"] == kv.OK and 0.15 < out["t"] < 1.0, out
assert c.wait(NS, K(5), 50) == kv.TIMEOUT
s = a.stats()
assert s["committed"] >= 3 and s["leases_active"] == 0, s
assert kv.status_name(3) == "FULL"
try:
    a.lookup(NS, [b"short"])
    raise SystemExit("short key accepted")
except ValueError:
    pass
print("all python client checks passed")
