"""Reconnect test for llmt_kv against a private test broker that this script starts, kills and restarts.
  python test_reconnect.py BROKER_BINARY SCRATCH_DIR
1. An engine that disconnects and reconnects (new Client) still hits entries it stored before.
2. Broker restart reusing the pool: the client reports BrokerGone, then BrokerReset on the call that reconnects,
   then works; same pool file, new epoch, empty index; reservations get FULL during the grace period; bytes written
   through the old mapping before the restart are still in the file (stale but harmless).
3. Broker restart with --new-pool: BrokerPoolChanged, and it stays that way.
"""
import os, signal, subprocess, sys, time
import llmt_kv

BIN, S = sys.argv[1], sys.argv[2]
SOCK, POOL = f"{S}/broker.sock", f"{S}/pool"
NS = 7
GRACE_MS = 1500


def start(*extra):
    p = subprocess.Popen([BIN, "--pool-gib", "0.015625", "--slot-kib", "256", "--pool-dir", POOL, "--socket", SOCK,
                          "--metrics", "off", "--low-water-pct", "0", "--reuse-grace-ms", str(GRACE_MS), *extra],
                         stderr=open(f"{S}/broker.log", "a"))
    for _ in range(100):
        if os.path.exists(SOCK):
            try:
                llmt_kv.Client(SOCK, engine_id=9, name="probe")
                return p
            except llmt_kv.BrokerError:
                pass
        time.sleep(0.05)
    raise SystemExit("broker did not start")


def stop(p):
    p.send_signal(signal.SIGKILL)
    p.wait()
    os.unlink(SOCK)


def key(i):
    return bytes([i]) * 32


def store(c, i, data):
    [(st, rng)] = c.reserve(NS, [{"key": key(i), "nbytes": len(data), "kind": 0}])
    assert st == llmt_kv.OK, llmt_kv.status_name(st)
    c.write(rng, data)
    assert c.commit(NS, [key(i)]) == [llmt_kv.OK]
    return rng


def expect(exc, fn):
    try:
        fn()
    except exc:
        return
    except llmt_kv.BrokerError as e:
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, call succeeded")


ok = 0
b = start()

# 1. engine reconnect keeps the broker's index
a = llmt_kv.Client(SOCK, engine_id=1, name="arc", timeout_ms=200)
a.pool_address()
rng = store(a, 1, b"x" * 1000)
pool_path, epoch0 = a.pool_path, a.epoch
del a
a = llmt_kv.Client(SOCK, engine_id=1, name="arc", timeout_ms=200, reconnect_ms=100)
a.pool_address()
n_hit, lease, ents = a.lookup(NS, [key(1)], acquire=True)
assert n_hit == 1 and a.read(ents[0].byte_ranges, 1000) == b"x" * 1000
a.release([lease])
print("1 engine reconnect: entry still there"); ok += 1

# 2. broker restart, pool reused
store(a, 2, b"y" * 1000)
stop(b)
expect(llmt_kv.BrokerGone, lambda: a.lookup(NS, [key(1)]))
assert not a.connected
expect(llmt_kv.BrokerGone, lambda: a.lookup(NS, [key(1)]))      # broker down: fast BrokerGone, no hang
b = start()
time.sleep(0.15)                                                 # past reconnect_ms
expect(llmt_kv.BrokerReset, lambda: a.lookup(NS, [key(1)]))      # this call reconnected
assert a.connected and a.reconnects == 1 and a.epoch != epoch0 and a.pool_path == pool_path
assert a.lookup(NS, [key(1), key(2)])[0] == 0                    # index empty after a broker restart
[(st, _)] = a.reserve(NS, [{"key": key(3), "nbytes": 100}])
assert st == llmt_kv.FULL, f"grace period: want FULL, got {llmt_kv.status_name(st)}"
time.sleep(GRACE_MS / 1000 + 0.2)
store(a, 3, b"z" * 100)
n_hit, lease, ents = a.lookup(NS, [key(3)], acquire=True)
assert n_hit == 1 and a.read(ents[0].byte_ranges, 100) == b"z" * 100
a.release([lease])
assert os.listdir(POOL) == [os.path.basename(pool_path)]
print("2 broker restart: Gone -> Reset -> works; same pool, new epoch, empty index, FULL during grace"); ok += 1

# 3. broker restart with a new pool file
stop(b)
b = start("--new-pool")
expect(llmt_kv.BrokerGone, lambda: a.lookup(NS, [key(3)]))       # finds the old connection dead
time.sleep(0.15)
expect(llmt_kv.BrokerPoolChanged, lambda: a.lookup(NS, [key(3)]))
time.sleep(0.15)
expect(llmt_kv.BrokerPoolChanged, lambda: a.lookup(NS, [key(3)]))
c = llmt_kv.Client(SOCK, engine_id=2, name="fresh")
assert c.pool_path != pool_path
print("3 new pool: BrokerPoolChanged (sticky); a fresh client gets the new pool"); ok += 1

stop(b)
print(f"PASS {ok}/3")
