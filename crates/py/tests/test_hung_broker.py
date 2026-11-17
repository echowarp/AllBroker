"""A hung broker must never stall the engine (a lock-order deadlock in the broker stalled the 27B's
decode until the broker was restarted by hand).

Run against a test broker started with LLMT_KV_TEST_HANG_MS (it holds its core lock for good after that many ms),
--watchdog-s and a supervisor that restarts it when it exits (systemd Restart=always / on-failure):

  python test_hung_broker.py SOCKET [SECONDS]

Engine-like traffic (reserve + commit of 64 keys, lookup, every 10 ms) with the engine's 2 ms timeout. Checks: no call
takes longer than 250 ms, the client dropped the hung connection after 16 timeouts, and from then on the tier is
off for good (BrokerPoolChanged: a dropped connection loses its leases, so the engine must not keep
loading from slots it looked up before, nor from a restarted broker's pool)."""
import os
import sys
import time

import llmt_kv

SOCK = sys.argv[1]
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
c = llmt_kv.Client(SOCK, engine_id=3, name="hung-test", timeout_ms=2.0, reconnect_ms=1000.0)
NS = 7
worst, n, ok_after_reset, errs = 0.0, 0, 0, {}
reset_seen = False
t_end = time.monotonic() + DUR
i = 0
while time.monotonic() < t_end:
    keys = [os.urandom(32) for _ in range(64)]
    for fn in (lambda: c.reserve(NS, [{"key": k, "parent": None, "nbytes": 100, "kind": 0, "format": 1, "meta": b""}
                                      for k in keys]),
               lambda: c.commit(NS, keys),
               lambda: c.lookup(NS, keys[:8], acquire=False)):
        t0 = time.monotonic()
        try:
            fn()
            if reset_seen:
                ok_after_reset += 1
        except llmt_kv.BrokerReset:
            reset_seen = True
            errs["reset"] = errs.get("reset", 0) + 1
        except llmt_kv.BrokerPoolChanged:
            errs["off"] = errs.get("off", 0) + 1
        except llmt_kv.BrokerError as e:
            errs[type(e).__name__] = errs.get(type(e).__name__, 0) + 1
        dt = time.monotonic() - t0
        worst = max(worst, dt)
        n += 1
    i += 1
    time.sleep(0.01)
print(f"calls {n}, worst {worst * 1000:.1f} ms, errors {errs}, timeout drops {c.timeout_drops}, "
      f"reconnects {c.reconnects}, ok after reconnect {ok_after_reset}")
fails = []
if worst > 0.25:
    fails.append(f"a call took {worst * 1000:.0f} ms")
if c.timeout_drops < 1:
    fails.append("the hung connection was never dropped")
if not errs.get("off"):
    fails.append("the tier did not go off after the drop")
if reset_seen or ok_after_reset:
    fails.append("the client came back to the broker after a drop (stale leases possible)")
print("FAIL: " + "; ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
