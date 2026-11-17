"""Pinning guard of the on-demand pool, against a private test broker that this script starts, kills and restarts.
  python test_pin_guard.py BROKER_BINARY SCRATCH_DIR      (SCRATCH_DIR on tmpfs: the on-demand pool needs it)
A pinning client (cuda_tier's SlotPinner) holds the pages a slot had when it registered it; a hole punch would
leave its GPU on pages the file no longer has. The broker must therefore:
1. punch freed slots while no client asked for the pool state (control: the guard is what keeps pages below);
2. never punch once a client asked for the pool state (POOL_STATE), even for slots that went free;
3. after a restart on the reused pool: keep every page (backed slots read back from the file, no punch at start),
   and keep them past the hold once a pool-state client reconnected within it;
4. after a restart with nobody reconnecting: punch the free slots once the hold is over.
"""
import os, signal, subprocess, sys, time
import llmt_kv

BIN, S = sys.argv[1], sys.argv[2]
SOCK, POOL = f"{S}/broker.sock", f"{S}/pool"
NS = 7
SLOT = 256 << 10
HOLD_S = 3


def start(*extra):
    p = subprocess.Popen([BIN, "--pool-gib", "0.015625", "--slot-kib", "256", "--pool-dir", POOL, "--socket", SOCK,
                          "--metrics", "off", "--low-water-pct", "0", "--reuse-grace-ms", "500", "--sweep-s", "1",
                          "--evict-age-s", "1", "--evict-step-pct", "100", "--warm-mib", "0",
                          "--pool-reuse-hold-s", str(HOLD_S), "--hotset-dir", "off", "--disk-dir", "off",
                          "--events-dir", "off", "--chains", "off", "--mem-reserve-gib", "0", *extra],
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


def pool_file():
    [f] = [os.path.join(POOL, n) for n in os.listdir(POOL) if n.startswith("pool-")]
    return f


def file_bytes():
    return os.stat(pool_file()).st_blocks * 512


def store(c, i):
    [(st, rng)] = c.reserve(NS, [{"key": bytes([i]) * 32, "nbytes": SLOT, "kind": 0}])
    assert st == llmt_kv.OK, llmt_kv.status_name(st)
    c.write(rng, bytes([i]) * SLOT)
    assert c.commit(NS, [bytes([i]) * 32]) == [llmt_kv.OK]


def settle(s):
    time.sleep(s)


def check(name, cond, detail):
    print(("ok   " if cond else "FAIL ") + name + f" ({detail})")
    assert cond, name


os.makedirs(POOL, exist_ok=True)
for n in os.listdir(POOL):
    os.unlink(os.path.join(POOL, n))

# 1. control: no pool-state client, aged-out entries are punched
p = start()
c = llmt_kv.Client(SOCK, engine_id=1, name="writer")
c.pool_address()
for i in range(1, 9):
    store(c, i)
before = file_bytes()
settle(4)                                    # age 1 s + sweeps every 1 s
check("1 no pinner: freed slots punched", file_bytes() < before // 2, f"{before} -> {file_bytes()} B")
stop(p)

# 2. a client asks for the pool state: nothing is punched any more
for n in os.listdir(POOL):
    os.unlink(os.path.join(POOL, n))
p = start()
c = llmt_kv.Client(SOCK, engine_id=1, name="writer")
c.pool_address()
pin = llmt_kv.Client(SOCK, engine_id=2, name="pinner")
pin.pool_state()
for i in range(1, 9):
    store(c, i)
before = file_bytes()
settle(4)
check("2 pinner: freed slots keep their pages", file_bytes() >= before, f"{before} -> {file_bytes()} B")

# 3. restart on the reused pool, the pinner reconnects within the hold
stop(p)
p = start()
check("3a restart: no punch at start", file_bytes() >= before, f"{before} -> {file_bytes()} B")
pin = llmt_kv.Client(SOCK, engine_id=2, name="pinner")
pin.pool_state()
settle(HOLD_S + 3)
check("3b pinner back within the hold: pages kept past it", file_bytes() >= before, f"{before} -> {file_bytes()} B")

# 4. restart, nobody reconnects: free slots are punched after the hold
stop(p)
p = start()
check("4a restart: no punch during the hold", file_bytes() >= before, f"{before} -> {file_bytes()} B")
settle(HOLD_S + 3)
check("4b nobody pinned: punched after the hold", file_bytes() < before // 2, f"{before} -> {file_bytes()} B")
stop(p)
print("all pin-guard checks passed")
