"""CPU tests for the lease-class tier client (load / resident / hold). No torch / vLLM / GPU: stubs for their imports.
Part 1 (fake broker): class caps and the sole-loader bypass. Part 2 (real llmt-kvbroker binary in $LLMT_KVBROKER_BIN, skipped without it or without the Rust client; private socket + tiny
pool in a temp dir, --lease-timeout-ms 1500): renewal keeps resident leases alive past the broker deadline.
  python3 -I test_leasecls_cpu.py [--no-broker]"""
import enum, os, subprocess, sys, tempfile, time, types

HERE = os.environ.get("PYTHONPATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
sys.path.insert(0, HERE)


def _stubs():
    def mod(name, **kw):
        m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m; return m
    mod("torch")
    import logging
    mod("vllm"); mod("vllm.logger", init_logger=lambda n: logging.getLogger(n))
    class LookupResult(enum.Enum):
        MISS = 0; HIT = 1; HIT_PENDING = 2; RETRY = 3
    class _O:  # generic stub base
        def __init__(self, *a, **kw): self.__dict__.update(kw)
    class ReqContext:
        def __init__(self, req_id, kv_transfer_params=None): self.req_id, self.kv_transfer_params = req_id, kv_transfer_params
    for n in ("vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.cpu"):
        mod(n)
    mod("vllm.v1.kv_offload.base", LoadStoreSpec=_O, LookupResult=LookupResult, OffloadingManager=object, OffloadKey=bytes,
        PrepareStoreOutput=_O, ReqContext=ReqContext, RequestOffloadingContext=_O)
    mod("vllm.v1.kv_offload.cpu.common", CPULoadStoreSpec=lambda slots: slots)
    mod("vllm.v1.kv_offload.cpu.gpu_worker", CPUOffloadingWorker=object)
    mod("vllm.v1.kv_offload.cpu.shared_offload_region", SharedOffloadRegion=object)
    mod("vllm.v1.kv_offload.cpu.spec", CPUOffloadingSpec=object)


_stubs()
from vllm.v1.kv_offload.base import LookupResult, ReqContext  # noqa: E402
import llmt_kv.vllm_tier as T  # noqa: E402

SLOT = 4096
key = lambda i, g=0: i.to_bytes(28, "big") + g.to_bytes(4, "big")
ok = 0


def check(cond, what):
    global ok
    if not cond:
        raise AssertionError(what)
    ok += 1
    print("ok", what)


class FakeEntry:
    def __init__(self, slot): self.byte_ranges = [(slot * SLOT, SLOT)]


class FakeClient:
    """lookup(ns, [bk], acquire) -> (n_hit, lease_id, [entry]); every key with index < present is in the pool."""
    connected, reconnects = True, 0
    def __init__(self, present): self.present, self.next, self.live, self.released = present, 1, set(), []
    def lookup(self, ns, bks, acquire=True):
        i = self.slot_of[bks[0]] if bks[0] in self.slot_of else None
        if i is None:
            return (0, 0, [])
        lid = self.next; self.next += 1; self.live.add(lid)
        return (1, lid, [FakeEntry(i)])
    def release(self, ids):
        for x in ids:
            self.live.discard(x); self.released.append(x)
        return [0] * len(ids)


def mgr_with(n_keys, lease_cap, resident_cap):
    c = FakeClient(n_keys)
    m = T.BrokerOffloadingManager(c, "t", SLOT, lease_cap=lease_cap, resident_cap=resident_cap, group_kinds=(0,))
    c.slot_of = {m._bk(key(i)): i for i in range(n_keys)}
    return m, c


def part1():
    os.environ["LLMT_KV_LEASE_RENEW_S"] = "0"
    # (a) the W2-A / tcs460 case: a session's own resident leases + holds no longer eat the load cap
    m, c = mgr_with(60, lease_cap=3, resident_cap=100)
    a = ReqContext("sess-old")
    for i in range(40):
        assert m.lookup_cls(key(i), a, "resident") == LookupResult.HIT
    h = ReqContext("arcH|sess-old")
    for i in range(40):
        m.lookup_cls(key(i), h, "hold")
    cc = m.class_counts()
    check(cc["hold"] == 40 and cc["load"] == 0, f"resident/hold leases classed apart from load: {cc}")
    b = ReqContext("sess-new")                     # next turn: restores 45 chunks; old hold still holds 0-39
    hits = sum(1 for i in range(45) if m.lookup(key(i), b) == LookupResult.HIT)
    check(hits == 45, f"next turn's lookup hits all 45 chunks with a load cap of 3 (sole loader): {hits}")
    check(m.n["lease_cap_bypass"] >= 1 and m.n["lease_cap"] == 0, "bypass counted, no cap miss")
    # (b) the cap still applies between concurrent loaders
    m, c = mgr_with(60, lease_cap=10, resident_cap=100)
    r1, r2 = ReqContext("r1"), ReqContext("r2")
    for i in range(8):
        m.lookup(key(i), r1)
    got2 = sum(1 for i in range(20, 30) if m.lookup(key(i), r2) == LookupResult.HIT)
    check(got2 == 2 and m.n["lease_cap"] == 8, f"second concurrent loader capped at the load cap (got {got2})")
    # (c) the resident cap bounds residency
    m, c = mgr_with(60, lease_cap=100, resident_cap=5)
    s = ReqContext("sess-s")
    got = sum(1 for i in range(10) if m.lookup_cls(key(i), s, "resident") == LookupResult.HIT)
    check(got == 5 and m.n["lease_cap_res"] == 5, f"resident cap enforced ({got})")
    # (d) resident leases survive the idle-step safety net, load leases do not
    m, c = mgr_with(10, lease_cap=100, resident_cap=100)
    m.lookup_cls(key(0), ReqContext("sess-z"), "resident"); m.lookup(key(1), ReqContext("w"))
    m.lease_steps = 1
    for _ in range(3):
        m.on_schedule_end(types.SimpleNamespace(new_req_ids=[]))
    check(m.lease_class(m._ik(key(0), ReqContext("x"))) == "resident" and m._ik(key(1), ReqContext("x")) not in m._leases,
          "idle net releases the load lease only")
    # (e) a finished resident holder: its leases drop normally
    m.on_request_finished(ReqContext("sess-z"))
    check(not m._leases, "resident leases end with their holder")
    # (f) renewal with the fake broker: new lease first, old released
    os.environ["LLMT_KV_LEASE_RENEW_S"] = "0.01"
    m, c = mgr_with(10, lease_cap=100, resident_cap=100)
    m.lookup_cls(key(3), ReqContext("sess-r"), "resident")
    old = m._leases[m._ik(key(3), ReqContext("x"))][0]
    time.sleep(0.02); m.on_schedule_end(types.SimpleNamespace(new_req_ids=[]))
    new = m._leases[m._ik(key(3), ReqContext("x"))][0]
    check(new != old and old in c.released and new in c.live and m.n["renewed"] == 1, "renewal swaps the lease id")


def part2():
    if not os.environ.get("LLMT_KVBROKER_BIN"):
        print("skip part 2 (no LLMT_KVBROKER_BIN)")
        return
    try:
        import llmt_kv
        llmt_kv.Client
    except (ImportError, AttributeError):
        print("skip part 2 (no Rust client)")
        return
    """Real broker: --lease-timeout-ms 1500. Leased entries must stay leased (no lease timeout) with renewal on."""
    exe = os.environ.get("LLMT_KVBROKER_BIN", "")
    d = tempfile.mkdtemp(prefix="leasecls-")
    res = {}
    for renew in ("0.4", "0"):
        sock, pool = f"{d}/b{renew}.sock", f"{d}/pool{renew}"
        os.makedirs(pool)
        p = subprocess.Popen([exe, "--pool-gib", "0.05", "--slot-kib", str(SLOT // 1024), "--pool-dir", pool, "--socket", sock,
                              "--no-hugepages-ctl", "--no-mlock", "--new-pool", "--hotset-dir", "off", "--events-dir", "off",
                              "--disk-dir", "off", "--chains", "off", "--mem-reserve-gib", "0", "--lease-timeout-ms", "1500",
                              "--metrics", "off"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if os.path.exists(sock):
                    break
                time.sleep(0.05)
            c = llmt_kv.Client(socket=sock, engine_id=7, name="leasecls-test", timeout_ms=200, reconnect_ms=200)
            os.environ["LLMT_KV_LEASE_RENEW_S"] = renew
            m = T.BrokerOffloadingManager(c, "t", c.slot_bytes, lease_cap=100, resident_cap=100, group_kinds=(0,))
            ctx = ReqContext("sess-live")
            keys = [key(i) for i in range(4)]
            st = m.prepare_store(keys, ctx)
            m.complete_store(keys, ctx, True)
            hits = sum(1 for k in keys if m.lookup_cls(k, ctx, "resident") == LookupResult.HIT)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 4.0:          # 2.7 x the broker deadline
                m.on_schedule_end(types.SimpleNamespace(new_req_ids=[]))
                time.sleep(0.1)
            s = c.stats()
            res[renew] = (hits, s, m.n["renewed"], m.n["renew_lost"])
        finally:
            p.terminate(); p.wait(5)
    def timeouts(s):
        if isinstance(s, dict):
            for k in ("lease_timeouts", "lease_timeouts_total"):
                if k in s:
                    return s[k]
        return getattr(s, "lease_timeouts", None)
    on, off = res["0.4"], res["0"]
    print("broker stats keys:", sorted(on[1].keys()) if isinstance(on[1], dict) else type(on[1]))
    check(on[0] == 4 and off[0] == 4, "4 stored chunks leased in both runs")
    check(on[2] >= 8 and on[3] == 0, f"renewals happened ({on[2]}), none lost")
    t_on, t_off = timeouts(on[1]), timeouts(off[1])
    check(t_on is not None and t_off is not None, f"broker reports lease timeouts (on {t_on}, off {t_off})")
    check(t_on == 0 and t_off >= 4, f"renewal ON: 0 broker lease timeouts; OFF: {t_off} (the R4 lapse)")


part1()
if "--no-broker" not in sys.argv:
    part2()
print(f"PASS {ok} checks")
