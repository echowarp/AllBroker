#!/usr/bin/env python3
"""fd-store restart test without systemd: this script plays systemd's part (NOTIFY_SOCKET datagrams with FDSTORE=1 /
FDSTOREREMOVE=1, then LISTEN_FDS / LISTEN_FDNAMES on the next start), kills the keeper with SIGKILL and restarts it.
  fdstore_sim.py <llmt-keeper binary> <scratch dir>
COMPAT="<command>" replaces the memfd client (keeper_compat.py) with another one taking <socket> <phase>, e.g. real GPU
memory: COMPAT="docker run --rm ... python3 /w/weights/tests/gpu_compat.py" (see docs/keeper.md, fd-store test)."""
import os, shlex, signal, socket, subprocess, sys, time

B, D = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
SOCK, STATE, NOTIFY = f"{D}/k.sock", f"{D}/state.json", f"{D}/notify.sock"
for p in (SOCK, STATE, NOTIFY):
    if os.path.exists(p): os.unlink(p)
n = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM); n.bind(NOTIFY); n.settimeout(0.05)
store = {}                                     # name -> fd (systemd's fd store)


def pump():
    while True:
        try: msg, fds, _, _ = socket.recv_fds(n, 4096, 253)
        except (TimeoutError, BlockingIOError): return
        kv = dict(l.split("=", 1) for l in msg.decode().splitlines() if "=" in l)
        if kv.get("FDSTORE") == "1":
            assert len(fds) == 1, fds
            store[kv["FDNAME"]] = fds[0]
        elif kv.get("FDSTOREREMOVE") == "1":
            fd = store.pop(kv["FDNAME"], None)
            if fd is not None: os.close(fd)


def start():
    names = list(store)
    fds = [store[k] for k in names]

    def pre():                                 # like systemd: fds at 3.., LISTEN_PID = the child
        tmp = [os.dup(f) for f in fds]
        for i, f in enumerate(tmp):
            os.dup2(f, 3 + i, inheritable=True)
        if fds:
            os.environ.update(LISTEN_PID=str(os.getpid()), LISTEN_FDS=str(len(fds)), LISTEN_FDNAMES=":".join(names))
    if os.path.exists(SOCK): os.unlink(SOCK)
    p = subprocess.Popen([B, "--socket", SOCK, "--fdstore", STATE], env=None, preexec_fn=pre, close_fds=False,
                         stderr=open(f"{D}/keeper.log", "a"))
    os.environ["NOTIFY_SOCKET"] = NOTIFY
    for _ in range(100):
        pump()
        if os.path.exists(SOCK): break
        time.sleep(0.05)
    return p


def compat(phase):
    cmd = shlex.split(os.environ["COMPAT"]) if os.environ.get("COMPAT") else [sys.executable, f"{HERE}/keeper_compat.py"]
    subprocess.run(cmd + [SOCK, phase], check=True)
    pump()


os.environ["NOTIFY_SOCKET"] = NOTIFY
p = start(); compat("put")
assert sorted(store) == ["whx-1-0", "whx-1-1", "whx-1-2"], store
p.send_signal(signal.SIGKILL); p.wait()
p = start(); compat("check"); print("restored after SIGKILL")
p.terminate(); p.wait()
store["whx-99-0"] = os.memfd_create("stray")    # an fd with no entry must be released on start
p = start(); compat("check"); pump()
assert "whx-99-0" not in store, store
compat("drop")
assert store == {}, f"fd store not emptied after drop: {store}"
p.terminate(); p.wait()
print("fdstore_sim OK")
