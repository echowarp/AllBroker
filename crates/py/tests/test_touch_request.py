"""The connector's TOUCH_REQUEST path: a request's KV groups, touched one by one by the scheduler, reach
the broker as one tagged TOUCH_REQUEST (kinds from group_kinds, groups in order); a request whose last group had nothing
to touch is sent at the end of the scheduler step; an llmt_kv build without touch_request sends plain TOUCH batches.
Same stubs as test_tier_statemachine (no vllm needed).
  python test_touch_request.py [path/to/vllm_tier.py]
"""
import os
import sys

if len(sys.argv) > 1:
    os.environ["TIER_PY"] = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tier_statemachine import FakeBroker, ReqContext, ScheduleEndContext, T  # noqa: E402

SLOT = 1000


def key(group: int, i: int) -> bytes:
    return bytes([i]) * 32 + group.to_bytes(4, "big")      # the vLLM key's last 4 bytes are its KV group


class Recorder(FakeBroker):
    def __init__(self, tagged=True):
        super().__init__()
        self.calls = []
        if tagged:
            self.touch_request = lambda ns, groups: self.calls.append(("touch_request", ns, groups))

    def touch(self, ns, keys):
        self.calls.append(("touch", ns, keys))
        return [0] * len(keys)


def manager(b):
    return T.BrokerOffloadingManager(b, "ns", SLOT, group_kinds=(1, 1, 0))   # two GDN state groups, then attention


def test_one_tagged_message_per_request():
    b = Recorder()
    m = manager(b)
    ctx = ReqContext("r1")
    m.touch([key(0, 3)], ctx)
    m.touch([key(1, 3)], ctx)
    assert b.calls == [], "buffered until the last group"
    m.touch([key(2, 1), key(2, 2), key(2, 3)], ctx)
    assert len(b.calls) == 1 and b.calls[0][0] == "touch_request"
    groups = b.calls[0][2]
    assert [g[0] for g in groups] == [1, 1, 0], groups
    assert [len(g[1]) for g in groups] == [1, 1, 3]
    assert groups[2][1] == [m._bk(key(2, i)) for i in (1, 2, 3)], "attention keys in chain order"


def test_missing_last_group_flushed_at_step_end():
    b = Recorder()
    m = manager(b)
    m.touch([key(0, 5)], ReqContext("a"))
    m.touch([key(1, 5)], ReqContext("a"))
    m.touch([key(0, 6)], ReqContext("b"))
    m.on_schedule_end(ScheduleEndContext(new_req_ids=()))
    sent = sorted(len(c[2]) for c in b.calls if c[0] == "touch_request")
    assert sent == [1, 2], b.calls
    assert not m._touch_buf


def test_older_client_plain_touch():
    b = Recorder(tagged=False)
    m = manager(b)
    ctx = ReqContext("r1")
    m.touch([key(0, 3)], ctx)
    m.touch([key(2, 1), key(2, 2), key(2, 3)], ctx)
    assert [c[0] for c in b.calls] == ["touch", "touch"]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
