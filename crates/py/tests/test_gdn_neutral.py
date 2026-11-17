"""GDN-neutral RAM layout of the vLLM tier: the copy rewrite (`_NeutralCopy`), the plan (`neutral_plan`)
and the tag/NaN-scan setup (`_install_neutral`), with a fake offloading handler whose op order matches vLLM 0.26's
SingleDirectionOffloadingHandler.transfer_async (groups in order, empty groups skipped, per ref `group_size` ops) and
a numpy byte array as memory (pointers = offsets). Needs vllm importable (engine image); no GPU.
    python3 test_gdn_neutral.py
"""
import types

import numpy as np
import torch

from llmt_kv import vllm_tier as T
import vllm.v1.kv_offload.cpu.gpu_worker as gw

ROW, SSM, ATT = 20, 64, 300          # conv row bytes, SSM bytes, attention page bytes (small stand-ins)
PAGE = 15 * ROW + SSM                 # GDN page: room for k=12 (3 + 12 rows) + SSM
MEM = np.zeros(1 << 20, np.uint8)


def check(cond, what):
    assert cond, what
    print("ok -", what, flush=True)


def segs(k):
    return [(0, 3 * ROW, torch.float16), ((3 + k) * ROW, SSM, torch.float32)]


class Ref:
    def __init__(self, page):
        self.page_size_bytes = page


class FakeHandler:
    """groups: [GDN (2 refs), attention (1 ref)]; block b of ref r on side s lives at base[s][g][r] + b * page."""

    def __init__(self, gpu_to_cpu, base):
        self.kv_cache_groups_data_refs = [[Ref(PAGE), Ref(PAGE)], [Ref(ATT)]]
        self.g2c, self.base = gpu_to_cpu, base
        self.done = []

    def _swap_blocks_batch(self, src, dst, sizes, is_src_access_order_any=False):
        for s, d, n in zip(src.numpy(), dst.numpy(), sizes.numpy()):
            MEM[d:d + n] = MEM[s:s + n]

    def transfer_async(self, job_id, src_spec, dst_spec):
        gpu = src_spec if self.g2c else dst_spec
        src, dst, sz = [], [], []
        for g, n in enumerate(gpu.group_sizes):
            if not n:
                continue
            for r, ref in enumerate(self.kv_cache_groups_data_refs[g]):
                for b in range(n):
                    gp = self.base["gpu"][g][r] + gpu.blocks[g][b] * ref.page_size_bytes
                    cp = self.base["cpu"][g][r] + gpu.cpu_blocks[g][b] * ref.page_size_bytes
                    src.append(gp if self.g2c else cp); dst.append(cp if self.g2c else gp); sz.append(ref.page_size_bytes)
        t = lambda x: torch.tensor(x, dtype=torch.int64)
        self._swap_blocks_batch(t(src), t(dst), t(sz), is_src_access_order_any=not self.g2c)
        self.done.append(job_id)
        return True

    def get_finished(self):
        out, self.done = [(j, True) for j in self.done], []
        return out


def layout(off):
    # gpu/cpu bases per group/ref, 8 blocks each
    b, out = off, {}
    for side in ("gpu", "cpu"):
        out[side] = []
        for pages in ([PAGE, PAGE], [ATT]):
            out[side].append([])
            for p in pages:
                out[side][-1].append(b)
                b += 8 * p
    return out


def spec(gsizes, gblocks, cblocks):
    return types.SimpleNamespace(group_sizes=gsizes, blocks=gblocks, cpu_blocks=cblocks)


rng = np.random.default_rng(0)
# engine A (k=6) and engine B (k=12) share the CPU (RAM) side: same cpu bases
A, B = layout(0), layout(400_000)
B["cpu"] = A["cpu"]
plan_a = T.neutral_plan([[segs(6), segs(6)], [[]]], [1, 0], [[PAGE, PAGE], [ATT]])
plan_b = T.neutral_plan([[segs(12), segs(12)], [[]]], [1, 0], [[PAGE, PAGE], [ATT]])
check(plan_a[1] is None and plan_a[0][0] == [(0, 0, 60), (180, 60, 64)], f"plan k=6: {plan_a[0][0]}")
check(plan_b[0][0] == [(0, 0, 60), (300, 60, 64)], f"plan k=12: {plan_b[0][0]}")

sa, la = FakeHandler(True, A), FakeHandler(False, A)
sb, lb = FakeHandler(True, B), FakeHandler(False, B)
wsa = T._NeutralCopy(sa, plan_a, True)
ZERO = MEM.size - 4096                # a zero region of the fake address space: the loads' zero source
ncs = {}
for name, h, p, g2c in (("la", la, plan_a, False), ("sb", sb, plan_b, True), ("lb", lb, plan_b, False)):
    ncs[name] = T._NeutralCopy(h, p, g2c)
    if not g2c:
        ncs[name].zero_ptr = ZERO
check(ncs["lb"].gaps[(0, 0)] == [(60, 240)] and ncs["la"].gaps[(0, 0)] == [(60, 120), (244, PAGE - 244)],
      f"load gaps: k=12 {ncs['lb'].gaps[(0, 0)]}, k=6 {ncs['la'].gaps[(0, 0)]}")
check(not ncs["sb"].gaps and ncs["sb"].zero is None, "stores: nothing zeroed")

MEM[:] = rng.integers(0, 256, MEM.size, dtype=np.uint8)
MEM[ZERO:] = 0
gpuA = lambda r, b: A["gpu"][0][r] + b * PAGE
gpuB = lambda r, b: B["gpu"][0][r] + b * PAGE
ram = lambda r, b: A["cpu"][0][r] + b * PAGE
# store: engine A GDN blocks 2,5 -> RAM blocks 1,3; attention block 4 -> RAM 0
sa.transfer_async(1, spec([2, 1], [[2, 5], [4]], [[1, 3], [0]]), None)
for r in range(2):
    for gb, cb in ((2, 1), (5, 3)):
        g, c = gpuA(r, gb), ram(r, cb)
        assert (MEM[c:c + 60] == MEM[g:g + 60]).all() and (MEM[c + 60:c + 124] == MEM[g + 180:g + 244]).all()
check(True, "store k=6: conv rows -> RAM 0..60, SSM (engine offset 180) -> RAM 60..124, both refs, both blocks")
att_g, att_c = A["gpu"][1][0] + 4 * ATT, A["cpu"][1][0]
check((MEM[att_c:att_c + ATT] == MEM[att_g:att_g + ATT]).all(), "attention page copied whole")
check(1 in wsa.inflight, "descriptor buffers held until the job is reported finished")
sa.get_finished()
check(1 not in wsa.inflight, "released after get_finished")

# load into engine B (k=12) blocks 0,7 from RAM 1,3; scratch rows must stay as they were
before = MEM.copy()
lb.transfer_async(2, None, spec([2, 0], [[0, 7], []], [[1, 3], []]))
for r in range(2):
    for gb, cb in ((0, 1), (7, 3)):
        g, c = gpuB(r, gb), ram(r, cb)
        assert (MEM[g:g + 60] == MEM[c:c + 60]).all(), "conv rows"
        assert (MEM[g + 300:g + 364] == MEM[c + 60:c + 124]).all(), "ssm"
        assert (MEM[g + 60:g + 300] == 0).all(), "scratch rows zeroed"
        assert (MEM[g + 364:g + PAGE] == 0).all(), "page tail zeroed"
check(True, "load into k=12: conv rows at 0, SSM at engine offset 300; scratch rows and page tail zeroed (the "
            "block's previous owner left random bytes there), as in a cold state")
other = B["gpu"][1][0]
check((MEM[other:other + 8 * ATT] == before[other:other + 8 * ATT]).all(), "nothing outside the loaded GDN pages written")
for r in range(2):
    a, b = gpuA(r, 2), gpuB(r, 0)
    assert (MEM[a:a + 60] == MEM[b:b + 60]).all() and (MEM[a + 180:a + 244] == MEM[b + 300:b + 364]).all()
check(True, "engine A (k=6) state == engine B (k=12) state on every real byte")

# round trip on the same engine
MEM[gpuA(0, 6):gpuA(0, 6) + PAGE] = 0
la.transfer_async(3, None, spec([1, 0], [[6], []], [[1], []]))
a, b = gpuA(0, 2), gpuA(0, 6)
check((MEM[a:a + 60] == MEM[b:b + 60]).all() and (MEM[a + 180:a + 244] == MEM[b + 180:b + 244]).all(),
      "round trip on k=6: store -> load restores conv rows and SSM")

# errors
for bad, what in (([[[(0, 60, None), (40, 64, None)]]], "overlapping segments"),
                  ([[[(0, 60, None), (PAGE - 10, 64, None)]]], "segments past the page"),
                  ([[[]]], "GDN ref without segments")):
    try:
        T.neutral_plan(bad, [1], [[PAGE]])
        check(False, what)
    except ValueError:
        check(True, f"neutral_plan rejects {what}")

# _install_neutral: tag + NaN-scan layout, and the fallback
W = types.SimpleNamespace(_store_handler=FakeHandler(True, A), _load_handler=FakeHandler(False, A))
T._STATE_TAG, T._STATE_K = "neutral3-ssm:float32-conv:auto", 6
gw._NC_LAYOUT = [[segs(6), segs(6)], [[]]]
check(T._install_neutral(W, [1, 0]) and T._STATE_TAG == "neutral3:60+64-ssm:float32-conv:auto", f"tag {T._STATE_TAG}")
check(gw._NC_LAYOUT[0][0] == [(0, 60, torch.float16), (60, 64, torch.float32)] and gw._NC_LAYOUT[1] == [[]],
      f"NaN scan uses packed RAM offsets: {gw._NC_LAYOUT[0][0]}")
W2 = types.SimpleNamespace(_store_handler=FakeHandler(True, A), _load_handler=FakeHandler(False, A))
T._STATE_TAG = "neutral3-ssm:float32-conv:auto"
gw._NC_LAYOUT = None
check(not T._install_neutral(W2, [1, 0]) and T._STATE_TAG == "k6-ssm:float32-conv:auto",
      f"no engine page layout -> RAM keeps the engine layout, tag falls back to {T._STATE_TAG}")
print("PASS gdn-neutral")
