"""CPU test of llmt_kv.vllm_hooks: registry semantics, and on a vLLM without vllm.v1.extension_hooks the fallback
wrappers (installed once per hook point; get_new_blocks override / default; lent blocks left out of get_usage).
Needs vLLM + torch (any build); no GPU.   python3 test_vllm_hooks.py"""
import contextlib

from llmt_kv import vllm_hooks as H

# ---- registry semantics (fallback registry; the fork's has the same contract)
H._REG.clear()
seen = []
H._REG["block_pool.touch"] = [lambda *a: seen.append(a), lambda *a: 1 / 0]
H.notify("block_pool.touch", "p", [1])           # an observer that raises is logged, not raised
assert seen == [("p", [1])]
H._REG["block_pool.get_new_blocks"] = [lambda p, n: None, lambda p, n: ["b"] * n, lambda p, n: ["x"]]
assert H.first("block_pool.get_new_blocks", None, 2) == ["b", "b"]
H._REG["worker.available_kv_memory"] = [lambda v, w: v + 1, lambda v, w: v * 2]
assert H.fold("worker.available_kv_memory", 3, None) == 8
log = []


@contextlib.contextmanager
def cm(tag):
    log.append("in " + tag)
    yield
    log.append("out " + tag)


H._REG["model_runner.execute_model"] = [lambda r: cm("a"), lambda r: None, lambda r: cm("b")]
with H.context("model_runner.execute_model", None):
    log.append("body")
assert log == ["in a", "in b", "body", "out b", "out a"], log
try:
    H.register("scheduler.no_such_point", print)
    raise AssertionError("unknown hook name accepted")
except ValueError:
    pass
H._REG.clear()
print("registry semantics OK")

# ---- fallback wrappers against the installed vLLM
if H.native() is not None:
    print("native extension hooks present (echowarp/vllm): fallback checks skipped; mode", H.mode())
else:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker import gpu_worker as gw

    calls = []
    BlockPool.get_new_blocks = lambda self, n: ["default"] * n        # stand-in for the real method
    BlockPool.get_num_free_blocks = lambda self: 2
    BlockPool.get_usage = lambda self: 1.0 - self.get_num_free_blocks() / (self.num_gpu_blocks - 1)
    take = {"on": False}
    assert H.register("block_pool.get_new_blocks", lambda p, n: (["placed"] * n) if take["on"] else None)
    assert H.register("block_pool.get_new_blocks", lambda p, n: calls.append(n))
    w1 = BlockPool.get_new_blocks
    assert hasattr(w1, "__llmt_hook_orig__")
    H.register("block_pool.get_new_blocks", lambda p, n: None)
    assert BlockPool.get_new_blocks is w1, "a hook point is wrapped once"

    class P:
        num_gpu_blocks = 11

        def get_num_free_blocks(self):
            return 2
    p = P()
    assert BlockPool.get_new_blocks(p, 2) == ["default"] * 2 and calls == [2]
    take["on"] = True
    assert BlockPool.get_new_blocks(p, 3) == ["placed"] * 3
    assert abs(BlockPool.get_usage(p) - 0.8) < 1e-9
    H.track_lent_blocks()
    p.num_lent_blocks = 4                             # 10 usable - 4 lent = 6, 2 free -> 4/6 used
    assert abs(BlockPool.get_usage(p) - (1 - 2 / 6)) < 1e-9, BlockPool.get_usage(p)
    assert not H.register("scheduler.admit", lambda s, r: False), "admit needs echowarp/vllm"
    for name in ("scheduler.init", "scheduler.before_schedule", "scheduler.after_schedule",
                 "kv_cache_manager.allocate_slots", "model_runner.kv_caches_initialized",
                 "model_runner.execute_model", "model_runner.model_forward", "model_runner.propose_draft",
                 "worker.available_kv_memory", "worker.warmed_up", "block_pool.cache_full_blocks",
                 "block_pool.touch", "block_pool.get_cached_block"):
        assert H.register(name, lambda *a: None), name
    assert hasattr(Scheduler.schedule, "__llmt_hook_orig__") and hasattr(Scheduler.__init__, "__llmt_hook_orig__")
    assert hasattr(gw.Worker.determine_available_memory, "__llmt_hook_orig__")
    assert hasattr(gw.Worker.compile_or_warm_up_model, "__llmt_hook_orig__")
    print("fallback wrappers OK")
print("test_vllm_hooks OK")
