"""Extension hooks for vLLM, with a fallback for stock vLLM.

The memory layer (KV lending, chain values) needs a few points inside vLLM's scheduler,
block pool, model runner and worker. echowarp/vllm (branches publish-20261113-*) has them as named hook points in
``vllm.v1.extension_hooks``; this module registers there when that module exists.

On a vLLM without it, the same hook names are emulated by wrapping the vLLM methods once (a monkeypatch per hook point,
installed on the first registration of that name). The semantics are the same in both cases, so plugin code is written
once, against the hook names:

  notify   every registered function is called, results ignored, exceptions logged
  first    the first non-None result wins (overrides); None from all = vLLM's default
  fold     value = fn(value, *args) for each registered function
  context  each function returns a context manager (or None); all are entered around the call

Hook points (name: kind, arguments). The table is the contract with echowarp/vllm's ``HOOKS``:

  scheduler.init                     notify  (scheduler)
  scheduler.before_schedule          notify  (scheduler)
  scheduler.after_schedule           notify  (scheduler, scheduler_output)
  scheduler.admit                    first   (scheduler, request) -> False keeps the request waiting this step
                                             (echowarp/vllm only: no fallback; register() returns False without it)
  block_pool.get_new_blocks          first   (pool, num_blocks) -> blocks taken, or None for the default
  block_pool.cache_full_blocks       notify  (pool, request, num_cached_blocks, num_full_blocks)
  block_pool.touch                   notify  (pool, blocks)
  block_pool.get_cached_block        notify  (pool, block_hash)
  kv_cache_manager.allocate_slots    context (manager, request)
  model_runner.kv_caches_initialized notify  (runner, kv_caches, kv_cache_config)
  model_runner.execute_model         context (runner)
  model_runner.model_forward         context (runner, args, kwargs)
  model_runner.propose_draft         context (runner)
  worker.available_kv_memory         fold    (bytes, worker) -> bytes
  worker.warmed_up                   notify  (worker)

Lent blocks: a plugin that takes blocks out of the pool for another use sets ``pool.num_lent_blocks``; echowarp/vllm's
``BlockPool.get_usage`` leaves them out of the total, and the fallback wraps ``get_usage`` to do the same.
"""
import contextlib
import functools
import logging

_log = logging.getLogger("llmt_kv.vllm_hooks")

HOOKS = {
    "scheduler.init": "notify",
    "scheduler.before_schedule": "notify",
    "scheduler.after_schedule": "notify",
    "scheduler.admit": "first",
    "block_pool.get_new_blocks": "first",
    "block_pool.cache_full_blocks": "notify",
    "block_pool.touch": "notify",
    "block_pool.get_cached_block": "notify",
    "kv_cache_manager.allocate_slots": "context",
    "model_runner.kv_caches_initialized": "notify",
    "model_runner.execute_model": "context",
    "model_runner.model_forward": "context",
    "model_runner.propose_draft": "context",
    "worker.available_kv_memory": "fold",
    "worker.warmed_up": "notify",
}

_REG = {}          # fallback registry: name -> [fn]
_PATCHED = set()   # fallback: hook names whose vLLM method is wrapped


def native():
    """vllm.v1.extension_hooks if this vLLM has it (echowarp/vllm), else None."""
    try:
        from vllm.v1 import extension_hooks
    except Exception:  # noqa: BLE001
        return None
    return extension_hooks


def mode():
    return "native" if native() is not None else "fallback"


# ---------------------------------------------------------------- fallback registry (same semantics as the fork's)
def _fns(name):
    return _REG.get(name, ())


def active(name):
    return bool(_REG.get(name))


def notify(name, *args):
    for fn in _fns(name):
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001
            _log.warning("hook %s: %s raised %r", name, getattr(fn, "__qualname__", fn), e)


def first(name, *args):
    for fn in _fns(name):
        r = fn(*args)
        if r is not None:
            return r
    return None


def fold(name, value, *args):
    for fn in _fns(name):
        value = fn(value, *args)
    return value


def context(name, *args):
    st = contextlib.ExitStack()
    for fn in _fns(name):
        cm = fn(*args)
        if cm is not None:
            st.enter_context(cm)
    return st


# ---------------------------------------------------------------- fallback patchers (one per hook point)
def _wrap(cls, attr, make):
    orig = getattr(cls, attr)
    w = make(orig)
    functools.update_wrapper(w, orig)
    w.__llmt_hook_orig__ = orig
    setattr(cls, attr, w)


def _p_scheduler(name):
    from vllm.v1.core.sched.scheduler import Scheduler
    if name == "scheduler.init":
        def make(orig):
            def __init__(self, *a, **k):
                orig(self, *a, **k)
                notify("scheduler.init", self)
            return __init__
        _wrap(Scheduler, "__init__", make)
    else:      # before / after share one wrapper
        if "scheduler.schedule" in _PATCHED:
            return
        _PATCHED.add("scheduler.schedule")

        def make(orig):
            def schedule(self, *a, **k):
                notify("scheduler.before_schedule", self)
                out = orig(self, *a, **k)
                notify("scheduler.after_schedule", self, out)
                return out
            return schedule
        _wrap(Scheduler, "schedule", make)


def _p_block_pool(name):
    from vllm.v1.core.block_pool import BlockPool
    if name == "block_pool.get_new_blocks":
        def make(orig):
            def get_new_blocks(self, num_blocks, *a, **k):
                if not a and not k:
                    r = first("block_pool.get_new_blocks", self, num_blocks)
                    if r is not None:
                        return r
                return orig(self, num_blocks, *a, **k)
            return get_new_blocks
        _wrap(BlockPool, "get_new_blocks", make)
    elif name == "block_pool.cache_full_blocks":
        def make(orig):
            def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, *a, **k):
                r = orig(self, request, blocks, num_cached_blocks, num_full_blocks, *a, **k)
                notify("block_pool.cache_full_blocks", self, request, num_cached_blocks, num_full_blocks)
                return r
            return cache_full_blocks
        _wrap(BlockPool, "cache_full_blocks", make)
    elif name == "block_pool.touch":
        def make(orig):
            def touch(self, blocks, *a, **k):
                notify("block_pool.touch", self, blocks)
                return orig(self, blocks, *a, **k)
            return touch
        _wrap(BlockPool, "touch", make)
    elif name == "block_pool.get_cached_block":
        if not hasattr(BlockPool, "get_cached_block"):
            return
        def make(orig):
            def get_cached_block(self, block_hash, *a, **k):
                notify("block_pool.get_cached_block", self, block_hash)
                return orig(self, block_hash, *a, **k)
            return get_cached_block
        _wrap(BlockPool, "get_cached_block", make)


def _p_lent_usage():
    """Fallback for BlockPool.num_lent_blocks: get_usage without the lent blocks."""
    from vllm.v1.core.block_pool import BlockPool
    if "lent_usage" in _PATCHED or not hasattr(BlockPool, "get_usage"):
        return
    _PATCHED.add("lent_usage")

    def make(orig):
        def get_usage(self):
            lent = getattr(self, "num_lent_blocks", 0)
            if not lent:
                return orig(self)
            total = max(1, self.num_gpu_blocks - 1 - lent)
            return max(0.0, min(1.0, 1.0 - self.get_num_free_blocks() / total))
        return get_usage
    _wrap(BlockPool, "get_usage", make)


def _p_kv_cache_manager(name):
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    def make(orig):
        def allocate_slots(self, request, *a, **k):
            with context("kv_cache_manager.allocate_slots", self, request):
                return orig(self, request, *a, **k)
        return allocate_slots
    _wrap(KVCacheManager, "allocate_slots", make)


def _p_model_runner(name):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if name == "model_runner.kv_caches_initialized":
        def make(orig):
            def initialize_kv_cache_tensors(self, kv_cache_config, *a, **k):
                kv = orig(self, kv_cache_config, *a, **k)
                notify("model_runner.kv_caches_initialized", self, kv, kv_cache_config)
                return kv
            return initialize_kv_cache_tensors
        _wrap(GPUModelRunner, "initialize_kv_cache_tensors", make)
    elif name == "model_runner.execute_model":
        def make(orig):
            def execute_model(self, *a, **k):
                with context("model_runner.execute_model", self):
                    return orig(self, *a, **k)
            return execute_model
        _wrap(GPUModelRunner, "execute_model", make)
    elif name == "model_runner.model_forward":
        def make(orig):
            def _model_forward(self, *a, **k):
                with context("model_runner.model_forward", self, a, k):
                    return orig(self, *a, **k)
            return _model_forward
        _wrap(GPUModelRunner, "_model_forward", make)
    elif name == "model_runner.propose_draft":
        if not hasattr(GPUModelRunner, "propose_draft_token_ids"):
            return
        def make(orig):
            def propose_draft_token_ids(self, *a, **k):
                with context("model_runner.propose_draft", self):
                    return orig(self, *a, **k)
            return propose_draft_token_ids
        _wrap(GPUModelRunner, "propose_draft_token_ids", make)


def _p_worker(name):
    from vllm.v1.worker import gpu_worker as gw
    if name == "worker.available_kv_memory":
        def make(orig):
            def determine_available_memory(self, *a, **k):
                return fold("worker.available_kv_memory", orig(self, *a, **k), self)
            return determine_available_memory
        _wrap(gw.Worker, "determine_available_memory", make)
    else:
        def make(orig):
            def compile_or_warm_up_model(self, *a, **k):
                r = orig(self, *a, **k)
                notify("worker.warmed_up", self)
                return r
            return compile_or_warm_up_model
        _wrap(gw.Worker, "compile_or_warm_up_model", make)


_PATCHERS = {
    "scheduler": _p_scheduler,
    "block_pool": _p_block_pool,
    "kv_cache_manager": _p_kv_cache_manager,
    "model_runner": _p_model_runner,
    "worker": _p_worker,
}


def register(name, fn):
    """Register fn at hook point name. Returns True when the hook will be called (natively or by the fallback)."""
    if name not in HOOKS:
        raise ValueError(f"unknown vLLM hook point {name!r}")
    h = native()
    if h is not None:
        h.register(name, fn)
        return True
    if name == "scheduler.admit":
        _log.warning("hook %s needs echowarp/vllm (vllm.v1.extension_hooks); not installed", name)
        return False
    if name not in _PATCHED:
        _PATCHERS[name.split(".", 1)[0]](name)
        _PATCHED.add(name)
    _REG.setdefault(name, []).append(fn)
    return True


def track_lent_blocks():
    """Call once if the plugin sets pool.num_lent_blocks (native vLLM handles it; the fallback wraps get_usage)."""
    if native() is None:
        _p_lent_usage()


def unregister(name, fn):
    h = native()
    if h is not None:
        h.unregister(name, fn)
        return
    lst = _REG.get(name, [])
    if fn in lst:
        lst.remove(fn)


def clear():
    """Tests: forget fallback registrations (wrappers stay installed and become no-ops)."""
    _REG.clear()
