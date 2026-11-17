# Hook-based replacements for the lend manager's vLLM wrappers (tools/lend-port/port.py swaps them in by function name).
# Each function registers at the named hook points of llmt_kv.vllm_hooks (native on echowarp/vllm, a one-time method
# wrapper on stock vLLM) and keeps the wrapped behaviour of the source it replaces.


def _patch_block_pool():
    from llmt_kv import vllm_hooks as H

    def cache_full_blocks(pool, request, num_cached_blocks, num_full_blocks):
        try:
            _chain_note(pool, request, num_cached_blocks, num_full_blocks)
        except Exception as e:  # noqa: BLE001
            _log("chain note: %r", e, level="warning")

    def get_cached_block(pool, block_hash):
        if _REGRET:
            try:
                _regret_check(block_hash)
            except Exception:  # noqa: BLE001
                pass

    def touch(pool, blocks):
        try:
            now = time.monotonic()
            for b in blocks:
                m = _CH.get(_rawh(getattr(b, "block_hash", None)))
                if m is not None:
                    m[3] += 1
                    m[4] = now
        except Exception:  # noqa: BLE001
            pass

    try:
        H.register("block_pool.cache_full_blocks", cache_full_blocks)
        H.register("block_pool.get_cached_block", get_cached_block)
        H.register("block_pool.touch", touch)
        H.track_lent_blocks()                      # KV usage of the pool without the lent blocks
    except Exception as e:  # noqa: BLE001
        _log("block pool hooks not installed (%r): compaction runs without chain values", e, level="warning")


def _patch_scheduler():
    from llmt_kv import vllm_hooks as H

    def init(sched):
        if getattr(sched, "lora_config", None) is not None:
            sched.lora_config = _LoraCfgView(sched.lora_config)
        try:
            _ST["block_tokens"] = int(sched.cache_config.block_size or 0)
        except Exception:  # noqa: BLE001
            pass

    def before(sched):
        v = getattr(sched, "lora_config", None)
        _ST["sched"] = sched
        try:
            _sess_note(sched)
        except Exception:  # noqa: BLE001
            pass
        with _LOCK:          # every engine with this plugin: LoRA and vision tenants (vision also without --enable-lora)
            try:
                _lend_tick(sched)
            except Exception as e:  # noqa: BLE001
                _log("lend tick: %r", e, level="warning")
            _lent_count(getattr(getattr(sched, "kv_cache_manager", None), "block_pool", None))
            try:
                _stats_tick()
            except Exception:  # noqa: BLE001
                pass
        if isinstance(v, _LoraCfgView):
            v._floor = len({r.lora_request.lora_int_id for r in sched.running
                            if r.lora_request and r.lora_request.lora_int_id > 0})

    def after(sched, out):
        v = getattr(sched, "lora_config", None)
        if not isinstance(v, _LoraCfgView):
            return
        try:
            nw = sum(1 for q in _sched_queues(sched)[1:] for r in q if getattr(r, "lora_request", None))
        except Exception:  # noqa: BLE001
            nw = 0
        if nw:
            _ST["n"]["lora_wait_steps"] = _ST["n"].get("lora_wait_steps", 0) + 1
            now = time.monotonic()
            if now - _ST.get("wait_log", 0) >= 30:
                _ST["wait_log"] = now
                try:
                    room = _room() if _ST["ready"] else float("nan")
                except Exception:  # noqa: BLE001
                    room = float("nan")
                _log("%d LoRA request(s) waiting; device slot capacity %d of %d (slot %.0f MiB + margin %d MiB, "
                     "room %.0f MiB); base requests are not held back", nw, v.max_loras, v._cfg.max_loras,
                     _slot_bytes() / 2**20, MARGIN >> 20, room / 2**20)

    try:
        H.register("scheduler.init", init)
        H.register("scheduler.before_schedule", before)
        H.register("scheduler.after_schedule", after)
    except Exception as e:  # noqa: BLE001
        _log("scheduler hooks not installed (%r): batches may ask for more adapters than the device holds", e,
             level="warning")
        return
    if ADMIT is not None:
        H.register("scheduler.admit", ADMIT)






def _patch_kv_layout():
    if not LEND:
        return
    from llmt_kv import vllm_hooks as H

    def kv_caches_initialized(runner, kv, kv_cache_config):
        try:
            _kv_layout_init(kv, int(kv_cache_config.num_blocks))
        except Exception as e:  # noqa: BLE001
            _KVL.update(ok=False, why=repr(e))
            _log("KV-block lending off: %r", e, level="warning")

    H.register("model_runner.kv_caches_initialized", kv_caches_initialized)


def _patch_memory_hooks():
    from llmt_kv import vllm_hooks as H

    def available_kv_memory(r, worker):
        _ST["act_reserve"] = int(getattr(worker, "peak_activation_memory", 0) or 0)
        _log("activation reserve from the profile run: %.2f GiB; LoRA slot layers %d (%.1f MiB when allocated)",
             _ST["act_reserve"] / 2**30, len(_ST["layers"]), sum(_layer_bytes(l) for l in _ST["layers"]) / 2**20)
        return r

    def warmed_up(worker):
        with _LOCK:
            if _ST["where"] != "off":
                _release("warm-up done")
            _sync()
            _ST["base_alloc"] = _acc().memory_allocated()
            _ST["ready"] = True
        threading.Thread(target=_reaper, name="llmt-lora-reaper", daemon=True).start()
        _log("ready (mode %s, ttl %.0f s, margin %d MiB, broker %s)", MODE, TTL, MARGIN >> 20,
             ("%s ns %r" % (_ST["broker"].sock, NS_NAME)) if _ST["broker"] else "off")

    H.register("worker.available_kv_memory", available_kv_memory)
    H.register("worker.warmed_up", warmed_up)


def _lent_count(pool):
    """pool.num_lent_blocks: blocks out of the KV pool for tenants (and returned ones not yet freed). The KV usage
    metric, the scheduler's length-cap wait and its tail-hold watermark leave them out (echowarp/vllm; the
    llmt_lent_blocks alias is kept for older engine builds)."""
    if pool is None:
        return 0
    n = sum(len(t["blocks"]) for t in _TENANTS.values()) + sum(len(b) for b in (_ST.get("to_return") or []))
    pool.num_lent_blocks = pool.llmt_lent_blocks = n
    return n


# ------------------------------------------------------------------------------------------ admission extension point
# Admission hold / priority preemption for a full KV pool (design: memory-lending section 5.2, not built yet).
# set_admission_policy(fn): fn(scheduler, request) -> False keeps a waiting request waiting this step (it goes to the
# scheduler's skipped queue and is offered again next step), True / None = admit as usual. Needs echowarp/vllm
# (hook point scheduler.admit); on stock vLLM the call logs a warning and returns False.
ADMIT = None


def set_admission_policy(fn):
    global ADMIT
    ADMIT = fn
    if _ST.get("installed"):
        from llmt_kv import vllm_hooks as H
        return H.register("scheduler.admit", fn)
    return True

