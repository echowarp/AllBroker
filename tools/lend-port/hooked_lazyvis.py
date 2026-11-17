def install():
    """Install the lazy tower (every vLLM process; the plugin entry point calls it). Model-loader points are wrapped
    (vLLM has no loader hook yet); KV sizing and warm-up use llmt_kv.vllm_hooks."""
    if not ON or _STATE.get("installed"):
        return
    _STATE["installed"] = True
    import vllm.model_executor.model_loader.base_loader as bl
    from llmt_kv import vllm_hooks as H
    orig_pwal = bl.process_weights_after_loading

    def pwal(model, *a, **k):
        r = orig_pwal(model, *a, **k)
        _offload(model, early=False)
        return r
    bl.process_weights_after_loading = pwal

    # earliest point: right after the model is built, before load_weights (base_loader calls log_model_inspection in
    # between). The tower's (still uninitialized) params move to the host copy and the checkpoint is read straight
    # into it, so the tower never holds device memory that the driver might keep pooled afterwards.
    orig_inspect = bl.log_model_inspection

    def inspect(model, *a, **k):
        r = orig_inspect(model, *a, **k)
        _offload(model, early=True)
        return r
    bl.log_model_inspection = inspect

    def available_kv_memory(r, worker):
        _STATE["act_reserve"] = int(getattr(worker, "peak_activation_memory", 0) or 0)
        _log("activation reserve from the profile run: %.2f GiB", _STATE["act_reserve"] / 2**30)
        return r

    def warmed_up(worker):
        _STATE["ready"] = True
        for lv in _STATE["towers"]:   # whatever warm-up loaded goes back out; the startup host copy is freed
            lv.mark_ready()
        _sync()
        _STATE["base_alloc"] = _ACC.memory_allocated()   # idle engine: weights + KV + persistent buffers

    H.register("worker.available_kv_memory", available_kv_memory)
    H.register("worker.warmed_up", warmed_up)
    _log("installed (mode %s, ttl %.0f s, broker %s, namespace %r)", MODE, TTL, BROKER, NS_NAME)
