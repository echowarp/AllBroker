# llmt_kv as a vLLM plugin

`pip install llmt-kv` (or the wheel from `crates/py`) makes AllBroker's engine side available to vLLM with no change
to vLLM's code:

| Part | Module | How vLLM finds it | Switch |
|---|---|---|---|
| Shared host-RAM KV tier | `llmt_kv.cuda_tier` / `llmt_kv.xpu_tier` (over `llmt_kv.vllm_tier`) | vLLM's `OffloadingConnector`, `spec_module_path` | `--kv-transfer-config` |
| Weight keeper loader | `llmt_kv.whx_loader` | `vllm.general_plugins` entry point `llmt` | `EXL3_WHX=<keeper socket>` |
| Lazy vision tower | `llmt_kv.lazyvis` | the same entry point | `EXL3_LAZY_VISION=1` |
| Lend manager (LoRA slots, vision tower in lent KV blocks) | `llmt_kv.lend` | the same entry point | `EXL3_LORA_HOT=1` |
| No-mmap safetensors loader | `llmt_kv.nommap_loader` | called by the engine plugin (`patch()`) | `EXL3_NOMMAP_LOAD` |

With none of the switches set, the plugin does nothing. `LLMT_KV_PLUGIN=0` disables the entry point entirely.

## KV tier

```sh
vllm serve <model> --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"spec_name":"BrokerOffloadingSpec","spec_module_path":"llmt_kv.cuda_tier",
  "cpu_bytes_to_use":1,"broker_socket":"/run/llmt-kv/broker.sock"}}'
```

Use `llmt_kv.xpu_tier` on Intel GPUs. The GPU-specific parts (how the pool is pinned: `cudaHostRegister` on NVIDIA, a
Level Zero host import on Intel) live in those two modules; everything else is shared. Every key the tier sends
includes the model, the KV format and the layout, so engines on different GPUs share entries only when their bytes
are interchangeable.

## Design note: plugin, hooks, or deep integration

The question was whether the memory layer should be a vLLM plugin or a deep integration into a vLLM fork. We chose a
plugin with small, named hooks in the fork:

- **The policy stays out of vLLM.** Lending, chain values, compaction, placement and the broker client are thousands
  of lines that change daily. Inside the fork they would make every vLLM upgrade a merge of that code; as a plugin they
  upgrade independently, and the same code serves the NVIDIA and the Intel engine.
- **vLLM gets only what a plugin cannot do from outside**: named extension points
  (`vllm/v1/extension_hooks.py` in [echowarp/vllm](https://github.com/echowarp/vllm), branches
  `publish-20261113-cuda` / `-xpu`) and two pieces of accounting (`BlockPool.num_lent_blocks` in KV usage, the
  scheduler's length-cap wait and tail-hold watermark). Each is a small commit that can go upstream on its own.
  With nothing registered every hook is one dict lookup.
- **Stock vLLM still works.** `llmt_kv.vllm_hooks` registers at the fork's hook points when they exist, and on a vLLM
  without them it emulates the same hook names by wrapping the vLLM method once per hook point. Plugin code is written
  once, against the hook names; the fallback is the old monkeypatching, kept in one module with the same semantics.
  The one point with no fallback is `scheduler.admit` (it needs a call inside the scheduler's waiting loop).
- **Deep integration was rejected** for the policy because it would tie the memory layer to one vLLM version and one
  fork. It was also rejected for the broker client: vLLM's `OffloadingConnector` already has the spec extension point,
  so the KV tier never needed a fork change.

Trade-offs:

- On stock vLLM the fallback wrappers depend on private method names and signatures (`BlockPool.get_new_blocks`,
  `GPUModelRunner._model_forward`, ...). A vLLM upgrade can break them silently; the fork's hooks cannot. The CPU test
  `crates/py/tests/test_vllm_hooks.py` checks that every wrapper lands.
- A plugin still reaches into vLLM objects (block pool free queue, punica wrapper, LoRA layers). Hooks fix where it is
  called, not what it touches.
- LoRA internals are still wrapped directly (no hook yet), listed below. They are the next hook candidates.

### Hook points used

| Hook | Used by |
|---|---|
| `scheduler.init`, `scheduler.before_schedule`, `scheduler.after_schedule` | lend manager: LoRA capacity view, lend ticks, session notes, stats |
| `scheduler.admit` | extension point for admission hold / priority preemption when the KV pool is full (`llmt_kv.lend.set_admission_policy`); no policy ships yet |
| `block_pool.cache_full_blocks`, `block_pool.touch`, `block_pool.get_cached_block` | prefix chain values, regret counters |
| `model_runner.kv_caches_initialized` | KV layout map for lending (block id to byte ranges) |
| `worker.available_kv_memory`, `worker.warmed_up` | activation reserve, ready state |

### Still wrapped directly

| Wrapped | Why | Hook it would need |
|---|---|---|
| `vllm.lora` layers (`create_lora_weights`, `_apply_lora_to_output`, `_get_lora_device`), punica `LoRAKernelMeta.prepare_tensors` | slot buffers on the meta device until an adapter is active; compacted LoRA rows | a LoRA slot-allocator hook |
| `LoRAModelRunnerMixin` (`maybe_setup_dummy_loras`, `_set_active_loras`), `CudagraphDispatcher.dispatch` / `_get_lora_cases`, `CUDAGraphWrapper.__call__`, `GPUModelRunner.execute_model` (LoRA graph pool) | no dummy adapters at startup, LoRA steps without the base graphs | a cudagraph-dispatch hook |
| `vllm.lora` model manager safetensors loading | adapter bytes from the broker's pool | a LoRA weight-source hook |
| `BaseModelLoader.load_model`, `GPUModelRunner.load_model` (keeper), `process_weights_after_loading` / `log_model_inspection` (lazy vision) | attach instead of load; tower offload before the checkpoint read | model-loader hooks |
| `EngineCoreProc._process_engine_step` (keeper load pacing, optional) | yield the GPU to a prefill engine while weights load | an engine-step hook |

## Re-syncing the lend manager

The lend manager and the lazy vision module are developed in an internal tree and ported here with
`tools/lend-port/port.py`. The port swaps the scheduler / block-pool / runner / worker wrappers for hook
registrations (`tools/lend-port/hooked*.py`), checks that every swapped function is unchanged at the source
(`MANIFEST.json`), scrubs comments, and records the source commit and file hashes in `tools/lend-port/SYNC.json`.
