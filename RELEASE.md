# AllBroker 2026-11-16: first public release

## What is in it

- **KV broker** (`llmt-kvbroker`, Rust): one shared host-RAM pool (SSD tier optional) for KV blocks and GDN states of
  every engine on the host; namespaces, leases, eviction, prefix hints for routers (HINT / WHERE).
- **Weight keeper** (`llmt-keeper`, `llmt_kv.whx_loader`): model weights stay in GPU memory across engine restarts;
  systemd unit and `llmt-keeper-ctl` in `contrib/`.
- **vLLM plugin** `llmt_kv` (entry point `vllm.general_plugins`): the RAM-tier connector (`llmt_kv.cuda_tier`,
  `llmt_kv.xpu_tier`) with load / resident / hold lease classes and broker-side renewal, the keeper loader, the lazy
  vision tower (`llmt_kv.lazyvis`, one broker entry shared by all engines of a model) and the lend manager
  (`llmt_kv.lend`: LoRA adapters and the vision tower in lent KV blocks). Works on stock vLLM 0.26 (one-time method
  wrappers) and natively on echowarp/vllm's extension hooks.
- broker: memory-pressure sweeps no longer hold the lock while freeing (requests could time out during a large sweep on unpinned pools).

## How it was validated

- Rust workspace: 203 passed, 0 failed. Python CPU suite (`crates/py/tests/lend/run-cpu.sh`): 17 passed, 0 failed in
  all four combinations (CUDA image and XPU image, each with stock vLLM and with the echowarp/vllm branch mounted).
- GPU runs with production images and this branch's files mounted (2026-11-15/16):
  - RTX 5090, fallback and native hooks: start OK, KV 482,846 tokens, vision tower lent and returned, 0 OOM /
    tracebacks under an agent simulation (79-82 agent turns, 0 errors) and a 6 x 60k-token + images stress run;
    log-probs inside the start-to-start noise floor (max |dlp| 0.070 vs 0.091 between two production starts).
  - Arc B70, fallback and native hooks: KV 497,302 tokens, LoRA decode 179.9 tok/s (1 stream) / 161.2 (2 streams),
    two adapters decoding together, 6 x 60k stress with 0 OOM, minimum free device memory 557 MiB.
- Keeper (docs/keeper.md): engine restart to healthy RTX 5090 226 s without the keeper, about 2 min with it; Arc B70
  150-200 s without, 85-137 s with.
- Vision-tower lending with the packed KV layout on the RTX 5090: KV 436,585 -> 482,846 tokens.

## Known limitations

- The `EXL3_LEND_ONLY` path (vision-tower lending without LoRA) and the lease-class / keeper-lazy-vision ports were
  validated with the module files mounted, not yet through the published package's plugin entry point.
- The tier's own modules (`vllm_tier`, `whx_loader` as published) were not part of the GPU runs above; production ran
  the same code paths from its own copies.
- `scheduler.admit` (admission hold) has no stock-vLLM fallback; it works only with echowarp/vllm.
- LoRA lending is off unless `EXL3_LORA_HOT=1`; AllRunner enables it on the Arc only.
- Chain-aware compaction is off by default (`EXL3_LEND_COMPACT=0`).
