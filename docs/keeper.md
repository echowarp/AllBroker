# Weight keeper: restart a vLLM engine in seconds

**Weights stay resident in GPU memory across engine restarts.** After the first load, the engine moves its processed
weights into GPU memory that can be shared by file descriptor and hands those descriptors to `llmt-keeper`, a small
process with no GPU context. The next engine start asks the keeper for them, maps the same physical VRAM, and skips
reading and processing the checkpoint: about 15 GiB attach in 3-5 s instead of one to several minutes of loading.

This is for faster development as much as for production: changing a launcher flag, a scheduler patch or a kernel
setting that does not touch the weights costs seconds of weight loading per restart, not minutes. The keeper works
on its own (no KV broker needed): `llmt-keeper` + `llmt_kv.whx_loader` + `EXL3_WHX=<socket>`.

| | |
|---|---|
| Attach | 15.06 GiB (+ 0.20 GiB side files) in 3.7 s on the Arc B70; 15.03 GiB in 2.8 s on the RTX 5090 |
| Cold load (same model, same disks) | 56-74 s on the Arc (slow USB disk), 56-57 s on the 5090, plus processing |
| Engine restart to healthy | RTX 5090: 226 s without the keeper, about 2 min with it (compile cache and graph capture included); Arc B70: 150-200 s without, 85-137 s with |
| GPUs | Intel (Level Zero IPC: dma-buf export / import), NVIDIA (CUDA VMM with POSIX-fd shareable handles) |

The rest of a start is compilation and graph capture, which vLLM's compile cache addresses separately. The keeper
never maps the memory itself; holding the descriptor is what keeps the allocation alive.

## Status

| GPU | Mechanism | State |
|---|---|---|
| Intel Arc Pro B70 (Level Zero) | dma-buf export / import (`weights/xpu/whx_ops.sycl`, `whx.so`) | in production use, one keeper per GPU; greedy output identical to a normal load |
| NVIDIA RTX 5090 (CUDA) | CUDA VMM with POSIX-fd shareable handles (libcuda over ctypes, no build step) | in production use (needs the exl3cuda plugin's keeper hooks); output within start-to-start noise (below) |
| Both | keeper restart through systemd's fd store | passes on both with a real unit (restart, SIGKILL, stop + start, drop frees the memory) |

Measured on the Arc B70 (vLLM 0.26, EXL3 4.0 bpw Qwen3.8-27B + MTP drafter, 15.26 GiB held, weights read from a slow
USB disk):

| Start | Weights phase | Engine healthy after | Greedy output |
|---|---|---|---|
| Normal (no keeper) | 56–74 s | 150–200 s | reference |
| First start with the keeper (export) | 56–74 s + 2 s export | 141–162 s | identical |
| **Attach** | **2.6–4.3 s** | **85–86 s** | identical |

Single-stream decode speed was unchanged (80.0–80.8 tok/s). The rest of the start time is compilation and graph
capture, which vLLM's compile cache addresses separately.

Measured on the RTX 5090 (vLLM 0.26, the same model with int4 KV, the full production stack in a lab container;
weights read from the internal disk):

| Start | Weights phase | Engine healthy after | KV cache |
|---|---|---|---|
| Normal (no keeper) | 56–57 s (target + drafter) | 458 s (fresh compile) | 439,476 tokens |
| First start with the keeper (export) | 66 s + 1.6 s export | 468 s (fresh compile) | 419,237 tokens |
| **Attach** (twice) | **2.8 s** | **236–241 s** (compile cache) | 436,585 tokens (-0.66 %; margin 0) |

- Held: 15.03 GiB (3,189 tensors; the model loads 16.5 GiB). The 1.78 GiB left in torch are the parts every start
  builds itself: the MTP draft heads and the DRAFT-DYN buffers.
- Output: on this GPU no two starts are bitwise equal, with or without the keeper (greedy text and logprobs vary from
  start to start). Teacher-forced prompt logprobs (`weights/tests/lp_probe.py`, 1.6k synthetic tokens): keeper-off vs
  attach mean |diff| 0.012 nats (median 0.00009), attach vs attach (same held bytes) 0.011; first-token argmax equal
  in 8/8 prompts for every pair. The keeper adds nothing measurable beyond that start-to-start noise.
- Decode speed: 214–271 tok/s single stream, with the same spread in keeper-off and keeper starts.
- Memory: an attached engine uses 0.03 GiB more than a normal one (weights + non-torch 16.79 vs 16.76 GiB), so the
  CUDA backend's KV margin is 0 (the Arc's 512 MiB margin cost 6.6% of the KV cache here); measured with 0: 436,585
  vs 439,476 tokens (-0.66 %). The export start itself keeps about 0.3 GiB of moved-from
  memory alive (4.6% less KV on that one start).

## How it works

**Export** (the keeper has no entry for this model):
1. Normal load.
2. After the target model and the drafter are loaded and shared weights are tied, every reachable GPU tensor moves
   onto exportable memory, one 1 GiB chunk at a time (peak = model + one chunk). `Tensor.set_` keeps the model
   unchanged.
3. Drafter tensors that share the target's memory are stored as references.
4. The descriptors, a manifest (tensor paths, shapes, strides, dtypes) and per-tensor checksums go to the keeper.

**Attach** (the keeper has a matching entry):
1. The model is built on the meta device and "loaded" from meta tensors of the checkpoint's shapes.
2. Tensors are matched to the manifest by path, with shape and dtype checks, and become views of the imported
   memory. Every checksum must match.
3. On any mismatch the keeper entry is dropped and the engine exits with a message; the next start loads normally
   and exports again. There is no in-process fallback, because a half-built meta model leaves global state behind.

**The key** identifies the weights: the model path, the checkpoint files' identity (inode, size, modification
time), the load-call index (target model, MTP drafter), the engine image and every `EXL3_*` / `VLLM_*` setting that
changes the weight tensors. Runtime-only settings are left out, so changing them never forces a full load or a
second copy: the KV tier, scheduling and speculation switches, the LoRA / KV lending / lazy-vision runtime settings
(`EXL3_LORA_*`, `EXL3_LEND_*`, `EXL3_LAZY_VISION_{TTL,MARGIN,BROKER}`,
`VLLM_ALLOW_RUNTIME_LORA_UPDATING`), the compile-cache location and switch (`VLLM_CACHE_ROOT`,
`VLLM_DISABLE_COMPILE_CACHE`) and every `EXL3_WHX_*` setting. The list is `_key_env` in `llmt_kv/whx_loader.py`.

**When a full load happens:** the first start; a new checkpoint (any file changed); a setting in the key changed;
the keeper restarted without its fd store, or its entry was dropped; an attach that failed a check (the entry is
dropped and the engine exits; the next start loads and exports again). Treat every miss for an unchanged model as a
key bug: the loader logs a `MISS` line with the environment difference to the held entry.

**Safety modes:**
- `EXL3_WHX_EXCLUSIVE=1` (one keeper per GPU, the production setting): a miss on the target model drops every held
  entry before the normal load, so a changed setting never loads next to a stale 15 GiB copy.
- `EXL3_WHX_READONLY=1` (a lab or test engine next to a production keeper): attach only, never put or drop; a miss
  stops the engine instead of loading. With `EXL3_WHX_SHARE=1` two live engines share one held copy.
- `drop`: `llmt-keeper-ctl <instance> drop-all` frees the held VRAM once no engine maps it (run it before anything
  else needs the whole GPU while the engine is stopped). Held weights are never evicted under memory pressure.
- An attach checks every tensor's shape, stride, dtype and checksum against the manifest; any mismatch drops the
  entry and stops the engine (no half-attached model).
- The lazy vision tower is not held: on attach it is offloaded and finalized at the same two points as on a normal
  start (`llmt_kv.lazyvis`), so the tower's VRAM goes to KV as without the keeper.
- On the Arc the torch allocator cap (`TORCH_CAP_MIB`, the out-of-resources guard) is lowered by the held MiB after
  an export or attach, because held weights live outside torch's allocator.

**Backends.** One loader (`llmt_kv/whx_loader.py`) with a device backend per vendor; the protocol, key, walk,
manifest and checksums are shared. It picks the backend from the package it is installed in (`exl3xpu` / `exl3cuda`;
`EXL3_WHX_BACKEND=xpu|cuda` overrides).

- **Intel (Level Zero):** `whx.so` (`weights/xpu/whx_ops.sycl`) allocates dma-buf exportable device memory in torch's
  SYCL context and imports the fds again.
- **NVIDIA (CUDA VMM):** `cuMemCreate` with POSIX-fd shareable handles, mapped into torch's primary context through
  libcuda over ctypes (no build step). The EXL3 linears of the CUDA plugin live in a module-level table, not in the
  model's parameters, so the plugin exposes keeper hooks (`whx_registry`, `whx_finish` in
  `exl3cuda/vllm_plugin.py`): the registry hands those tensors to the walk by module name; after an export
  `whx_finish` rebuilds the fused-decode pointer tables that still pointed at the moved-from memory; on attach the
  meta build registers placeholders and defers the draft-head and DRAFT-DYN steps (they read weight data), and
  `whx_finish` builds them from the held views at the table ids reserved for them, so every id equals a normal load's.
  The draft heads are not held; every attach rebuilds them (the same work as a normal load). Without the hooks the
  CUDA loader stays off rather than export half of the weights.

Side files (the environment record of each key, weight-derived CPU tensors, the DRAFT-DYN host table of about 0.7 GB)
go next to the socket, or to `EXL3_WHX_DIR`; keep them off `/run` when RAM is tight. `EXL3_WHX_*` settings are never
part of the key; every other `EXL3_*` / `VLLM_*` setting is unless listed as runtime-only in the loader, so test
switches that must not change the key need other names.

**Memory accounting:** vLLM's startup check counts held bytes as free, and they show up later as non-torch memory,
like a load. On attach, a margin (`EXL3_WHX_MARGIN_MB`; Arc 512 MiB, CUDA 0) comes off the KV budget to keep the
total equal to a normal start.

## The keeper process

```sh
llmt-keeper --socket /run/llmt-whx/keeper.sock [--max-gib N] [--fdstore STATE_FILE]
```

- One request per connection: an 8-byte little-endian length carrying the descriptors (SCM_RIGHTS), then a JSON
  body. Ops: `put` (key, chunk sizes, manifest + one descriptor per chunk), `get`, `list`, `drop`.
- **Held weights are not a cache.** Nothing is evicted under memory pressure. `--max-gib` (default: no cap) drops the
  oldest entry only when a new, different model would exceed it.
- **Surviving a keeper restart:** with `--fdstore`, every held descriptor is also parked in systemd's file descriptor
  store and the metadata goes to a state file, so restarting or upgrading the keeper keeps the weights in VRAM. Use
  the unit in `contrib/systemd/llmt-keeper.service`, or one instance per GPU with the template
  `contrib/systemd/llmt-keeper@.service` (`llmt-keeper@gpu0`, socket `/run/llmt-keeper-gpu0/keeper.sock`;
  `Type=notify`, `FileDescriptorStoreMax`, `FileDescriptorStorePreserve=yes`). Without the fd store, stopping the
  keeper frees the memory once no engine maps it. With it, stopping the unit does not free it:
  `systemctl clean --what=fdstore` after the stop does.
- Engines: install `llmt_kv` into the engine's environment (the vLLM plugin entry point installs the loader) and set
  `EXL3_WHX=/run/llmt-keeper-gpu0/keeper.sock` (plus `EXL3_WHX_EXCLUSIVE=1`). The XPU backend needs `whx.so`
  (`EXL3_WHX_LIB`, built from `weights/xpu/whx_ops.sycl`); the CUDA backend needs nothing extra.
- `contrib/bin/llmt-keeper-ctl <instance> list | held-mib | drop-all`.

## Tests

- `weights/tests/test_whx_loader.py`: CPU only (torch, no GPU, no vLLM). memfds stand in for device memory and a fake
  keeper speaks the wire protocol: key contents, walk paths, export + attach round trip with bit-exact contents,
  checksum and tensor-set mismatches, references to an earlier model's storage, no leaked fds.
- `crates/keeper/tests/keeper_compat.py`, `fdstore_sim.py`: the keeper's wire format and its fd-store restore (the
  simulator plays systemd: FDSTORE notifications, LISTEN_FDS on restart, SIGKILL in between).
- `weights/tests/gpu_compat.py`: the same client phases with real device memory through the loader's backend;
  `COMPAT="<container command> gpu_compat.py" fdstore_sim.py ...` runs the restore path on a GPU without systemd.
- `contrib/systemd/keeper-fdstore-test.sh`: the real systemd path on a GPU (transient unit: restart, SIGKILL,
  stop + start with `FileDescriptorStorePreserve=yes`, drop frees the memory). It starts a unit, so it refuses to run
  without `KEEPER_FDSTORE_TEST_OK=1`.

## Weight loading without mmap

`llmt_kv.nommap_loader` replaces vLLM's memory-mapped safetensors loading. Call `llmt_kv.nommap_loader.patch()` in the
engine process before the model loads (for example from a vLLM plugin); `EXL3_NOMMAP_LOAD=0` turns it off. It parses
the safetensors header itself and reads one tensor at a time with large sequential `preadv` calls into a private
buffer, then the engine copies it to the GPU.

What it is for: **reliable loading on slow disks or under memory pressure**, not a speed-up on a fast machine.

- No page-fault stalls. Reads are large and sequential, so a load does not depend on how page faults behave while a
  RAM KV tier and shared-memory pools crowd the page cache.
- Bounded memory: peak extra RAM is the largest single tensor (about 2.5 GB for a 27B model). vLLM's non-mmap
  alternative (`--safetensors-load-strategy eager`) reads a whole shard into memory, about twice the shard's size.
- No reliance on the checkpoint already being in the page cache.

What we saw, on a host whose root disk is a USB stick shared with swap: a memory-mapped load stalled at about 1 MB/s
through page faults; with this loader, plus a smaller block-device readahead (512 KB instead of 2048 KB) and
`vm.swappiness=1`, loads ran at 110–230 MB/s (16.6 GiB in about two minutes). Those three changes went in together,
so the loader's own share is not measured. On a fast NVMe drive with spare RAM, mmap loading is already fast and the
gain is probably small. The weight keeper is what removes load time on restarts.

`EXL3_NOMMAP_DROP=1` also drops each range from the page cache after reading it (every start is then a cold read).
