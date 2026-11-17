# KV lending (`llmt_kv.lend`)

VRAM is KV cache by default: vLLM sizes the KV pool as if nothing else needed memory. Other users of VRAM borrow KV
blocks while they need them and return them when they are done. KV always wins: a borrower that cannot get room waits
(LoRA requests queue; base requests keep running) or runs a slower path (the vision tower streams), it never pushes
an engine into an out-of-memory error.

Status: lab-tested on an Intel Arc Pro B70 and an RTX 5090 (vLLM 0.26, Qwen3.8-27B EXL3); not in production yet.
The CPU tests (`crates/py/tests/lend/`) cover the policy; GPU validation items are listed at the end.

## Borrowers

| Borrower | Switch | What it borrows | When it gives back |
|---|---|---|---|
| LoRA adapters | `EXL3_LORA_HOT=1` (+ vLLM `--enable-lora --max-loras N --max-lora-rank R`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`) | slot buffers for active adapters | `EXL3_LORA_TTL` s (300) without a LoRA step, or 1 s under memory pressure |
| Vision tower | `EXL3_LAZY_VISION=1`, `EXL3_VIS_LEND=1` | the tower's weights, placed in a lent block run | the lazy tower's TTL |

- **LoRA by name.** With `EXL3_LORA_DIR` (default `/loras`), a request whose model is `<name>` and
  `<dir>/<name>/adapter_config.json` exists (PEFT LoRA, rank within `--max-lora-rank`) loads that adapter on first
  use. The adapter file's bytes are a broker weights entry (kind 5) shared by every engine with the same file.
- **Compiled graphs stay the base model.** Stock vLLM traces the model once with no adapter active, so LoRA terms are
  traced out of the compiled graph. Base traffic keeps those graphs; steps with LoRA tokens run eagerly or from a
  separate LoRA graph pool (`EXL3_LORA_GRAPHS`).
- **LoRA-step MLP.** A step with LoRA tokens runs the MLP unfused (plain path through the LoRA layers), whose
  intermediate the profiled activation reserve does not cover for a large prefill chunk. With more than
  `EXL3_LORA_MLP_ROWS` (2048) rows, rows without an adapter take the fused base MLP and the adapter rows go through
  the plain + LoRA path in chunks of that many rows (the plain intermediate stays bounded; the result is exact).

## Policy

- **Chains.** Prefix-cache blocks form request chains; each chain has a value from its request class (`agent`,
  `normal`, `low`, `lowest`, from the router's tags), its reuse and whether its session is still live.
- **Region scoring and compaction.** Before a lend, `score_region(pool, lo, n)` prices clearing a block range (free
  blocks cost nothing, cached blocks cost their chain's value). The cheapest run is cleared by moving blocks out of it
  (copy + identity swap, block tables remapped).
- **Victim eviction.** Free blocks first, then cached blocks of the lowest-value chains. Blocks in use by a running
  request are moved only with `EXL3_LEND_MOVE_INUSE=1` (off by default until a GPU check shows moved blocks are
  bit-identical).
- **Move guard.** With async scheduling a step that is already queued may still read a block; pending KV-RING copies
  to or from the RAM tier too. Such blocks are never moved or lent until that step / copy is done.
- **Regret counters.** A chain block dropped by a lend that a lookup asks for again soon after counts as a regret
  (`LEND-STATS` log line); the number tells whether lending costs prefix hits.
- **Admission hold (extension point).** `llmt_kv.lend.set_admission_policy(fn)`, where `fn(scheduler, request)`
  returning `False` keeps a waiting request waiting this step (needs echowarp/vllm's `scheduler.admit` hook). Planned
  use: hold or preempt low-priority work when the KV pool is full. No policy ships yet.

## Settings

All settings are environment variables, so one plugin build serves every GPU.

Knobs and their defaults: `EXL3_LEND_STATS_S` 30, `EXL3_LORA_MLP_ROWS` 2048, `EXL3_LORA_MARGIN_MB` 256, `EXL3_LORA_MODE`
auto|device|host, `EXL3_LORA_BROKER` auto|off|<socket>, `EXL3_LORA_NS` weights:lora, `EXL3_LORA_LEND` 1,
`EXL3_LEND_COMPACT` 0, `EXL3_LEND_CHAIN_TAU_S` 300, `EXL3_VIS_LEND` 1. The full list is at the top of each section of
`llmt_kv/lend.py`.

## Open items that need a GPU

- Forced in-use block moves bit-identical on both GPUs (then `EXL3_LEND_MOVE_INUSE` can default on).
- The hook-based port (this package) against the lab results of the internal version: same KV token counts, same
  claim / regret counters, no OOM under the mixed-batch replay, on both GPUs, with stock vLLM (fallback wrappers) and
  with echowarp/vllm (native hooks).
