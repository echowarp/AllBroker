# Prefix-affinity hints for routers (HINT / WHERE)

Goal: a router in front of several engines (here an Arc B70 and an RTX 5090 serving the same model) sends a request to
the GPU that already holds its prefix: in VRAM best, in the broker's RAM tier next, else the least loaded. A 22-28k-token
agent prompt costs ~2.7-3.8 s cold prefill on the 5090 and ~0.73 s as a RAM-tier load; while one ~20k prefill runs, the
other agents decoding on that GPU drop to ~7 % of their rate for ~2.7 s.

## Why the router cannot compute the keys

Broker keys are `blake2b(namespace | key format | compat tag | vLLM offload key)` (see `llmt_kv/vllm_tier.py`), where
the offload key is vLLM's block hash chain plus the KV group index. A router sees only the OpenAI request JSON. To
compute the keys it would need the engine's chat template and tokenizer (exactly as the engine renders tools, reasoning
and history), vLLM's block-hash function (sha256 over a pickled/CBOR tuple of parent hash, token ids and extra keys
such as `cache_salt` and multimodal hashes, seeded from `PYTHONHASHSEED`), the offload chunk size, the per-GPU group
layout, and each GPU's compat tag (slot size, KV dtype, K-smoothing table sha256, layout, GDN state tag, model id).
That couples the router to engine internals and breaks on any engine upgrade. So the keys stay engine-side and the
router uses a **fingerprint of its own** (the anchor); engines map anchors to their keys.

## Design

- **Anchor** = 16 opaque bytes. The router already hashes each request at every message boundary (cumulative 16-byte
  blake2b; the root covers model, tools and chat-template options). The anchor of a request = its hash at the last
  message boundary (the full prompt). The router puts it into `kv_transfer_params.llmt_anchor` (hex).
- **HINT** (engine -> broker, op 30): when a request carrying `llmt_anchor` finishes, the engine's connector sends its
  keys per KV group in token order (the keys it looked up or stored for that request, deduplicated), with the group
  kinds, tokens per key and flags (bit 0 `HINT_VRAM`: the engine had just served it, so it is in its VRAM prefix cache
  now). One message per finished request, after the request: nothing on the request path.
- **Side table**: anchor -> at most 8 hints (one per engine + class + namespace; a new HINT from the same engine
  replaces its old one), bounded by `--hint-mib` (default 32, 0 = off: the feature is not offered); the oldest hint goes
  first. It is never part of the entry index: hints do not keep entries alive, count as hits, or change eviction.
- **WHERE** (router -> broker, op 31): the request's anchors, newest boundary first (the router sends up to 64). The
  next turn of an agent repeats the previous turn's messages, so one of its boundary hashes equals the previous turn's
  anchor. The broker answers per (engine, class, namespace) the hint with the most leading chunks committed in RAM
  **now** (ties: the longer anchor): `anchor_idx, engine_id, name, namespace, age_ms, chunk_tokens, n_chunks,
  n_resident, flags`. Residency follows the engine's resume rule: attention groups need every key up to the boundary,
  GDN/Mamba state groups (kind 1) need their key at the boundary (`Core::resident_prefix`). WHERE is read-only: no
  lease, no hit or lookup counter, no recency/heat, no SSD promotion; it checks at most 4 hints per engine (the longest
  anchors) under each class's core lock once.
- **Observer connections**: the router HELLOs with engine id `OBSERVER_ENGINE_ID` (0xbb01). Like the background worker
  it is not a client: its requests are not traffic for the quiet-time rules (compaction, checksums), it does not keep
  the broker from its idle exit, and its HELLO does not trigger the SSD warm-up.
- **Feature bit** `FEAT_PREFIX_HINTS` (bit 5) in the broker's FEATURES reply. Clients send HINT / WHERE only to a broker
  that offers it; an older broker answers FEATURES without it (or BAD_REQUEST), and the router and engines then simply
  never use the ops.

VRAM residency: engines do not report VRAM evictions (vLLM's prefix cache has no cheap per-request signal here), so
the router's own prefix map (last backend per prefix, `affinity_ttl_s`) stays the VRAM estimate; the broker's
`HINT_VRAM` flag and `age_ms` are reported for logging (`vram_ttl_s`). A later version could have engines re-HINT
on VRAM eviction (vLLM KV events) and let the router trust `HINT_VRAM`.

## Router policy (llmt-router `prefix_hints`, off by default)

Per request, before choosing: WHERE (5 ms timeout) and inject `llmt_anchor`. Then, per backend:

- cached tokens = the router's own affinity match (VRAM estimate) on the affinity backend; with a WHERE answer, the
  RAM-resident tokens the broker reports for that backend's engine replace the `shared_tier` guess (a fraction of the
  match credited to every other backend, which overstates reality while compat tags differ per GPU), and their load time
  (`kv_bytes_per_token / load_gbps`) is added;
- a conversation the router no longer knows (TTL, restart, weak match) whose prefix a backend holds in RAM (>=
  `min_tokens`, `affinity_min_tokens` and `affinity_min_frac` x prompt) is not "new": no `balance_new`; reason `hint`
  when that backend wins the cost comparison.

Order of precedence is unchanged: usable backends only (drain, hold, stall), `ctx` / `max_images`, the finish-time
cost (now with hint-based cached tokens), `prefer_until`, `summary_target`, `balance_new` (skipped for RAM holders),
`spill`. Caps keep their penalty. So: VRAM holder > RAM holder > least loaded, as far as the cost model agrees, and
never against caps, drains or holds.

## Failure modes

| Failure | Effect |
|---|---|
| broker down, socket missing | WHERE fails in < 5 ms, retried at most every 1 s; routing is exactly as without hints (`shared_tier` credit) |
| broker slow / hung | 5 ms timeout per request; after 8 timeouts in a row the connection is dropped and retried after 1 s |
| broker without `FEAT_PREFIX_HINTS` | hints off, asked again after 60 s; engines send no HINT |
| broker restart | side table empty (like the index); WHERE answers nothing until engines hint again |
| engine without the connector change | no HINT; WHERE never reports it; the router falls back to its own map and `shared_tier` |
| stale hint (entries evicted) | `n_resident` is computed live from the index, so an evicted prefix reports 0 |
| anchor collision / template re-render | WHERE is an upper bound; the engine's LOOKUP decides what is loaded; worst case = one mis-placed request paying a cold prefill, as today |
| side table full | oldest hints dropped (`llmt_kv_hints_dropped_total`) |

Costs: a HINT is one frame per finished request (~32 B x keys, e.g. 4 groups x 15 keys for a 25k prompt). WHERE with 64
anchors: ~0.2 ms round trip measured from the router's asyncio client against a debug broker build.

## Metrics

Broker: `llmt_kv_hints_total`, `llmt_kv_hints_refused_total`, `llmt_kv_hints_dropped_total`, `llmt_kv_hint_entries`,
`llmt_kv_hint_bytes`, `llmt_kv_where_requests_total`, `llmt_kv_where_anchors_total`, `llmt_kv_where_places_total`,
`llmt_kv_where_places_resident_total`, `llmt_kv_where_micros_total`; STATS `hint_entries`, `hint_bytes`, `hints`,
`hints_refused`, `hints_dropped`, `where_requests`, `where_places_resident`. Router:
`llmt_router_hint_where_total{result}`, `llmt_router_hint_where_seconds_total`, reason `hint`, decision-log
`hint=<backend>:<tokens>[v]`, trace field `hint`.

## Expected value (offline estimate from router traces)

25.3 days of production router traces (58,568 routed requests; aggregate counts only). The traces carry only the first
three prefix hashes of a conversation, no per-request residency and no in-flight counts at arrival, so this is an
estimate of the opportunity, not a replay:

- 22,255 requests have an earlier finished request of the same conversation (the likely holder); 21,325 (96 %) already
  went to the holder's GPU. 930 went to the other GPU: steal 540, spill 125, new 117, balance 116, ctx 14, prefer 10,
  affinity(-down) 8.
- With the production thresholds (holder prefix >= 32,768 tokens and >= half the prompt, holder finished <= 15 min
  earlier) prefix hints would move **12** new/balance requests (11 at 5 min, 19 at 60 min retention): ~160 s of cold
  prefill saved in total (~13 s each, mostly Arc prefills moved to the 5090). With a lower `min_tokens` floor of 8,192
  (and `affinity_min_tokens` lowered to match): 200 requests, ~1,155 s.
- 160 `steal` decisions went against a RAM holder (149 of them to the 5090 away from an Arc holder): ~1,500 s of
  prefill, but moving them back would cost ~2,200 s of decode at 1-stream rates (the Arc decodes ~4.6x slower), so the
  cost model would keep most of them where they went. Their real gain is the externality (the stall a cold prefill
  causes the agents already decoding there), which the traces cannot show.

The router's own prefix map already captures almost all of the benefit for same-GPU follow-ups. Prefix hints mainly
(1) replace the `shared_tier` assumption with what is actually resident (cross-GPU hits are rare while compat tags
differ per GPU), (2) keep affinity across router restarts and past `affinity_ttl_s` while the RAM tier still holds a
conversation, and (3) are the hook for cross-GPU routing once both GPUs share one compat tag (then one engine's stored keys are
loadable by the other, and WHERE would need to report residency per compat tag rather than per hinting engine: a
later extension).
