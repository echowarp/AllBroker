# N-gram (PLE) table store: design

Useful in every layout: Flash-Next alone on a Strix Halo, Flash-Next split across an Arc B70 + Strix Halo, or a 27B
on the Arc with Flash-Next on the Strix Halo. All of them read the same 16 rows per token at layer 1.

## Facts

| | Production GGUF (UD-Q4_K_XL) | EXL3 4.05 (`ngram_embedding.safetensors`) |
|---|---|---|
| Rows | 320,001,536 x 160 (16 hashed heads: 8 bigram + 8 trigram, ~20 M rows each) | same |
| Row format | IQ4_NL: 5 x (f16 scale + 16 B nibbles) = **90 B** | K6 trellis, `I16 [rows, 61]` = **122 B**, + `head_bias [16 x 160]` f16 |
| Table size | 26.8 GiB | 36.4 GiB |
| Self-contained rows | yes | yes |

Hash (gufo `HashNgramRows`, ported in `src/hash.rs`): 64-bit multiply/xor of the token and up to two predecessors,
modulo each head's prime vocab, plus the head offset. **Rust port is bit-identical** (`parity/`: gufo's function
compiled from its source, 200 random sequences with EOS, 0 mismatches).

## Measurements (990 PRO, uniform random rows = no locality, no cache; Flash-Next running alongside)

| Gather | io_uring p50 / p99 | threaded pread p50 / p99 |
|---|---|---|
| 1 token (16 rows) | 137 / 217 us | 278 / 470 us (baseline respawns threads per gather: flatters io_uring) |
| 8 tokens (128 rows) | 394 / 559 us | 733 / 1,009 us |
| 2,048-token prefill chunk (32,768 rows) | 83 ms | 87 ms (both at the drive's ~385 k random reads/s) |

Live gufo (trace): its 8 MB direct-mapped cache misses ~80% of unique rows.

## Disk behaviour (990 PRO, random aligned reads inside the table; `ngram-bench sizes`)

| Read | QD1 p50 / p99 | 16 at once p50 | QD64 |
|---|---|---|---|
| 4 KiB | 68 / 78 us | 133 us | 249 k IOPS, 1.0 GB/s |
| 8 KiB | 71 / 86 us | 203 us | 128 k IOPS, 1.0 GB/s |
| **16 KiB** | **78 / 90 us** | 235 us | 108 k IOPS, **1.8 GB/s** |
| 32 KiB | 90 / 106 us | 302 us | 82 k IOPS, 2.7 GB/s |
| 64 KiB | 107 / 139 us | 422 us | 55 k IOPS, 3.6 GB/s |

A 16 KiB read costs ~15% more time than 4 KiB for 4x the data (the NAND page). **Disk unit: 16 KiB bins, 16 KiB
aligned.** (The drive reports 512 B logical/physical sectors, so this was measured, not read from sysfs.)

## Layout: n-gram bundles in clustered 16 KiB bins

A token's 16 rows come from two keys: its bigram (8 heads) and its trigram (8 heads). Rows are a pure function of the
n-gram, so precomputed copies are exact.

1. **Bundles:** the 8 rows of one frequent bigram or trigram, concatenated (720 B IQ4_NL, 976 B EXL3).
2. **Pairing:** a bigram bundle (b,c) sits next to its frequent trigram bundles (a,b,c): one read per token.
3. **Chains:** each bin holds ~22 (IQ4_NL) / ~16 (EXL3) bundles along a frequent token sequence (corpus successor
   statistics, greedy chain building as in Pettis-Hansen code layout). One read serves the next several tokens.
4. **Replication:** hot bundles may appear in several chains (disk is cheap); the directory points each n-gram at its
   best bin.
5. **Tiers:** hottest bins in one contiguous region, read sequentially into RAM at start; warm bins on demand; rare
   n-grams fall back to row reads from the original table (4 KiB, unclustered: the hash scatters rows).
6. **Directory in RAM:** n-gram key (u64) -> bin, a compact set-associative table (tags of a bucket in one 64 B line).
   B-tree-like in spirit: index levels in RAM, leaves (bins) on disk, exactly one read per miss.
7. **Cache:** reading a bin inserts all of its bundles; the RAM cache is set-associative with per-bucket CLOCK /
   TinyLFU, keyed per n-gram.

Metric for comparing layouts (replay tool): **reads per token and exposed latency per token** (decode, verify,
prefill), per RAM budget, not plain hit rate.

## Where it lives: inside the RAM-tier broker (kvbroker)

The table service is part of the RAM-tier broker. Reasons:
- **Warm across engine restarts.** Engines restart often (new builds, model switches); an in-engine cache starts cold
  every time. The broker outlives engines.
- **One memory budget.** KV/GDN state and n-gram rows share one pool; the broker can trade one for the other.
- **Shared** by every engine that uses the same table (several gufo instances, the future split engine).
- **One place for stats** (the broker already exports Prometheus text).

### Implemented (kvbroker `crates/broker/src/ngram.rs`)

Differences from the first sketch below: the row cache is broker-private memory (not a pool region) and the staging
windows live in their own sparse shared file `<pool-dir>/ngram-<epoch>`, so the KV pool, its dynamic sizing and its
pinning are untouched. Behaviour (dynamic):
- Tables (`--ngram-table NAME=GGUF`) load lazily on the first `NGRAM_OPEN`; the cache grows in ~6 MiB segments as
  rows are inserted, up to `--ngram-cache-gib`.
- Idle unload after `--ngram-idle-s` (default 600) without users: cache and reader dropped, memory returned.
- Memory pressure: below the broker's `--mem-reserve-gib`, the n-gram cap shrinks by the deficit + 256 MiB at once
  (before the KV pool culls: an n-gram miss is a disk read, a KV miss a re-prefill); regrows 512 MiB per
  `--pool-regrow-s` of calm with 3 GiB spare.
- Fixed pin locations: one staging window per connection at a fixed offset, never moved or
  punched while owned, punched on disconnect. An engine can register its window with the GPU once.
- Engines send token runs (2-token history + tokens); the broker hashes rows (bit-identical to gufo) so bundles can
  later be served without engine changes.
- Verified on a test broker: 102,400 rows compared with direct reads, 0 mismatches; warm 256-row gather p50 12 us,
  cold 697 us; lazy load, idle unload (35 -> 15 MB RSS) and pressure shrink (cap 1 GiB -> 0, gathers still exact).

### Bins in the broker (`src/bins.rs`, `ngram-binfile`, broker `--ngram-bins NAME=PATH`)

- **Bins file** (`.ngbins`, written by `ngram-binfile <layout> <model.gguf> <out>`): 16 KiB header, then the
  layout's 16 KiB bins (hottest first), then its 4 KiB bins, then the directory (sorted `(u64 key, u32 bin)`).
  A bin: `u32 count | u32 0 | count x u64 key | count x bundle`; a bundle is the key's 8 rows in head order, copied
  from the table. The header carries the table identity (rows, row bytes, hash multipliers); the broker refuses a
  file built for another table. `ngram-binfile` re-reads a sample of bundles and compares them with the table.
- **Broker read path:** gathers hash token runs into (key, 8 rows) groups. Cache misses are split: n-grams in the
  directory read their bin (O_DIRECT, own io_uring), every bundle of that bin goes into the row cache; other n-grams
  read their missing rows from the table as before, submitted while the bins are in flight. The cache stays
  row-granular and engines are unchanged. Directory: binary search in RAM (12 B per n-gram; ~240 MB at 20 M).
- **Counters:** `row_reads_total`, `bin_reads_total{size}`, `bin_rows_total` (rows cached from bins), next to
  `disk_rows_total` (requested rows that missed). Grafana: "Disk reads per second", "Rows per disk read".
- **First check** (test layout from an 8 M-token sample: 600 k n-grams, 0.49 GB; 3,000 one-token gathers each on
  held-out code and article text, cold broker): 0 mismatches in 192 k rows; disk I/Os 67.9 k -> 41.8 k (-38%); cold
  gather p50 code 102 -> 22 us, article 132 -> 93 us. The real profile (~1.5-2 B tokens) comes next.

### Pool layout (first sketch)

The broker's pool file (`/dev/shm/llmt-kv/pool-<epoch>`) gets a second region after the KV slots:

```
[ KV slots (2 MiB each) ............ ][ n-gram row arena: fixed row slots (96 B IQ4_NL / 128 B EXL3) ][ staging ]
```

- **Row arena:** row slots of `row_bytes` rounded up to 32 B (no straddling a cache line pair). Index row -> slot is
  broker-private (open addressing, u32 keys).
- **Pinned hot set:** the top rows of an offline profile (`ngram-profile`: tokenize local code/docs with the model's
  tokenizer, count rows; never prompts), loaded at broker start with sequential reads, never evicted.
- **Dynamic part:** W-TinyLFU (count-min sketch admission + segmented LRU). Prompts are mostly one-off n-grams;
  frequency-gated admission keeps them from flushing hot rows (plain LRU and the page cache can't).
- **Staging:** one area per client connection; the broker copies requested rows there (2 KB per decode token,
  up to 4 MB per prefill chunk), so rows need no leases and the arena can evict freely.

### Protocol additions (to protocol v1)

| Request | Reply | Notes |
|---|---|---|
| `NGRAM_OPEN {table identity}` | arena + staging offsets, row bytes, codec | identity = file path + size + mtime + tensor offset |
| `NGRAM_GATHER {rows[]}` | done (rows in staging, request order) | broker serves hits from the arena, misses via io_uring, admits per policy |
| `NGRAM_PREFETCH {rows[]}` | none (one-way) | warm the arena: prefill chunk ahead, MTP drafts, prompt-lookup candidates, sampler top-k |

Engines keep hashing (they already have the history); the crate exposes the same hash for tools and tests.
I/O never happens under the index lock: a dedicated io_uring thread completes reads into the arena.

### Engine side (gufo)

`NgramTable::StartRead/WaitRead` call the broker client (C ABI from `crates/client`) when `GUFO_NGRAM_BROKER` is set
and the socket answers; otherwise gufo's own direct-I/O reader runs unchanged (the broker's failure model: no broker
= today's behavior). gufo still decodes rows on its side (IQ4_NL today), so the change in gufo is small. The container
needs the pool directory and socket bind-mounted.

### Split engine client + dev broker

**Engine** (`engine/ple_rows.h`, `plerows::Source`; CLI, `--serve` and `libsplitengine.so` alike):
- `PLE_BROKER=<socket>` (opt-in): every PLE row request goes to the broker as token runs (NGRAM_GATHER, one run per
  session / segment) and is sent before the embedding reads, so they overlap; the rows land in this connection's
  staging window and are copied out. Without it, or on any broker error (logged once; reconnect tried every 30 s),
  rows are pread from the table file as before. At connect, a fixed 6-token gather is compared byte for byte with
  pread (catches a broker serving another table of the same shape).
- PREFETCH: the next prefill step (up to two chunks) once the current chunk's rows are in; MTP verify rows as they
  become known (`next` before the first MTP step, each draft but the last after its step; `se_mtp_draft`,
  `se_mtp_draft_cand` likewise). `PLE_PREFETCH=0` turns it off.
- `PLE_TABLE` / `PLE_OFFSET` replace the hardcoded GGUF path. `PLE_FORMAT=auto` (default: bf16z if
  `rows.ngz(.complete)` exists, else bf16 if `rows.bin(.complete)`, else iq4) | `iq4` | `k6` | `bf16` | `bf16z` picks
  the table (below); k6 / bf16 / bf16z are decoded on the host into the fp16 embedding x 2^18 the PLE kernels start
  from (5120 B per token instead of 1440 B of IQ4_NL rows; the B70 kernels skip `ple::emb` and the projections scale
  their fp32 result by 2^-18). The scale keeps every value an fp16 normal (bf16 rows reach 6.5e-12; values below 2^-32
  are flushed to 0 on the host), so no subnormal reaches the device.
- A line `ple rows (exit): ...` at exit: gathers, host us/token, broker disk rows/token, prefetches, fallbacks.

**Tables** (same hashing and row ids, 320,001,536 rows x 160):

| PLE_FORMAT | file | row | vs bf16 (4,000 random rows): rel. rms err / cosine |
|---|---|---|---|
| iq4 | GGUF shard 2 @ 682,453,280 | 90 B IQ4_NL | 7.6% / 0.9971 |
| k6 | EXL3 `ngram_embedding.safetensors` (trellis @ 6,328, head_bias) | 122 B + bias | 1.7% / 0.99986 |
| bf16 | `<model dir>/rows.bin` (`tools/fetch_bf16_table.py`) | 320 B | 0 |
| bf16z | `<model dir>/rows.ngz` (NGZ1, below; 74.24 GB) | 232 B slot | 0 (lossless) |

The bf16 rows come from the original checkpoint (`Qwen/Qwen3.8-Flash-Next` @ de4b8e4d, 128 shards of 2,500,012 rows
in model-00005..00037; row r of rows.bin = global row r: checked against the K6 / IQ4 rows of the same ids).
The K6 host decoder is bit-identical to a transcription of exllamav3's `ngram_codec.py`.

**NGZ1** (`src/ngz.rs`, `ngram-ngz build|verify`, engine `Ngz` in ple_rows.h): bf16 rows losslessly in fixed 232 B
slots (72.5%): byte 0 the row's largest exponent E, 160 bytes sign << 7 | 7-bit mantissa, 71 bytes the exponent
offsets E - e Huffman coded (canonical, LSB-first; offsets >= 15 as ESC + 6 raw bits; 2.60 bits per value, entropy
2.56: one scale per row, Gaussian-like values). Rows that don't fit go raw into an escape section (12 of 320 M); their
slot has E = 0. Fixed slots keep random access, bins and the broker cache unchanged (rows are opaque to the broker).
Decode goes straight to the scaled fp16 (sign << 15 | (e - 94) << 10 | m << 3 for e >= 95, else 0): 483 ns per row,
7.7 us per token, bit-identical to the bf16 path. Full-table verify: 0 mismatches. Mantissas are incompressible
(6.97 of 7 bits); value delta / XOR coding loses (values uncorrelated). Per bundle (8 rows, 1,856 B): 8 per 16 KiB
bin (bf16: 6), 2 per 4 KiB page (bf16: 1); broker cache 261 B per row (bf16: 341).

**Broker** additions: `--ngram-rows NAME=FILE@OFFSET/ROW_BYTES` (a table whose rows come from another file / codec;
hashing and row count from its `--ngram-table` GGUF; bins built from the same rows with `ngram-binfile --rows`), `--ngram-resident NAME` (the whole table read into RAM
with O_DIRECT in the background when MemAvailable allows it + reserve + 4 GiB; gathers are then plain copies; dropped
first when MemAvailable falls below the reserve; metric `llmt_kv_ngram_resident_bytes`).

**Dev broker** (`ngram-store/dev_broker.sh`, user units `split-kvbroker-dev.{socket,service}`): a second llmt-kvbroker
with its own binary (`llmt-kvbroker-dev`), socket `$XDG_RUNTIME_DIR/llmt-kv-dev/broker.sock`, staging
`/dev/shm/llmt-kv-dev`, metrics 127.0.0.1:9113. **Socket-activated** (dev services do not run
always-on): only the `.socket` is enabled (a listener that costs nothing); the first connect starts the broker
(first connect + table load with the bins directory: ~0.7 s, once), and it exits after `IDLE_EXIT_S` (600) without
connections, removing its pool and staging files (`--idle-exit-s`, LISTEN_FDS support in the broker). Its RAM cache
survives engine restarts and test runs within a dev session (any gap under 10 min), not beyond. Tables: `qwen3.8-flash-next-bf16z` (NGZ1 + bins ngz-20m30m, 102 GB),
`qwen3.8-flash-next` (IQ4_NL + bins i-20m30m), `qwen3.8-flash-next-k6` (not resident by default: RAM goes to KV);
config in `split-kvbroker-dev.env` (cache budget `NGRAM_CACHE_GIB`, default 8: the budget curve plateaus
there;
capped at start to MemAvailable - max(0, `GPU_PEAK_GIB` 70 - Strix Halo GTT in use) - 4 GiB; at run time the
`--mem-reserve-gib 6` shrink). Agents:

```
ngram-store/dev_broker.sh status | stats | start | stop | restart     # install: rebuild + install binary and unit
PLE_BROKER=$XDG_RUNTIME_DIR/llmt-kv-dev/broker.sock ./b70_engine ...    # bf16z by default; PLE_FORMAT=iq4|k6 others
GUFO_NGRAM_BROKER=$XDG_RUNTIME_DIR/llmt-kv-dev/broker.sock GUFO_NGRAM_TABLE=qwen3.8-flash-next gufo ...   # IQ4 only
```
`restart` empties the cache (cold measurements); otherwise leave it running. Never point tests at the production
socket (`$XDG_RUNTIME_DIR/llmt-kv/broker.sock`).

The engine uses the dev broker **by default** when its socket exists (`PLE_BROKER` unset; `off` / `0`
forces pread); the socket is enabled, so the first connect starts the broker. Without a broker, pread goes through an in-process row
cache (`PLE_LOCAL_CACHE_MIB`, default 256) filled by a prefetch thread from the same PREFETCH hooks (prefill chunk
ahead, MTP drafts), so the reads leave the critical path.

## Steps

1. [done] crate: GGUF reader, exact hash (parity-tested), IQ4_NL codec, io_uring O_DIRECT reader, `ngram-bench`.
2. `ngram-profile` (corpus -> n-gram counts + successor statistics), `ngram-bins` (bundles, chains, replication, hot
   region, directory), `ngram-replay` (corpus token streams -> reads/token and exposed latency per layout and RAM
   budget: raw rows vs bundles vs chained bins), and the set-associative cache library.
3. Broker integration: pool region, the three requests, io_uring thread, Prometheus counters
   (`kvbroker_ngram_{hits,misses,disk_reads,prefetch,latency}`), client C ABI.
4. gufo: `GUFO_NGRAM_BROKER`, plus prefetch hooks (prefill chunk-ahead, MTP drafts, top-k). Deploy on Flash-Next,
   compare in Grafana (decode p50 step time, misses on the critical path).
5. EXL3 trellis codec (122 B rows + head bias) for the split engine.
