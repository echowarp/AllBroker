# AllBroker protocol (version 1)

The contract between the broker and its clients. Wire encodings are pinned by tests in `crates/proto`.

## Architecture

- **Data plane:** one shared-memory file `/dev/shm/llmt-kv/pool-<epoch>` per slot class, divided into fixed-size
  **slots**. Its size is the RAM ceiling; with `--pool-lazy on` only slots that hold entries take memory, and the
  ceiling can change at runtime (`POOL_STATE` / `POOL_ACK` tell pinning clients). The broker creates it (and removes older `pool-*` files
  in that directory, which only it creates); engines `mmap` it (directory bind-mounted into their containers), register
  it once (`cudaHostRegister` / `zexDriverImportExternalPointer`) and copy blocks in and out with their own GPU DMA.
  **KV bytes never pass through the broker.**
- **Control plane:** the broker process. It owns the index (key → slot ranges + state), allocation, leases, eviction
  and stats, and serves a Unix stream socket `/run/llmt-kv/broker.sock` (directory bind-mounted into the engines).
- **Epoch:** a random u64 per broker start, in the pool file name and the HELLO reply. A broker restart drops every
  connection; the client reconnects, sees a new epoch and treats the tier as empty (its old mapping points at an
  unlinked file, so late DMA from before the restart cannot corrupt live slots).
- **Failure model:** if the broker is down or slow, engines get errors or timeouts and run **without** the shared tier
  (normal prefill). Client default timeout 2 ms per request; the broker never blocks on I/O inside the index lock.


## Keys and entries

- `key`: 32 bytes, opaque to the broker. The client computes it as a hash over (namespace, weights id, canonical
  format version, token ids up to a canonical boundary, parent key).
  **Compat tag:** every key also hashes slot size (block size × KV row format), KV
  dtype, sha256 of the K-smoothing table, the GDN-recompute flag and the entry format id, so engines with mismatched
  settings cannot hit each other even in a shared namespace. Token-id hashing
  is vLLM's sha256 block hash; every engine sharing a namespace must run with the same `PYTHONHASHSEED`, or no prefix
  ever matches across engines or restarts.
- `namespace`: u32. Private (namespace = engine) or shared (namespace = model+weights). Keys are per namespace.
- An **entry** = one key → slot ranges plus: `bytes` (u64), `kind` (u8: 0 = attention KV, 1 = GDN/Mamba state,
  2 = KV + state at one boundary, 3 = other, 4 = partial tail: attention pages past the last full block + GDN
  state at the resume point, written when an engine's VRAM tail hold ends; evicted before any other entry by
  RESERVE, `--tail-age-div` (4) times faster in every age / heat order, no TTL, SSD only after one resume like kind 1;
  5 = model weights, e.g. a multimodal model's vision tower (mmproj), one entry of many slots in a namespace shared
  by every engine of the model: an ordinary RAM entry for eviction (no pinning, no TTL, low water and pressure
  apply), valued per slot against KV: its idle time counts `--weights-kv-slot-ms` (1300, one KV slot's recompute
  ms) × slots / `cost_hint` (the engine's reload-from-disk ms) times in every age / heat order, and RESERVE /
  pressure take it before the policy's KV victim once it is colder than that victim (`cost_hint` 0 = plain recency);
  never persisted: not in the hot set, never written to the SSD tier, a file of one is deleted and never restored,
  never compressed in place), `producer` (u16 engine id), `format` (u16), `parent` (key or none),
  `cost_hint` (u32, client's recompute ms; for kind 5 the reload ms), `meta` (≤ 1024 opaque bytes: e.g. segment offsets/sizes inside the entry),
  `created`/`last_used`, hit count, flags (bit 0 = ephemeral).
- Slots per entry = ceil(bytes / slot_bytes), allocated as few contiguous ranges as possible (in a fragmented pool
  a large entry gets several ranges; a client copies each one).
- Entries are immutable once committed. A key is never rewritten; a new version needs a new format id in the key.
- **Chains:** an entry with `parent` set requires the parent to exist (reserved or committed) at RESERVE, else
  `NO_PARENT`. Only **leaves** (no reserved or committed children) are evictable, so chains shrink from the tail and
  shared roots stay. Aborting or timing out a reservation also aborts its reserved descendants.
- **Ephemeral entries** (flag bit 0, `ttl_ms`): per-request tails (e.g. KV + state at the exact prompt end for prefill
  offload). Evicted first once their TTL passes or their first lease is released, whichever comes first.


## Entry states and race rules

`reserved(writer) → committed → (leased by N readers) → evicted`

1. **Two-phase store:** `RESERVE` → ranges, or `EXISTS` (already committed), `BUSY` (another writer holds it; the
   second writer must not write), `FULL` (nothing evictable frees enough), `NO_PARENT`, `TOO_BIG`. The writer copies
   into the ranges, then `COMMIT`. Only the reserving connection may commit or abort (`NOT_OWNER`). Readers never see
   reserved entries. Reservations end on `ABORT`, on the writer's socket closing, or after a timeout (default 300 s: it only matters for a hung writer, since a closed socket already frees them; it must exceed the longest first prefill).
2. **Leases:** `LOOKUP` with `acquire = 1` leases the longest committed prefix of the key list and returns its ranges;
   leased entries are never evicted. `RELEASE` ends a lease; so do the socket closing and a timeout (default 60 s).
   `LOOKUP` with `acquire = 0` returns the hit length only (for vLLM's repeated scheduler checks); no ranges, no lease.
3. **Eviction** takes committed, unleased leaves only, by policy order. It runs in `RESERVE` when free slots are short
   and in the background to keep a low-water mark of free slots (default 1%).
4. **WAIT:** blocks (asynchronously; other requests on the connection continue) until a key is committed, or
   returns `TIMEOUT`. Used by an engine that waits for another engine's store (prefill offload).
5. All state changes happen under one index lock; requests are small. Socket I/O happens outside the lock.


## Wire protocol

Unix stream socket. Frame = `u32 len | u16 op | u16 flags | u64 req_id | payload`, little-endian; `len` counts the
bytes after itself (12 + payload). Max frame 4 MiB. Responses echo `op` and `req_id` and set flag bit 0 (`RESP`);
a response payload starts with `u16 status`. Pipelining allowed; responses may come out of order (`WAIT`).
Strings and blobs: `u16 len | bytes`. Keys: 32 raw bytes. Ranges: `u32 n | n × (u32 start_slot, u32 n_slots)`.

| op | Request | Response (after `status`) |
|---|---|---|
| 1 `HELLO` | `engine_id u16, protocol_version u16, name str` | `protocol_version u16, broker_version str, epoch u64, pool_path str, pool_bytes u64, slot_bytes u64, n_slots u32` |
| 2 `LOOKUP` | `namespace u32, acquire u8, n u32, keys[n]` | `n_hit u32, lease_id u64` (0 if none) `, n_hit × entry` (only if acquire) |
| 3 `RESERVE` | `namespace u32, n u32, n × item` | `n × (status u16, ranges)` |
| 4 `COMMIT` | `namespace u32, n u32, keys[n]` | `n × status u16` |
| 5 `ABORT` | `namespace u32, n u32, keys[n]` | `n × status u16` |
| 6 `RELEASE` | `n u32, lease_ids[n] u64` | `n × status u16` |
| 7 `TOUCH` | `namespace u32, n u32, keys[n]` | `n × status u16` |
| 8 `STATS` | — | `n u32, n × (name str, value u64)` |
| 9 `WAIT` | `namespace u32, key, timeout_ms u32` | — (status `OK` or `TIMEOUT`) |
| 10-17 | `POOL_STATE`, `POOL_ACK`, `NGRAM_OPEN`, `NGRAM_GATHER`, `NGRAM_PREFETCH`, `DOMAIN`, `PROBE`, `HEAT`: see `crates/proto` | |
| 18 `FEATURES` | `features u64, n u32, n × (format str, version u16)` (what the client reads) | same shape: the broker's features and the formats it writes |
| 19 `TOUCH_REQUEST` | `namespace u32, n u32, n × (kind u8, n_keys u32, keys)`: one request's KV groups in order, kind 0 attention / 1 GDN state / other unknown | — |
| 29 `TOUCH_TAGGED` | `namespace u32, session u64, parent u64, class u8, agent u8, flags u8, n u32, n × (kind u8, n_keys u32, keys)`: `TOUCH_REQUEST` plus the request's tags (FEAT_REQUEST_TAGS): opaque session / parent ids (0 = none), class 0 unknown / 1 low / 2 normal / 3 agent / 4 lowest, agent letter, flags bit 0 = the session's terminal request | — |

`TOUCH` and `TOUCH_REQUEST` are also the persistent hot set's use signal (`--hotset-rank use`): every key of a batch,
in RAM or not, counts one use per request chain (re-touches of the same request continue its chain and do not count),
decayed over `--hotset-use-half-life-s`. Lookup hits count only for connections that never touch. Send the whole
chain from the root in each attention batch, VRAM-served blocks included.

**Value classes** (FEAT_REQUEST_TAGS): RESERVE flags bit 2 `ENTRY_LOW`, bit 3 `ENTRY_LOWEST` mark a block of a
low-value request (background or one-shot work). Such entries are admitted and aged like any other while there is room;
under room pressure (RESERVE, memory pressure) the coldest lowest-class, then low-class unit goes before the policy's
victim unless it is more than `--low-evict-margin-s` hotter, so a reused low-class prefix still outlives a cold normal
entry. The hot set ranks normal units first, then low, then lowest. A touch raises an entry to the toucher's class (plain
`TOUCH` / `TOUCH_REQUEST` = normal; a RESERVE of an existing key with a higher class too). `TOUCH_TAGGED` names the
broker session by its tag (no chain-end inference, linked from the first request); its terminal flag drops the touched
entries only that session used (not shared, < 2 chains' use) to lowest. An older broker ignores the flag bits and
answers `TOUCH_TAGGED`'s absence in FEATURES: the client sends `TOUCH_REQUEST` instead.
| 30 `HINT` | `namespace u32, anchor (16 bytes), chunk_tokens u32, flags u8, n u32, n × (kind u8, n_keys u32, keys)`: a router anchor covers these keys of this engine (`FEAT_PREFIX_HINTS`; flags bit 0 = in VRAM now) | — |
| 31 `WHERE` | `n u32, n × anchor (16 bytes)` (longest prefix first) | `n u32, n × place` |

- `item` = `key, has_parent u8, parent (32 bytes if has_parent), bytes u64, kind u8, format u16, cost_hint u32,
  flags u8, ttl_ms u32, meta blob`.
- `entry` = `bytes u64, kind u8, format u16, producer u16, flags u8, meta blob, ranges`.
- Status: 0 `OK`, 1 `EXISTS`, 2 `BUSY`, 3 `FULL`, 4 `NOT_FOUND`, 5 `TIMEOUT`, 6 `BAD_REQUEST`, 7 `NOT_OWNER`,
  8 `NO_PARENT`, 9 `TOO_BIG`, 10 `NO_HELLO`, 11 `VERSION`.
- A connection must `HELLO` first (`NO_HELLO` otherwise). `producer` = the connection's `engine_id`.
- `place` = `anchor_idx u32, engine_id u16, name str, namespace u32, age_ms u64, chunk_tokens u32, n_chunks u32,
  n_resident u32, flags u8`: per (engine, pool class, namespace) the hinted anchor with the most leading chunks
  committed in RAM now. HINT / WHERE: prefix-affinity hints for routers, `docs/prefix-hints.md`. They live in a bounded
  side table (`--hint-mib`, default 32, 0 = off), never in the entry index; WHERE is read-only (no lease, no hit, no
  heat, no SSD promotion). A router HELLOs with engine id `OBSERVER_ENGINE_ID` (0xbb01): not a client (no SSD warm-up,
  not traffic, no hold on the idle exit).


### Background worker

The broker's CPU-heavy idle work (integrity checksums, in-place slot compression) runs in a separate process,
`llmt-kvbroker --bg-worker SOCKET` (run it in its own low-weight unit: CPUWeight=1, nice 19, idle I/O, bound to the broker's unit), so the broker (CPUWeight=1000) only answers requests and frees memory. The worker connects to each
pool class, declares `FEAT_BG_WORKER` in `FEATURES` (its requests do not count as traffic for the quiet-time rules)
and maps the pools like an engine.

| op | Request | Response |
|---|---|---|
| 21 `BG_JOBS` | `kind u8` (1 checksum, 2 compress), `max u32` | `cksum_bytes_s u64, chunk u64, n u32, classes[n] str, m u32, m × (namespace u32, key, bytes u64, guard)` |
| 22 `BG_CHECKSUM` | `namespace u32, key, guard, has u8, sum u64` | `result u8` (0 stored, 1 verified, 2 corrupt: evicted, 3 raced, 4 busy) |
| 23 `BG_REWRITE` | `phase u8` (1 begin, 2 end, 3 skip), `namespace u32, key, guard, len u64` | `result u8` (0 ok, 3 refused) |
| 24 `BG_SSD_JOBS` | `max u32` | `n u32, n × (namespace u32, key, guard, kind u8, format u16, producer u16, hits u64, bytes u64, last_used_ms u64, meta blob, path str, codec u8)` |
| 26 `BG_COMPACT_JOBS` | `max u32` (one move at a time) | `n u32, n × (namespace u32, key, guard, new ranges)` |
| 27 `BG_COMPACT_DONE` | `namespace u32, key, copied u8` | `result u8` (0 moved, 3 not moved) |
| 25 `BG_SSD_DONE` | `namespace u32, key, outcome u8` (0 written, 1 raced, 2 corrupt, 3 failed), `file_bytes u64` | `result u8` (0 recorded, 3 not in flight) |

SSD tier writes : the broker plans them (admission, ancestors first, daily budget) into a queue; the worker
copies the slots, checks the copy against the entry's RAM checksum (BG_CHECKSUM: a copy that fails is never written),
compresses and writes the file, and reports; only then the SSD index lists it. A worker that disconnects with writes
in flight: their temporary files are removed and the next flush plans them again. Promotions (SSD to RAM) stay in the
broker (an engine waits for them), below its request threads (nice 10).

Compaction : the sweep gives the worker a byte budget (only when quiet, without memory pressure, not idle);
the broker picks the move and allocates its new slots under the core lock, backs them outside it, and the worker
copies outside it while the entry stays readable at its old slots (the copy used to run under the core lock, several
ms per slot). The broker moves the entry only if it is unchanged and unleased. Slots a departed worker may still be
writing (a torn rewrite's slot, a compaction move's targets) go to quarantine (60 s), never straight back to use.

`guard` = `entry_id u64, ranges, compressed u8, rewriting u8`: the entry's identity when the job was handed out; a
result or a rewrite only counts while the entry is still exactly that one. Jobs are only handed out while the broker
is quiet, and a rewrite only begins then. Between begin and end the entry is unreadable; a worker that disconnects
in between leaves a possibly torn slot, which the broker evicts (with the rest of its chain). Without a worker the
work pauses; `--bg internal` runs it inside the broker (rollback).


### Versioning

Old and new components must never misread each other; a mismatch degrades to "feature not used" or "entry dropped",
never to wrong bytes.

- **Wire.** `PROTOCOL_VERSION` (HELLO) changes only for an incompatible change; HELLO refuses a mismatch (`VERSION`).
  Every compatible addition is a new op or a new feature bit; an existing op's encoding never changes (decoders refuse
  trailing bytes, and clients in other languages speak these bytes). `crates/proto` pins the exact
  bytes of every op (`tests::wire_format_pinned`, `WIRE_PINNED`: lines are only ever added).
- **Negotiation.** New clients send `FEATURES` right after every HELLO (a reconnect may reach another build): their
  feature bits and the formats they read. A broker without it answers `BAD_REQUEST` (an unknown op; the connection
  goes on): no features, and the client uses nothing new (`touch_request` falls back to plain `TOUCH`). A client that
  never sends `FEATURES` is a legacy client: it gets nothing new and reads slot codec 1 only. Feature bits are only
  added, never reused: bit 0 `TOUCH_REQUEST`, bit 1 `TRACE_DUMP`, bit 2 `BG_WORKER`, bit 3 `INVALIDATE`, bit 4
  `REQUEST_TAGS`, bit 5 `PREFIX_HINTS`.
- **Formats a client reads** (`FEATURES` format list): `slot_codec` = compressed RAM slot images (`LKVZ<v>`). The
  broker writes a version only while every client of the pool class declared it (`Shared::class_reads`; metric
  `ram_compress_no_reader`), and readers refuse an `LKVZ` image of a version they do not know instead of copying it as
  raw KV (`kvslotz::unknown_format`).
- **Files the broker alone reads** carry a magic + version (or an algorithm byte) and are read by explicit version:
  SSD entry files (`LKVD1`, data checksum algorithm in header byte 74: 0 old mix, 1 XXH3-64), hot-set files (`LKVH1`
  old mix, `LKVH2` XXH3-64, `LKVH3` = `LKVH2` + the entry's use and chain relation, so a restored prefix keeps its
  rank and its GDN state anchor; an older broker drops `LKVH3` files as damaged). A file of an unknown version is dropped (a cold cache), never interpreted. Compatibility
  with the files of the previous build is tested against that binary (`failpoints.rs` `old_files_still_read`,
  `OLD_KVBROKER=`); the same goes for any new file format (snapshots, traces).

