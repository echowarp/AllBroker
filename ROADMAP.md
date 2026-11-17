# Roadmap

The broker's job is to decide what lives in VRAM, host RAM and on the SSD at any moment, across every engine on a
host, by the value of keeping it there. KV entries are the first kind of data it manages; these come next.

## mmproj (vision encoder weights)

The vision tower of a multimodal model is needed only for requests with images, but it takes VRAM that KV could use.
The broker manages it as an entry with a cost (load time) and a heat (how often images arrive), next to KV.

Done (broker side): entry kind 5 (weights, [protocol](docs/protocol.md#keys-and-entries)).

- Kept in host RAM only while it is worth more than the KV it displaces there, under the same pressure rules as KV:
  the same age / heat / policy orders, low water and memory pressure, no pinning, no TTL. It is valued per slot: its
  idle time counts `--weights-kv-slot-ms` × slots / `cost_hint` times (`cost_hint` = the engine's reload-from-disk
  ms; one ~16-slot tower reloaded in ~1 s ages ~20 times faster than KV, so a burst of image requests keeps it and
  it leaves soon after the images stop), and RESERVE takes it before the policy's KV victim once it is colder.
- Shared by every engine of the same model on the host: one entry in a namespace named from the model and the tower
  layout; engines of either GPU hit it (two leases at once are fine), a second store of the same bytes is `EXISTS`.
- Never persisted: not in the hot set, never on the SSD tier, a file of one is never restored (the model file is
  on disk already; the engine reloads from it on a miss and stores the entry again).

Engine side (done, lab-tested): `llmt_kv.lazyvis`. An image request looks the entry up, copies it to VRAM and
releases it; on a miss it reloads from the model file and stores it. With KV lending the tower is placed into lent
KV blocks (`docs/lend.md`). Next: GPU validation of the plugin build on both GPUs.

## LoRA adapters

Engine side (lab-tested): `llmt_kv.lend`. Adapters load by name on first use, their bytes are a shared broker
weights entry, and their slot buffers live in lent KV blocks only while an adapter is active (`docs/lend.md`).
Next: GPU validation of the hook-based port; in-use block moves on by default once they are shown bit-identical;
admission hold / priority preemption for a full KV pool (extension point `set_admission_policy`).

## Weight keeper

- Done: per-GPU unit template (`contrib/systemd/llmt-keeper@.service`, one keeper per GPU with
  `EXL3_WHX_EXCLUSIVE=1`), read-only attach for test engines, `llmt-keeper-ctl`.
- Next: model-loader hooks in vLLM so the loader needs no method wrapping; more GPUs than the two tested.

## KV tier

- Prefix-affinity hints for routers (`HINT` / `WHERE`): send a request to the GPU that already holds its prefix.
  Implemented (ops, side table, connector HINT, kvbench `hints`; design and trace estimate: `docs/prefix-hints.md`);
  open: VRAM-eviction hints from engines, residency per compat tag for cross-GPU sharing.
- Measured admission tuning on more production traffic (what is worth storing matters more than the eviction order).
- Hot set by use (done, broker side): ranked by request chains that touch an entry, not RAM hits; saved as loadable
  units. Next: the engine stores again a touched block the broker reports missing (TOUCH status `NOT_FOUND`), so a
  prompt that lives only in VRAM after a broker restart reaches RAM and the hot set; and a GDN state at the branch
  point of a shared prompt when no request's prompt ended there (`hotset_stateless_entries`).

## Packaging

- Wheels for `llmt_kv` on PyPI; release binaries for the broker and the keeper.
