"""Client for llmt-kvbroker, the shared host-RAM KV tier (protocol: docs/protocol.md).

    import llmt_kv
    c = llmt_kv.Client("/run/llmt-kv/broker.sock", engine_id=1, name="rtx5090", timeout_ms=2)
    base = c.pool_address()                       # register base..base+c.pool_bytes with the GPU runtime once
    n_hit, lease, entries = c.lookup(ns, keys, acquire=True)   # entries[i].byte_ranges -> copy pool -> GPU
    c.release([lease])
    [(st, byte_ranges)] = c.reserve(ns, [{"key": k, "parent": p, "nbytes": n, "kind": 2}])
    # copy GPU -> pool at byte_ranges, then
    c.commit(ns, [k])

Any BrokerError means: skip the shared tier for this step and prefill normally. The client reconnects by itself
(at most every reconnect_ms) after the broker restarts or drops the connection:
  BrokerGone         not connected right now
  BrokerReset        this call reconnected; reservations and leases (after a broker restart: the whole index) are
                     gone, so forget in-flight transfer bookkeeping
  BrokerPoolChanged  the broker serves a different pool file: stop using the tier until the engine restarts

The pure-Python parts import without the native client (a build without the Rust extension, or a test): the vLLM
plugin (llmt_kv.plugin), its hooks (llmt_kv.vllm_hooks), the lend manager (llmt_kv.lend), the lazy vision tower
(llmt_kv.lazyvis), the weight keeper loader (llmt_kv.whx_loader) and the no-mmap loader (llmt_kv.nommap_loader).
Every broker user treats a missing client like a broker that is down.
"""
try:
    from ._llmt_kv import *  # noqa: F401,F403
    from ._llmt_kv import ChainsClient  # noqa: F401  (lookup chains)
    from ._llmt_kv import Client, Entry, BrokerError, BrokerTimeout, BrokerGone, BrokerReset, BrokerPoolChanged, status_name, PROTOCOL_VERSION  # noqa: F401
    NATIVE_ERROR = None
except ImportError as _e:  # pure-Python use only
    NATIVE_ERROR = _e


def native_available():
    """True when the Rust broker client (llmt_kv._llmt_kv) is importable."""
    return NATIVE_ERROR is None
