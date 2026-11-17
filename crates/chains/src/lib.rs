//! Lookup chains for speculative decoding: engines stream each request's tokens; when
//! the newest tokens repeat an earlier span, the broker pushes the continuation as a hint (extra draft tokens; the
//! target verifies them, so outputs stay exact). Runs inside llmt-kvbroker on its own socket and threads, with no state
//! shared with the KV index (`--chains off` by default). Phase 1: the per-request index; the global token-tree store
//! (permanent + hot tiers, idle GC, disk snapshot) comes next.
pub mod client;
pub mod engine;
pub mod index;
pub mod proto;
pub mod server;
pub mod store;
