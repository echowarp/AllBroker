//! Store for the Qwen3.8-Flash-Next n-gram (PLE) embedding table: 320,001,536 rows x 160, read at layer 1,
//! 16 rows per token. No mmap: rows come from disk with O_DIRECT + io_uring into a row-granular RAM cache.
//! Design: DESIGN.md.

pub mod bins;
pub mod codec;
pub mod gguf;
pub mod hash;
pub mod layout;
pub mod ngz;
pub mod reader;

pub const TABLE_TENSOR: &str = "per_layer_token_embd.weight";
