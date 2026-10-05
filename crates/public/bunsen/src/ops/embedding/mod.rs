//! # Embedding operations
//!
//! Operations on embedding tables, as functions:
//!
//! - [`unembed`] is the tied-embedding output head. It converts a model's final
//!   `[batch, seq_len, d_model]` hidden state into `[batch, seq_len, n_vocab]`
//!   logits by multiplying with the transpose of an embedding weight, so a
//!   model can reuse its input embedding as its output projection (Whisper's
//!   text decoder does). [`EmbeddingArg`] lets it take a burn [`Embedding`], or
//!   that embedding's bare `[n_vocab, d_model]` weight.
//! - [`inverse_frequency_table`] and [`positional_frequency_table`] build the
//!   rotary position embedding (`RoPE`) angle tables.
//!
//! burn's [`Embedding`] module is the lookup; these are the operations around
//! it. The embedding *modules* bunsen adds (`RotaryEmbedding`, which caches
//! the `RoPE` tables, and the `iota_embedding` / `identity_embedding` test
//! fixtures) are blocks, in [`crate::blocks::transformers::embedding`].
//!
//! [`Embedding`]: burn::nn::Embedding

mod inverse;
mod rotary_tables;

pub use inverse::*;
pub use rotary_tables::*;
