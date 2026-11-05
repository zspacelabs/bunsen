//! # Embedding blocks
//!
//! - [`RotaryEmbedding`] is rotary position embedding (`RoPE`) as a module.
//!   Built by [`RotaryEmbeddingConfig`], it caches the `cos` and `sin` of
//!   [`positional_frequency_table`] as bare tensors (no parameters).
//!   [`apply`](RotaryEmbedding::apply) rotates a `[B, T, H, D]` query or key,
//!   and [`clip_range`](RotaryEmbedding::clip_range) narrows the tables to the
//!   positions of a decode step. [`CausalSelfAttention`] takes one in its
//!   `forward`; the `NanoChat` GPT holds one as a field of its module tree.
//! - [`RotarySet`] is a stack's rotary tables: one or more, and which one each
//!   layer applies (Gemma 3's local and global tables).
//! - [`TokenEmbedding`] looks tokens up in a table, then, as its
//!   [`TokenEmbeddingConfig`] says, normalizes and scales them: nanochat's
//!   normed embedding, Gemma's scaled one.
//! - `testing` (behind the `testing` feature) holds `iota_embedding` and
//!   `identity_embedding`, burn `Embedding`s with weights you can check by
//!   hand, as fixtures for tests.
//!
//! The table builders, and [`unembed`] for a tied output head, are operations
//! in [`crate::ops::embedding`].
//!
//! [`positional_frequency_table`]: crate::ops::embedding::positional_frequency_table
//! [`unembed`]: crate::ops::embedding::unembed
//! [`CausalSelfAttention`]: crate::blocks::transformers::attention::csa::CausalSelfAttention
//! [`Embedding`]: burn::nn::Embedding

mod rotary;
mod rotary_set;
mod token;

pub use rotary::*;
pub use rotary_set::*;
pub use token::*;

#[cfg(any(test, feature = "testing"))]
pub mod testing;
