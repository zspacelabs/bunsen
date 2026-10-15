//! # Qwen3
//!
//! The Qwen3 decoder-only language model, as `transformers`' `Qwen3Model`
//! defines it: grouped-query causal self-attention with per-head query/key
//! RMS normalization before a rotary embedding, a `SwiGLU` feed-forward, and a
//! head tied to the embedding. [`Qwen3Model`] is built from a
//! [`Qwen3Config`] out of the shared
//! [`CausalSelfAttention`](crate::blocks::transformers::attention::csa::CausalSelfAttention),
//! [`GatedMlp`](crate::blocks::transformers::mlp::GatedMlp) and
//! [`RotaryEmbedding`](crate::blocks::transformers::embedding::RotaryEmbedding),
//! and decodes through a
//! [`KVCache`](crate::ops::transformers::attention::`KVCache`).
//!
//! It is the text model of
//! [`kits::speech::qwen3_asr`](crate::kits::speech::qwen3_asr).

mod model;

pub use model::*;
