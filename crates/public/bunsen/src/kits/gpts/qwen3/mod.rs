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
//! [`KVCache`](crate::ops::transformers::attention::KVCache).
//!
//! It is the model of the dense Qwen3 checkpoints up to 4B (`Qwen3-0.6B`,
//! `Qwen3-1.7B`, `Qwen3-4B`, their `-Base` builds, and `Qwen3-4B`'s `2507`
//! builds), whose heads are tied, and the text model of
//! [`kits::speech::qwen3_asr`](crate::kits::speech::qwen3_asr).
//!
//! - [`blocks`]: the model, built from a [`Qwen3Config`].
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `Qwen3ForCausalLM` state dict, and reading its `config.json`.
//! - [`pretrained`] (features `store_safetensors` and `cache`): the named
//!   checkpoints, `qwen/qwen3-0.6b` the default, and the factory that loads
//!   one.
//!
//! A tokenizer, a chat template and a sampler are not part of the kit: what
//! a caller decodes the model with is its own.

pub mod blocks;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(feature = "store_safetensors", feature = "cache"))]
pub mod pretrained;

pub use blocks::model::*;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;
