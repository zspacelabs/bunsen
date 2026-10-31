//! # Qwen3
//!
//! The Qwen3 decoder-only language model, as `transformers`' `Qwen3Model`
//! defines it: grouped-query causal self-attention with per-head query/key
//! RMS normalization before a rotary embedding, a `SwiGLU` feed-forward, and a
//! head tied to the embedding or a linear map of its own
//! (`tie_word_embeddings`). [`Qwen3Model`] is built from a
//! [`Qwen3Config`] out of the shared
//! [`CausalSelfAttention`](crate::blocks::transformers::attention::csa::CausalSelfAttention),
//! [`GatedMlp`](crate::blocks::transformers::mlp::GatedMlp) and
//! [`RotaryEmbedding`](crate::blocks::transformers::embedding::RotaryEmbedding),
//! and decodes through a
//! [`KVCache`](crate::ops::transformers::attention::KVCache).
//!
//! It is the model of the dense Qwen3 checkpoints up to 8B: `Qwen3-0.6B`,
//! `Qwen3-1.7B`, `Qwen3-4B`, their `-Base` builds and `Qwen3-4B`'s `2507`
//! builds, whose heads are tied, and `Qwen3-8B` and its `-Base`, whose head is
//! its own. It is also the model of any dense Qwen3 an `hf:` repo holds, and
//! the text model of
//! [`kits::speech::qwen3_asr`](crate::kits::speech::qwen3_asr).
//!
//! - [`blocks`]: the model, built from a [`Qwen3Config`].
//! - [`Qwen3DecodeState`]: a decode's cache and rotary table; the model is a
//!   [`CachedStep`](crate::kits::decoding::CachedStep), so
//!   [`decode_cached`](crate::kits::decoding::decode_cached) decodes it under
//!   any search.
//! - [`Qwen3Bundle`]: a model with the positions it was trained on and its
//!   checkpoint's tokenizer and generation files, unread.
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `Qwen3ForCausalLM` state dict, and reading its `config.json`.
//! - [`pretrained`] (features `store_safetensors` and `cache`): the named
//!   checkpoints, `qwen/qwen3-0.6b` the default, and the factory that loads
//!   one.
//!
//! A tokenizer and a chat template are not part of the kit; the searches are
//! [`kits::decoding`](crate::kits::decoding)'s.

pub mod blocks;

mod bundle;
mod decode;
pub use bundle::*;
pub use decode::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(feature = "store_safetensors", feature = "cache"))]
pub mod pretrained;

pub use blocks::model::*;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;
