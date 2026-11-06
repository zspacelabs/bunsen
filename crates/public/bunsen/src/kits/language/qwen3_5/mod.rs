//! # Qwen3.5
//!
//! The dense Qwen3.5 language models (their text models), as `transformers`'
//! `Qwen3_5TextModel` defines them: a hybrid
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! whose layers are mostly linear attention, a
//! [`GatedDeltaNet`](crate::blocks::transformers::mixers::GatedDeltaNet), and
//! every few full attention, with QK-norm, a sigmoid gate on its output and
//! a rotary embedding over a quarter of each head; a `SwiGLU` in every
//! layer, and `1 + w` RMS norms. One [`Qwen3_5Model`] is built from a
//! [`Qwen3_5Config`].
//!
//! - [`blocks`]: the model, built from a [`Qwen3_5Config`].
//! - Decoding: the model is a
//!   [`StackedLm`](crate::kits::support::decoding::StackedLm), its decode state
//!   its stack's
//!   [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache):
//!   a KV cache for the full-attention layers, and a
//!   [`DeltaNetState`](crate::ops::transformers::linear_attention::DeltaNetState)
//!   for each linear one. A prefill runs the delta rule a chunk at a time, a
//!   decode step one position.
//! - [`Qwen3_5Construct`]: a model with the positions it was trained on and its
//!   checkpoint's tokenizer and template files, unread.
//! - [`pretrained`] (features `store_safetensors` and `cache`): the named
//!   checkpoints, `qwen/qwen3.5-0.8b` the default, and the factory that loads
//!   one.
//! - [`weights`] (feature `store`): loading tensors named as a `transformers`
//!   Qwen3.5 state dict (a multimodal checkpoint's text model, its vision tower
//!   and multi-token-prediction head dropped), and reading its `config.json`.
//!
//! A tokenizer and a chat template are not part of the kit; the searches are
//! [`kits::support::decoding`](crate::kits::support::decoding)'s.

pub mod blocks;

mod construct;
mod decode;
pub use construct::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(feature = "store_safetensors", feature = "cache"))]
pub mod pretrained;

pub use blocks::model::*;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;
