//! # Qwen3.5
//!
//! The dense Qwen3.5 language models (their text models), as `transformers`'
//! `Qwen3_5TextModel` defines them: a hybrid
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! whose layers are mostly linear attention, a
//! [`GatedDeltaNet`](crate::blocks::transformers::mixers::GatedDeltaNet), and
//! every few full attention, with QK-norm, a sigmoid gate on its output and
//! a rotary embedding over a quarter of each head; a `SwiGLU` in every
//! layer, and `1 + w` RMS norms. A [`Qwen3_5ContractConfig`] builds it as an
//! [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel).
//!
//! - [`Qwen3_5ContractConfig`]: the policy the model,
//!   [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel),
//!   is built from; its prefabs are [`pretrained`]'s.
//! - Decoding: the model, with its linear-attention layers, is a
//!   [`StackedLm`](crate::kits::support::decoding::StackedLm), its decode state
//!   its stack's
//!   [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache):
//!   a KV cache for the full-attention layers, and a
//!   [`DeltaNetState`](crate::ops::transformers::linear_attention::DeltaNetState)
//!   for each linear one. A prefill runs the delta rule a chunk at a time, a
//!   decode step one position.
//! - [`Qwen3_5Construct`]: a model with the positions it was trained on and its
//!   checkpoint's tokenizer and template files, unread.
//! - [`pretrained`] (features `store` and `cache`): the named checkpoints,
//!   `qwen/qwen3.5-0.8b` the default, and the factory that loads one.
//! - [`interop`]: its checkpoints' ecosystem, [`interop::transformers`]: a
//!   `transformers` Qwen3.5's `config.json` and state-dict names (a multimodal
//!   checkpoint's text model, its vision tower and multi-token-prediction head
//!   dropped).
//!
//! A tokenizer and a chat template are not part of the kit; the searches are
//! [`kits::support::decoding`](crate::kits::support::decoding)'s.

mod contract;

pub mod interop;

pub mod pretrained;
pub use contract::*;
#[cfg(feature = "store")]
pub use pretrained::{
    CHAT_TEMPLATE,
    GENERATION_CONFIG,
    Qwen3_5Construct,
    TOKENIZER,
    TOKENIZER_CONFIG,
};

#[cfg(all(test, feature = "store"))]
mod toy_tests;
