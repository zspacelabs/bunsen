//! # Gemma
//!
//! Google's Gemma decoder-only language models, generations 1, 2 and 3
//! (their text models), as `transformers`' `GemmaModel`, `Gemma2Model` and
//! `Gemma3TextModel` define them. One [`GemmaContractConfig`] covers the three,
//! each built as an [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel)
//! whose stack is `GeGLU` layers with `1 + w` RMS norms; from Gemma 2, a norm
//! on each sublayer's output too (the sandwich), sliding-window layers among
//! full ones, and a `query_pre_attn_scalar`; Gemma 2's attention and final
//! soft-caps; Gemma 3's QK-norm and its two rotary tables, a local one for
//! the sliding layers and a global one, linearly scaled in the larger
//! models. Every generation scales its embedding by `√d_model` and ties its
//! head to it.
//!
//! - [`GemmaContractConfig`]: the policy the model,
//!   [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel),
//!   is built from; its prefabs are [`pretrained`]'s.
//! - Decoding: the model wraps an attention-only
//!   [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel),
//!   and is a [`StackedLm`](crate::kits::support::decoding::StackedLm), its
//!   decode state its stack's
//!   [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache),
//!   whose sliding layers keep only their window.
//! - [`GemmaConstruct`]: a model with the positions it was trained on and its
//!   checkpoint's tokenizer, generation and template files, unread.
//! - [`pretrained`] (features `store` and `cache`): the named checkpoints, on
//!   the `unsloth/` mirrors, `gemma/gemma-3-1b-it` the default, and the factory
//!   that loads one.
//! - [`interop`]: its checkpoints' ecosystem, [`interop::transformers`]: a
//!   `transformers` Gemma's `config.json` and state-dict names (a causal LM's,
//!   or a multimodal Gemma 3's text model's, its vision tower dropped).
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
    END_OF_TURN,
    GENERATION_CONFIG,
    GemmaConstruct,
    TOKENIZER,
    TOKENIZER_CONFIG,
};

#[cfg(all(test, feature = "store"))]
mod toy_tests;
