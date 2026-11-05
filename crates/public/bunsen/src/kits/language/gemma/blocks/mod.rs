//! # Gemma blocks.
//!
//! The model, [`GemmaModel`] with its [`GemmaConfig`] and [`GemmaMeta`]. Its
//! layers are the generic
//! [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)
//! in a [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! of two kinds, sliding and full ([`GemmaLayerKind`]). The [`gemma`](super)
//! module docs cover the model.

pub(crate) mod model;

pub use model::*;
