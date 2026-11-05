//! # `ModernBERT` blocks.
//!
//! The model, [`ModernBertModel`] with its [`ModernBertConfig`] and
//! [`ModernBertMeta`]. Its layers are the generic
//! [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock),
//! with bidirectional attention, in a
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! of three kinds ([`ModernBertLayerKind`], and the first layer). The
//! [`modernbert`](super) module docs cover the model.

pub(crate) mod model;

pub use model::*;
