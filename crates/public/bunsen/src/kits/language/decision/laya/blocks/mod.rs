//! # Laya blocks.
//!
//! The model, [`LayaModel`] with its [`LayaConfig`]: a `ModernBERT` encoder
//! and the decision head, its [`LayaScorer`] and its act head (a
//! [`BunsenMlp`](crate::blocks::mlp::BunsenMlp)). The head's layers are a
//! generic encoder
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack),
//! and the scorer's and act head's MLPs `BunsenMlp`s.
//! The [`laya`](super) module docs cover the model.

pub(crate) mod model;

pub use model::*;
