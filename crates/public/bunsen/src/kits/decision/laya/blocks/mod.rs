//! # Laya blocks.
//!
//! The model, [`LayaModel`] with its [`LayaConfig`]: a `ModernBERT` encoder
//! and the decision head, its [`LayaScorer`] and [`LayaActHead`]. The head's
//! layers are the generic
//! [`ResidualEncoderAttentionBlock`](crate::blocks::transformers::encoders::residual::ResidualEncoderAttentionBlock).
//! The [`laya`](super) module docs cover the model.

pub(crate) mod model;

pub use model::*;
