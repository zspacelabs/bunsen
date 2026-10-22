//! # `NanoChat` blocks.
//!
//! The model, [`NanoChatGpt`] with its configs and [`NanoChatGptMeta`]. Its
//! repeated layer is the generic
//! [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock),
//! with RMS norms and a `ReLU`² `Mlp`; it and the rotary embedding come from
//! [`crate::blocks::transformers`]. The
//! [`nanochat`](super) module docs cover the lifecycle.

pub(crate) mod model;

pub use model::*;
