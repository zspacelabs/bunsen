//! # Qwen3.5 blocks.
//!
//! The model, [`Qwen3_5Model`] with its [`Qwen3_5Config`] and [`Qwen3_5Meta`].
//! Its layers are the generic
//! [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)
//! in a hybrid
//! [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//! of two kinds, linear and full ([`Qwen3_5LayerKind`]). The
//! [`qwen3_5`](super) module docs cover the model.

pub(crate) mod model;

pub use model::*;
