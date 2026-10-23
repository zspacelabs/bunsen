//! # Qwen3 blocks.
//!
//! The model, [`Qwen3Model`] with its [`Qwen3Config`] and [`Qwen3Meta`]. Its
//! repeated layer is the generic
//! [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock),
//! with RMS norms, per-head query/key RMS norms and a `SwiGLU`
//! [`GatedMlp`](crate::blocks::transformers::mlp::GatedMlp); it and the rotary
//! embedding come from [`crate::blocks::transformers`]. The
//! [`qwen3`](super) module docs cover the model.

pub(crate) mod model;

pub use model::*;
