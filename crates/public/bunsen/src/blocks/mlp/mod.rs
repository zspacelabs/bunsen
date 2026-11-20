//! # MLPs: position-wise two-layer networks, over any rank
//!
//! - [`BunsenMlp`]: `linear2(act(linear1(x)))`, built from a
//!   [`BunsenMlpConfig`]: its input, hidden and output widths, its activation,
//!   whether its projections carry a bias, and a dropout; over the last axis of
//!   a tensor of any rank. The general form of burn's
//!   `PositionWiseFeedForward`.
//! - [`BunsenGatedMlp`]: `down(act(gate(x)) * up(x))` (`SwiGLU`, `GeGLU`),
//!   built from a [`BunsenGatedMlpConfig`], over the last axis of any rank. The
//!   general form of burn's `SwiGlu`; its fields carry the names `transformers`
//!   gives them.
//!
//! A transformer's feed-forward sublayer is
//! [`transformers::mlp`](super::transformers::mlp)'s, built on these.

mod gated;
mod plain;

pub use gated::*;
pub use plain::*;
