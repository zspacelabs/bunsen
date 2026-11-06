//! # Token mixers
//!
//! What a decoder layer mixes positions with: [`TokenMixer`], causal
//! self-attention ([`CausalSelfAttention`]) or a [`GatedDeltaNet`], linear
//! attention whose state is a summary of the prefix, not a history. A
//! hybrid stack (Qwen3.5's) interleaves the two by layer.
//!
//! [`CausalSelfAttention`]: crate::blocks::transformers::attention::csa::CausalSelfAttention

mod gated_delta_net;
mod token_mixer;

pub use gated_delta_net::*;
pub use token_mixer::*;
