//! # Decoder blocks
//!
//! Causal transformer decoder layers, each taking its key/value cache:
//!
//! - [`causal`]: the decoder-only layer,
//!   [`CausalDecoderBlock`](causal::CausalDecoderBlock): pre-normalized causal
//!   self-attention and a pre-normalized feed-forward, each residual;
//!   nanochat's and Qwen3's, by config.

pub mod causal;
