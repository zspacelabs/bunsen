//! # Decoder blocks
//!
//! Causal transformer decoder layers, each taking its key/value cache:
//!
//! - [`causal`]: the decoder-only layer,
//!   [`CausalDecoderBlock`](causal::CausalDecoderBlock): pre-normalized causal
//!   self-attention and a pre-normalized feed-forward, each residual;
//!   nanochat's and Qwen3's, by config.
//! - [`stack`]: a stack of causal layers and a final norm,
//!   [`DecoderStack`](stack::DecoderStack), with the rotary embedding they
//!   share and their KV cache's geometry: Qwen3's decoder, by config.
//! - [`cross`]: the encoder-decoder layer,
//!   [`CrossDecoderBlock`](cross::CrossDecoderBlock): pre-normalized causal
//!   self-attention, cross-attention over the encoder output, and an MLP, each
//!   residual; Whisper's.

pub mod causal;
pub mod cross;
pub mod stack;
