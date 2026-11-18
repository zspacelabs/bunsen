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
//! - [`attn_res`]: a decoder stack's layers without a residual stream,
//!   [`AttnResStack`](attn_res::AttnResStack): each sublayer reads a learned
//!   mixture of the stack's input and every earlier sublayer's output (Full
//!   Attention Residuals); ZONOS3's.
//! - [`cross`]: the encoder-decoder layer,
//!   [`CrossDecoderBlock`](cross::CrossDecoderBlock): pre-normalized causal
//!   self-attention, cross-attention over the encoder output, and an MLP, each
//!   residual; Whisper's.

pub mod attn_res;
pub mod causal;
pub mod cross;
pub mod stack;
