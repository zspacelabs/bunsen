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
//! - [`attn_res`]: a decoder stack's routes in place of its residual stream,
//!   [`AttnResRoutes`](attn_res::AttnResRoutes): each sublayer reads a learned
//!   mixture of the stack's input and every earlier sublayer's output (Full
//!   Attention Residuals), a [`DecoderStack`](stack::DecoderStack) with routes;
//!   ZONOS3's.
//! - [`depth`]: a depth decoder, [`DepthDecoder`](depth::DepthDecoder): a small
//!   causal decoder over one frame's codebooks after the first, from the
//!   backbone's rows, each codebook its own table and head; Qwen3-TTS's code
//!   predictor (CSM's, Kyutai DSM's and Dia2's have its shape).
//! - [`projected`]: a decoder stack between a projection in and an optional
//!   projection out,
//!   [`ProjectedDecoderStack`](projected::ProjectedDecoderStack), over a whole
//!   sequence: a codec's transformer trunk; ZONOS3's codec stages and
//!   Qwen3-TTS's codec transformer.
//! - [`cross`]: the encoder-decoder layer,
//!   [`CrossDecoderBlock`](cross::CrossDecoderBlock): pre-normalized causal
//!   self-attention, cross-attention over the encoder output, and an MLP, each
//!   residual; Whisper's.

pub mod attn_res;
pub mod causal;
pub mod cross;
pub mod depth;
pub mod projected;
pub mod stack;
