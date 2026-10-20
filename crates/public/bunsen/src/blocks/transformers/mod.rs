//! # Transformer blocks
//!
//! The parts of transformer models, as modules:
//!
//! - [`attention`]:
//!   [`CausalSelfAttention`](attention::csa::CausalSelfAttention), bunsen's
//!   causal self-attention layer (QK-norm, rotary embeddings, grouped-query
//!   attention, an optional KV cache), and the encoders' bidirectional
//!   attentions, relative-position and rotary.
//! - [`embedding`]: [`RotaryEmbedding`](embedding::RotaryEmbedding), the rotary
//!   position embedding tables as a module, and two fixture embeddings for
//!   tests, [`iota_embedding`](embedding::iota_embedding) and
//!   [`identity_embedding`](embedding::identity_embedding).
//! - [`encoder_block`]: the pre-norm encoder layer of Whisper's and Qwen3-ASR's
//!   audio encoders,
//!   [`ResidualEncoderAttentionBlock`](encoder_block::ResidualEncoderAttentionBlock).
//! - [`rope_encoder`]: a pre-norm encoder with rotary attention,
//!   [`RopeTransformerEncoder`](rope_encoder::RopeTransformerEncoder), over
//!   [`RopeEncoderBlock`](rope_encoder::RopeEncoderBlock)s:
//!   `Nemotron-3-Diarization`'s.
//! - [`mlp`]: [`Mlp`](mlp::Mlp), the position-wise feed-forward block (expand,
//!   activate, project back), [`layer_norm_mlp`](mlp::layer_norm_mlp), which
//!   runs it after a `LayerNorm`, and [`GatedMlp`](mlp::GatedMlp), the `SwiGLU`
//!   feed-forward.
//!
//! The operations these layers call, and the key/value cache state, are in
//! [`crate::ops::transformers`].

pub mod attention;
pub mod embedding;
pub mod encoder_block;
pub mod mlp;
pub mod rope_encoder;
