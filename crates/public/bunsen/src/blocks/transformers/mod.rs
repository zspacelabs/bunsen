//! # Transformer blocks
//!
//! The parts of transformer models, and the layers and stacks built from
//! them. A *part* is one sublayer ([`attention`], [`embedding`], [`mlp`]); a
//! *layer* wraps parts in normalization and residuals; a *stack* repeats a
//! layer. A layer or stack lives here when it is generic (what tells its
//! users apart is config: norm placement, which attention, which MLP) and
//! more than one kit builds it; a named model, with its own weights and
//! quirks, stays in its kit and is assembled from these.
//!
//! The modules:
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
//! - [`encoders`]: bidirectional encoder layers and encoders:
//!   [`ResidualEncoderAttentionBlock`](encoders::residual::ResidualEncoderAttentionBlock),
//!   the pre- or post-norm layer of Whisper's, Qwen3-ASR's and Sortformer v2's
//!   encoders, and its stack,
//!   [`ResidualEncoderTransformer`](encoders::residual::ResidualEncoderTransformer);
//!   [`RopeTransformerEncoder`](encoders::rope::RopeTransformerEncoder), a
//!   pre-norm encoder with rotary attention, `Nemotron-3-Diarization`'s; and
//!   [`ConformerEncoderConfig`](encoders::conformer::ConformerEncoderConfig),
//!   the `FastConformer`, Parakeet's and Sortformer v2's.
//!   [`EncoderTransformer`](encoders::EncoderTransformer) is any of their
//!   stacks, chosen by config.
//! - [`decoders`]: causal decoder layers:
//!   [`CausalDecoderBlock`](decoders::causal::CausalDecoderBlock), the
//!   decoder-only layer, nanochat's and Qwen3's by config.
//! - [`mlp`]: [`Mlp`](mlp::Mlp), the position-wise feed-forward block (expand,
//!   activate, project back), [`layer_norm_mlp`](mlp::layer_norm_mlp), which
//!   runs it after a `LayerNorm`, and [`GatedMlp`](mlp::GatedMlp), the `SwiGLU`
//!   feed-forward.
//!
//! The operations these layers call, and the key/value cache state, are in
//! [`crate::ops::transformers`].

pub mod attention;
pub mod decoders;
pub mod embedding;
pub mod encoders;
pub mod mlp;
