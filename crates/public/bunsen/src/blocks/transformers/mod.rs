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
//! - [`conditioning`]: [`AdaptiveScale`](conditioning::AdaptiveScale), a
//!   per-channel scale learned from a conditioning vector (adaptive
//!   normalization), which a decoder layer can apply to its feed-forward's
//!   input.
//! - [`embedding`]: [`RotaryEmbedding`](embedding::RotaryEmbedding), the rotary
//!   position embedding tables as a module, and, in `embedding::testing`
//!   (behind the `testing` feature), two fixture embeddings for tests.
//! - [`encoders`]: bidirectional encoder layers and encoders:
//!   [`ResidualEncoderAttentionBlock`](encoders::residual::ResidualEncoderAttentionBlock),
//!   the pre- or post-norm layer of Whisper's, Qwen3-ASR's and Sortformer v2's
//!   encoders, and its stack,
//!   [`ResidualEncoderTransformer`](encoders::residual::ResidualEncoderTransformer);
//!   a bidirectional [`DecoderStack`](decoders::stack::DecoderStack), a
//!   pre-norm encoder with rotary attention, `Nemotron-3-Diarization`'s; and
//!   [`ConformerEncoderConfig`](encoders::conformer::ConformerEncoderConfig),
//!   the `FastConformer`, Parakeet's and Sortformer v2's.
//!   [`EncoderTransformer`](encoders::EncoderTransformer) is any of their
//!   stacks, chosen by config.
//! - [`decoders`]: causal decoder layers:
//!   [`CausalDecoderBlock`](decoders::causal::CausalDecoderBlock), the
//!   decoder-only layer, nanochat's and Qwen3's by config; and
//!   [`CrossDecoderBlock`](decoders::cross::CrossDecoderBlock), the
//!   encoder-decoder layer with cross-attention, Whisper's.
//! - [`head`]: [`LmHead`](head::LmHead), a language model's head: tied to the
//!   token table or its own, with an optional slice, soft-cap and logits dtype.
//! - [`mlp`]: [`TransformerMlp`](mlp::TransformerMlp), the position-wise
//!   feed-forward block (expand, activate, project back),
//!   [`norm_mlp`](mlp::norm_mlp), which runs it after a `LayerNorm`, and
//!   [`TransformerGatedMlp`](mlp::TransformerGatedMlp), the gated feed-forward
//!   (`SwiGLU`, `GeGLU`).
//!
//! The operations these layers call, and the key/value cache state, are in
//! [`crate::ops::transformers`].

pub mod attention;
pub mod conditioning;
pub mod decoders;
pub mod embedding;
pub mod encoders;
pub mod head;
pub mod mlp;
