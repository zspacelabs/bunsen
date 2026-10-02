//! # Transformer blocks
//!
//! The parts of transformer models, as modules:
//!
//! - [`attention`]:
//!   [`CausalSelfAttention`](attention::csa::CausalSelfAttention), bunsen's
//!   causal self-attention layer (QK-norm, rotary embeddings, grouped-query
//!   attention, an optional KV cache).
//! - [`embedding`]: [`RotaryEmbedding`](embedding::RotaryEmbedding), the rotary
//!   position embedding tables as a module, and two fixture embeddings for
//!   tests, [`iota_embedding`](embedding::iota_embedding) and
//!   [`identity_embedding`](embedding::identity_embedding).
//! - [`mlp`]: [`Mlp`](mlp::Mlp), the position-wise feed-forward block (expand,
//!   activate, project back), and [`layer_norm_mlp`](mlp::layer_norm_mlp),
//!   which runs it after a `LayerNorm`.
//!
//! The operations these layers call, and the key/value cache state, are in
//! [`crate::ops::transformers`].

pub mod attention;
pub mod embedding;
pub mod mlp;
