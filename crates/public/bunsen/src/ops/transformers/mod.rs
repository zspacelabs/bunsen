//! # Transformer operations
//!
//! The operations inside transformer layers, and the state they carry:
//!
//! - [`attention`]: scaled dot-product attention, attention over burn's
//!   [`MultiHeadAttention`] weights, and the key/value caches that decoding
//!   carries between steps;
//! - [`linear_attention`]: the gated `DeltaNet`'s convolution and delta rule,
//!   and the per-layer state a hybrid model's decode carries;
//! - [`attn_res`]: attention residuals' routing, a learned mixture over a bank
//!   of earlier outputs in place of a residual stream.
//!
//! burn's counterpart is the [`MultiHeadAttention`] module; these are the
//! pieces below and around a module like it. The transformer layers
//! themselves are blocks: see [`crate::blocks::transformers`].
//!
//! [`MultiHeadAttention`]: burn::nn::attention::MultiHeadAttention

pub mod attention;
pub mod attn_res;
pub mod linear_attention;
