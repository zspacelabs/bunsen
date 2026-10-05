//! # Transformer operations
//!
//! The operations inside transformer layers, and the state they carry, in
//! [`attention`]: scaled dot-product attention, attention over burn's
//! [`MultiHeadAttention`] weights, and the key/value caches that decoding
//! carries between steps.
//!
//! burn's counterpart is the [`MultiHeadAttention`] module; these are the
//! pieces below and around a module like it. The transformer layers
//! themselves are blocks: see [`crate::blocks::transformers`].
//!
//! [`MultiHeadAttention`]: burn::nn::attention::MultiHeadAttention

pub mod attention;
