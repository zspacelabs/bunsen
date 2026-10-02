//! # Normalization operations
//!
//! [`rms_norm`] is root-mean-square normalization with no trainable
//! parameters: `x / sqrt(mean(x²) + eps)` over the last axis, with the mean
//! taken in `F32` and the result in the input's dtype. [`RmsNormOptions`]
//! carries `eps` (default `1e-5`) as a value object, and
//! [`RmsNormOptions::norm`] applies it.
//!
//! burn's [`RmsNorm`] module computes the same normalization and then scales
//! by a learnable `gamma`. Use `RmsNorm` when the scale should train; use
//! `rms_norm` for a parameter-free norm (`PyTorch`'s `F.rms_norm` with no
//! weight).
//!
//! [`RmsNorm`]: burn::nn::RmsNorm

mod rms_norm_impl;
pub use rms_norm_impl::*;
