//! # Normalization operations
//!
//! [`RmsNormOptions::norm`] is root-mean-square normalization with no
//! trainable parameters: `x / sqrt(mean(x²) + eps)` over the last axis, the
//! result in the input's dtype. [`RmsNormOptions`] is its value object: `eps`
//! (default `1e-5`), and the dtypes the input is divided in and its mean of
//! squares taken in (by default, the mean in f32 and the division in the
//! input's dtype; the norm seam's `Rms` kind, in `blocks::norm`, widens a half
//! input to f32 for both, as `transformers` does).
//!
//! burn's [`RmsNorm`] module computes the same normalization and then scales
//! by a learnable `gamma`. Use `RmsNorm` when the scale should train; use
//! `RmsNormOptions::norm` for a parameter-free norm (`PyTorch`'s `F.rms_norm`
//! with no weight).
//!
//! [`RmsNorm`]: burn::nn::RmsNorm

mod rms_norm_impl;
pub use rms_norm_impl::*;
