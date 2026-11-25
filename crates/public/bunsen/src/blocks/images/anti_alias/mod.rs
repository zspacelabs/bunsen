//! # Anti-aliased downsampling
//!
//! A strided conv aliases: it samples before it low-passes. An anti-aliased
//! net moves the stride off the conv onto a pool that smooths first
//! ("Making Convolutional Networks Shift-Invariant Again",
//! [arXiv:1904.11486](https://arxiv.org/abs/1904.11486)). The pools are the
//! kinds of the [`AntiAlias`] seam:
//!
//! - [`BlurPool2d`]: a fixed binomial blur, then the stride (timm's
//!   `BlurPool2d`): the input reflect-padded, each channel convolved with the
//!   outer product of `C(n-1, k) / 2^(n-1)`.
//! - an average pool of the stride (timm's `nn.AvgPool2d` as `aa_layer`).
//!
//! Neither has parameters: a blur's filter is computed in its forward, in the
//! input's dtype, as timm's non-persistent buffer is never saved.

mod anti_alias_pool;
mod blur_pool;

pub use anti_alias_pool::*;
pub use blur_pool::*;
