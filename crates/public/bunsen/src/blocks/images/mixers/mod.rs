//! # Spatial mixers
//!
//! The spatial mixer of a conv net's residual unit is the conv that strides,
//! a bottleneck's `3x3`. These blocks take its place, each more than one conv:
//!
//! - [`Res2Conv`]: `Res2Net`'s scale split ([arXiv:1904.01169](https://arxiv.org/abs/1904.01169);
//!   timm's `Bottle2neck`'s `convs`). The channels split into `scale` chunks;
//!   each chunk but the last is convolved, after the first adding the previous
//!   branch's output, so the receptive field grows branch by branch.
//!
//! Each keeps its upstream's parameter names where the conv net's
//! conventions allow (`convs.{k}.conv`, `convs.{k}.norm`), and is a kind of
//! the conv net's
//! [`ResidualConvConfig`](crate::blocks::images::models::conv_net::ResidualConvConfig)
//! seam, chosen by its
//! [`ConvMixerKind`](crate::blocks::images::models::conv_net::ConvMixerKind).

mod res2;

pub use res2::*;
