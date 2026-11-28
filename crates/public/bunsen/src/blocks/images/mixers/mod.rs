//! # Spatial mixers
//!
//! The spatial mixer of a conv net's residual unit is the conv that strides,
//! a bottleneck's `3x3`. These blocks take its place, each more than one conv:
//!
//! - [`Res2Conv`]: a scale split (timm's `Bottle2neck`'s `convs`). The channels
//!   split into `scale` chunks; each chunk but the last is convolved, after the
//!   first adding the previous branch's output, so the receptive field grows
//!   branch by branch.
//! - [`SplitAttn`]: split attention (timm's `SplitAttn`). A grouped conv widens
//!   to `radix` splits, and an attention over their sum weighs them per
//!   channel.
//! - [`SelectiveKernel`]: a selective kernel (timm's `SelectiveKernel`). Paths
//!   of two receptive fields, and an attention over their sum that weighs them
//!   per channel.
//!
//! Each keeps its upstream's parameter names where the conv net's
//! conventions allow (`convs.{k}.conv`, `convs.{k}.norm`), and is a kind of
//! the conv net's
//! [`ResidualConvConfig`](crate::blocks::images::models::conv_net::ResidualConvConfig)
//! seam, chosen by its
//! [`ConvMixerKind`](crate::blocks::images::models::conv_net::ConvMixerKind).
//!
//! ## Discussion
//!
//! - [`Res2Conv`] is `Res2Net`'s scale split ([arXiv:1904.01169](https://arxiv.org/abs/1904.01169)).
//! - [`SplitAttn`] is `ResNeSt`'s split attention ([arXiv:2004.08955](https://arxiv.org/abs/2004.08955)).
//! - [`SelectiveKernel`] is `SKNet`'s selective kernel ([arXiv:1903.06586](https://arxiv.org/abs/1903.06586)).

mod res2;
mod selective_kernel;
mod split_attn;

pub use res2::*;
pub use selective_kernel::*;
pub use split_attn::*;
