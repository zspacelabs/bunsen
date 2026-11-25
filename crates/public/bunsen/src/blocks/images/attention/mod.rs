//! # Channel attention
//!
//! Blocks that rescale a feature map's channels by gates computed from the
//! map itself: `x * sigmoid(f(mean(x)))`, `f` a small net over the channels'
//! spatial means. Each is a kind of the [`ChannelAttention`] seam, the
//! attention slot of a conv net's residual unit:
//!
//! - [`SqueezeExcite`]: a `1x1` conv down to a reduction, an activation, and
//!   a `1x1` conv back (Squeeze-and-Excitation,
//!   [arXiv:1709.01507](https://arxiv.org/abs/1709.01507); timm's
//!   `SEModule`).
//! - [`Eca`]: one `1d` conv across the channels, its kernel sized from their
//!   count (ECA-Net, [arXiv:1910.03151](https://arxiv.org/abs/1910.03151);
//!   timm's `EcaModule`).
//!
//! The configs size themselves as timm's do from a channel count
//! ([`SqueezeExciteConfig::timm`], [`EcaConfig::timm`]); their parameters
//! are named as timm's (`fc1`, `fc2`; `conv`).

mod channel_attention;
mod eca;
mod squeeze_excite;

pub use channel_attention::*;
pub use eca::*;
pub use squeeze_excite::*;
