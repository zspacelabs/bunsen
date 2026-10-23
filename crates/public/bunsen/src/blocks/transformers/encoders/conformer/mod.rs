//! # Conformer blocks
//!
//! The parts of the Conformer (Gulati et al. 2020) in its `FastConformer` form
//! (Rybakov et al. 2023), the encoder of NVIDIA's Parakeet and Canary speech
//! models:
//!
//! - [`ConformerEncoderConfig`]: the encoder, as a policy that lowers to a
//!   [`SubsampledEncoderConfig`](crate::blocks::transformers::encoders::SubsampledEncoderConfig):
//!   [`DwStridingSubsampling`] (8x over time by default), then a
//!   [`ConformerTransformer`]: the stack of [`ConformerBlock`]s sharing one
//!   relative-position table.
//! - [`ConformerBlock`]: two half-step [`ConformerFeedForward`] modules around
//!   [`RelPosSelfAttention`](crate::blocks::transformers::attention::rel_pos::RelPosSelfAttention)
//!   and a [`ConformerConvolution`], each pre-normalized and residual.
//! - [`RelPosSelfAttention`](crate::blocks::transformers::attention::rel_pos::RelPosSelfAttention): Transformer-XL relative-position self-attention,
//!   over [`rel_position_table`](crate::ops::transformers::attention::rel_position_table)
//!   and [`rel_shift`](crate::ops::transformers::attention::rel_shift).
//!
//! Every forward takes the batch's per-item lengths as host values, and keeps
//! padding from leaking into valid frames: the subsampling zeroes padding
//! after each convolution, attention never attends to a padding key, and the
//! convolution module zeroes padding before its depthwise convolution.
//! [`length_mask`](crate::ops::transformers::attention::length_mask) builds
//! the `[batch, time]` validity mask from them.

mod block;
mod convolution;
mod encoder;
mod feed_forward;
mod subsampling;
mod transformer;

pub use block::*;
pub use convolution::*;
pub use encoder::*;
pub use feed_forward::*;
pub use subsampling::*;
pub use transformer::*;
