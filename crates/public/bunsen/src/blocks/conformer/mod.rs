//! # Conformer blocks
//!
//! The parts of the Conformer (Gulati et al. 2020) in its `FastConformer` form
//! (Rybakov et al. 2023), the encoder of NVIDIA's Parakeet and Canary speech
//! models:
//!
//! - [`ConformerEncoder`]: [`DwStridingSubsampling`] (8x over time by default),
//!   then a stack of [`ConformerBlock`]s sharing one relative-position table.
//! - [`ConformerBlock`]: two half-step [`ConformerFeedForward`] modules around
//!   [`RelPosSelfAttention`] and a [`ConformerConvolution`], each
//!   pre-normalized and residual.
//! - [`RelPosSelfAttention`]: Transformer-XL relative-position self-attention,
//!   over [`rel_position_table`](crate::ops::transformers::attention::rel_position_table)
//!   and [`rel_shift`](crate::ops::transformers::attention::rel_shift).
//!
//! Every forward takes the batch's per-item lengths as host values, and keeps
//! padding from leaking into valid frames: the subsampling zeroes padding
//! after each convolution, attention never attends to a padding key, and the
//! convolution module zeroes padding before its depthwise convolution.
//! [`time_mask`] builds the `[batch, time]` validity mask from them.

mod block;
mod convolution;
mod encoder;
mod feed_forward;
mod rel_pos_attention;
mod subsampling;

pub use block::*;
pub use convolution::*;
pub use encoder::*;
pub use feed_forward::*;
pub use rel_pos_attention::*;
pub use subsampling::*;
