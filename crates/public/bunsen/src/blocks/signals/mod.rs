//! # Signal blocks
//!
//! Learned layers over a signal's time axis: the counterpart of
//! [`ops::signal`](crate::ops::signal), whose front ends (STFT, windows,
//! log-mel) hold no trainable weights. The frame-rate changes, each by a
//! whole factor:
//!
//! - [`FeatureStacking`]: `factor` consecutive frames stacked into one and
//!   projected, a learned downsampling (`NeMo`'s `FeatureStacking` pre-encode).
//! - [`DwStridingSubsampling`]: the `FastConformer`'s depthwise-striding
//!   convolutions, `2^n_stages` over time, onto `d_model`.
//! - [`SubpixelUpsample1d`]: a convolution to `factor` times the channels, each
//!   frame's channels then split into `factor` frames, a learned upsampling
//!   (sub-pixel convolution, Shi et al. 2016).
//!
//! And [`Subsampling`], an encoder's front end chosen by config, over the
//! two downsamplings: what a
//! [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder)
//! runs before its stack. Each keeps the input layout its upstream used.
//!
//! `Nemotron-3-Diarization` stacks 10 ms mel frames into 80 ms encoder frames
//! and upsamples them back to 10 ms before its classifier.

mod dw_striding;
mod feature_stacking;
mod subpixel;
mod subsampling;

pub use dw_striding::*;
pub use feature_stacking::*;
pub use subpixel::*;
pub use subsampling::*;
