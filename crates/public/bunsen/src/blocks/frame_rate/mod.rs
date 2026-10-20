//! # Frame-rate changes
//!
//! Layers that change a sequence's frame rate by a whole factor, as modules:
//!
//! - [`FeatureStacking`]: `factor` consecutive frames stacked into one and
//!   projected, a learned downsampling (`NeMo`'s `FeatureStacking` pre-encode).
//! - [`SubpixelUpsample1d`]: a convolution to `factor` times the channels, each
//!   frame's channels then split into `factor` frames, a learned upsampling
//!   (sub-pixel convolution, Shi et al. 2016).
//!
//! `Nemotron-3-Diarization` stacks 10 ms mel frames into 80 ms encoder frames
//! and upsamples them back to 10 ms before its classifier.

mod feature_stacking;
mod subpixel;

pub use feature_stacking::*;
pub use subpixel::*;
