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
//! And the waveform out of frames: [`DacDecoder`], DAC's decoder (Descript
//! Audio Codec, Kumar et al. 2023), a convolution into its width, blocks
//! that each upsample by a factor through a causal transposed convolution
//! and run residual units of `SnakeBeta` and dilated causal convolutions
//! ([`DacDecoderBlock`], [`DacResidualUnit`]), and a convolution to one
//! channel; Qwen3-TTS's codec vocoder. It also runs as a stream, over chunks
//! of latents through a [`DacDecoderStreamContext`], each chunk's samples
//! final at once and, joined, the whole forward's.
//!
//! And [`EncoderSubsampling`], an encoder's front end chosen by config, over
//! the two downsamplings: what a
//! [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder)
//! runs before its stack. Each keeps the input layout its upstream used.
//!
//! `Nemotron-3-Diarization` stacks 10 ms mel frames into 80 ms encoder frames
//! and upsamples them back to 10 ms before its classifier.

mod dac;
mod dw_striding;
mod feature_stacking;
mod subpixel;
mod subsampling;

pub use dac::*;
pub use dw_striding::*;
pub use feature_stacking::*;
pub use subpixel::*;
pub use subsampling::*;
