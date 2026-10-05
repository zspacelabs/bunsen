//! # Signal processing
//!
//! Audio front ends: short-time Fourier analysis, analysis windows, and the
//! waveform-to-log-mel conversion that speech models consume. They hold no
//! trainable weights, but they do use the Module / Config machinery, to hold
//! cached tables and stream state; that is the [ops
//! rule](crate::ops#ops-and-blocks) at work.
//!
//! ## Config, coefficients, stream context
//!
//! Each analyzer is split three ways:
//!
//! 1. A config value ([`SlidingStftConfig`],
//!    [`PerceptiveAudioConverterOptions`]) is validated and builds the
//!    coefficients through `ModuleInit`.
//! 2. A coefficients `Module` ([`SlidingStft`], [`PerceptiveAudioConverter`])
//!    holds fixed tables: windows, DFT tables, a mel filterbank. They are bare
//!    tensors, not `Param`s, so they move with `to_device` but are not
//!    recorded, and a `ModuleMapper` pass does not see them. The module is
//!    stateless, so one instance can serve any number of streams.
//! 3. A stream context ([`SlidingStftContext`], built by
//!    [`SlidingStft::init_state`]; [`PerceptiveAudioConversionContext`], built
//!    by [`PerceptiveAudioConverter::new_context`]) carries the samples a
//!    stream still needs across calls. It is injected state: the caller holds
//!    one per stream (each batch row is an independent stream), and resets or
//!    drops it when the stream ends.
//!
//! ## Map
//!
//! - **STFT.** [`SlidingStftConfig`] -> [`SlidingStft`] ->
//!   [`SlidingStftContext`], with [`SlidingStftMeta`] on all three. Spectra are
//!   `numpy.fft.rfft`-compatible; see [`SlidingStft`]'s *Spectrum convention*.
//! - **Windows.** [`SamplingWindowBuilder`] materializes a window as a
//!   `Vec<f64>` or a tensor. [`CosineWindowConfig`] (Hann, Hamming) and
//!   [`DualCosineWindow`] (Blackman) are the families, and [`StftWindowConfig`]
//!   picks one as a config value.
//! - **Log-mel.** [`perceptive_audio`] converts waveforms to log-mel
//!   spectrograms (Whisper's front end; defaults reproduce `librosa`).
//! - **Testing.** The `testing` submodule (with the `testing` feature) has
//!   assertions for window builders.
//!
//! burn's counterpart is `burn::tensor::signal` (`stft`, `rfft`), which
//! [`SlidingStft`] is built on.
//!
//! [`PerceptiveAudioConverterOptions`]: perceptive_audio::PerceptiveAudioConverterOptions
//! [`PerceptiveAudioConverter`]: perceptive_audio::PerceptiveAudioConverter
//! [`PerceptiveAudioConverter::new_context`]: perceptive_audio::PerceptiveAudioConverter::new_context
//! [`PerceptiveAudioConversionContext`]: perceptive_audio::PerceptiveAudioConversionContext

pub mod perceptive_audio;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

mod cosine_window;
mod sliding_stft;
mod stft_window;
mod window_builder;

pub use cosine_window::*;
pub use sliding_stft::*;
pub use stft_window::*;
pub use window_builder::*;
