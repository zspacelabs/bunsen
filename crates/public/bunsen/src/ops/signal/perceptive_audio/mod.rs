//! # Waveform to log-mel conversion
//!
//! The log-mel spectrogram front end that speech models consume (Whisper's
//! among them), with defaults that reproduce `librosa`.
//!
//! ## Lifecycle
//!
//! - [`PerceptiveAudioConverterOptions`] configures the pipeline: framing,
//!   window, spectrum, mel bank, log. [`ModuleInit`] validates it and builds
//!   the converter.
//! - [`PerceptiveAudioConverter`] holds the precomputed constants (window, DFT
//!   tables, mel filterbank) as bare tensors. It is stateless:
//!   [`forward`](PerceptiveAudioConverter::forward) converts a whole signal in
//!   one call, and one converter can serve any number of streams.
//! - [`PerceptiveAudioConverter::new_context`] opens a
//!   [`PerceptiveAudioConversionContext`] per stream (each batch row is one
//!   stream). It carries the samples a stream has not finished with, so
//!   hop-aligned chunks produce exactly the frames one call would;
//!   [`finish`](PerceptiveAudioConversionContext::finish) flushes the end
//!   padding.
//!
//! [`PerceptiveAudioConverterMeta`] reads the framing geometry the same way
//! from the options, the converter, or a context.
//!
//! ## The mel bank
//!
//! [`MelFilterbankConfig`] builds the triangular filterbank on the host, in
//! `Vec<f64>`, following `librosa.filters.mel`; [`MelScale`] picks the
//! frequency-to-mel curve and [`FilterNorm`] the triangle normalization. The
//! converter lifts it to the device at init.
//!
//! ## Dynamic-range packaging is the caller's step
//!
//! [`RangeClamp`] (Whisper's "keep 8 dB below the max") and
//! [`AffineCompress`] (Whisper's `(x + 4) / 4`) are applied by the caller to a
//! finished spectrogram, not configured on the converter. A clamp relative to
//! the maximum reduces over whatever it is handed, so folding it into a
//! streaming converter would make the chunking observable.
//!
//! [`ModuleInit`]: crate::burner::module::ModuleInit

mod filterbank;
mod perceptive_audio_context;
mod perceptive_audio_converter;

#[cfg(test)]
mod cross_test;

pub use filterbank::*;
pub use perceptive_audio_context::*;
pub use perceptive_audio_converter::*;
