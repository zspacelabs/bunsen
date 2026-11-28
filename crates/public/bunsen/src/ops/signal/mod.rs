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
//!    [`PerceptiveAudioConverterConfig`]) is validated and builds the
//!    coefficients through `ModuleInit`.
//! 2. A coefficients `Module` ([`SlidingStft`], [`PerceptiveAudioConverter`])
//!    holds fixed tables: windows, DFT tables, a mel filterbank. They are bare
//!    tensors, not `Param`s, so they move with `to_device` but are not
//!    recorded, and a `ModuleMapper` pass does not see them. The module is
//!    stateless, so one instance can serve any number of streams.
//! 3. A stream context ([`SlidingStftStreamContext`],
//!    [`PerceptiveAudioConverterStreamContext`], each from its module's
//!    `new_stream_context`) carries the samples a stream still needs across
//!    calls; the module's `{verb}_stream` methods advance it. It is injected
//!    state: the caller holds one per stream (each batch row is an independent
//!    stream), and resets or drops it when the stream ends.
//!
//! ## Map
//!
//! - **STFT.** [`SlidingStftConfig`] -> [`SlidingStft`], with
//!   [`SlidingStftMeta`] on both, and a [`SlidingStftStreamContext`] per
//!   stream; any FFT length, by [`SpectrumImpl`] (burn's `rfft` for a power of
//!   two, a DFT by matmul for any). The log-mel converter frames and transforms
//!   by one. Spectra are `numpy.fft.rfft`-compatible; see [`SlidingStft`]'s
//!   *Spectrum convention*.
//! - **Inverse STFT.** [`InverseStftConfig`] -> [`InverseStft`]: spectra back
//!   to a waveform, as `torch.istft` computes it, at any FFT size; also as a
//!   stream of frames, through an [`InverseStftStreamContext`], each sample
//!   returned once no later frame reaches it.
//! - **Kernels.** [`kernels`]: the coefficient math the operators apply, one
//!   family to a module, each built on the host and on a device and the two
//!   cross-tested there. [`kernels::window`]: the analysis windows (Hann,
//!   Hamming, Blackman), picked as a config value by
//!   [`StftWindowConfig`](kernels::window::StftWindowConfig).
//! - **Log-mel.** [`perceptive_audio`] converts waveforms to log-mel
//!   spectrograms (Whisper's front end; defaults reproduce `librosa`).
//! - **Kaldi filterbanks.** [`KaldiFbankConfig`] -> [`KaldiFbank`], with
//!   [`KaldiFbankMeta`] on both: whole signals to Kaldi's `fbank` features, as
//!   `torchaudio.compliance.kaldi.fbank` computes them (speaker encoders' front
//!   end).
//! - **Rate conversion.** [`resample`]:
//!   [`StreamingResampler`](resample::StreamingResampler), a streaming
//!   Blackman-windowed sinc, and
//!   [`HannSincResampler`](resample::HannSincResampler), `torchaudio`'s
//!   `sinc_interp_hann` conversion; both on [`kernels::sinc`].
//! - **Long audio.** [`quiet_cuts`] cuts a long recording into bounded windows
//!   at its quietest points, for models that take bounded utterances.
//!
//! burn's counterpart is `burn::tensor::signal` (`stft`, `rfft`, power-of-two
//! only); [`SlidingStft`]'s [`Rfft`](SpectrumImpl::Rfft) is built on it.
//!
//! [`PerceptiveAudioConverterConfig`]: perceptive_audio::PerceptiveAudioConverterConfig
//! [`PerceptiveAudioConverter`]: perceptive_audio::PerceptiveAudioConverter
//! [`PerceptiveAudioConverterStreamContext`]: perceptive_audio::PerceptiveAudioConverterStreamContext

pub mod kernels;
pub mod perceptive_audio;
pub mod resample;

mod inverse_stft;
mod kaldi_fbank;
mod quiet_cuts;
mod sliding_stft;

pub use inverse_stft::*;
pub use kaldi_fbank::*;
pub use quiet_cuts::*;
pub use sliding_stft::*;
