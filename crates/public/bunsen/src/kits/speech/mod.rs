//! # Speech models
//!
//! Two kits that work on audio, and the way one feeds the other:
//!
//! - [`whisper`]: `OpenAI`'s Whisper, speech to text. The model, an index of
//!   pretrained checkpoints, the decoders, and a stream driver that turns audio
//!   written in chunks into timed transcript events.
//! - [`silero_vad`]: the Silero voice-activity detector. A small recurrent
//!   model that gives, per chunk of audio, the probability that it holds
//!   speech.
//!
//! # How they connect
//!
//! Silero is the voice-activity gate of the Whisper stream driver. The
//! real-time emission presets
//! ([`conservative`](whisper::driver::EmissionPolicy::conservative) and
//! [`responsive`](whisper::driver::EmissionPolicy::responsive)) decode at
//! the end of each speech region, so their driver needs a
//! [`SileroVad`](silero_vad::SileroVad), attached with
//! [`with_vad`](whisper::driver::WhisperStreamDriver::with_vad) together
//! with a
//! [`VoiceActivityFilterConfig`](whisper::driver::VoiceActivityFilterConfig)
//! that turns the model's probabilities into
//! [`SpeechRegion`](whisper::driver::SpeechRegion)s. Each stream then runs
//! Silero in its own context, decodes each closed region as a unit of its
//! own, and skips windows of silence. The offline preset needs no gate.
//!
//! Both kits load the same way, through the data layer's
//! [`pretrained`](crate::data::pretrained) machinery: a
//! [`PretrainedCache`](crate::data::pretrained::PretrainedCache) and a
//! kit factory,
//! [`default_whisper_factory`](whisper::pretrained::default_whisper_factory)
//! (named, bundled and Hugging Face checkpoints) or
//! [`default_silero_factory`](silero_vad::pretrained::default_silero_factory)
//! (one bundled row, with the `silero-weights` feature). Whisper's text
//! comes through the shared [`Detokenizer`](crate::kits::tokens::Detokenizer)
//! seam in [`kits::tokens`](crate::kits::tokens); its ids need no tokenizer.
//!
//! # Stream state is injected
//!
//! Neither kit's model owns the state of a stream. Each model is immutable
//! once loaded, and a stream's state lives in a context the caller holds
//! and passes back in, so one loaded model serves several streams, or
//! several caches, in one process:
//!
//! | Shared, immutable | Per stream or per decode |
//! |---|---|
//! | [`SileroVad`](silero_vad::SileroVad) | [`SileroVadContext`](silero_vad::SileroVadContext): the last chunk's tail and the recurrent state |
//! | [`TextDecoder`](whisper::blocks::TextDecoder) | [`TextDecoderCache`](whisper::blocks::TextDecoderCache): one decode's keys and values |
//! | [`PerceptiveAudioConverter`](crate::ops::signal::perceptive_audio::PerceptiveAudioConverter) | [`PerceptiveAudioConversionContext`](crate::ops::signal::perceptive_audio::PerceptiveAudioConversionContext): the mel front end's carry |
//! | [`WhisperStreamDriver`](whisper::driver::WhisperStreamDriver) | [`WhisperStreamContext`](whisper::driver::WhisperStreamContext): everything above, for one transcription |
//!
//! The Whisper driver's docs ([`whisper::driver`]) cover the design: the
//! `write_read` fold, the emission policy, the stream clock, the clamp
//! policy, and voice-activity regions as sub-streams.

pub mod silero_vad;

pub mod whisper;
