//! # Speech models
//!
//! Five kits that work on audio, the way two of them connect,
//! [`asr_driver`], the parts of a speech-to-text stream driver that do not
//! depend on the model, and [`diarization`], the parts of a diarizer that do
//! not:
//!
//! - [`whisper`]: `OpenAI`'s Whisper, speech to text. The model, an index of
//!   pretrained checkpoints, the decoders, and a stream driver that turns audio
//!   written in chunks into timed transcript events.
//! - [`parakeet`]: NVIDIA's Parakeet TDT, speech to text: a `FastConformer`
//!   encoder and a token-and-duration transducer.
//! - [`qwen3_asr`]: Qwen's Qwen3-ASR, speech to text: a Whisper-style audio
//!   encoder feeding a Qwen3 language model.
//! - [`sortformer`]: NVIDIA's Sortformer, speaker diarization: per-frame
//!   activity of each speaker, in the order they first speak.
//! - [`silero_vad`]: the Silero voice-activity detector. A small recurrent
//!   model that gives, per chunk of audio, the probability that it holds
//!   speech.
//!
//! # How they connect
//!
//! Silero is the voice-activity gate of the Whisper stream driver. The
//! real-time emission presets
//! ([`conservative`](asr_driver::EmissionPolicy::conservative) and
//! [`responsive`](asr_driver::EmissionPolicy::responsive)) decode at
//! the end of each speech region, so their driver needs a
//! [`SileroVad`](silero_vad::SileroVad), attached with
//! [`with_vad`](whisper::driver::WhisperStreamDriver::with_vad) together
//! with a
//! [`VoiceActivityFilterConfig`](asr_driver::VoiceActivityFilterConfig)
//! that turns the model's probabilities into
//! [`SpeechRegion`](asr_driver::SpeechRegion)s. Each stream then runs
//! Silero in its own context, decodes each closed region as a unit of its
//! own, and skips windows of silence. The offline preset needs no gate.
//!
//! A diarizer attaches to any of the stream drivers the same way, with
//! [`with_speakers`](asr_driver::AsrStreamDriver::with_speakers): any
//! [`SpeakerActivityModel`](diarization::SpeakerActivityModel), and an
//! [`SpeakerAttribution`](asr_driver::SpeakerAttribution) rule that gives
//! every emitted segment a speaker.
//!
//! Both kits load the same way, through the data layer's
//! [`pretrained`](crate::data::pretrained) machinery: a
//! [`PretrainedCache`](crate::data::pretrained::PretrainedCache) and a
//! kit factory,
//! [`default_whisper_factory`](whisper::pretrained::default_whisper_factory)
//! (named, bundled and Hugging Face checkpoints) or
//! [`default_silero_factory`](silero_vad::pretrained::default_silero_factory)
//! (upstream's 16 kHz safetensors, `silero/vad-16k`). Whisper's text
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

pub mod asr_driver;
pub mod diarization;
pub mod nemo;
pub mod parakeet;
pub mod qwen3_asr;
pub mod silero_vad;
pub mod sortformer;

pub mod whisper;

#[cfg(feature = "testing")]
pub mod testing;
