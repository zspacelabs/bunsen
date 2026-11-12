//! # Speech models
//!
//! Six kits that work on audio, grouped by the task each does. Each task
//! module holds its kits and the parts of the task that do not depend on the
//! model:
//!
//! - [`asr`]: speech to text. [`asr::driver`] is the model-generic part of a
//!   stream driver; the kits:
//!   - [`whisper`](asr::whisper): `OpenAI`'s Whisper. The model, an index of
//!     pretrained checkpoints, the decoders, and a stream driver that turns
//!     audio written in chunks into timed transcript events.
//!   - [`parakeet`](asr::parakeet): NVIDIA's Parakeet TDT: a `FastConformer`
//!     encoder and a token-and-duration transducer.
//!   - [`qwen3_asr`](asr::qwen3_asr): Qwen's Qwen3-ASR: a Whisper-style audio
//!     encoder feeding a Qwen3 language model.
//!   - [`voxtral`](asr::voxtral): Mistral's Voxtral Realtime, streaming: a
//!     causal audio encoder feeding a Mistral text model, one decoder step per
//!     80 ms of audio.
//! - [`diarization`]: who spoke when. The diarizer seam, turns and RTTM; the
//!   kit:
//!   - [`sortformer`](diarization::sortformer): NVIDIA's Sortformer: per-frame
//!     activity of each speaker, in the order they first speak, streamed
//!     through a speaker cache. `Nemotron-3-Diarization` and the streaming
//!     Sortformer v2 and v2.1.
//! - [`vad`]: voice activity; the kit:
//!   - [`silero_vad`](vad::silero_vad): the Silero voice-activity detector. A
//!     small recurrent model that gives, per chunk of audio, the probability
//!     that it holds speech.
//!
//! Beside the tasks, [`nemo`] holds the parts NVIDIA's `NeMo` models share
//! across them (the mel front end, `.nemo` checkpoints): Parakeet's and
//! Sortformer's.
//!
//! # How they connect
//!
//! Silero is the voice-activity gate of the Whisper stream driver. The
//! real-time emission presets
//! ([`conservative`](asr::driver::EmissionPolicy::conservative) and
//! [`drafting`](asr::driver::EmissionPolicy::drafting)) decode at
//! the end of each speech region, so their driver needs a
//! [`SileroVad`](vad::silero_vad::SileroVad), attached with
//! [`with_vad`](asr::whisper::driver::WhisperStreamDriver::with_vad) together
//! with a
//! [`VoiceActivityFilterConfig`](asr::driver::VoiceActivityFilterConfig)
//! that turns the model's probabilities into
//! [`SpeechRegion`](asr::driver::SpeechRegion)s. Each stream then runs
//! Silero in its own context, decodes each closed region as a unit of its
//! own, and skips windows of silence. The offline preset needs no gate.
//!
//! A diarizer attaches to any of the stream drivers the same way, with
//! [`with_speakers`](asr::driver::AsrStreamDriver::with_speakers): any
//! [`SpeakerActivityModel`](diarization::SpeakerActivityModel), and an
//! [`SpeakerAttribution`](asr::driver::SpeakerAttribution) rule that gives
//! every emitted segment a speaker.
//!
//! Both kits load the same way, through the data layer's
//! [`pretrained`](crate::data::pretrained) machinery: a
//! [`PretrainedCache`](crate::data::resources::PretrainedCache) and a
//! kit factory,
//! [`default_whisper_factory`](asr::whisper::pretrained::default_whisper_factory)
//! (named, bundled and Hugging Face checkpoints) or
//! [`default_silero_factory`](vad::silero_vad::pretrained::default_silero_factory)
//! (upstream's 16 kHz safetensors, `silero/vad-16k`). Whisper's text
//! comes through the shared
//! [`Detokenizer`](crate::kits::support::tokens::Detokenizer)
//! seam in [`kits::support::tokens`](crate::kits::support::tokens); its ids
//! need no tokenizer.
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
//! | [`SileroVad`](vad::silero_vad::SileroVad) | [`SileroVadContext`](vad::silero_vad::SileroVadContext): the last chunk's tail and the recurrent state |
//! | [`TextDecoder`](asr::whisper::blocks::TextDecoder) | [`KVCache`](crate::ops::transformers::attention::KVCache): one decode's keys and values, self- and cross-attention |
//! | [`PerceptiveAudioConverter`](crate::ops::signal::perceptive_audio::PerceptiveAudioConverter) | [`PerceptiveAudioConversionContext`](crate::ops::signal::perceptive_audio::PerceptiveAudioConversionContext): the mel front end's carry |
//! | [`WhisperStreamDriver`](asr::whisper::driver::WhisperStreamDriver) | [`WhisperStreamContext`](asr::whisper::driver::WhisperStreamContext): everything above, for one transcription |
//!
//! The Whisper driver's docs ([`whisper::driver`](asr::whisper::driver)) cover
//! the design: the `write_read` fold, the emission policy, the stream clock,
//! the clamp policy, and voice-activity regions as sub-streams.

pub mod asr;
pub mod diarization;
pub mod nemo;
pub mod vad;

#[cfg(feature = "testing")]
pub mod testing;
