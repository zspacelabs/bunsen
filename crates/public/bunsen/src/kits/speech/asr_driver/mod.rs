//! # ASR stream driving: the parts that do not depend on the model
//!
//! A stream driver turns audio, written in pieces as it arrives, into timed
//! transcript events. Everything about that which does not depend on the
//! model lives here; a model plugs in as an [`AsrEngine`].
//!
//! - [`AsrStreamDriver`] and [`AsrStreamContext`]: the shared, immutable half
//!   of a transcription and one stream in progress. The context buffers the
//!   engine's features, runs the voice-activity gate, skips silence, decides
//!   when to decode, commits and drafts, and places what it emits on the
//!   stream's clock; [`advance_ready`] batches the decodes of many streams
//!   through one engine.
//! - [`AsrEngine`]: what a model gives the core: its streaming features
//!   ([`StreamingFeatures`]; [`LogMelFeatures`] streams the log-mels the
//!   Whisper family reads), the decode of a window ([`DecodeRequest`] to
//!   [`WindowDecode`]), its text, and any per-stream carry.
//! - [`TranscriptSegment`] and [`TranscriptEvent`]: what a driver emits, a
//!   committed segment or a draft one.
//! - [`EmissionPolicy`]: when a driver decodes ([`DecodeTriggers`]) and what it
//!   commits ([`CommitRule`]), with presets for the three deployments (offline,
//!   conservative real time, responsive real time).
//! - [`StreamClock`]: a stream's sample index to media time.
//! - [`VoiceActivityModel`] and [`VoiceActivityFilter`]: per-chunk speech
//!   probabilities, and the gate that turns them into [`SpeechRegion`]s.
//!   [`SileroVad`](crate::kits::speech::silero_vad::SileroVad) is a model.
//! - Cutting a full window: an engine that times its words places the cut
//!   between two [`WordSpan`]s ([`pause_cut`]) and makes each a segment
//!   ([`word_segments`]); one that cannot cuts at the window's quietest stretch
//!   before decoding it ([`quiet_cut`]).
//! - Speakers: with a diarizer attached
//!   ([`with_speakers`](AsrStreamDriver::with_speakers), any
//!   [`SpeakerActivityModel`](crate::kits::speech::diarization::SpeakerActivityModel)),
//!   every segment is attributed to a speaker ([`SpeakerLabel`]) by a
//!   [`SpeakerAttribution`] rule: [`MeanActivity`], `NeMo`'s, or
//!   [`TurnOverlap`].
//! - [`LevelPolicy`] and [`LevelStatistic`]: what a window's features are
//!   scaled against, the window's own frames or everything the stream has seen,
//!   summarized as the model needs ([`MaxLevel`], [`MomentsLevel`]).
//!
//! [`support`] holds the internals underneath; nothing in it is part of the
//! API.
//!
//! [`whisper::driver`](crate::kits::speech::whisper::driver) is a driver
//! built on these parts.

pub mod support;

mod emission_policy;
mod engine;
mod level;
mod log_mel_features;
mod quiet_cut;
mod speaker_attribution;
mod speech_region;
mod stream_clock;
mod stream_context;
mod stream_driver;
mod transcript;
mod voice_activity_filter;
mod voice_activity_model;
mod word_spans;

pub use emission_policy::*;
pub use engine::*;
pub use level::*;
pub use log_mel_features::*;
pub use quiet_cut::*;
pub use speaker_attribution::*;
pub use speech_region::*;
pub use stream_clock::*;
pub use stream_context::*;
pub use stream_driver::*;
pub use transcript::*;
pub use voice_activity_filter::*;
pub use voice_activity_model::*;
pub use word_spans::*;
