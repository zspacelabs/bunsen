//! # ASR stream driving: the parts that do not depend on the model
//!
//! The pieces a speech-to-text stream driver is built from, whatever model
//! it drives:
//!
//! - [`TranscriptSegment`] and [`TranscriptEvent`]: what a driver emits, a
//!   committed segment or a draft one.
//! - [`EmissionPolicy`]: when a driver decodes ([`DecodeTriggers`]) and what it
//!   commits ([`CommitRule`]), with presets for the three deployments (offline,
//!   conservative real time, responsive real time).
//! - [`StreamClock`]: a stream's sample index to media time.
//! - [`VoiceActivityFilter`]: a voice-activity model's per-chunk probabilities
//!   to [`SpeechRegion`]s.
//!
//! [`whisper::driver`](crate::kits::speech::whisper::driver) is a driver
//! built on them.

mod emission_policy;
mod speech_region;
mod stream_clock;
mod transcript;
mod voice_activity_filter;

pub use emission_policy::*;
pub use speech_region::*;
pub use stream_clock::*;
pub use transcript::*;
pub use voice_activity_filter::*;
