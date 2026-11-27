//! # Voice-activity detection
//!
//! What a voice-activity detector is, apart from any one model:
//!
//! - [`VoiceActivityModel`] and [`VoiceActivityStream`]: per-chunk speech
//!   probabilities, a stream each.
//! - [`VoiceActivityFilter`] ([`VoiceActivityFilterConfig`]): the hysteresis
//!   gate that turns a probability track into [`SpeechRegion`]s, padded by
//!   [`pad_regions`].
//!
//! The ASR stream drivers gate their decoding on these
//! ([`AsrStreamDriver`](crate::kits::speech::asr::driver::AsrStreamDriver)).
//!
//! And the models:
//!
//! - [`silero_vad`]: the Silero voice-activity detector.

pub mod silero_vad;

mod activity;
mod filter;
mod speech_region;

pub use activity::*;
pub use filter::*;
pub use speech_region::*;
