//! Testing support for speech kits: inputs, and scores.
//!
//! - [`ToneConfig`]: synthetic tones.
//! - [`ScriptedDiarizer`]: a diarizer that reads a script of who speaks when.
//! - [`RecordedVoiceActivity`]: a voice-activity model that replays recorded
//!   probabilities.
//! - [`wer`]: word error rates, of a transcript and of a speaker-attributed one
//!   (cpWER).
//! - [`der`]: diarization error rates.
//! - [`tts_asr_round_trip`]: speech synthesis judged by speech recognition, any
//!   synthesizer and any recognizer, scored by word error rate;
//!   [`tts_synthesizer_round_trip`] through a TTS driver.

pub mod der;
pub mod wer;

mod diarizer;
mod round_trip;
mod tones;
mod voice_activity;

pub use diarizer::*;
pub use round_trip::*;
pub use tones::*;
pub use voice_activity::*;
