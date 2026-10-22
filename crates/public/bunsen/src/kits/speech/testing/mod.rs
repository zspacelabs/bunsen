//! Testing support for speech kits: inputs, and scores.
//!
//! - [`ToneConfig`]: synthetic tones.
//! - [`ScriptedDiarizer`]: a diarizer that reads a script of who speaks when.
//! - [`RecordedVoiceActivity`]: a voice-activity model that replays recorded
//!   probabilities.
//! - [`wer`]: word error rates, of a transcript and of a speaker-attributed one
//!   (cpWER).
//! - [`der`]: diarization error rates.

pub mod der;
pub mod wer;

mod diarizer;
mod tones;
mod voice_activity;

pub use diarizer::*;
pub use tones::*;
pub use voice_activity::*;
