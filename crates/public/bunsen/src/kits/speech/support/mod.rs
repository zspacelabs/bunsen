//! # What more than one speech task shares
//!
//! The parts of speech kits that no one task owns:
//!
//! - [`StreamingFeatures`]: the seam a streaming front end implements, audio in
//!   pieces of any size to feature frames; the ASR engines and Sortformer read
//!   through it.
//! - [`LevelStatistic`]: what a front end scales a window against
//!   ([`MaxLevel`], [`MomentsLevel`]).
//! - [`log_mel`]: the log-mel front end the Whisper family shares: Whisper's,
//!   Qwen3-ASR's and Voxtral's (ASR).
//! - [`nemo`]: what NVIDIA's `NeMo` models share across tasks (the mel front
//!   end): Parakeet's (ASR) and Sortformer's (diarization).
//! - `testing` (feature `testing`): speech test machinery, the word and
//!   diarization error rates, test tones, and stand-in diarizers and engines.

pub mod log_mel;
pub mod nemo;

mod features;
mod level;

pub use features::*;
pub use level::*;

#[cfg(feature = "testing")]
pub mod testing;
