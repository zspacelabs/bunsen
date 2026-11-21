//! # What more than one speech task shares
//!
//! The parts of speech kits that no one task owns:
//!
//! - [`nemo`]: what NVIDIA's `NeMo` models share across tasks (the mel front
//!   end): Parakeet's (ASR) and Sortformer's (diarization).
//! - `testing` (feature `testing`): speech test machinery, the word and
//!   diarization error rates, test tones, and stand-in diarizers and engines.

pub mod nemo;

#[cfg(feature = "testing")]
pub mod testing;
