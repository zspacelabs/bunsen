//! Speech Models

pub mod pitch;

/// A fully functional Silero VAD model.
pub mod silero_vad;

/// A ten-vad model, with a full audio pre-processing driver.
///
/// The pitch feature is stubbed; see [`ten_vad::context`].
pub mod ten_vad;

/// A structural whisper model.
/// Loads, runs; no driver implementation yet.
pub mod whisper;
