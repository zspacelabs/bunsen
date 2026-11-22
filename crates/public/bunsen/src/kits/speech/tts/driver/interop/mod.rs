//! # The TTS driver's file formats
//!
//! - [`safetensors`] (feature `store`): the voice file, a
//!   [`TtsVoicePrint`](super::TtsVoicePrint) as one safetensors file
//!   ([`TtsVoicePrint::to_bytes`](super::TtsVoicePrint::to_bytes),
//!   [`from_bytes`](super::TtsVoicePrint::from_bytes),
//!   [`write`](super::TtsVoicePrint::write),
//!   [`read`](super::TtsVoicePrint::read)): its tensors, and its identity as
//!   the header's only metadata.

#[cfg(feature = "store")]
pub mod safetensors;
