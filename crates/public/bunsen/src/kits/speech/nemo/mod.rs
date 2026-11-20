//! # What `NeMo`'s speech models share
//!
//! NVIDIA's `NeMo` models (Parakeet, the Sortformer diarizers) are built from
//! the same parts. The ones more than one kit here uses live in this module:
//!
//! - [`NemoMelFrontEnd`]: `AudioToMelSpectrogramPreprocessor` at inference,
//!   waveforms to log-mels, built from a [`NemoMelFrontEndConfig`] whose
//!   presets are the models'
//!   ([`parakeet_tdt`](NemoMelFrontEndConfig::parakeet_tdt),
//!   [`nemotron_diarization`](NemoMelFrontEndConfig::nemotron_diarization));
//!   [`NemoMelFeatures`] streams it.
//!
//! What their files mean (the YAML config sections, the bunsen configs they
//! give, the state-dict names) is
//! [`support::interop::nemo`](crate::kits::support::interop::nemo); the
//! `.nemo` container is [`data::formats::nemo`](crate::data::formats::nemo).

mod front_end;

pub use front_end::*;
