//! # `nemo`: what `NeMo`'s speech models share
//!
//! NVIDIA's `NeMo` models (Parakeet, the Sortformer diarizers) are built from
//! the same parts and ship the same files. What more than one kit here uses,
//! by use:
//!
//! - [`front_end`]: the mel front end,
//!   [`NemoMelFrontEnd`](front_end::NemoMelFrontEnd),
//!   `AudioToMelSpectrogramPreprocessor` at inference, waveforms to log-mels,
//!   built from a [`NemoMelFrontEndConfig`](front_end::NemoMelFrontEndConfig)
//!   whose presets are the models'
//!   ([`parakeet_tdt`](front_end::NemoMelFrontEndConfig::parakeet_tdt),
//!   [`nemotron_diarization`](front_end::NemoMelFrontEndConfig::nemotron_diarization));
//!   [`NemoMelFeatures`](front_end::NemoMelFeatures) streams it.
//! - [`interop`]: what their files mean: the YAML config sections, the bunsen
//!   configs they give, and the state-dict names.
//!
//! The `.nemo` container is [`data::formats::nemo`](crate::data::formats::nemo).

pub mod front_end;
pub mod interop;
