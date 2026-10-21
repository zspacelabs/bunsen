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
//! - `NemoArchive` (feature `nemo`): a `.nemo` checkpoint, `NeMo`'s tar of a
//!   YAML config and a `PyTorch` state dict; `read_nemo_weights` reads the
//!   state dict, and `parse_nemo_yaml` the config, into typed sections
//!   (`NemoPreprocessorConfig`, `NemoConformerEncoderConfig`,
//!   `NemoTransformerEncoderConfig`) that give the bunsen configs they
//!   describe. A section's fields are its YAML keys, spelled as `NeMo` spells
//!   them, so a field greps to `NeMo`'s source and to a `.nemo`'s config;
//!   bunsen's own names start in the configs a section gives.
//! - [`nemo_conformer_to_bunsen_path`] and [`nemo_transformer_to_bunsen_path`]:
//!   `NeMo`'s state-dict names as bunsen's module paths.
//!
//! The `.nemo` fixture in `testdata/nemo/` is a toy Sortformer v2 that `NeMo`
//! saved itself, written by `tools/gen_nemo_fixtures.py`.

#[cfg(feature = "nemo")]
mod archive;
#[cfg(feature = "nemo")]
mod config;
mod front_end;
mod renames;

#[cfg(all(test, feature = "nemo"))]
mod archive_tests;

#[cfg(feature = "nemo")]
pub use archive::*;
#[cfg(feature = "nemo")]
pub use config::*;
pub use front_end::*;
pub use renames::*;
