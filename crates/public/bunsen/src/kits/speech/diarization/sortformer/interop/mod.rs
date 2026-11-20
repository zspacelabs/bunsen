//! # Sortformer's interop: its checkpoints' ecosystems
//!
//! - [`transformers`]: a `Nemotron3DiarizationForAudioFrameClassification`
//!   checkpoint (the Nemotron family), its `config.json` and state-dict names.
//! - [`nemo`]: a `SortformerEncLabelModel` `.nemo` (Sortformer v2 and v2.1),
//!   its YAML config and state-dict names.

pub mod nemo;
pub mod transformers;
