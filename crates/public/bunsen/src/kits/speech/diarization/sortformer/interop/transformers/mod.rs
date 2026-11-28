//! # Sortformer in `transformers`' names
//!
//! A `Nemotron3DiarizationForAudioFrameClassification` checkpoint as
//! bunsen's Nemotron
//! [`Sortformer`](crate::kits::speech::diarization::sortformer::Sortformer):
//!
//! - **Config:** [`nemotron_diarization_config`] of a parsed `config.json`, and
//!   [`nemotron_diarization_config_from_str`] /
//!   [`nemotron_diarization_config_from_path`].
//! - **Names:** [`nemotron_diarization_rekey`], from [`embedder_rekey`] and
//!   [`encoder_rekey`].
//! - **Loading** (feature `store`): `load_nemotron_diarization_tensors`; with
//!   `cache`, `load_nemotron_diarization`, a resource set's checkpoint.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod config;
mod rekey;

pub use config::*;
pub use rekey::*;

#[cfg(test)]
mod tests;
