//! # Whisper as `OpenAI`'s `whisper` package saves it
//!
//! `OpenAI`'s `.pt`: a `PyTorch` state dict under `model_state_dict`, with
//! upstream's names, as bunsen's
//! [`Whisper`](crate::kits::speech::asr::whisper::Whisper). It has no config
//! to convert: the geometry is read from the tensors' shapes.
//!
//! - **Loading** (feature `store`): `whisper_config_from_path`, the geometry
//!   from the state dict; `load_whisper_tensors` and `load_whisper`, the model
//!   from its tensors, under the store's remaps of upstream's names onto
//!   bunsen's canonical ones.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;
