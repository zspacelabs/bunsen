//! # Whisper in `transformers`' names
//!
//! A `WhisperForConditionalGeneration` checkpoint as bunsen's
//! [`Whisper`](crate::kits::speech::asr::whisper::Whisper). It has no config
//! to convert: the geometry is read from the tensors' shapes.
//!
//! - **Names:** [`whisper_rekey`], its layers by `transformers`' Whisper layer
//!   names
//!   ([`whisper_encoder_layer_rekey`](crate::kits::support::transformers::interop::whisper_encoder_layer_rekey),
//!   [`whisper_decoder_layer_rekey`](crate::kits::support::transformers::interop::whisper_decoder_layer_rekey)).
//! - **Loading** (feature `store`): `whisper_config_from_headers`, the geometry
//!   from the files' headers; `load_whisper_tensors` and `load_whisper`, the
//!   model from their tensors.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod rekey;

pub use rekey::*;
