//! # Gemma in `transformers`' names
//!
//! A `transformers` Gemma checkpoint (Gemma 1, 2 or 3; a causal LM's, or a
//! multimodal Gemma 3's text model, its vision tower dropped) as bunsen's
//! [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel):
//!
//! - **Config:** [`gemma_config`] of a parsed `config.json`, and
//!   [`gemma_config_from_str`] / [`gemma_config_from_path`].
//! - **Names:** [`gemma_rekey`], its layers by [`text_layer_rekey`].
//! - **Loading** (feature `store`): `load_gemma_tensors`; with `cache`,
//!   `load_gemma`, a resource set's checkpoint.

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
