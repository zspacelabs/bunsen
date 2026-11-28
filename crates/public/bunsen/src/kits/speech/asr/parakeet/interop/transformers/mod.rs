//! # Parakeet in `transformers`' names
//!
//! A `ParakeetForTDT` checkpoint as bunsen's
//! [`ParakeetTdt`](crate::kits::speech::asr::parakeet::ParakeetTdt), the
//! names every Parakeet checkpoint here is read under:
//!
//! - **Config:** [`parakeet_tdt_config`] of a parsed `config.json`, and
//!   [`parakeet_tdt_config_from_str`] / [`parakeet_tdt_config_from_path`].
//! - **Names:** [`parakeet_tdt_rekey`], over the encoder's ([`encoder_rekey`]:
//!   [`subsampling_rekey`], [`encoder_block_rekey`]).
//! - **Loading** (feature `store`): `load_parakeet_tdt_tensors`; with `cache`,
//!   `load_parakeet_tdt`, a resource set's checkpoint and `tokenizer.json`
//!   vocabulary.

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
