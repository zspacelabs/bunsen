//! # `ModernBERT` in `transformers`' names
//!
//! A `transformers` `ModernBertModel` checkpoint (a bare model's, or a
//! `ModernBertFor*`'s under `model.`, its head dropped) as bunsen's
//! [`ModernBertModel`](crate::kits::language::encoders::modernbert::ModernBertModel):
//!
//! - **Config:** [`modernbert_config`] of a parsed `config.json`, and
//!   [`modernbert_config_from_str`] / [`modernbert_config_from_path`].
//! - **Names:** [`modernbert_rekey`], after the fused projections' row splits
//!   ([`modernbert_row_splits`]), its layers by [`modernbert_layer_keys`].
//! - **Loading** (feature `store`): `load_modernbert_tensors`. The kit has no
//!   pretrained of its own: a model built on it (Laya) loads it.

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
