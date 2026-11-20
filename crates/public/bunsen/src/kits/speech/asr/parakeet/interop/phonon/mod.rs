//! # Parakeet as `FermionResearch`'s Phonon-2 archive packs it
//!
//! A Phonon-2 archive holds a Parakeet TDT's weights, in a five-value
//! container (`model.fermion`), and its `NeMo` model config (`config.json`),
//! whose `labels` are the vocabulary. The container is
//! [`data::formats::fermion`](crate::data::formats::fermion); its tensors are
//! named as `transformers`' `ParakeetForTDT` state dict, which
//! [`interop::transformers`](super::transformers)' loader maps onto the model.
//!
//! - **Loading** (feature `store`): `read_phonon2_archive`; with `cache`,
//!   `load_parakeet_tdt`, a resource set's archive.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;
