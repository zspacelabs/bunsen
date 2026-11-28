//! # Sortformer as `NeMo` saves it
//!
//! A `.nemo` of a `SortformerEncLabelModel` (Sortformer v2 and v2.1) as
//! bunsen's [`Sortformer`](crate::kits::speech::diarization::sortformer::Sortformer): its `model_config.yaml`'s mirror,
//! [`NemoSortformerMirror`] (whose sections are the shared
//! [`support::interop::nemo`](crate::kits::support::interop::nemo)'s), the
//! configs it gives
//! ([`sortformer_structure_config`](NemoSortformerMirror::sortformer_structure_config),
//! [`mel_front_end_config`](NemoSortformerMirror::mel_front_end_config)),
//! and its state dict's names ([`sortformer_v2_rekey`]).
//!
//! - **Loading** (feature `store`): `load_sortformer_v2_tensors`; with `cache`,
//!   `load_sortformer_v2`, a resource set's archive. A `.nemo` carries its own
//!   config, so the loader reads it and returns the model's front end with the
//!   model.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod mirror;
mod rekey;

pub use mirror::*;
pub use rekey::*;

#[cfg(test)]
mod tests;
