//! # Pretrained Parakeet models
//!
//! From a model's name to a loaded [`ParakeetBundle`](super::ParakeetBundle),
//! through the data layer's pretrained machinery
//! ([`data::pretrained`](crate::data::pretrained)). The whole module needs
//! the `store_safetensors` and `cache` features; the Phonon archive reader
//! needs `fermion` as well.
//!
//! # The index
//!
//! [`default_parakeet_factory`] is the index a caller holds, over one
//! provider, [`WELL_KNOWN_TABLE`]:
//!
//! - `fermion/phonon-2` (`phonon-2`, `phonon`): `FermionResearch`'s Phonon-2,
//!   NVIDIA's `parakeet-tdt-0.6b-v3` with its encoder re-trained to a
//!   five-value alphabet, as one 164 MB archive. The default.
//! - `nvidia/parakeet-tdt-0.6b-v3` (`parakeet-tdt-0.6b-v3`): the base model,
//!   `transformers`' 2.5 GB `model.safetensors` and its `tokenizer.json`.
//!
//! Every file is pinned to its digest and fetched from a fixed revision of
//! its Hugging Face repo.
//!
//! # Loading
//!
//! [`load_bundle`](crate::data::pretrained::PretrainedFactory::load_bundle)
//! resolves a name to a deferred model carrying [`ParakeetConstruct`], the
//! hook its checkpoint's `kind` calls for, brings every file local through
//! the cache, and reads them into a `parakeet-tdt-0.6b-v3`-shaped
//! [`ParakeetTdt`](super::ParakeetTdt) through
//! [`load_hf_tensors`](super::weights::load_hf_tensors).

mod construct;
mod providers;

pub use construct::*;
pub use providers::*;
