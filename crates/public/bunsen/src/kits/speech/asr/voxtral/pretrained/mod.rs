//! # Pretrained Voxtral Realtime models
//!
//! From a model's name to a loaded
//! [`VoxtralConstruct`](super::VoxtralConstruct), through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)). The
//! whole module needs the `store_safetensors` and `cache` features.
//!
//! # The index
//!
//! [`default_voxtral_factory`] is the index a caller holds, over two
//! providers. First the well-known table, [`VOXTRAL_CATALOG`], whose row
//! names its prefab in [`VOXTRAL_PREFABS`]:
//!
//! - `mistralai/voxtral-mini-4b-realtime-2602` (`voxtral-mini-4b-realtime`):
//!   Mistral's 4B realtime model, its 8.9 GB `bf16` `model.safetensors` and its
//!   `tekken.json`. The default.
//!
//! Then `hf:org/repo` names any Hugging Face repo in the published layout:
//! its safetensors checkpoint (one file or shards), its `config.json`, which
//! gives the geometry
//! ([`voxtral_config_from_hf`](super::weights::voxtral_config_from_hf)),
//! and its `tekken.json`, which gives the vocabulary and the stream's layout.
//!
//! Every file is pinned to its digest and fetched from a fixed revision of
//! its Hugging Face repo.
//!
//! # Loading
//!
//! [`load_construct`](crate::kits::support::construct::PretrainedFactory::load_construct)
//! resolves a name to a deferred model carrying [`VoxtralConstructConfig`],
//! brings every file local through the cache, and reads them into the row's
//! prefab through [`load_hf_tensors`](super::weights::load_hf_tensors).

mod construct;
mod prefabs;
mod providers;

pub use construct::*;
pub use prefabs::*;
pub use providers::*;
