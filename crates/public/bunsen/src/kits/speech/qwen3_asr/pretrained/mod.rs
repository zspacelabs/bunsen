//! # Pretrained Qwen3-ASR models
//!
//! From a model's name to a loaded [`Qwen3AsrBundle`](super::Qwen3AsrBundle),
//! through the data layer's pretrained machinery
//! ([`data::pretrained`](crate::data::pretrained)). The whole module needs
//! the `store_safetensors` and `cache` features; the Phonon archive reader
//! needs `fermion` as well.
//!
//! # The index
//!
//! [`default_qwen3_asr_factory`] is the index a caller holds, over two
//! providers. First the well-known table, [`WELL_KNOWN_TABLE`], whose rows
//! name their prefab in [`QWEN3_ASR_PREFABS`]:
//!
//! - `fermion/phonon-1` (`phonon-1`): `FermionResearch`'s Phonon-1, an English
//!   fine-tune of `Qwen3-ASR-0.6B` (other speech comes out as broken English),
//!   its text decoder in a five-symbol code and its audio tower 6-bit, as one
//!   415 MB archive. The default.
//! - `fermion/phonon-1-micro` (`phonon-1-micro`): the 285 MB build, its audio
//!   tower and embedding 4-bit.
//! - `fermion/phonon-1-big` (`phonon-1-big`): the 581 MB build, its audio tower
//!   and embedding `bf16`.
//! - `qwen/qwen3-asr-0.6b` (`qwen3-asr-0.6b`): the base model, `transformers`'
//!   1.6 GB `bf16` `model.safetensors` and its `tokenizer.json`.
//! - `qwen/qwen3-asr-1.7b` (`qwen3-asr-1.7b`): the 1.7B model, 4.1 GB `bf16`,
//!   the `qwen3-asr-1.7b` prefab.
//!
//! Then `hf:org/repo` names any Hugging Face repo in
//! `transformers`' `Qwen3ASRForConditionalGeneration` layout: its
//! safetensors checkpoint (one file or shards), its `config.json`, which
//! gives the geometry
//! ([`qwen3_asr_config_from_hf`](super::weights::qwen3_asr_config_from_hf)),
//! and its `tokenizer.json`.
//!
//! Every file is pinned to its digest and fetched from a fixed revision of
//! its Hugging Face repo.
//!
//! # Loading
//!
//! [`load_bundle`](crate::data::pretrained::PretrainedFactory::load_bundle)
//! resolves a name to a deferred model carrying [`Qwen3AsrConstruct`], the
//! hook its checkpoint's `kind` calls for, brings every file local through
//! the cache, and reads them into the row's prefab through
//! [`load_hf_tensors`](super::weights::load_hf_tensors).

mod construct;
mod prefabs;
mod providers;

pub use construct::*;
pub use prefabs::*;
pub use providers::*;
