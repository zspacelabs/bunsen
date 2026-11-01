//! # Pretrained Qwen3 models
//!
//! From a model's name to a loaded [`Qwen3Bundle`](super::Qwen3Bundle), through
//! the data layer's pretrained machinery
//! ([`data::pretrained`](crate::data::pretrained)). The whole module needs the
//! `store_safetensors` and `cache` features.
//!
//! # The index
//!
//! [`default_qwen3_factory`] is the index a caller holds, over two providers.
//! First the well-known table, [`WELL_KNOWN_TABLE`], whose rows name their
//! prefab in [`QWEN3_PREFABS`]; each is a `transformers` checkpoint (bf16),
//! its `config.json`, `tokenizer.json`, `tokenizer_config.json` and
//! `generation_config.json`:
//!
//! - `qwen/qwen3-0.6b` (`qwen3-0.6b`): the default, 1.5 GB.
//! - `qwen/qwen3-1.7b`, `qwen/qwen3-4b`: the larger chat models, 4.1 and 8.0
//!   GB.
//! - `qwen/qwen3-0.6b-base`, `qwen/qwen3-1.7b-base`, `qwen/qwen3-4b-base`: the
//!   pretrained models before chat tuning.
//! - `qwen/qwen3-4b-instruct-2507`, `qwen/qwen3-4b-thinking-2507`: `Qwen3-4B`'s
//!   2507 builds, non-thinking and thinking, the `qwen3-4b-2507` prefab.
//! - `qwen/qwen3-8b`, `qwen/qwen3-8b-base`: the 8B chat and pretrained models,
//!   16.4 GB, whose head is their own (the `qwen3-8b` prefab). In bf16 they fit
//!   one 24 GB device; in f32 they do not.
//! - `qwen/qwen3-4b-saferl`: `Qwen3-4B` tuned for safety by reinforcement
//!   learning, 8.0 GB.
//! - `qwen/qwen3guard-gen-0.6b`, `qwen/qwen3guard-gen-4b`,
//!   `qwen/qwen3guard-gen-8b`: safety moderation on the 0.6B, 4B and 8B
//!   geometries (the 8B's head its own). Their chat template asks the model to
//!   rate the last turn, a prompt or a reply: `Safety: Safe`, `Unsafe` or
//!   `Controversial`, and the categories it falls under.
//!
//! Then `hf:org/repo` names any Hugging Face repo in `transformers`'
//! `Qwen3ForCausalLM` layout: its safetensors checkpoint (one file or
//! shards), its `config.json`, which gives the geometry
//! ([`qwen3_config_from_hf`](super::weights::qwen3_config_from_hf)), and
//! those of its tokenizer and generation files it has.
//!
//! Every file of a well-known row is pinned to its digest and fetched from a
//! fixed revision of its Hugging Face repo.
//!
//! # Loading
//!
//! [`load_bundle`](crate::data::pretrained::PretrainedFactory::load_bundle)
//! resolves a name to a deferred model carrying [`Qwen3Construct`], brings
//! every file local through the cache, and reads the checkpoint into its
//! geometry through [`load_hf_tensors`](super::weights::load_hf_tensors). The
//! tokenizer and generation files are handed back as paths, unread: the
//! kit's model is ids in and logits out.

mod construct;
mod prefabs;
mod providers;

pub use construct::*;
pub use prefabs::*;
pub use providers::*;
