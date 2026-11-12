//! # Pretrained Qwen3 models
//!
//! From a model's name to a loaded [`Qwen3Construct`](super::Qwen3Construct),
//! through the data layer's pretrained machinery
//! ([`data::pretrained`](crate::data::pretrained)). The whole module needs the
//! `store` and `cache` features.
//!
//! # The index
//!
//! [`default_qwen3_factory`] is the index a caller holds, over two providers.
//! First the catalog rows, the built-in catalog's builds of
//! [`QWEN3_CATALOG_CONTRACT`], whose geometries are
//! [`Qwen3Config`](crate::kits::language::qwen3::Qwen3Config)'s prefabs'; each
//! is a `transformers` checkpoint (bf16), its `config.json`, `tokenizer.json`,
//! `tokenizer_config.json` and `generation_config.json`:
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
//! ([`qwen3_config_from_transformers`](super::weights::qwen3_config_from_transformers)), and
//! those of its tokenizer and generation files it has.
//!
//! Every file of a catalog row is pinned to its digest and fetched from a
//! fixed revision of its Hugging Face repo.
//!
//! # Presets
//!
//! A construct carries the [`Qwen3Policy`](super::Qwen3Policy) its presets
//! resolved to: its
//! [`GenerationPolicy`](crate::kits::support::model_config::GenerationPolicy),
//! for a chat driver. Every row's base holds what its `generation_config.json`
//! names (the stop tokens, and the sampling), probed when the catalog is
//! built. The hybrid chat models (`qwen3-0.6b` through `qwen3-8b`, and
//! `qwen3-4b-saferl`) are the catalog's `hybrid` family, with a
//! `thinking-mode` axis: `[thinking]`, the default (the chat template's own,
//! which the validation crate checks), and `[no-thinking]`, which switches
//! the template's `enable_thinking` off and samples at the card's
//! non-thinking settings. The pretrained models, the guard models and the
//! fixed-mode 2507 builds have no axis. An `hf:` repo's
//! `generation_config.json` is probed into its base when it loads.
//!
//! # Loading
//!
//! [`load_construct`](crate::data::builds::BuildFactory::load_construct)
//! resolves a name to a deferred model carrying [`Qwen3ConstructConfig`],
//! brings every file local through the cache, and reads the checkpoint into its
//! geometry through
//! [`load_transformers_tensors`](super::weights::load_transformers_tensors).
//! The tokenizer and generation files are handed back as paths, unread: the
//! kit's model is ids in and logits out.

mod construct;
mod prefabs;
mod providers;

pub use construct::*;
pub use providers::*;
