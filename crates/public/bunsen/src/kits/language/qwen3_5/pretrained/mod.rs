//! # Pretrained Qwen3.5 models
//!
//! From a model's name to a loaded
//! [`Qwen3_5Construct`](super::Qwen3_5Construct), through the data layer's
//! pretrained machinery ([`data::pretrained`](crate::data::pretrained)). The
//! whole module needs the `store` and `cache` features.
//!
//! # The index
//!
//! [`default_qwen3_5_factory`] is the index a caller holds, over two
//! providers. First the well-known rows, the built-in catalog's builds of
//! [`QWEN3_5_CATALOG_CONTRACT`], whose geometries are [`QWEN3_5_PREFABS`]';
//! each is a `transformers` checkpoint in bf16 (a
//! `Qwen3_5ForConditionalGeneration`'s, whose text model alone is loaded), its
//! `config.json`, `tokenizer.json` and `tokenizer_config.json`, and a chat
//! model's `chat_template.jinja`:
//!
//! - `qwen/qwen3.5-0.8b`, `-2b`, `-4b` and `-9b`, and their `-base` builds;
//!   `qwen/qwen3.5-0.8b` is the default;
//! - `qwen/qwen3.5-27b`, which no card holds in bf16.
//!
//! Then `hf:org/repo` names any Hugging Face repo in a `transformers`
//! Qwen3.5's layout: its safetensors checkpoint (one file or shards), its
//! `config.json`, which gives the geometry
//! ([`qwen3_5_config_from_transformers`](super::weights::qwen3_5_config_from_transformers)),
//! and those of its tokenizer, generation and template files it has.
//!
//! Every file of a well-known row is pinned to its digest and fetched from a
//! fixed revision.
//!
//! # Presets
//!
//! A construct carries the [`Qwen3_5Policy`](super::Qwen3_5Policy) its presets
//! resolved to. The dense repos have no `generation_config.json`: every row's
//! base holds its `tokenizer_config.json`'s stop tokens (its `eos_token`, and
//! a chat row's `<|endoftext|>`) and its chat template, probed when the
//! catalog is built; a chat row's family adds the model card's sampling under
//! its `thinking-mode` axis, by default what the row's own template does
//! unset: `qwen3.5` (the 4B and up) thinks, `qwen3.5-small` (the 0.8B and 2B)
//! answers at once.
//!
//! # Loading
//!
//! [`load_construct`](crate::kits::support::construct::PretrainedFactory::load_construct)
//! resolves a name to a deferred model carrying [`Qwen3_5ConstructConfig`],
//! brings every file local through the cache, and reads the checkpoint into
//! its geometry through
//! [`load_transformers_tensors_as`](super::weights::load_transformers_tensors_as).
//! The tokenizer and template files are handed back as paths, unread: the
//! kit's model is ids in and logits out.

mod construct;
mod prefabs;
mod providers;

pub use construct::*;
pub use prefabs::*;
pub use providers::*;
