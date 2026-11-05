//! # Pretrained Gemma models
//!
//! From a model's name to a loaded [`GemmaConstruct`](super::GemmaConstruct),
//! through the data layer's pretrained machinery
//! ([`data::pretrained`](crate::data::pretrained)). The whole module needs the
//! `store_safetensors` and `cache` features.
//!
//! # The index
//!
//! [`default_gemma_factory`] is the index a caller holds, over two providers.
//! First the well-known table, [`GEMMA_CATALOG`], whose rows name their
//! prefab in [`GEMMA_PREFABS`]; each is a `transformers` checkpoint (bf16,
//! but Gemma 2's base models, f32), its `config.json`, `tokenizer.json`,
//! `tokenizer_config.json` and `generation_config.json`, and an
//! instruction-tuned Gemma 3's `chat_template.jinja`, fetched from the
//! `unsloth/` mirror of Google's repo (Google's are gated; the mirrors' files
//! are the same weights, with unsloth's edits to their configs):
//!
//! - Gemma 3: `gemma/gemma-3-270m`, `gemma/gemma-3-1b-pt` and
//!   `gemma/gemma-3-4b-pt`, and their `-it` builds; `gemma/gemma-3-1b-it` is
//!   the default. The 4B is multimodal; its text model alone is loaded.
//! - Gemma 2: `gemma/gemma-2-2b` and `gemma/gemma-2-9b`, and their `-it`
//!   builds.
//! - Gemma 1: `gemma/gemma-2b` and `gemma/gemma-7b`, and their `-it` builds.
//!
//! Then `hf:org/repo` names any Hugging Face repo in a `transformers` Gemma's
//! layout: its safetensors checkpoint (one file or shards), its
//! `config.json`, which gives the geometry
//! ([`gemma_config_from_hf`](super::weights::gemma_config_from_hf)), and
//! those of its tokenizer, generation and template files it has.
//!
//! Every file of a well-known row is pinned to its digest and fetched from a
//! fixed revision of its mirror.
//!
//! # Presets
//!
//! A construct carries the [`GemmaPolicy`](super::GemmaPolicy) its presets
//! resolved to: its
//! [`GenerationPolicy`](crate::kits::support::model_config::GenerationPolicy),
//! for a chat driver. Every row's base holds what its `generation_config.json`
//! names (the stop tokens, and the sampling), probed when the catalog is
//! built; an `hf:` repo's is probed when it loads.
//!
//! # Loading
//!
//! [`load_construct`](crate::kits::support::construct::PretrainedFactory::load_construct)
//! resolves a name to a deferred model carrying [`GemmaConstructConfig`],
//! brings every file local through the cache, and reads the checkpoint into
//! its geometry through
//! [`load_hf_tensors_as`](super::weights::load_hf_tensors_as). The
//! tokenizer, generation and template files are handed back as paths,
//! unread: the kit's model is ids in and logits out.

mod construct;
mod prefabs;
mod providers;

pub use construct::*;
pub use prefabs::*;
pub use providers::*;
