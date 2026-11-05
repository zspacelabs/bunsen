//! # Pretrained Laya models
//!
//! From a model's name to a loaded [`LayaConstruct`](super::LayaConstruct),
//! through the data layer's pretrained machinery
//! ([`data::pretrained`](crate::data::pretrained)). The whole module needs the
//! `store_safetensors` and `cache` features.
//!
//! # The index
//!
//! [`default_laya_factory`] is the index a caller holds, over the well-known
//! table, [`LAYA_CATALOG`]: the three checkpoints of
//! `convaiinnovations/laya`, each its `model.safetensors` (f16), its
//! encoder's `config.json`, its head's `rl_agent_config.json` and its
//! tokenizer, pinned to a fixed revision:
//!
//! - `laya/laya` (also `laya-en`), the default: English, on `ModernBERT-large`;
//! - `laya/laya-multilingual`: on `mmBERT-base`, with Gemma's 256k vocabulary;
//! - `laya/laya-typed-decisions`: the English model trained on typed questions.
//!
//! # Loading
//!
//! [`load_construct`](crate::kits::support::construct::PretrainedFactory::load_construct)
//! resolves a name to a deferred model carrying [`LayaConstructConfig`],
//! brings every file local through the cache, builds the model from its two
//! configs ([`laya_config`](super::weights::laya_config)), and reads the
//! checkpoint into it
//! ([`load_laya_tensors`](super::weights::load_laya_tensors)) in the dtype the
//! load policy resolves. The tokenizer files are handed back as paths, unread.

mod construct;
mod providers;

pub use construct::*;
pub use providers::*;
