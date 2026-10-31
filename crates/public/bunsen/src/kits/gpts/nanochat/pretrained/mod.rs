//! # Pretrained nanochat models
//!
//! From a model's name to a loaded [`NanoChatBundle`](super::NanoChatBundle),
//! through the data layer's pretrained machinery
//! ([`data::pretrained`](crate::data::pretrained)). The whole module needs the
//! `store_pytorch` and `cache` features.
//!
//! # The index
//!
//! [`default_nanochat_factory`] is the index a caller holds, over the
//! well-known table, [`WELL_KNOWN_TABLE`], whose rows name their prefab in
//! [`NANOCHAT_PREFABS`]. There are no official nanochat releases; each row is
//! a community upload of upstream's own checkpoint files, unconverted: a
//! `model_<step>.pt` (about 2 GB), its `meta_<step>.json`, and the tokenizer
//! (`tokenizer.pkl`, `token_bytes.pt`).
//!
//! - `nanochat-students/rl-d20` (`nanochat-d20-rl`): the default, a d20 (561M)
//!   of upstream's 2025 model after its RL stage.
//! - `nanochat-students/base-d20`, `nanochat-students/mid-d20`
//!   (`nanochat-d20-base`, `nanochat-d20-mid`): the same run after pretraining
//!   and after midtraining.
//! - `nitishpandey04/nanochat-d18-base`, `nitishpandey04/nanochat-d18-sft`
//!   (`nanochat-d18-base`, `nanochat-d18-sft`): a d18 of upstream's 2026 model,
//!   after pretraining and after SFT.
//!
//! Every file is pinned to its digest and fetched from a fixed revision of
//! its Hugging Face repo. nanochat repos follow no one layout (the file
//! names carry the training step; some keep the checkpoint in a
//! subdirectory), so there is no `hf:` provider.
//!
//! # Loading
//!
//! [`load_bundle`](crate::data::pretrained::PretrainedFactory::load_bundle)
//! resolves a name to a deferred model carrying [`NanoChatConstruct`], brings
//! every file local through the cache, and reads the checkpoint through
//! [`load_pt_checkpoint`](super::weights::load_pt_checkpoint): the meta's
//! geometry, which of upstream's models the state dict is, and the weights.
//! The tokenizer files are handed back as paths, unread: the kit's model is
//! ids in and logits out.

mod construct;
mod prefabs;
mod providers;

pub use construct::*;
pub use prefabs::*;
pub use providers::*;
