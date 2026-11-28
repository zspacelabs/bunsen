//! # nanochat as upstream's checkpoint manager saves it
//!
//! Upstream nanochat (`karpathy/nanochat`) writes a step as two files: the
//! model's state dict, `model_<step>.pt` (`torch.save`), and its
//! `meta_<step>.json`, whose `model_config` is the model's geometry. As
//! bunsen's [`NanoChatGpt`](crate::kits::language::generation::nanochat::NanoChatGpt):
//!
//! - **Mirrors:** [`NanoChatMetaMirror`] (the meta) and
//!   [`NanoChatModelConfigMirror`] (its `model_config`, upstream's
//!   `GPTConfig`); [`generation`](NanoChatMetaMirror::generation) tells from
//!   the state dict's names which of upstream's models it is
//!   ([`NanoChatGeneration`]), and
//!   [`gpt_config`](NanoChatMetaMirror::gpt_config) is the config for both.
//! - **Names:** [`nanochat_gpt_rekey`].
//! - **Loading** (feature `store`): `load_nanochat_gpt_tensors`, and
//!   `load_nanochat_gpt_from_paths`, a meta and a state dict; with `cache`,
//!   `load_nanochat_gpt`, a resource set's.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod config;
mod mirror;
mod rekey;

pub use config::*;
pub use mirror::*;
pub use rekey::*;

#[cfg(test)]
mod tests;
