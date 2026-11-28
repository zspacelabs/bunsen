//! # Laya as its release saves it
//!
//! A Laya checkpoint (`convaiinnovations/laya`): a `transformers`
//! `ModernBERT` encoder's `encoder/config.json`, the head's
//! `rl_agent_config.json`, and one `model.safetensors` (the encoder under
//! `encoder.`, the head's `PyTorch` layers under `head.layers.`), as bunsen's
//! [`LayaModel`](crate::kits::language::decision::laya::LayaModel):
//!
//! - **Mirror:** [`LayaAgentMirror`] (`rl_agent_config.json`), its
//!   [`agent_config`](LayaAgentMirror::agent_config) the head and calibration;
//!   [`laya_config`] the model's, from both files.
//! - **Names:** [`laya_rekey`], the encoder's by [`modernbert_rekey`](crate::kits::language::encoders::modernbert::interop::transformers::modernbert_rekey).
//! - **Loading** (feature `store`): `load_laya_tensors`; with `cache`,
//!   `load_laya`, a resource set's checkpoint.

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
