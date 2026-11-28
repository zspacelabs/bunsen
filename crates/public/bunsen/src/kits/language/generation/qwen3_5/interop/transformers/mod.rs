//! # Qwen3.5 in `transformers`' names
//!
//! A dense `transformers` Qwen3.5 checkpoint (a
//! `Qwen3_5ForConditionalGeneration`'s text model, its vision tower and
//! multi-token-prediction head dropped, or a bare `Qwen3_5ForCausalLM`) as
//! bunsen's [`Qwen3_5Model`](crate::kits::language::generation::qwen3_5::Qwen3_5Model):
//!
//! - **Config:** [`qwen3_5_config`] of a parsed `config.json`, and
//!   [`qwen3_5_config_from_str`] / [`qwen3_5_config_from_path`].
//! - **Names:** [`qwen3_5_rekey`], its layers by [`text_layer_rekey`].
//! - **Loading** (feature `store`): `load_qwen3_5_tensors`; with `cache`,
//!   `load_qwen3_5`, a resource set's checkpoint.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod config;
mod rekey;

pub use config::*;
pub use rekey::*;

#[cfg(test)]
mod tests;
