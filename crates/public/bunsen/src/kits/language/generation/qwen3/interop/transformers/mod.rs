//! # Qwen3 in `transformers`' names
//!
//! A `Qwen3ForCausalLM` checkpoint as bunsen's [`Qwen3Model`]:
//!
//! - **Config:** [`qwen3_config`] (a `config.json` section, or the
//!   `text_config` of a model a Qwen3 is the text model of), and
//!   [`qwen3_config_from_str`] / [`qwen3_config_from_path`] for a whole file.
//! - **Names:** [`text_layer_rekey`], a decoder layer's names, which the kits
//!   whose text model is a Qwen3 mount; [`qwen3_rekey`], the whole state
//!   dict's.
//! - **Loading** (feature `store`): [`load_qwen3_tensors`].
//!
//! [`Qwen3Model`]: crate::kits::language::generation::qwen3::Qwen3Model

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
