//! # Qwen3-ASR in `transformers`' names
//!
//! A `Qwen3ASRForConditionalGeneration` checkpoint as bunsen's
//! [`Qwen3Asr`](crate::kits::speech::asr::qwen3_asr::Qwen3Asr):
//!
//! - **Config:** [`qwen3_asr_config`] of a parsed `config.json`, and
//!   [`qwen3_asr_config_from_str`] / [`qwen3_asr_config_from_path`].
//! - **Front end:** [`Qwen3AsrFeatureExtractorMirror`], a
//!   `processor_config.json`'s (`Qwen3AsrProcessorMirror`) or a
//!   `preprocessor_config.json`'s feature extractor.
//! - **Names:** [`qwen3_asr_rekey`]; its text model's layers are Qwen3's
//!   ([`text_layer_rekey`](crate::kits::language::generation::qwen3::interop::transformers::text_layer_rekey)).
//! - **Loading** (feature `store`): `load_qwen3_asr_tensors`; with `cache`,
//!   `load_qwen3_asr`, a resource set's checkpoint and `tokenizer.json`
//!   vocabulary.

mod feature_extractor;
pub use feature_extractor::*;

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
