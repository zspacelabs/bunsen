//! # `transformers` interop: a `config.json`, for any kit
//!
//! - [`TransformersConfigJson`]: a `transformers` `config.json`, or one section
//!   of it, read field by field, each error naming the field's dotted path. A
//!   kit's `interop::transformers` converts it into the kit's configs.
//! - [`TransformersDecoderSection`]: the fields every decoder section has.
//! - [`TransformersGenerationConfigMirror`] (`generation_config.json`: the stop
//!   ids and the sampling) and [`TransformersTokenizerConfigMirror`]
//!   (`tokenizer_config.json`: the tokens and the chat template), the files the
//!   catalog's probes read.
//! - [`llama_layer_keys`]: the Llama line's decoder-layer key set, which
//!   `blocks`' generic [`CausalDecoderBlock::layer_rekey`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock::layer_rekey) maps onto a block.
//! - [`config_from_str`] and [`config_from_path`]: a conversion over a
//!   `config.json`'s text or file, the second naming the file in its errors:
//!   what a kit's `{target}_config_from_str` and `{target}_config_from_path`
//!   wrap.
//! - [`lm_max_positions`]: the positions a language model was trained on.
//! - **Loading** (feature `store`, with `cache`): `load_transformers_lm`, the
//!   language-model loader a kit's `load_{target}` wraps.

mod config_json;
mod decoder_section;
mod generation_config;
mod layer_keys;
mod tokenizer_config;

pub use config_json::*;
pub use decoder_section::*;
pub use generation_config::*;
pub use layer_keys::*;
pub use tokenizer_config::*;

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod config;

pub use config::*;

#[cfg(test)]
mod tests;
