//! # `transformers`: a `config.json`, for any kit
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
//!   `blocks`' generic [`CausalDecoderBlock::layer_rekey`] maps onto a block.
//! - [`config_from_str`] and [`config_from_path`]: a conversion over a
//!   `config.json`'s text or file, the second naming the file in its errors:
//!   what a kit's `{target}_config_from_str` and `{target}_config_from_path`
//!   wrap.
//! - [`lm_max_positions`]: the positions a language model was trained on.
//! - **Loading** (feature `store`, with `cache`): `load_transformers_lm`, the
//!   language-model loader a kit's `load_{target}` wraps.

use std::path::Path;

#[cfg(doc)]
use crate::blocks::transformers::decoders::causal::CausalDecoderBlock;
use crate::errors::{
    BunsenResult,
    sys_at,
};

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

/// The positions a language model was trained on: its
/// `max_position_embeddings`, its `text_config`'s where it has one (a
/// multimodal model's); `None` when the file names none.
pub fn lm_max_positions(c: &TransformersConfigJson) -> Option<usize> {
    let text = c.section("text_config").ok();
    text.as_ref()
        .unwrap_or(c)
        .usize("max_position_embeddings")
        .ok()
}

/// `convert` over a `config.json`'s text.
///
/// # Errors
/// As [`TransformersConfigJson::parse`], and as `convert`.
pub fn config_from_str<T>(
    json: &str,
    convert: impl FnOnce(&TransformersConfigJson) -> BunsenResult<T>,
) -> BunsenResult<T> {
    convert(&TransformersConfigJson::parse(json)?)
}

/// `convert` over the `config.json` at `path`; every error names the file.
///
/// # Errors
/// When the file can't be read, and as [`config_from_str`].
pub fn config_from_path<T>(
    path: &Path,
    convert: impl FnOnce(&TransformersConfigJson) -> BunsenResult<T>,
) -> BunsenResult<T> {
    let json = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
    config_from_str(&json, convert).map_err(|e| e.context(path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::testing::{
        ErrorMatcher,
        text,
    };

    /// The text helper parses and converts; a conversion's error passes
    /// through.
    #[test]
    fn test_config_from_str() {
        let n = config_from_str(r#"{"vocab_size": 7}"#, |t| t.usize("vocab_size")).unwrap();
        assert_eq!(n, 7);
        ErrorMatcher::invalid_resource()
            .display(text::contains("missing_field"))
            .assert_err(&config_from_str(r#"{}"#, |t| t.usize("missing_field")));
    }

    /// The file helper names the file in a conversion's error, and in a
    /// read's.
    #[test]
    fn test_config_from_path_names_the_file() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let missing = dir.join("no-such-config.json");
        assert!(
            config_from_path(&missing, |t| t.usize("vocab_size"))
                .unwrap_err()
                .to_string()
                .contains("no-such-config.json")
        );
    }
}
