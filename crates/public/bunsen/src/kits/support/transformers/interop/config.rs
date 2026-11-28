//! Reading a `config.json`.

use std::path::Path;

use super::config_json::TransformersConfigJson;
#[cfg(doc)]
use crate::blocks::transformers::decoders::causal::CausalDecoderBlock;
use crate::errors::{
    BunsenResult,
    sys_at,
};

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
