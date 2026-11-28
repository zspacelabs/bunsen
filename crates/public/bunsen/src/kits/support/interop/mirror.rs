//! Reading a JSON mirror.

use std::path::Path;

use crate::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};

/// The JSON file at `path` as the typed mirror `T`: a file a kit reads
/// whole (a `generation_config.json`, a codec's `config.json`). Fields `T`
/// does not name are ignored.
///
/// # Errors
/// When the file can't be read;
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the file, with the parser's message, when it does not fit `T`.
pub fn json_mirror_from_path<T: serde::de::DeserializeOwned>(path: &Path) -> BunsenResult<T> {
    let text = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
    serde_json::from_str(&text)
        .map_err(|e| BunsenError::invalid_resource(format!("{}", path.display())).with_cause(e))
}
