//! A checkpoint's generation defaults.
use std::path::Path;

use bunsen::{
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::decoding::SearchConfig,
};
use serde_json::Value;

/// What a checkpoint's `generation_config.json` says a generation does: the
/// tokens that end it, and how it samples.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationDefaults {
    /// The tokens that end a reply (`eos_token_id`, one or several).
    pub stop_ids: Vec<i64>,

    /// The search: sampling at the file's temperature, top-k and top-p when
    /// it samples (`do_sample`), the argmax otherwise.
    pub search: SearchConfig,
}

impl GenerationDefaults {
    /// The defaults of a `generation_config.json` file.
    ///
    /// # Errors
    /// [`Sys`](bunsen::errors::BunsenErrorKind::Sys) or
    /// [`Lookup`](bunsen::errors::BunsenErrorKind::Lookup) when it cannot be
    /// read; as [`from_json`](Self::from_json).
    pub fn from_file(path: &Path) -> BunsenResult<Self> {
        let json = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
        Self::from_json(&json).map_err(|e| e.context(path.display()))
    }

    /// The defaults of a `generation_config.json`.
    ///
    /// # Errors
    /// [`InvalidResource`](bunsen::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse, or names no stop token.
    pub fn from_json(json: &str) -> BunsenResult<Self> {
        let c: Value = serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("a generation_config.json").with_cause(e))?;
        let stop_ids: Vec<i64> = match c.get("eos_token_id") {
            Some(Value::Array(ids)) => ids.iter().filter_map(Value::as_i64).collect(),
            Some(id) => id.as_i64().into_iter().collect(),
            None => Vec::new(),
        };
        if stop_ids.is_empty() {
            return Err(BunsenError::invalid_resource(
                "a generation_config.json without `eos_token_id`",
            ));
        }
        let sample = c.get("do_sample").and_then(Value::as_bool).unwrap_or(false);
        let search = if sample {
            SearchConfig::new()
                .with_temperature(c.get("temperature").and_then(Value::as_f64).unwrap_or(1.0))
                .with_top_k(c.get("top_k").and_then(Value::as_u64).map(|k| k as usize))
                .with_top_p(c.get("top_p").and_then(Value::as_f64))
                .with_min_p(c.get("min_p").and_then(Value::as_f64))
        } else {
            SearchConfig::new()
        };
        Ok(Self { stop_ids, search })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_json() {
        let qwen3 = r#"{"bos_token_id": 151643, "do_sample": true, "eos_token_id": [151645, 151643],
            "pad_token_id": 151643, "temperature": 0.6, "top_k": 20, "top_p": 0.95}"#;
        let d = GenerationDefaults::from_json(qwen3).unwrap();
        assert_eq!(d.stop_ids, [151645, 151643]);
        assert_eq!(d.search.temperature, 0.6);
        assert_eq!(d.search.top_k, Some(20));
        assert_eq!(d.search.top_p, Some(0.95));

        let base = r#"{"bos_token_id": 151643, "do_sample": false, "eos_token_id": 151643}"#;
        let d = GenerationDefaults::from_json(base).unwrap();
        assert_eq!(d.stop_ids, [151643]);
        assert_eq!(d.search, SearchConfig::new());

        assert!(GenerationDefaults::from_json("{}").is_err());
    }
}
