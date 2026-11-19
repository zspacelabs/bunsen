//! A `tokenizers` normalizer behind [`TextNormalizer`].

use alloc::{
    borrow::Cow,
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

use serde_json::Value;
use tokenizers::{
    NormalizedString,
    Normalizer,
    normalizers::NormalizerWrapper,
};

use super::{
    BunsenLanguageTag,
    TextNormalizer,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};

/// A normalizer of the `tokenizers` crate (feature `tokenizer`), run on its
/// own: Unicode forms (`NFC`, `NFKC`, ...), `Lowercase`, `Strip`,
/// `StripAccents`, `Replace`, `Prepend`, a model's `Precompiled` table, and
/// `Sequence`s of them. Any language: these normalizers read none.
///
/// Built from a spec in `tokenizer.json`'s syntax, or from a
/// `tokenizer.json`'s `normalizer` ([`TokenizersSpec`](super::TokenizersSpec)
/// names either as config). A tokenizer that runs its own normalizer
/// inside `encode` already does this; this is for running one as a text
/// stage, before another normalizer or in place of the tokenizer's.
#[derive(Debug)]
pub struct TokenizersTextNormalizer {
    normalizer: NormalizerWrapper,
    name: String,
}

impl TokenizersTextNormalizer {
    /// The normalizer of `spec`, in `tokenizer.json`'s syntax.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a spec that
    /// is not a normalizer.
    pub fn from_spec(spec: &Value) -> BunsenResult<Self> {
        let normalizer =
            serde_json::from_value::<NormalizerWrapper>(spec.clone()).map_err(|error| {
                BunsenError::illegal(format!("not a tokenizers normalizer: {error}"))
            })?;
        Ok(Self {
            normalizer,
            name: format!("tokenizers({})", spec_name(spec)),
        })
    }

    /// The `normalizer` of the `tokenizer.json` at `path`.
    ///
    /// # Errors
    /// [`sys_at`]'s for a file that cannot be read (a missing one is a
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup));
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for one that is not JSON, or has no normalizer; and
    /// [`from_spec`](Self::from_spec)'s.
    pub fn from_tokenizer_json(path: &std::path::Path) -> BunsenResult<Self> {
        let text = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
        let json: Value = serde_json::from_str(&text).map_err(|error| {
            BunsenError::invalid_resource(format!("{} is not JSON: {error}", path.display()))
        })?;
        match json.get("normalizer") {
            Some(spec) if !spec.is_null() => Self::from_spec(spec),
            _ => Err(BunsenError::invalid_resource(format!(
                "{} has no normalizer",
                path.display()
            ))),
        }
    }
}

/// A spec's name: its `type`, with a `Sequence`'s parts.
fn spec_name(spec: &Value) -> String {
    let kind = spec
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string();
    match spec.get("normalizers").and_then(Value::as_array) {
        Some(parts) if kind == "Sequence" => {
            let parts: Vec<String> = parts.iter().map(spec_name).collect();
            parts.join(", ")
        }
        _ => kind,
    }
}

impl TextNormalizer for TokenizersTextNormalizer {
    /// `tokenizers(NFKC, Lowercase)`: the spec's types.
    fn name(&self) -> String {
        self.name.clone()
    }

    fn normalize<'a>(
        &self,
        text: &'a str,
        _language: Option<&BunsenLanguageTag>,
    ) -> BunsenResult<Cow<'a, str>> {
        let mut normalized = NormalizedString::from(text);
        self.normalizer
            .normalize(&mut normalized)
            .map_err(|error| BunsenError::illegal(format!("{}: {error}", self.name)))?;
        Ok(Cow::Owned(normalized.get().to_string()))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::errors::BunsenErrorKind;

    #[test]
    fn test_from_spec() {
        let normalizer = TokenizersTextNormalizer::from_spec(&json!({
            "type": "Sequence",
            "normalizers": [{"type": "NFKC"}, {"type": "Lowercase"}, {"type": "StripAccents"}],
        }))
        .unwrap();
        assert_eq!(
            normalizer.name(),
            "tokenizers(NFKC, Lowercase, StripAccents)"
        );
        // NFKC folds the ligature and the full-width letters; the accent is
        // kept, as StripAccents strips only decomposed marks.
        assert_eq!(
            normalizer.normalize("ﬁne ＡＢ Café", None).unwrap(),
            "fine ab café"
        );

        let error = TokenizersTextNormalizer::from_spec(&json!({"type": "Nope"})).unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Illegal);
    }

    #[test]
    fn test_from_tokenizer_json() {
        let dir = tempfile::tempdir().unwrap();
        let with = dir.path().join("with.json");
        std::fs::write(
            &with,
            json!({"normalizer": {"type": "Lowercase"}, "model": {}}).to_string(),
        )
        .unwrap();
        let normalizer = super::super::TokenizersSpec::TokenizerJson(with.display().to_string())
            .init()
            .unwrap();
        assert_eq!(normalizer.name(), "tokenizers(Lowercase)");
        assert_eq!(normalizer.normalize("ABC", None).unwrap(), "abc");

        let without = dir.path().join("without.json");
        std::fs::write(&without, json!({"normalizer": null}).to_string()).unwrap();
        let error = TokenizersTextNormalizer::from_tokenizer_json(&without).unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::InvalidResource);

        let error = TokenizersTextNormalizer::from_tokenizer_json(&dir.path().join("missing.json"))
            .unwrap_err();
        assert_eq!(error.kind(), BunsenErrorKind::Lookup);
    }
}
