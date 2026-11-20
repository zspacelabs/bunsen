//! `transformers`' `tokenizer_config.json`.
use alloc::{
    collections::BTreeMap,
    string::String,
};

use serde::Deserialize;

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A `transformers` `tokenizer_config.json`, as it names its fields: the
/// tokens and the chat template bunsen reads. Every field is optional;
/// every other key is ignored.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct TransformersTokenizerConfigMirror {
    /// The end-of-sequence token, by its spelling.
    #[serde(default)]
    pub eos_token: Option<TransformersTokenSpelling>,

    /// The added tokens, by id (as the file's keys spell it).
    #[serde(default)]
    pub added_tokens_decoder: Option<BTreeMap<String, TransformersAddedTokenMirror>>,

    /// The chat template: Jinja source, or named templates.
    #[serde(default)]
    pub chat_template: Option<serde_json::Value>,
}

/// A token as `transformers` names one: its spelling, or an added token's
/// record of it.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum TransformersTokenSpelling {
    /// Its spelling.
    Plain(String),

    /// An added token's record, its spelling its `content`.
    Added {
        /// Its spelling.
        #[serde(default)]
        content: Option<String>,
    },
}

/// One of `added_tokens_decoder`'s records.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TransformersAddedTokenMirror {
    /// Its spelling.
    #[serde(default)]
    pub content: Option<String>,
}

impl TransformersTokenizerConfigMirror {
    /// Parses a `tokenizer_config.json`'s bytes.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it is not a JSON object of these fields' types.
    pub fn from_slice(json: &[u8]) -> BunsenResult<Self> {
        let value: serde_json::Value = serde_json::from_slice(json)
            .map_err(|e| BunsenError::invalid_resource("a tokenizer_config.json").with_cause(e))?;
        if !value.is_object() {
            return Err(BunsenError::invalid_resource(
                "a tokenizer_config.json that is not an object",
            ));
        }
        serde_json::from_value(value)
            .map_err(|e| BunsenError::invalid_resource("a tokenizer_config.json").with_cause(e))
    }

    /// The end-of-sequence token's spelling, when the file names one (an
    /// added token's record without a `content` spells it empty).
    pub fn eos_spelling(&self) -> Option<&str> {
        self.eos_token.as_ref().map(|t| match t {
            TransformersTokenSpelling::Plain(s) => s.as_str(),
            TransformersTokenSpelling::Added { content } => content.as_deref().unwrap_or(""),
        })
    }

    /// The id of the added token spelled `spelling`: the first, in the
    /// order of the file's id keys, whose id parses.
    pub fn added_token_id(
        &self,
        spelling: &str,
    ) -> Option<i64> {
        self.added_tokens_decoder
            .as_ref()?
            .iter()
            .find_map(|(id, t)| {
                (t.content.as_deref() == Some(spelling))
                    .then(|| id.parse::<i64>().ok())
                    .flatten()
            })
    }

    /// Whether the file holds a chat template's Jinja source.
    pub fn holds_chat_template(&self) -> bool {
        self.chat_template
            .as_ref()
            .is_some_and(serde_json::Value::is_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_config() {
        let c = TransformersTokenizerConfigMirror::from_slice(
            br#"{"eos_token": "<|im_end|>", "chat_template": "{{ x }}",
                "added_tokens_decoder": {"151643": {"content": "<|endoftext|>"},
                                         "151645": {"content": "<|im_end|>"}}}"#,
        )
        .unwrap();
        assert_eq!(c.eos_spelling(), Some("<|im_end|>"));
        assert_eq!(c.added_token_id("<|im_end|>"), Some(151645));
        assert_eq!(c.added_token_id("?"), None);
        assert!(c.holds_chat_template());
        let object =
            TransformersTokenizerConfigMirror::from_slice(br#"{"eos_token": {"content": "</s>"}}"#)
                .unwrap();
        assert_eq!(object.eos_spelling(), Some("</s>"));
        assert!(!object.holds_chat_template());
    }
}
