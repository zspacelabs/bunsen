//! # Tags: what a pretrained model is, as facts a selection can read
//!
//! A [`Pretrained`](super::Pretrained) row carries [`Tags`]: a typeless map
//! from keys to JSON values (`thinking: "switch"`, `language: ["en", "*"]`,
//! `parameters: 600000000`). Tags are distinct from presets: a preset is a
//! delta over the config a model is built from; a tag is a fact about the
//! model, for listings and for choosing among models.
//!
//! The keys form a loosely closed, shared vocabulary. Shared keys are bare
//! (`language`); a key only one kit uses is prefixed with the kit's name and a
//! dot (`whisper.multilingual`). Values are any JSON; each key's shape is a
//! convention of the vocabulary. Nothing at run time refuses a tag.
//!
//! A build catalog writes a row's tags from its build's tag fields, and a
//! [`BuildCatalogProvider`](super::BuildCatalogProvider) writes the row's
//! [`KIT`] tag. A factory writes the [`KIT`] tag on a row of any provider
//! that lacks one, an `hf:` repo's among them.
//!
//! ## The vocabulary
//!
//! Each shared key is a [`TagDef`]: its key, its description, the shape of
//! its value ([`TagShape`]), and who writes it ([`TagAuthor`]). They are
//! data and docs; bunsen runs no checks of them.
//!
//! | key | shape | written by |
//! |---|---|---|
//! | [`kit`](KIT) | text: the kit's name | the catalog, and a factory |
//! | [`groups`](GROUPS) | a list of names: the variation groups the model is part of | the row |
//! | [`version`](VERSION) | `{group, version}`: a release in a sequence | the row |
//! | [`language`](LANGUAGE) | a list of canonical BCP 47 tags; `"*"` for unspecified others | the row |
//! | [`parameters`](PARAMETERS) | a count: total parameters, tied weights once | the row |
//! | [`active_parameters`](ACTIVE_PARAMETERS) | a count: per-token active parameters (mixture of experts) | the row |
//! | [`dtype`](DTYPE) | `f32`, `f16` or `bf16`: the stored weights' element type | the catalog, else the row |
//! | [`scores`](SCORES) | `{name: number}` | the row |
//! | [`reasoning`](REASONING) | bool: trained to reason | the row |
//! | [`thinking`](THINKING) | `always`, `switch` or `never`: a thinking block, and whether it switches | the row |
//! | [`base`](BASE) | bool: a pretrained model, not tuned to follow instructions | the row |
//! | [`translate`](TRANSLATE) | bool: it can translate | the row |

use alloc::string::{
    String,
    ToString,
};

use serde::{
    Deserialize,
    Serialize,
    de::DeserializeOwned,
};
use serde_json::{
    Map,
    Value,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    support::merge_patch,
};

mod vocabulary;

pub use vocabulary::*;

/// A pretrained row's tags: keys to JSON values.
///
/// Serialized as the map itself (`{"thinking": "switch"}`). Keys are kept in
/// key order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Tags(Map<String, Value>);

impl Tags {
    /// No tags.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// How many there are.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// The value of `key`.
    pub fn get(
        &self,
        key: impl AsRef<str>,
    ) -> Option<&Value> {
        self.0.get(key.as_ref())
    }

    /// Whether `key` is present.
    pub fn contains_key(
        &self,
        key: impl AsRef<str>,
    ) -> bool {
        self.0.contains_key(key.as_ref())
    }

    /// Sets `key` to `value`, and returns what it was.
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Value>,
    ) -> Option<Value> {
        self.0.insert(key.into(), value.into())
    }

    /// These tags with `key` set to `value`.
    pub fn with(
        mut self,
        key: impl Into<String>,
        value: impl Into<Value>,
    ) -> Self {
        self.insert(key, value);
        self
    }

    /// Removes `key`, and returns what it was.
    pub fn remove(
        &mut self,
        key: impl AsRef<str>,
    ) -> Option<Value> {
        self.0.remove(key.as_ref())
    }

    /// The tags, in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The value of `key`, read as a `T`: `None` when it is absent.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it is present but not a `T`: a present value of the wrong shape
    /// is an error, not a miss.
    pub fn get_as<T: DeserializeOwned>(
        &self,
        key: impl AsRef<str>,
    ) -> Option<BunsenResult<T>> {
        let key = key.as_ref();
        let value = self.0.get(key)?;
        Some(serde_json::from_value(value.clone()).map_err(|e| {
            BunsenError::invalid_resource(format!("the tag {key:?}: {value}")).with_cause(e)
        }))
    }

    /// Whether `key` is `true`.
    pub fn is_true(
        &self,
        key: impl AsRef<str>,
    ) -> bool {
        self.0.get(key.as_ref()) == Some(&Value::Bool(true))
    }

    /// The value of `key` as text, when it is a string.
    pub fn text(
        &self,
        key: impl AsRef<str>,
    ) -> Option<&str> {
        self.0.get(key.as_ref()).and_then(Value::as_str)
    }

    /// Whether `key` is the string `text`, or a list holding it.
    pub fn contains_text(
        &self,
        key: impl AsRef<str>,
        text: &str,
    ) -> bool {
        match self.0.get(key.as_ref()) {
            Some(Value::String(s)) => s == text,
            Some(Value::Array(items)) => items.iter().any(|v| v.as_str() == Some(text)),
            _ => false,
        }
    }

    /// The value of `key` as a number, when it is one.
    pub fn number(
        &self,
        key: impl AsRef<str>,
    ) -> Option<f64> {
        self.0.get(key.as_ref()).and_then(Value::as_f64)
    }

    /// These tags laid over `base`: an RFC 7386 merge patch
    /// ([`merge_patch::apply`]), as presets lay a row's over its family's.
    /// A key of these replaces `base`'s, a `null` removes `base`'s, and two
    /// objects merge by key (a family's `scores` and a row's combine).
    pub fn over(
        &self,
        base: &Tags,
    ) -> Tags {
        let mut merged = Value::Object(base.0.clone());
        merge_patch::apply(&mut merged, &Value::Object(self.0.clone()));
        match merged {
            Value::Object(map) => Tags(map),
            _ => unreachable!("an object patch over an object is an object"),
        }
    }
}

impl<K: ToString, V: Into<Value>> FromIterator<(K, V)> for Tags {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Tags(
            iter.into_iter()
                .map(|(k, v)| (k.to_string(), v.into()))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tags(value: Value) -> Tags {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn test_reads() {
        let t = tags(json!({
            "thinking": "switch",
            "reasoning": true,
            "language": ["en", "*"],
            "parameters": 600000000u64,
        }));
        assert_eq!(t.len(), 4);
        assert_eq!(t.text("thinking"), Some("switch"));
        assert!(t.is_true("reasoning"));
        assert!(!t.is_true("thinking"));
        assert!(!t.is_true("absent"));
        assert!(t.contains_text("language", "en"));
        assert!(t.contains_text("language", "*"));
        assert!(!t.contains_text("language", "de"));
        assert!(t.contains_text("thinking", "switch"));
        assert_eq!(t.number("parameters"), Some(6e8));
        assert_eq!(t.number("thinking"), None);

        assert_eq!(t.get_as::<u64>("parameters").unwrap().unwrap(), 600_000_000);
        assert!(t.get_as::<u64>("absent").is_none());
        assert!(
            t.get_as::<u64>("thinking").unwrap().is_err(),
            "the wrong shape"
        );
    }

    /// A row's tags over its family's: keys replace, `null` removes, and
    /// objects merge by key.
    #[test]
    fn test_over() {
        let family = tags(json!({
            "thinking": "never",
            "base": false,
            "scores": {"a": 1.0, "b": 2.0},
        }));
        let row = tags(json!({
            "thinking": "switch",
            "base": null,
            "scores": {"b": 3.0, "c": 4.0},
        }));
        assert_eq!(
            row.over(&family),
            tags(json!({
                "thinking": "switch",
                "scores": {"a": 1.0, "b": 3.0, "c": 4.0},
            }))
        );
        assert_eq!(Tags::new().over(&family), family);
        assert_eq!(row.over(&Tags::new()).get("base"), None, "a null removes");
    }

    #[test]
    fn test_round_trip_and_build() {
        let t = Tags::new().with(KIT, "qwen3").with("parameters", 8u64);
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, r#"{"kit":"qwen3","parameters":8}"#);
        assert_eq!(serde_json::from_str::<Tags>(&json).unwrap(), t);
        let collected: Tags = [("kit", json!("qwen3")), ("parameters", json!(8))]
            .into_iter()
            .collect();
        assert_eq!(collected, t);
        let mut t = t;
        assert_eq!(t.remove(KIT), Some(json!("qwen3")));
        assert!(!t.contains_key(KIT));
        assert_eq!(t.iter().map(|(k, _)| k).collect::<Vec<_>>(), ["parameters"]);
    }
}
