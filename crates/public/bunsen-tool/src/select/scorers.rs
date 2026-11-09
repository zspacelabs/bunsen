//! The select steps a selection expression's `require` and `prefer` compile to.
use bunsen::{
    data::pretrained::{
        selection::SelectionCandidate,
        tags::LANGUAGE,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};
use serde_json::Value;

use crate::select::{
    LanguageMatch,
    match_language,
};

/// What a scorer makes of a candidate that lacks its key: the call point
/// says, and no value is ever implied. An expression's is `Drop` unless it
/// says otherwise.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Absent {
    /// The candidate is dropped: a score of 0.
    #[default]
    Drop,

    /// The candidate is scored as though it carried this value.
    Default(Value),
}

impl Absent {
    /// The value to score: `value`, or what is absent's stand-in.
    fn resolve<'a>(
        &'a self,
        value: Option<&'a Value>,
    ) -> Option<&'a Value> {
        match (value, self) {
            (Some(v), _) => Some(v),
            (None, Absent::Drop) => None,
            (None, Absent::Default(v)) => Some(v),
        }
    }
}

/// What a `require` or `prefer` entry asks of a tag's value.
#[derive(Clone, Debug, PartialEq)]
pub enum Want {
    /// The value is this one, or a list holding it (`groups: qwen3`).
    Is(Value),

    /// The value is one of these, or a list holding one.
    OneOf(Vec<Value>),

    /// The value is a number within the bounds, either open.
    Range {
        /// The least, if any.
        min: Option<f64>,
        /// The most, if any.
        max: Option<f64>,
    },

    /// The `language` tag handles this BCP 47 tag ([`match_language`]).
    Language(String),
}

impl Want {
    /// What `spec`, an entry's value for `key`, asks: a text for `language`
    /// is a language; a list, one of its members; `{min, max}` (either), a
    /// range; anything else, itself.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for a `language`
    /// that is not text, or a range bound that is not a number.
    pub fn from_spec(
        key: &str,
        spec: &Value,
    ) -> BunsenResult<Want> {
        if key == LANGUAGE.key {
            return match spec {
                Value::String(tag) => Ok(Want::Language(tag.clone())),
                other => Err(BunsenError::policy(format!(
                    "a language is asked for by its BCP 47 tag, not {other}"
                ))),
            };
        }
        match spec {
            Value::Array(members) => Ok(Want::OneOf(members.clone())),
            Value::Object(bounds)
                if !bounds.is_empty() && bounds.keys().all(|k| k == "min" || k == "max") =>
            {
                let bound = |name: &str| -> BunsenResult<Option<f64>> {
                    match bounds.get(name) {
                        None => Ok(None),
                        Some(v) => v.as_f64().map(Some).ok_or_else(|| {
                            BunsenError::policy(format!(
                                "the tag {key:?}'s {name} is not a number: {v}"
                            ))
                        }),
                    }
                };
                Ok(Want::Range {
                    min: bound("min")?,
                    max: bound("max")?,
                })
            }
            other => Ok(Want::Is(other.clone())),
        }
    }

    /// Whether `value` is what is wanted (a language at full strength:
    /// [`LanguageMatch::Filtered`]).
    pub fn matches(
        &self,
        value: &Value,
    ) -> bool {
        let holds = |wanted: &Value| match value {
            Value::Array(items) => value == wanted || items.contains(wanted),
            _ => value == wanted,
        };
        match self {
            Want::Is(wanted) => holds(wanted),
            Want::OneOf(members) => members.iter().any(holds),
            Want::Range { min, max } => value
                .as_f64()
                .is_some_and(|v| min.is_none_or(|m| v >= m) && max.is_none_or(|m| v <= m)),
            Want::Language(tag) => match_language(tag, value) == LanguageMatch::Filtered,
        }
    }
}

/// The `require` step: a candidate scores 1 when every entry's tag is what
/// is wanted, 0 otherwise.
#[derive(Clone, Debug, PartialEq)]
pub struct Require {
    /// The entries: a tag's key, and what is wanted of it.
    pub entries: Vec<(String, Want)>,

    /// What a candidate lacking an entry's key is.
    pub absent: Absent,
}

impl Require {
    /// The score of `candidate`.
    pub fn score(
        &self,
        candidate: &SelectionCandidate,
    ) -> f64 {
        let all = self.entries.iter().all(|(key, want)| {
            self.absent
                .resolve(candidate.get(key))
                .is_some_and(|value| want.matches(value))
        });
        if all { 1.0 } else { 0.0 }
    }
}

/// A `prefer` step: a candidate scores 1 when its tag is what is wanted, and
/// `otherwise` when it is not. For `language`, the match is graded: 1 for a
/// filtered match, `truncated` for one through a truncated request,
/// `otherwise` for the row's `"*"` alone, and 0 for none.
#[derive(Clone, Debug, PartialEq)]
pub struct Prefer {
    /// The tag's key.
    pub key: String,

    /// What is wanted of it.
    pub want: Want,

    /// The score when it is not.
    pub otherwise: f64,

    /// For `language`, the score of a match through a truncated request.
    pub truncated: f64,

    /// What a candidate lacking the key is.
    pub absent: Absent,
}

impl Prefer {
    /// The score of `candidate`.
    pub fn score(
        &self,
        candidate: &SelectionCandidate,
    ) -> f64 {
        let Some(value) = self.absent.resolve(candidate.get(&self.key)) else {
            return 0.0;
        };
        match &self.want {
            Want::Language(tag) => match match_language(tag, value) {
                LanguageMatch::Filtered => 1.0,
                LanguageMatch::Truncated => self.truncated,
                LanguageMatch::Wildcard => self.otherwise,
                LanguageMatch::Missed => 0.0,
            },
            want if want.matches(value) => 1.0,
            _ => self.otherwise,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::select::testing::candidate;

    #[test]
    fn test_wants() {
        let want = |key: &str, spec: Value| Want::from_spec(key, &spec).unwrap();
        assert!(want("base", json!(false)).matches(&json!(false)));
        assert!(!want("base", json!(false)).matches(&json!(true)));
        assert!(want("groups", json!("qwen3")).matches(&json!(["qwen3", "qwen3-2507"])));
        assert!(want("thinking", json!(["always", "switch"])).matches(&json!("switch")));
        assert!(!want("thinking", json!(["always", "switch"])).matches(&json!("never")));
        assert!(want("kit", json!(["qwen3", "gemma"])).matches(&json!("gemma")));
        let range = want("parameters", json!({"max": 8e9}));
        assert!(range.matches(&json!(6e8)) && !range.matches(&json!(9e9)));
        assert!(!range.matches(&json!("big")));
        assert!(want("parameters", json!({"min": 1, "max": 2})).matches(&json!(2)));
        assert!(want("language", json!("en")).matches(&json!(["en-US"])));
        assert!(!want("language", json!("en")).matches(&json!(["*"])));
        assert!(Want::from_spec("language", &json!(["en"])).is_err());
        assert!(Want::from_spec("parameters", &json!({"max": "big"})).is_err());
        assert_eq!(
            want("scores", json!({"wer": 1})),
            Want::Is(json!({"wer": 1}))
        );
    }

    #[test]
    fn test_require_and_absent() {
        let require = |absent: Absent| Require {
            entries: vec![("thinking".into(), Want::Is(json!("never")))],
            absent,
        };
        let never = candidate("a", json!({"thinking": "never"}));
        let switch = candidate("b", json!({"thinking": "switch"}));
        let silent = candidate("c", json!({}));
        assert_eq!(require(Absent::Drop).score(&never), 1.0);
        assert_eq!(require(Absent::Drop).score(&switch), 0.0);
        assert_eq!(require(Absent::Drop).score(&silent), 0.0);
        assert_eq!(require(Absent::Default(json!("never"))).score(&silent), 1.0);
    }

    /// A language preference is graded; any other degrades to `otherwise`.
    #[test]
    fn test_prefer() {
        let language = Prefer {
            key: "language".into(),
            want: Want::Language("en".into()),
            otherwise: 0.8,
            truncated: 0.9,
            absent: Absent::Drop,
        };
        let score = |tags: Value| language.score(&candidate("x", tags));
        assert_eq!(score(json!({"language": ["en"]})), 1.0);
        assert_eq!(score(json!({"language": ["*"]})), 0.8);
        assert_eq!(score(json!({"language": ["de"]})), 0.0);
        assert_eq!(score(json!({})), 0.0);
        let gb = Prefer {
            want: Want::Language("en-GB".into()),
            ..language.clone()
        };
        assert_eq!(gb.score(&candidate("x", json!({"language": ["en"]}))), 0.9);

        let translate = Prefer {
            key: "translate".into(),
            want: Want::Is(json!(true)),
            otherwise: 0.5,
            truncated: 1.0,
            absent: Absent::Default(json!(false)),
        };
        assert_eq!(
            translate.score(&candidate("x", json!({"translate": true}))),
            1.0
        );
        assert_eq!(
            translate.score(&candidate("x", json!({"translate": false}))),
            0.5
        );
        assert_eq!(translate.score(&candidate("x", json!({}))), 0.5);
    }
}
