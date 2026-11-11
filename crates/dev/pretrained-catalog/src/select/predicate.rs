//! Typed predicates: what a `where` entry asks of a declared axis or tag.
//!
//! An entry's key resolves to a declaration ([`Subject`]), and the
//! declaration says what its values may be:
//!
//! | subject | `x` | `[x, y]` | object forms |
//! |---|---|---|---|
//! | bool axis | `true` / `false`: offers that member | n/a | `switch`, `always`, `never` (as values) |
//! | set-valued (categorical axis offers, `text_list`, `languages`) | contains `x` | contains all | `{any: [..]}`, `{only: [..]}` |
//! | scalar-valued (bool, `count`, `text`, `one_of`, a derived key) | equals `x` | equals any | `{min, max}` (count, derived) |
//!
//! `{is: x}` is `x`, everywhere. A value that doesn't fit its subject is an
//! error when the query compiles.

use serde_json::Value;

use crate::{
    document::TagType,
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogResult,
    },
    name::CatalogName,
    select::core::SelectionCandidate,
};

/// What a `where` key resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Subject {
    /// A bool axis, by its full name.
    BoolAxis {
        /// The axis.
        axis: CatalogName,
    },
    /// A categorical axis: what a build's use of it offers.
    CategoricalAxis {
        /// The axis.
        axis: CatalogName,
        /// Its members.
        members: Vec<String>,
    },
    /// A declared tag.
    Tag {
        /// The tag's full name.
        key: String,
        /// Its type.
        tag_type: TagType,
    },
    /// A key a derive step declares: untyped, tested as a scalar.
    Derived {
        /// The key.
        key: String,
    },
}

impl Subject {
    /// The key, for messages.
    pub fn key(&self) -> String {
        match self {
            Subject::BoolAxis { axis } | Subject::CategoricalAxis { axis, .. } => axis.to_string(),
            Subject::Tag { key, .. } | Subject::Derived { key } => key.clone(),
        }
    }
}

/// A bool axis's tested state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolState {
    /// Offers the positive member.
    Positive,
    /// Offers the negative member.
    Negative,
    /// Offers both.
    Switch,
    /// Offers the positive member only.
    Always,
    /// Offers the negative member only.
    Never,
}

/// The test a predicate applies.
#[derive(Clone, Debug, PartialEq)]
pub enum Test {
    /// A bool axis's state.
    Bool(BoolState),
    /// A set contains every one of `all`, at least one of `any` (when there
    /// are any), and, with `only`, nothing else.
    Set {
        /// Members it must contain.
        all: Vec<Value>,
        /// Members it must contain one of.
        any: Vec<Value>,
        /// The exact set, when given.
        only: Option<Vec<Value>>,
    },
    /// A value equals one of these.
    Equals(Vec<Value>),
    /// A number within the bounds.
    Range {
        /// The least, if any.
        min: Option<f64>,
        /// The most, if any.
        max: Option<f64>,
    },
}

/// A compiled `where` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct Predicate {
    /// What it tests.
    pub subject: Subject,
    /// How.
    pub test: Test,
}

fn bad(
    key: &str,
    why: impl std::fmt::Display,
) -> CatalogError {
    CatalogError::new(CatalogErrorKind::Selection, format!("`where.{key}`: {why}"))
}

/// `{is: x}` is `x`.
fn unwrap_is(value: &Value) -> &Value {
    match value {
        Value::Object(fields) if fields.len() == 1 && fields.contains_key("is") => &fields["is"],
        other => other,
    }
}

impl Predicate {
    /// Compiles `value` for `subject`, checking it against the subject's
    /// declaration.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] when the value doesn't fit.
    pub fn compile(
        subject: Subject,
        value: &Value,
    ) -> CatalogResult<Predicate> {
        let key = subject.key();
        let value = unwrap_is(value);
        let test = match &subject {
            Subject::BoolAxis { .. } => Test::Bool(match value {
                Value::Bool(true) => BoolState::Positive,
                Value::Bool(false) => BoolState::Negative,
                Value::String(s) if s == "switch" => BoolState::Switch,
                Value::String(s) if s == "always" => BoolState::Always,
                Value::String(s) if s == "never" => BoolState::Never,
                other => {
                    return Err(bad(
                        &key,
                        format!(
                            "a bool axis takes `true`, `false`, `switch`, `always` or `never`, \
                             not {other}"
                        ),
                    ));
                }
            }),
            Subject::CategoricalAxis { members, .. } => {
                let test = set_test(&key, value)?;
                if let Test::Set { all, any, only } = &test {
                    for member in all.iter().chain(any).chain(only.iter().flatten()) {
                        let ok = member
                            .as_str()
                            .is_some_and(|m| members.iter().any(|x| x == m));
                        if !ok {
                            return Err(bad(
                                &key,
                                format!("{member} is not one of the axis's members {members:?}"),
                            ));
                        }
                    }
                }
                test
            }
            Subject::Tag { tag_type, .. } => match tag_type {
                TagType::TextList | TagType::Languages => {
                    let test = set_test(&key, value)?;
                    if let Test::Set { all, any, only } = &test
                        && !all
                            .iter()
                            .chain(any)
                            .chain(only.iter().flatten())
                            .all(Value::is_string)
                    {
                        return Err(bad(&key, "its members are text"));
                    }
                    test
                }
                TagType::Bool => {
                    scalar_test(&key, value, &|v| v.is_boolean(), false, "`true` or `false`")?
                }
                TagType::Count => scalar_test(&key, value, &|v| v.is_number(), true, "a number")?,
                TagType::Text => scalar_test(&key, value, &|v| v.is_string(), false, "text")?,
                TagType::OneOf(choices) => scalar_test(
                    &key,
                    value,
                    &|v| v.as_str().is_some_and(|s| choices.iter().any(|c| c == s)),
                    false,
                    &format!("one of {choices:?}"),
                )?,
                TagType::Scores => {
                    return Err(bad(&key, "a `scores` tag can't be tested; name one score"));
                }
            },
            Subject::Derived { .. } => scalar_test(&key, value, &|_| true, true, "a value")?,
        };
        Ok(Predicate { subject, test })
    }

    /// Whether `candidate` passes.
    pub fn holds(
        &self,
        candidate: &SelectionCandidate,
    ) -> bool {
        self.grade(candidate, 0.0, 0.0) == 1.0
    }

    /// `candidate`'s score as a preference: 1 when it passes, `otherwise`
    /// when it doesn't (or lacks the subject). A language is graded: a match
    /// through a truncated request scores `truncated`, a match only through
    /// the build's `"*"` scores `otherwise`; with several languages, the
    /// least of their grades.
    pub fn grade(
        &self,
        candidate: &SelectionCandidate,
        truncated: f64,
        otherwise: f64,
    ) -> f64 {
        let yes = |b: bool| if b { 1.0 } else { otherwise };
        match (&self.subject, &self.test) {
            (Subject::BoolAxis { axis }, Test::Bool(state)) => {
                let Some(using) = candidate.build.presets.axis_use(axis) else {
                    return otherwise;
                };
                let positive = axis.leaf().to_string();
                let on = using.offers.contains(&positive);
                let off = using.offers.iter().any(|o| *o != positive);
                yes(match state {
                    BoolState::Positive => on,
                    BoolState::Negative => off,
                    BoolState::Switch => on && off,
                    BoolState::Always => on && !off,
                    BoolState::Never => off && !on,
                })
            }
            (Subject::CategoricalAxis { axis, .. }, test) => {
                let Some(using) = candidate.build.presets.axis_use(axis) else {
                    return otherwise;
                };
                let offers: Vec<Value> = using.offers.iter().cloned().map(Value::String).collect();
                yes(set_holds(test, &offers, &|a, b| a == b))
            }
            (
                Subject::Tag {
                    key,
                    tag_type: TagType::Languages,
                },
                Test::Set { all, any, only },
            ) => {
                let Some(value) = candidate.get(key) else {
                    return otherwise;
                };
                if only.is_none() && any.is_empty() && !all.is_empty() {
                    // Graded, the least of each requested language's grade.
                    return all
                        .iter()
                        .filter_map(Value::as_str)
                        .map(|lang| match match_language(lang, value) {
                            LanguageMatch::Filtered => 1.0,
                            LanguageMatch::Truncated => truncated,
                            LanguageMatch::Wildcard => otherwise,
                            LanguageMatch::Missed => 0.0,
                        })
                        .fold(1.0, f64::min);
                }
                let items = value.as_array().cloned().unwrap_or_default();
                yes(set_holds(&self.test, &items, &|wanted, have| {
                    let (Some(w), Some(_)) = (wanted.as_str(), have.as_str()) else {
                        return false;
                    };
                    match_language(w, &Value::Array(vec![have.clone()])) == LanguageMatch::Filtered
                }))
            }
            (Subject::Tag { key, .. } | Subject::Derived { key }, test) => {
                let Some(value) = candidate.get(key) else {
                    return otherwise;
                };
                yes(match test {
                    Test::Set { .. } => {
                        let items = value.as_array().cloned().unwrap_or_default();
                        set_holds(test, &items, &|a, b| a == b)
                    }
                    Test::Equals(any) => any.contains(value),
                    Test::Range { min, max } => value
                        .as_f64()
                        .is_some_and(|v| min.is_none_or(|m| v >= m) && max.is_none_or(|m| v <= m)),
                    Test::Bool(_) => false,
                })
            }
            (Subject::BoolAxis { .. }, _) => otherwise,
        }
    }
}

/// The test for a set-valued subject: `x` contains it; a list contains all;
/// `{any}` and `{only}`.
fn set_test(
    key: &str,
    value: &Value,
) -> CatalogResult<Test> {
    match value {
        Value::Array(items) => Ok(Test::Set {
            all: items.clone(),
            any: Vec::new(),
            only: None,
        }),
        Value::Object(fields) => {
            let mut test = Test::Set {
                all: Vec::new(),
                any: Vec::new(),
                only: None,
            };
            for (k, v) in fields {
                let Value::Array(items) = v else {
                    return Err(bad(key, format!("`{k}` takes a list, not {v}")));
                };
                let Test::Set { any, only, .. } = &mut test else {
                    unreachable!()
                };
                match k.as_str() {
                    "any" => *any = items.clone(),
                    "only" => *only = Some(items.clone()),
                    other => {
                        return Err(bad(
                            key,
                            format!("a set takes `{{any}}` or `{{only}}`, not `{other}`"),
                        ));
                    }
                }
            }
            Ok(test)
        }
        scalar => Ok(Test::Set {
            all: vec![scalar.clone()],
            any: Vec::new(),
            only: None,
        }),
    }
}

/// Whether `items` passes a set test, with `same` deciding membership.
fn set_holds(
    test: &Test,
    items: &[Value],
    same: &dyn Fn(&Value, &Value) -> bool,
) -> bool {
    let Test::Set { all, any, only } = test else {
        return false;
    };
    let contains = |wanted: &Value| items.iter().any(|have| same(wanted, have));
    all.iter().all(contains)
        && (any.is_empty() || any.iter().any(contains))
        && only.as_ref().is_none_or(|only| {
            only.iter().all(contains) && items.iter().all(|have| only.iter().any(|w| same(w, have)))
        })
}

/// The test for a scalar-valued subject: equals `x`; equals any of a list;
/// `{min, max}` when `ranged`.
fn scalar_test(
    key: &str,
    value: &Value,
    fits: &dyn Fn(&Value) -> bool,
    ranged: bool,
    what: &str,
) -> CatalogResult<Test> {
    match value {
        Value::Array(items) => {
            if let Some(item) = items.iter().find(|v| !fits(v)) {
                return Err(bad(key, format!("{item} is not {what}")));
            }
            Ok(Test::Equals(items.clone()))
        }
        Value::Object(bounds)
            if ranged && !bounds.is_empty() && bounds.keys().all(|k| k == "min" || k == "max") =>
        {
            let bound = |name: &str| -> CatalogResult<Option<f64>> {
                match bounds.get(name) {
                    None => Ok(None),
                    Some(v) => v
                        .as_f64()
                        .map(Some)
                        .ok_or_else(|| bad(key, format!("its {name} is not a number: {v}"))),
                }
            };
            Ok(Test::Range {
                min: bound("min")?,
                max: bound("max")?,
            })
        }
        Value::Object(_) => Err(bad(
            key,
            if ranged {
                "an object here is `{min, max}`".to_string()
            } else {
                format!("this tag takes {what}, or a list of them")
            },
        )),
        scalar if fits(scalar) => Ok(Test::Equals(vec![scalar.clone()])),
        other => Err(bad(key, format!("{other} is not {what}"))),
    }
}

/// How a requested language matches a build's languages, best first
/// (RFC 4647).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LanguageMatch {
    /// A tag the request filters to (`en` matches `en`, `en-US`).
    Filtered,
    /// A tag matched only once the request is truncated (`en-GB` to `en`).
    Truncated,
    /// Only the build's `"*"`: unspecified other languages.
    Wildcard,
    /// None of the build's languages.
    Missed,
}

/// `request` (a BCP 47 tag, or `"*"`) against a `languages` value,
/// case-insensitively.
pub fn match_language(
    request: &str,
    languages: &Value,
) -> LanguageMatch {
    let tags: Vec<String> = match languages {
        Value::String(s) => vec![s.to_ascii_lowercase()],
        Value::Array(items) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_ascii_lowercase)
            .collect(),
        _ => Vec::new(),
    };
    let request = request.to_ascii_lowercase();
    let wildcard = tags.iter().any(|t| t == "*");
    let named: Vec<&str> = tags
        .iter()
        .map(String::as_str)
        .filter(|t| *t != "*")
        .collect();
    if request == "*" {
        return if tags.is_empty() {
            LanguageMatch::Missed
        } else {
            LanguageMatch::Filtered
        };
    }
    let filters = |range: &str, tag: &str| {
        tag == range || (tag.starts_with(range) && tag.as_bytes().get(range.len()) == Some(&b'-'))
    };
    if named.iter().any(|t| filters(&request, t)) {
        return LanguageMatch::Filtered;
    }
    let mut range = request.as_str();
    while let Some(cut) = range.rfind('-') {
        range = &range[..cut];
        if let Some(cut) = range.rfind('-')
            && range.len() - cut == 2
        {
            range = &range[..cut];
        }
        if named.contains(&range) {
            return LanguageMatch::Truncated;
        }
    }
    if wildcard {
        LanguageMatch::Wildcard
    } else {
        LanguageMatch::Missed
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::select::testing::candidate;

    fn thinking() -> Subject {
        Subject::BoolAxis {
            axis: CatalogName::root("thinking").unwrap(),
        }
    }

    fn task() -> Subject {
        Subject::CategoricalAxis {
            axis: CatalogName::root("task").unwrap(),
            members: vec!["transcribe".into(), "translate".into()],
        }
    }

    fn tag(
        key: &str,
        tag_type: TagType,
    ) -> Subject {
        Subject::Tag {
            key: key.into(),
            tag_type,
        }
    }

    fn holds(
        subject: Subject,
        value: Value,
        build: &str,
    ) -> bool {
        Predicate::compile(subject, &value)
            .unwrap()
            .holds(&candidate(build))
    }

    #[test]
    fn test_bool_axis_states() {
        // qwen3-0.6b switches; the base never thinks; 2507 always thinks.
        assert!(holds(thinking(), json!(false), "qwen3-0.6b"));
        assert!(holds(thinking(), json!(false), "qwen3-0.6b-base"));
        assert!(!holds(thinking(), json!(false), "qwen3-4b-thinking-2507"));
        assert!(holds(thinking(), json!("switch"), "qwen3-0.6b"));
        assert!(!holds(thinking(), json!("switch"), "qwen3-0.6b-base"));
        assert!(holds(thinking(), json!("always"), "qwen3-4b-thinking-2507"));
        assert!(holds(thinking(), json!("never"), "qwen3-0.6b-base"));
        // A build without the axis offers nothing on it.
        assert!(!holds(thinking(), json!(true), "large-v3"));
        assert!(Predicate::compile(thinking(), &json!("maybe")).is_err());
    }

    #[test]
    fn test_categorical_axis_sets() {
        assert!(holds(task(), json!("translate"), "large-v1"));
        assert!(holds(
            task(),
            json!(["transcribe", "translate"]),
            "large-v1"
        ));
        assert!(!holds(
            task(),
            json!(["transcribe", "translate"]),
            "large-v3-turbo"
        ));
        assert!(holds(
            task(),
            json!({"only": ["transcribe"]}),
            "large-v3-turbo"
        ));
        assert!(!holds(task(), json!({"only": ["transcribe"]}), "large-v1"));
        assert!(holds(
            task(),
            json!({"any": ["translate", "transcribe"]}),
            "large-v3-turbo"
        ));
        let err = Predicate::compile(task(), &json!("summarize")).unwrap_err();
        assert!(
            err.message.contains("not one of the axis's members"),
            "{err}"
        );
    }

    #[test]
    fn test_scalar_tags() {
        let params = || tag("parameters", TagType::Count);
        assert!(holds(params(), json!({"max": 1e9}), "qwen3-0.6b"));
        assert!(!holds(params(), json!({"max": 1e9}), "large-v3"));
        assert!(holds(params(), json!([600000000u64, 1]), "qwen3-0.6b"));
        let instruct = || tag("instruct", TagType::Bool);
        assert!(holds(instruct(), json!(false), "qwen3-0.6b-base"));
        assert!(holds(instruct(), json!({"is": true}), "qwen3-0.6b"));
        for bad_value in [json!("yes"), json!({"min": 1}), json!([true, "no"])] {
            assert!(
                Predicate::compile(instruct(), &bad_value).is_err(),
                "{bad_value}"
            );
        }
        assert!(Predicate::compile(params(), &json!("big")).is_err());
        assert!(Predicate::compile(tag("scores", TagType::Scores), &json!(1)).is_err());
    }

    #[test]
    fn test_languages() {
        let lang = || tag("language", TagType::Languages);
        assert!(holds(lang(), json!("en"), "large-v1"));
        assert!(!holds(lang(), json!("en"), "qwen3-0.6b"));
        let p = Predicate::compile(lang(), &json!("en-GB")).unwrap();
        assert_eq!(p.grade(&candidate("large-v1"), 0.75, 0.25), 0.75);
        let p = Predicate::compile(lang(), &json!("de")).unwrap();
        assert_eq!(p.grade(&candidate("qwen3-0.6b"), 0.75, 0.25), 0.25);
        let both = Predicate::compile(lang(), &json!(["en", "de"])).unwrap();
        assert_eq!(both.grade(&candidate("large-v1"), 0.75, 0.25), 0.25);
    }

    #[test]
    fn test_language_matching() {
        let langs = json!(["en", "zh-Hans", "*"]);
        assert_eq!(match_language("en", &langs), LanguageMatch::Filtered);
        assert_eq!(match_language("zh", &langs), LanguageMatch::Filtered);
        assert_eq!(match_language("en-GB", &langs), LanguageMatch::Truncated);
        assert_eq!(match_language("fr", &langs), LanguageMatch::Wildcard);
        assert_eq!(match_language("fr", &json!(["en"])), LanguageMatch::Missed);
        assert_eq!(
            match_language("de-x-y", &json!(["de"])),
            LanguageMatch::Truncated
        );
    }
}
