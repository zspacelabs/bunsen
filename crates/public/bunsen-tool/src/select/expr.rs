//! Selection expressions: the `--select {..}` object, and what it compiles to.
use bunsen::{
    data::pretrained::selection::{
        Order,
        Selection,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    support::merge_patch,
};
use serde::Deserialize;
use serde_json::{
    Map,
    Value,
};

use crate::select::{
    Absent,
    Latest,
    Prefer,
    Require,
    Want,
};

/// The keys a `prefer` entry may carry beside its one tag.
const PREFER_PARAMS: &[&str] = &["otherwise", "truncated", "absent"];

/// A selection expression: a YAML or JSON object of fixed keys, each
/// optional, applied in one order whatever the order they are written in:
/// every `require` and `prefer`, then `latest`, then `order`, then `take`.
///
/// ```yaml
/// {
///   name: qwen3,                        # rows that answer to the name
///   require: {base: false, thinking: [always, switch], parameters: {max: 8e9}},
///   prefer: [{language: en, truncated: 0.9, otherwise: 0.8}],
///   latest: {step: 0.05, floor: 0.0},   # or `true`, for these
///   absent: drop,                       # or {default: value}
///   order: [weight, -parameters],       # most significant first; `-` descending
///   take: 0.5                           # the cutoff; absent, 0
/// }
/// ```
///
/// - `require`: per tag, a value (`base: false`; a list's member, `groups:
///   qwen3`), a list of acceptable values, or `{min, max}` bounds; a `language`
///   is a BCP 47 tag, matched at full strength. A candidate that fails one is
///   dropped.
/// - `prefer`: a list of entries, each one tag as in `require`, with
///   `otherwise` (the score when it is not met; default 0), `truncated` (for
///   `language`, a match through a truncated request; default 1) and its own
///   `absent`. A `language` preference grades: a filtered match 1, a truncated
///   one `truncated`, the row's `"*"` alone `otherwise`, none 0.
/// - `latest`: [`Latest`], each release sequence's newest first.
/// - `absent`: what a candidate lacking a key is, for every entry without its
///   own: `drop` (the default) or `{default: value}`.
/// - `order`: tag keys whose values are numbers, and `weight`.
/// - `take`: the least weight the taken candidate has.
///
/// Expressions combine as RFC 7386 merge patches ([`merge`]): a later one's
/// `require` entries replace an earlier one's of the same tag, `null`
/// removes one, and every other key replaces whole.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectionExpr {
    /// The name rows answer to, matched as `--select NAME` is.
    pub name: Option<String>,

    /// The `require` entries, in key order.
    pub require: Vec<(String, Value)>,

    /// The `prefer` entries.
    pub prefer: Vec<PreferEntry>,

    /// The `latest` step, when on.
    pub latest: Option<Latest>,

    /// The default for absent keys.
    pub absent: Absent,

    /// The order keys, most significant first.
    pub order: Vec<(String, Order)>,

    /// The cutoff.
    pub take: Option<f64>,
}

/// One `prefer` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferEntry {
    /// The tag.
    pub key: String,

    /// What is wanted of it, as written.
    pub spec: Value,

    /// The score when it is not met.
    pub otherwise: f64,

    /// For `language`, the score of a truncated match.
    pub truncated: f64,

    /// Its own `absent`, if it has one.
    pub absent: Option<Absent>,
}

/// The object as written, before its parts are read.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    require: Map<String, Value>,
    #[serde(default)]
    prefer: Vec<Map<String, Value>>,
    #[serde(default)]
    latest: Option<Value>,
    #[serde(default)]
    absent: Option<Value>,
    #[serde(default)]
    order: Vec<String>,
    #[serde(default)]
    take: Option<f64>,
}

/// An `absent` value: `drop`, or `{default: value}`.
fn parse_absent(value: &Value) -> BunsenResult<Absent> {
    match value {
        Value::String(s) if s == "drop" => Ok(Absent::Drop),
        Value::Object(fields) if fields.len() == 1 && fields.contains_key("default") => {
            Ok(Absent::Default(fields["default"].clone()))
        }
        other => Err(BunsenError::policy(format!(
            "`absent` is `drop` or `{{default: value}}`, not {other}"
        ))),
    }
}

/// A score parameter: a number from 0 to 1.
fn parse_score(
    what: &str,
    value: &Value,
) -> BunsenResult<f64> {
    value
        .as_f64()
        .filter(|v| (0.0..=1.0).contains(v))
        .ok_or_else(|| BunsenError::policy(format!("{what} is a score from 0 to 1, not {value}")))
}

impl SelectionExpr {
    /// The expression `value` writes.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for an object with
    /// an unknown key, a part of the wrong shape, or a `prefer` entry with
    /// other than one tag.
    pub fn from_value(value: &Value) -> BunsenResult<Self> {
        let raw: Raw = serde_json::from_value(value.clone())
            .map_err(|e| BunsenError::policy(format!("a selection expression: {e}")))?;
        let absent = raw
            .absent
            .as_ref()
            .map(parse_absent)
            .transpose()?
            .unwrap_or_default();
        let mut prefer = Vec::new();
        for entry in raw.prefer {
            let tags: Vec<(&String, &Value)> = entry
                .iter()
                .filter(|(k, _)| !PREFER_PARAMS.contains(&k.as_str()))
                .collect();
            let [(key, spec)] = tags.as_slice() else {
                return Err(BunsenError::policy(format!(
                    "a `prefer` entry names one tag, beside {}: {}",
                    PREFER_PARAMS.join(", "),
                    Value::Object(entry.clone())
                )));
            };
            prefer.push(PreferEntry {
                key: (*key).clone(),
                spec: (*spec).clone(),
                otherwise: entry
                    .get("otherwise")
                    .map(|v| parse_score("`otherwise`", v))
                    .transpose()?
                    .unwrap_or(0.0),
                truncated: entry
                    .get("truncated")
                    .map(|v| parse_score("`truncated`", v))
                    .transpose()?
                    .unwrap_or(1.0),
                absent: entry.get("absent").map(parse_absent).transpose()?,
            });
        }
        let latest = match raw.latest {
            None | Some(Value::Null) | Some(Value::Bool(false)) => None,
            Some(Value::Bool(true)) => Some(Latest::default()),
            Some(Value::Object(params)) if params.keys().all(|k| k == "step" || k == "floor") => {
                let mut latest = Latest::default();
                if let Some(step) = params.get("step") {
                    latest.step = parse_score("`latest.step`", step)?;
                }
                if let Some(floor) = params.get("floor") {
                    latest.floor = parse_score("`latest.floor`", floor)?;
                }
                Some(latest)
            }
            Some(other) => {
                return Err(BunsenError::policy(format!(
                    "`latest` is `true`, `false` or `{{step, floor}}`, not {other}"
                )));
            }
        };
        let order = raw
            .order
            .iter()
            .map(|key| match key.strip_prefix('-') {
                Some(key) => (key.to_string(), Order::Descending),
                None => (key.to_string(), Order::Ascending),
            })
            .collect();
        if let Some(take) = raw.take
            && !(take.is_finite() && take >= 0.0)
        {
            return Err(BunsenError::policy(format!(
                "`take` is a weight, 0 or more, not {take}"
            )));
        }
        let expr = SelectionExpr {
            name: raw.name,
            require: raw.require.into_iter().collect(),
            prefer,
            latest,
            absent,
            order,
            take: raw.take,
        };
        // Every entry's want must read now, not when a candidate meets it.
        expr.steps()?;
        Ok(expr)
    }

    /// The `require` and `prefer` steps.
    fn steps(&self) -> BunsenResult<(Require, Vec<Prefer>)> {
        let require = Require {
            entries: self
                .require
                .iter()
                .map(|(key, spec)| Ok((key.clone(), Want::from_spec(key, spec)?)))
                .collect::<BunsenResult<_>>()?,
            absent: self.absent.clone(),
        };
        let prefer = self
            .prefer
            .iter()
            .map(|entry| {
                Ok(Prefer {
                    key: entry.key.clone(),
                    want: Want::from_spec(&entry.key, &entry.spec)?,
                    otherwise: entry.otherwise,
                    truncated: entry.truncated,
                    absent: entry.absent.clone().unwrap_or_else(|| self.absent.clone()),
                })
            })
            .collect::<BunsenResult<_>>()?;
        Ok((require, prefer))
    }

    /// Runs the expression's steps over `selection`, all but `take`.
    pub fn apply(
        &self,
        selection: Selection,
    ) -> Selection {
        let (require, prefer) = self.steps().expect("checked when the expression was read");
        let mut selection = if require.entries.is_empty() {
            selection
        } else {
            selection.select_each(|c| require.score(c))
        };
        for step in &prefer {
            selection = selection.select_each(|c| step.score(c));
        }
        if let Some(latest) = &self.latest {
            selection = selection.select(latest);
        }
        // A tuple order is stable sorts, least significant first.
        for (key, direction) in self.order.iter().rev() {
            selection = if key == "weight" {
                selection.order(&[*direction], |_, w| [Some(w)])
            } else {
                selection.order(&[*direction], |c, _| [c.number(key)])
            };
        }
        selection
    }

    /// The cutoff `take` names; 0 when it names none, the first survivor.
    pub fn cutoff(&self) -> f64 {
        self.take.unwrap_or(0.0)
    }
}

/// `objects` merged in order, each an RFC 7386 merge patch over those before
/// it: a selection's defaults, then the expressions over them.
pub fn merge<'a>(objects: impl IntoIterator<Item = &'a Value>) -> Value {
    let mut merged = Value::Object(Map::new());
    for object in objects {
        merge_patch::apply(&mut merged, object);
    }
    merged
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::select::testing::{
        candidate,
        names,
    };

    fn expr(value: Value) -> SelectionExpr {
        SelectionExpr::from_value(&value).unwrap()
    }

    fn rows() -> Selection {
        Selection::new([
            candidate(
                "qwen3-0.6b",
                json!({"base": false, "thinking": "switch", "parameters": 6e8}),
            ),
            candidate(
                "qwen3-0.6b-base",
                json!({"base": true, "thinking": "never", "parameters": 6e8}),
            ),
            candidate(
                "qwen3-8b",
                json!({"base": false, "thinking": "switch", "parameters": 8e9}),
            ),
            candidate(
                "gemma",
                json!({"base": false, "thinking": "never", "parameters": 1e9,
                                      "language": ["en", "*"]}),
            ),
            candidate("nameless", json!({"base": false})),
        ])
    }

    #[test]
    fn test_require_order_take() {
        let e = expr(json!({
            "require": {"base": false, "thinking": ["always", "switch"]},
            "order": ["-parameters"],
        }));
        let s = e.apply(rows());
        assert_eq!(names(&s), ["qwen3-8b", "qwen3-0.6b"]);
        assert_eq!(e.cutoff(), 0.0);

        let small = expr(json!({"require": {"parameters": {"max": 2e9}}, "order": ["parameters"]}));
        assert_eq!(
            names(&small.apply(rows())),
            ["qwen3-0.6b", "qwen3-0.6b-base", "gemma"]
        );
    }

    /// `absent` decides what a candidate lacking a key is: dropped, or given a
    /// value.
    #[test]
    fn test_absent() {
        let drop = expr(json!({"require": {"thinking": "never"}}));
        assert_eq!(names(&drop.apply(rows())), ["qwen3-0.6b-base", "gemma"]);
        let default = expr(json!({
            "require": {"thinking": "never"},
            "absent": {"default": "never"},
        }));
        assert_eq!(
            names(&default.apply(rows())),
            ["qwen3-0.6b-base", "gemma", "nameless"]
        );
    }

    /// A preference degrades instead of refusing; `weight` orders the
    /// degraded after the rest.
    #[test]
    fn test_prefer_and_weight_order() {
        let e = expr(json!({
            "require": {"base": false},
            "prefer": [{"thinking": "never", "otherwise": 0.5, "absent": {"default": "switch"}}],
            "order": ["-weight", "-parameters"],
        }));
        let s = e.apply(rows());
        assert_eq!(names(&s), ["gemma", "qwen3-8b", "qwen3-0.6b", "nameless"]);
        let weights: Vec<f64> = s.iter().map(|(_, w)| w).collect();
        assert_eq!(weights, [1.0, 0.5, 0.5, 0.5]);
        assert_eq!(e.take.map(|_| ()), None, "no take: the first survivor");
    }

    #[test]
    fn test_refusals() {
        for bad in [
            json!({"select": {}}),
            json!({"prefer": [{"a": 1, "b": 2}]}),
            json!({"prefer": [{"otherwise": 0.5}]}),
            json!({"prefer": [{"a": 1, "otherwise": 2}]}),
            json!({"latest": "yes"}),
            json!({"latest": {"step": 0.1, "speed": 1}}),
            json!({"absent": "keep"}),
            json!({"take": -1}),
            json!({"require": {"language": ["en"]}}),
            json!({"require": {"parameters": {"max": "big"}}}),
        ] {
            assert!(SelectionExpr::from_value(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn test_parts() {
        let e = expr(json!({
            "name": "qwen3",
            "latest": {"step": 0.1},
            "order": ["weight", "-parameters"],
            "take": 0.5,
        }));
        assert_eq!(e.name.as_deref(), Some("qwen3"));
        assert_eq!(
            e.latest,
            Some(Latest {
                step: 0.1,
                floor: 0.0
            })
        );
        assert_eq!(
            e.order,
            [
                ("weight".to_string(), Order::Ascending),
                ("parameters".to_string(), Order::Descending)
            ]
        );
        assert_eq!(e.cutoff(), 0.5);
        assert_eq!(
            expr(json!({"latest": true})).latest,
            Some(Latest::default())
        );
        assert_eq!(expr(json!({"latest": false})).latest, None);
        assert_eq!(expr(json!({})), SelectionExpr::default());
    }

    /// Defaults merge under the expressions: a `require` entry replaces the
    /// default's of the same tag, `null` removes one, and the rest replace
    /// whole.
    #[test]
    fn test_merge() {
        let defaults = json!({"require": {"base": false}, "latest": true});
        let merged = merge([&defaults, &json!({"require": {"thinking": "switch"}})]);
        assert_eq!(
            merged,
            json!({"require": {"base": false, "thinking": "switch"}, "latest": true})
        );
        let merged = merge([
            &defaults,
            &json!({"require": {"base": null}, "latest": null}),
        ]);
        assert_eq!(merged, json!({"require": {}}));
        let merged = merge([&defaults, &json!({"require": {"base": true}})]);
        assert_eq!(merged["require"]["base"], json!(true));
        assert_eq!(merge([]), json!({}));
    }
}
