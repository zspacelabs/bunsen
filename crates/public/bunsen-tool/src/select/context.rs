//! A command's selection context, and the choice it makes.
use bunsen::{
    data::pretrained::selection::{
        Selection,
        SelectionCandidate,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};
use serde_json::Value;

use crate::select::{
    NameMatch,
    SelectionExpr,
    match_name,
    merge,
};

/// A selection a command names: `--select en`.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedSelection {
    /// Its name.
    pub name: String,

    /// A line for a listing.
    pub description: String,

    /// The selection, a [`SelectionExpr`]'s object, merged over the
    /// command's defaults.
    pub selection: Value,
}

/// What a command chooses its model with: the default selection every
/// choice merges over, and the selections it names.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectionContext {
    /// The default selection: applied unless the choice states its keys
    /// (a merge patch over it), and not at all to a row named exactly. A
    /// weighting step (`latest`, a `prefer`) chooses only when an `order`
    /// puts the heavier first (`order: [-weight]`): `take` takes the first
    /// candidate, in order, at or above its cutoff.
    pub defaults: Value,

    /// The named selections.
    pub named: Vec<NamedSelection>,
}

/// One `--select`: a name, or a selection object.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectArg {
    /// A name: a named selection, a row, or a prefix, group or kit of rows.
    Name(String),

    /// A selection object.
    Object(Value),
}

impl SelectArg {
    /// `text` as a `--select` reads it: an object when it starts with `{`
    /// (no name does), a name otherwise.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for an object that
    /// is not a YAML mapping.
    pub fn parse(text: &str) -> BunsenResult<SelectArg> {
        let text = text.trim();
        if !text.starts_with('{') {
            return Ok(SelectArg::Name(text.to_string()));
        }
        let value: Value = serde_saphyr::from_str(text)
            .map_err(|e| BunsenError::policy(format!("`--select {text}` is not YAML: {e}")))?;
        if !value.is_object() {
            return Err(BunsenError::policy(format!(
                "`--select {text}` is not a mapping"
            )));
        }
        Ok(SelectArg::Object(value))
    }
}

/// What a choice took, and how.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    /// The taken candidate.
    pub candidate: SelectionCandidate,

    /// Its weight.
    pub weight: f64,

    /// How it was chosen, for a log line.
    pub why: String,
}

impl SelectionContext {
    /// The named selection called `name`.
    pub fn named(
        &self,
        name: &str,
    ) -> Option<&NamedSelection> {
        self.named
            .iter()
            .find(|n| n.name.eq_ignore_ascii_case(name))
    }

    /// Chooses a candidate of `universe` by `selects` (each `--select`, in
    /// order) and `presets` (each `--preset`):
    ///
    /// 1. At most one name, from a `--select` or an object's `name`: a named
    ///    selection is its selection, merged first; a row named exactly is that
    ///    row alone, and no defaults apply; a fuzzy match is the rows it
    ///    matches ([`match_name`]). No name: the whole universe.
    /// 2. A candidate lacking any of `presets` is dropped.
    /// 3. The defaults (unless a row was named exactly), then the named
    ///    selection's, then the objects, merged in order ([`merge`]), are
    ///    applied, and the first candidate whose weight reaches the cutoff is
    ///    taken.
    ///
    /// # Errors
    /// [`Policy`](bunsen::errors::BunsenErrorKind::Policy) for a malformed
    /// `--select`, two names, a name nothing answers to, or a selection that
    /// leaves no candidate.
    pub fn choose(
        &self,
        universe: Vec<SelectionCandidate>,
        selects: &[String],
        presets: &[String],
    ) -> BunsenResult<Choice> {
        let (name, objects) = Self::parse_selects(selects)?;
        self.choose_parsed(universe, name, objects, presets)
    }

    /// `selects`, each `--select`, as at most one name and the objects.
    fn parse_selects(selects: &[String]) -> BunsenResult<(Option<String>, Vec<Value>)> {
        let mut name = None;
        let mut objects = Vec::new();
        for text in selects {
            match SelectArg::parse(text)? {
                SelectArg::Name(n) => {
                    if let Some(first) = name.replace(n.clone()) {
                        return Err(BunsenError::policy(format!(
                            "two names to select by: {first:?} and {n:?}"
                        )));
                    }
                }
                SelectArg::Object(object) => {
                    if let Some(n) = object.get("name").and_then(Value::as_str)
                        && let Some(first) = name.replace(n.to_string())
                    {
                        return Err(BunsenError::policy(format!(
                            "two names to select by: {first:?} and {n:?}"
                        )));
                    }
                    objects.push(object);
                }
            }
        }
        Ok((name, objects))
    }

    /// [`choose`](Self::choose) with the `--select`s already read: `name`, at
    /// most one, and `objects`, merged in order over the defaults.
    ///
    /// # Errors
    /// As [`choose`](Self::choose).
    pub fn choose_parsed(
        &self,
        universe: Vec<SelectionCandidate>,
        name: Option<String>,
        objects: Vec<Value>,
        presets: &[String],
    ) -> BunsenResult<Choice> {
        let mut layers: Vec<Value> = Vec::new();
        let mut why = Vec::new();
        let mut defaults = true;
        // A named selection is its selection, merged first; a name in it
        // matches rows as a `--select` name does.
        let mut row_name = name.clone();
        if let Some(n) = name.as_deref()
            && let Some(named) = self.named(n)
        {
            why.push(format!("the selection {:?}", named.name));
            layers.push(named.selection.clone());
            row_name = named
                .selection
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        let candidates: Vec<SelectionCandidate> = match row_name.as_deref() {
            None => universe,
            Some(n) => match match_name(&universe, n) {
                NameMatch::Exact(i) => {
                    defaults = false;
                    why.push(format!("the row {:?}", universe[i].pretrained.name));
                    vec![universe[i].clone()]
                }
                NameMatch::Fuzzy(found) => {
                    why.push(format!("{} rows matching {n:?}", found.len()));
                    found.into_iter().map(|i| universe[i].clone()).collect()
                }
                NameMatch::Missing => {
                    let mut known: Vec<&str> = self.named.iter().map(|n| n.name.as_str()).collect();
                    known.extend(universe.iter().map(|c| c.pretrained.name.as_str()));
                    return Err(BunsenError::policy(format!(
                        "no selection, row, group or kit answers to {n:?}; there are {}",
                        known.join(", ")
                    )));
                }
            },
        };
        let candidates: Vec<SelectionCandidate> = candidates
            .into_iter()
            .filter(|c| {
                presets
                    .iter()
                    .all(|p| c.pretrained.presets.names().any(|name| name == p))
            })
            .collect();
        if !presets.is_empty() {
            why.push(format!("having the presets {}", presets.join(", ")));
        }

        if defaults {
            layers.insert(0, self.defaults.clone());
        }
        layers.extend(objects);
        let mut merged = merge(&layers);
        if let Some(fields) = merged.as_object_mut() {
            // The name was matched above; the rest is the expression.
            fields.remove("name");
        }
        let expr = SelectionExpr::from_value(&merged)?;
        let selection = expr.apply(Selection::new(candidates));
        let ranked: Vec<String> = selection
            .iter()
            .map(|(c, w)| format!("{} ({w:.3})", c.pretrained.name))
            .collect();
        let cutoff = expr.cutoff();
        let taken = selection
            .iter()
            .find(|(_, w)| *w >= cutoff)
            .map(|(c, w)| (c.clone(), w));
        match taken {
            Some((candidate, weight)) => {
                let why = format!(
                    "{} by {}: {}",
                    candidate.pretrained.name,
                    if why.is_empty() {
                        "the defaults".to_string()
                    } else {
                        why.join(", ")
                    },
                    ranked.join(", ")
                );
                log::info!("selected {why}");
                Ok(Choice {
                    candidate,
                    weight,
                    why,
                })
            }
            None => Err(BunsenError::policy(format!(
                "the selection {merged} leaves no candidate{}",
                if ranked.is_empty() {
                    String::new()
                } else {
                    format!(" at or above {cutoff}: {}", ranked.join(", "))
                }
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::select::testing::candidate;

    fn universe() -> Vec<SelectionCandidate> {
        let mut rows = vec![
            candidate(
                "openai/large-v2",
                json!({"kit": "whisper", "base": false, "parameters": 1.5e9,
                       "version": {"group": "whisper-large", "version": "2.0.0"}}),
            ),
            candidate(
                "openai/large-v3",
                json!({"kit": "whisper", "base": false, "parameters": 1.5e9,
                       "version": {"group": "whisper-large", "version": "3.0.0"}}),
            ),
            candidate(
                "openai/tiny",
                json!({"kit": "whisper", "base": false, "parameters": 3.9e7}),
            ),
            candidate(
                "qwen/qwen3-0.6b-base",
                json!({"kit": "qwen3", "base": true, "parameters": 6e8}),
            ),
            candidate(
                "qwen/qwen3-0.6b",
                json!({"kit": "qwen3", "base": false, "parameters": 6e8}),
            ),
        ];
        rows[1].pretrained.presets = serde_json::from_value(json!({
            "presets": [{"name": "translate", "description": "", "patch": {}}]
        }))
        .unwrap();
        rows
    }

    fn context() -> SelectionContext {
        SelectionContext {
            defaults: json!({"require": {"base": false}, "latest": true, "order": ["-weight"]}),
            named: vec![NamedSelection {
                name: "small".into(),
                description: "the smallest".into(),
                selection: json!({"order": ["parameters"]}),
            }],
        }
    }

    fn choose(selects: &[&str]) -> BunsenResult<String> {
        choose_with(selects, &[])
    }

    fn choose_with(
        selects: &[&str],
        presets: &[&str],
    ) -> BunsenResult<String> {
        let selects: Vec<String> = selects.iter().map(|s| s.to_string()).collect();
        let presets: Vec<String> = presets.iter().map(|s| s.to_string()).collect();
        context()
            .choose(universe(), &selects, &presets)
            .map(|c| c.candidate.pretrained.name)
    }

    #[test]
    fn test_defaults_apply_without_a_name() {
        assert_eq!(
            choose(&[]).unwrap(),
            "openai/large-v3",
            "large-v2 is not the latest"
        );
        assert_eq!(
            choose(&["{order: [-weight, -parameters]}"]).unwrap(),
            "openai/large-v3"
        );
        assert_eq!(
            choose(&["small"]).unwrap(),
            "openai/tiny",
            "a named selection"
        );
    }

    /// A row named exactly is that row, the defaults aside; a fuzzy name is
    /// a set the defaults choose within.
    #[test]
    fn test_names() {
        assert_eq!(
            choose(&["qwen3-0.6b-base"]).unwrap(),
            "qwen/qwen3-0.6b-base"
        );
        assert_eq!(choose(&["large-v2"]).unwrap(), "openai/large-v2");
        assert_eq!(choose(&["large"]).unwrap(), "openai/large-v3");
        assert_eq!(
            choose(&["qwen3"]).unwrap(),
            "qwen/qwen3-0.6b",
            "base: false"
        );
        assert_eq!(
            choose(&["qwen3", "{require: {base: null}, order: [parameters]}"]).unwrap(),
            "qwen/qwen3-0.6b-base"
        );
        assert_eq!(
            choose(&["whisper", "{require: {parameters: {max: 1e9}}}"]).unwrap(),
            "openai/tiny"
        );
        assert_eq!(choose(&["{name: tiny}"]).unwrap(), "openai/tiny");
    }

    /// A named selection that names a row is that row, the defaults aside.
    #[test]
    fn test_a_named_selection_naming_a_row() {
        let mut context = context();
        context.named.push(NamedSelection {
            name: "default".into(),
            description: "the default".into(),
            selection: json!({"name": "qwen3-0.6b-base"}),
        });
        let choice = context
            .choose(universe(), &["default".to_string()], &[])
            .unwrap();
        assert_eq!(choice.candidate.pretrained.name, "qwen/qwen3-0.6b-base");
        assert!(
            choice.why.contains("the selection \"default\", the row"),
            "{}",
            choice.why
        );
    }

    #[test]
    fn test_presets_narrow_first() {
        assert_eq!(
            choose_with(&["whisper"], &["translate"]).unwrap(),
            "openai/large-v3"
        );
        assert!(choose_with(&["tiny"], &["translate"]).is_err());
    }

    #[test]
    fn test_refusals() {
        assert!(
            choose(&["gemma"])
                .unwrap_err()
                .to_string()
                .contains("no selection, row")
        );
        assert!(
            choose(&["tiny", "large"])
                .unwrap_err()
                .to_string()
                .contains("two names")
        );
        assert!(choose(&["tiny", "{name: large}"]).is_err());
        assert!(
            choose(&["{require: {base: maybe}}"])
                .unwrap_err()
                .to_string()
                .contains("leaves no")
        );
        assert!(choose(&["{nonsense: 1}"]).is_err());
        assert!(choose(&["{not yaml"]).is_err());
    }

    #[test]
    fn test_the_choice_says_why() {
        let choice = context()
            .choose(universe(), &["large".to_string()], &[])
            .unwrap();
        assert_eq!(choice.weight, 1.0);
        assert!(
            choice
                .why
                .starts_with("openai/large-v3 by 2 rows matching \"large\""),
            "{}",
            choice.why
        );
        assert!(
            choice.why.contains("openai/large-v2 (0.950)"),
            "{}",
            choice.why
        );
    }
}
