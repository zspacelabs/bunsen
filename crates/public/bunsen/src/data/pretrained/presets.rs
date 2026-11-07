//! A pretrained's presets: named, described merge patches over its config.

use alloc::{
    collections::BTreeSet,
    string::String,
    vec::Vec,
};
use core::fmt;

use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    support::merge_patch,
};

/// One preset of a pretrained: a named, described delta over the config it
/// is built from.
///
/// The delta is a [merge patch](crate::support::merge_patch) over that
/// config's serialized form. Which fields it may set is the config's
/// business: a field the construction reads changes what is built, and one a
/// driver reads changes how it runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    /// Its name, as a lookup's box names it: ASCII letters, digits, `-`,
    /// `_` and `.`.
    pub name: String,

    /// A line for a listing and for errors. Required, but a row's preset
    /// laid over a family's of the same name may leave it empty to keep the
    /// family's ([`PresetSet::over`]).
    #[serde(default)]
    pub description: String,

    /// The merge patch it applies.
    #[serde(default = "empty_patch", skip_serializing_if = "is_empty_patch")]
    pub patch: Value,
}

fn empty_patch() -> Value {
    Value::Object(Default::default())
}

fn is_empty_patch(patch: &Value) -> bool {
    patch.as_object().is_some_and(|members| members.is_empty())
}

/// Presets that exclude one another: a lookup selects at most one member.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetAxis {
    /// Its name, in the same alphabet as a preset's.
    pub name: String,

    /// A line for a listing and for errors: what the axis decides.
    pub description: String,

    /// Its members, by preset name: two or more.
    pub members: Vec<String>,
}

/// What a pretrained offers: its presets, the axes that group some of them,
/// and its default selection.
///
/// The default is explicit: a pretrained's config with no presets selected
/// is its config with these applied. A lookup that selects a member of an
/// axis replaces the default's member of that axis. The
/// [`base`](Self::base) patch holds what every selection keeps, applied
/// before the default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetSet {
    /// A patch applied under every preset: what any selection keeps, such
    /// as the stop tokens a checkpoint's `generation_config.json` names.
    #[serde(default = "empty_patch", skip_serializing_if = "is_empty_patch")]
    pub base: Value,

    /// The presets, in listing order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub presets: Vec<Preset>,

    /// The axes, in listing order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub axes: Vec<PresetAxis>,

    /// The presets applied when a lookup selects none, by name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub default: Vec<String>,
}

impl Default for PresetSet {
    fn default() -> Self {
        Self {
            base: empty_patch(),
            presets: Vec::new(),
            axes: Vec::new(),
            default: Vec::new(),
        }
    }
}

impl PresetSet {
    /// Whether the set declares nothing: no base, presets, axes or default.
    pub fn is_empty(&self) -> bool {
        is_empty_patch(&self.base)
            && self.presets.is_empty()
            && self.axes.is_empty()
            && self.default.is_empty()
    }

    /// The preset `name`, if there is one.
    pub fn get(
        &self,
        name: &str,
    ) -> Option<&Preset> {
        self.presets.iter().find(|p| p.name == name)
    }

    /// The axis `name` is a member of, if any.
    pub fn axis_of(
        &self,
        name: &str,
    ) -> Option<&PresetAxis> {
        self.axes
            .iter()
            .find(|a| a.members.iter().any(|m| m == name))
    }

    /// The presets' names, in listing order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.presets.iter().map(|p| p.name.as_str())
    }

    /// `self` merged over `family`: a family's presets with a row's own laid
    /// over them.
    ///
    /// A preset of `self` is merged over the family's of the same name: its
    /// patch laid over the family's, its description the family's when it
    /// gives none. An axis of `self` replaces the family's of the same name.
    /// The rest of each are kept in order, the family's first. The default is
    /// `self`'s when it names any, else the family's; the base is `self`'s
    /// merged over the family's. The result is not checked:
    /// [`validate`](Self::validate) it.
    pub fn over(
        &self,
        family: &PresetSet,
    ) -> PresetSet {
        fn merged<T: Clone>(
            base: &[T],
            mine: &[T],
            name: impl Fn(&T) -> &str,
        ) -> Vec<T> {
            let mut out: Vec<T> = base
                .iter()
                .map(|b| {
                    mine.iter()
                        .find(|m| name(m) == name(b))
                        .unwrap_or(b)
                        .clone()
                })
                .collect();
            out.extend(
                mine.iter()
                    .filter(|m| !base.iter().any(|b| name(b) == name(m)))
                    .cloned(),
            );
            out
        }
        let mut base = family.base.clone();
        merge_patch::apply(&mut base, &self.base);
        let presets = merged(&family.presets, &self.presets, |p| &p.name)
            .into_iter()
            .map(|mut preset| {
                if let Some(theirs) = family.presets.iter().find(|f| f.name == preset.name)
                    && self.presets.iter().any(|m| m.name == preset.name)
                {
                    let mut patch = theirs.patch.clone();
                    merge_patch::apply(&mut patch, &preset.patch);
                    preset.patch = patch;
                    if preset.description.trim().is_empty() {
                        preset.description = theirs.description.clone();
                    }
                }
                preset
            })
            .collect();
        PresetSet {
            base,
            presets,
            axes: merged(&family.axes, &self.axes, |a| &a.name),
            default: if self.default.is_empty() {
                family.default.clone()
            } else {
                self.default.clone()
            },
        }
    }

    /// The preset a lookup's `name` selects: the one so named, ignoring
    /// ASCII case.
    pub fn find(
        &self,
        name: &str,
    ) -> Option<&Preset> {
        self.get(name).or_else(|| {
            self.presets
                .iter()
                .find(|p| p.name.eq_ignore_ascii_case(name))
        })
    }

    /// Checks `selection`'s presets against the set, and names them as the
    /// set declares them.
    ///
    /// Each must be a preset of the set (ignoring ASCII case), named once;
    /// at most one member of an axis may be selected; and the selected
    /// presets' patches must set disjoint paths, so that their order never
    /// matters.
    ///
    /// # Errors
    /// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup): a preset the set
    ///   lacks (its candidates are the set's presets), or one selected twice.
    /// - [`Policy`](crate::errors::BunsenErrorKind::Policy): two members of one
    ///   axis, or two presets setting the same path.
    pub fn check_selection(
        &self,
        selection: &PresetSelection,
    ) -> BunsenResult<Vec<String>> {
        let mut selected: Vec<String> = Vec::new();
        for name in &selection.presets {
            let Some(preset) = self.find(name) else {
                return Err(BunsenError::lookup(
                    LookupError::missing("preset", name.clone()).with_candidates(self.names()),
                ));
            };
            let name = &preset.name;
            if selected.contains(name) {
                return Err(BunsenError::lookup(LookupError::duplicate(
                    "preset",
                    name.clone(),
                )));
            }
            if let Some(axis) = self.axis_of(name)
                && let Some(other) = selected
                    .iter()
                    .find(|s| axis.members.iter().any(|m| m == *s))
            {
                return Err(BunsenError::policy(format!(
                    "`{other}` and `{name}` are both on the axis `{}` ({}): select one",
                    axis.name, axis.description
                )));
            }
            selected.push(name.clone());
        }

        let patches: Vec<&Value> = selected
            .iter()
            .map(|name| &self.get(name).expect("found above").patch)
            .collect();
        if let Some(conflict) = merge_patch::conflicts(&patches).first() {
            let who: Vec<&str> = conflict
                .patches
                .iter()
                .map(|&i| selected[i].as_str())
                .collect();
            return Err(BunsenError::policy(format!(
                "the presets {} each set `{}`: select one",
                quoted(&who),
                conflict.path
            )));
        }
        Ok(selected)
    }

    /// The presets a checked selection applies, in order: the default's
    /// that it does not displace (a default whose axis it selects a member
    /// of is displaced), then the selected.
    pub fn applied(
        &self,
        selected: &[String],
    ) -> Vec<String> {
        let mut applied: Vec<String> = self
            .default
            .iter()
            .filter(|name| {
                let displaced = self.axis_of(name).is_some_and(|axis| {
                    selected.iter().any(|s| axis.members.iter().any(|m| m == s))
                });
                !displaced && !selected.contains(name)
            })
            .cloned()
            .collect();
        applied.extend(selected.iter().cloned());
        applied
    }

    /// Checks the set is well formed.
    ///
    /// The base is an object; names are unique, non-empty, and in the preset
    /// alphabet; every preset
    /// and axis has a description; an axis has two or more members, each a
    /// preset, and a preset is on at most one axis; the default names
    /// presets, each once, at most one per axis, and their patches set
    /// disjoint paths.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), naming the first
    /// rule broken.
    pub fn validate(&self) -> BunsenResult<()> {
        if !self.base.is_object() {
            return Err(BunsenError::illegal("the base patch is not an object"));
        }
        let mut names = BTreeSet::new();
        for preset in &self.presets {
            check_name("preset", &preset.name)?;
            check_description("preset", &preset.name, &preset.description)?;
            if !names.insert(preset.name.as_str()) {
                return Err(BunsenError::illegal(format!(
                    "the preset `{}` is declared twice",
                    preset.name
                )));
            }
        }

        let mut axis_names = BTreeSet::new();
        let mut on_an_axis = BTreeSet::new();
        for axis in &self.axes {
            check_name("axis", &axis.name)?;
            check_description("axis", &axis.name, &axis.description)?;
            if !axis_names.insert(axis.name.as_str()) {
                return Err(BunsenError::illegal(format!(
                    "the axis `{}` is declared twice",
                    axis.name
                )));
            }
            if axis.members.len() < 2 {
                return Err(BunsenError::illegal(format!(
                    "the axis `{}` has fewer than two members",
                    axis.name
                )));
            }
            for member in &axis.members {
                if !names.contains(member.as_str()) {
                    return Err(BunsenError::illegal(format!(
                        "the axis `{}` names `{member}`, which is not a preset",
                        axis.name
                    )));
                }
                if !on_an_axis.insert(member.as_str()) {
                    return Err(BunsenError::illegal(format!(
                        "the preset `{member}` is on more than one axis"
                    )));
                }
            }
        }

        let mut seen = BTreeSet::new();
        let mut axes_used = BTreeSet::new();
        for name in &self.default {
            if !names.contains(name.as_str()) {
                return Err(BunsenError::illegal(format!(
                    "the default names `{name}`, which is not a preset"
                )));
            }
            if !seen.insert(name.as_str()) {
                return Err(BunsenError::illegal(format!(
                    "the default names `{name}` twice"
                )));
            }
            if let Some(axis) = self.axis_of(name)
                && !axes_used.insert(axis.name.as_str())
            {
                return Err(BunsenError::illegal(format!(
                    "the default selects two members of the axis `{}`",
                    axis.name
                )));
            }
        }
        let patches: Vec<&Value> = self
            .default
            .iter()
            .filter_map(|name| self.get(name).map(|p| &p.patch))
            .collect();
        if let Some(conflict) = merge_patch::conflicts(&patches).first() {
            let who: Vec<&str> = conflict
                .patches
                .iter()
                .map(|&i| self.default[i].as_str())
                .collect();
            return Err(BunsenError::illegal(format!(
                "the default's presets {} each set `{}`",
                quoted(&who),
                conflict.path
            )));
        }
        Ok(())
    }
}

/// What a lookup asks of a pretrained's presets: the presets it names, and
/// a final patch over the result.
///
/// `Display` writes it as a lookup expression's tail: the box, then the
/// patch as JSON, each only when present (`[thinking,yarn-128k]{"beam":1}`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PresetSelection {
    /// The presets named, in the order given.
    pub presets: Vec<String>,

    /// A merge patch applied after every preset.
    pub patch: Option<Value>,
}

impl PresetSelection {
    /// Selects `presets`.
    pub fn new<I, S>(presets: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            presets: presets.into_iter().map(Into::into).collect(),
            patch: None,
        }
    }

    /// Sets the final patch.
    pub fn with_patch(
        mut self,
        patch: Value,
    ) -> Self {
        self.patch = Some(patch);
        self
    }

    /// Whether it selects nothing and patches nothing.
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty() && self.patch.is_none()
    }
}

impl fmt::Display for PresetSelection {
    /// The selection as a lookup expression's tail: the box, `[a,b]`, then
    /// the patch as a YAML flow mapping, `{beam_size: 5, language: de}`,
    /// which parses back to the same selection.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if !self.presets.is_empty() {
            write!(f, "[{}]", self.presets.join(","))?;
        }
        if let Some(patch) = &self.patch {
            write_flow(f, patch)?;
        }
        Ok(())
    }
}

/// `value` in YAML's flow style: `{key: value, ...}` and `[item, ...]`,
/// each string plain where YAML reads it back as that string
/// ([`is_plain`]), and double-quoted, with JSON's escapes, where not.
fn write_flow(
    f: &mut fmt::Formatter<'_>,
    value: &Value,
) -> fmt::Result {
    match value {
        Value::Object(entries) => {
            f.write_str("{")?;
            for (i, (key, value)) in entries.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write_flow_str(f, key)?;
                f.write_str(": ")?;
                write_flow(f, value)?;
            }
            f.write_str("}")
        }
        Value::Array(items) => {
            f.write_str("[")?;
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write_flow(f, item)?;
            }
            f.write_str("]")
        }
        Value::String(s) => write_flow_str(f, s),
        // Null, booleans and numbers write the same in YAML as in JSON.
        other => write!(f, "{other}"),
    }
}

fn write_flow_str(
    f: &mut fmt::Formatter<'_>,
    s: &str,
) -> fmt::Result {
    if is_plain(s) {
        f.write_str(s)
    } else {
        write!(f, "{}", Value::String(s.to_string()))
    }
}

/// Whether YAML reads `s`, unquoted in a flow collection, back as the
/// string `s`: it starts with a letter or `_`, holds only letters, digits,
/// `_`, `-`, `.` and `/`, and is none of the words a YAML reader may take
/// for a boolean, a null or a special float, in any case.
fn is_plain(s: &str) -> bool {
    const WORDS: [&str; 12] = [
        "true", "false", "null", "yes", "no", "on", "off", "y", "n", "nan", "inf", "infinity",
    ];
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/'))
        && !WORDS.iter().any(|w| w.eq_ignore_ascii_case(s))
}

/// Whether `name` is in the preset alphabet: non-empty ASCII letters,
/// digits, `-`, `_` and `.`.
pub fn is_preset_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn check_name(
    what: &str,
    name: &str,
) -> BunsenResult<()> {
    if is_preset_name(name) {
        Ok(())
    } else {
        Err(BunsenError::illegal(format!(
            "the {what} name {name:?} is not ASCII letters, digits, `-`, `_` and `.`"
        )))
    }
}

fn check_description(
    what: &str,
    name: &str,
    description: &str,
) -> BunsenResult<()> {
    if description.trim().is_empty() {
        Err(BunsenError::illegal(format!(
            "the {what} `{name}` has no description"
        )))
    } else {
        Ok(())
    }
}

/// `a`, `b` and `c`, each in backticks.
pub(crate) fn quoted(names: &[&str]) -> String {
    let names: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
    match names.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    fn preset(
        name: &str,
        patch: Value,
    ) -> Preset {
        Preset {
            name: name.to_string(),
            description: format!("the {name} preset"),
            patch,
        }
    }

    fn axis(
        name: &str,
        members: &[&str],
    ) -> PresetAxis {
        PresetAxis {
            name: name.to_string(),
            description: format!("the {name} axis"),
            members: members.iter().map(|m| m.to_string()).collect(),
        }
    }

    fn thinking_set() -> PresetSet {
        PresetSet {
            presets: vec![
                preset("thinking", json!({"thinking": true})),
                preset("no-thinking", json!({"thinking": false})),
                preset("yarn-128k", json!({"yarn": {"factor": 4.0}})),
            ],
            axes: vec![axis("thinking-mode", &["thinking", "no-thinking"])],
            default: vec!["thinking".to_string()],
            ..Default::default()
        }
    }

    fn assert_illegal(
        set: &PresetSet,
        message: &str,
    ) {
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_eq(message)
            .assert_err(&set.validate());
    }

    #[test]
    fn test_lookups() {
        let set = thinking_set();
        set.validate().unwrap();
        assert!(!set.is_empty());
        assert!(PresetSet::default().is_empty());
        assert_eq!(
            set.get("thinking").unwrap().patch,
            json!({"thinking": true})
        );
        assert!(set.get("nope").is_none());
        assert_eq!(set.axis_of("no-thinking").unwrap().name, "thinking-mode");
        assert!(set.axis_of("yarn-128k").is_none());
        assert_eq!(
            set.names().collect::<Vec<_>>(),
            vec!["thinking", "no-thinking", "yarn-128k"]
        );
    }

    #[test]
    fn test_serde_round_trip_omits_empties() {
        let set = PresetSet {
            presets: vec![preset("plain", json!({}))],
            ..Default::default()
        };
        let text = serde_json::to_string(&set).unwrap();
        assert_eq!(
            text,
            r#"{"presets":[{"name":"plain","description":"the plain preset"}]}"#
        );
        assert_eq!(serde_json::from_str::<PresetSet>(&text).unwrap(), set);
        assert_eq!(
            serde_json::from_str::<PresetSet>("{}").unwrap(),
            PresetSet::default()
        );

        let set = thinking_set();
        let text = serde_json::to_string(&set).unwrap();
        assert_eq!(serde_json::from_str::<PresetSet>(&text).unwrap(), set);
    }

    #[test]
    fn test_preset_names() {
        assert!(is_preset_name("yarn-128k"));
        assert!(is_preset_name("faster_whisper.v2"));
        assert!(!is_preset_name(""));
        assert!(!is_preset_name("a,b"));
        assert!(!is_preset_name("a b"));
        assert!(!is_preset_name("x[y]"));
    }

    #[test]
    fn test_validate_rejects_bad_presets() {
        let mut set = thinking_set();
        set.presets[0].description = "  ".to_string();
        assert_illegal(&set, "the preset `thinking` has no description");

        let mut set = thinking_set();
        set.presets.push(preset("thinking", json!({})));
        assert_illegal(&set, "the preset `thinking` is declared twice");

        let mut set = thinking_set();
        set.presets[2].name = "yarn 128k".to_string();
        assert_illegal(
            &set,
            "the preset name \"yarn 128k\" is not ASCII letters, digits, `-`, `_` and `.`",
        );
    }

    #[test]
    fn test_validate_rejects_bad_axes() {
        let mut set = thinking_set();
        set.axes[0].description = String::new();
        assert_illegal(&set, "the axis `thinking-mode` has no description");

        let mut set = thinking_set();
        set.axes[0].members.pop();
        assert_illegal(&set, "the axis `thinking-mode` has fewer than two members");

        let mut set = thinking_set();
        set.axes[0].members.push("fast".to_string());
        assert_illegal(
            &set,
            "the axis `thinking-mode` names `fast`, which is not a preset",
        );

        let mut set = thinking_set();
        set.axes.push(axis("again", &["thinking", "yarn-128k"]));
        assert_illegal(&set, "the preset `thinking` is on more than one axis");

        let mut set = thinking_set();
        set.axes
            .push(axis("thinking-mode", &["yarn-128k", "thinking"]));
        assert_illegal(&set, "the axis `thinking-mode` is declared twice");
    }

    #[test]
    fn test_validate_rejects_bad_defaults() {
        let mut set = thinking_set();
        set.default.push("fast".to_string());
        assert_illegal(&set, "the default names `fast`, which is not a preset");

        let mut set = thinking_set();
        set.default.push("thinking".to_string());
        assert_illegal(&set, "the default names `thinking` twice");

        let mut set = thinking_set();
        set.default.push("no-thinking".to_string());
        assert_illegal(
            &set,
            "the default selects two members of the axis `thinking-mode`",
        );

        let mut set = thinking_set();
        set.presets
            .push(preset("think-hard", json!({"thinking": true})));
        set.default.push("think-hard".to_string());
        assert_illegal(
            &set,
            "the default's presets `thinking` and `think-hard` each set `thinking`",
        );
    }

    #[test]
    fn test_quoted() {
        assert_eq!(quoted(&[]), "");
        assert_eq!(quoted(&["a"]), "`a`");
        assert_eq!(quoted(&["a", "b"]), "`a` and `b`");
        assert_eq!(quoted(&["a", "b", "c"]), "`a`, `b` and `c`");
    }

    #[test]
    fn test_check_selection_folds_case_and_applies_over_the_default() {
        let set = thinking_set();
        assert_eq!(
            set.check_selection(&PresetSelection::new(["No-Thinking", "YARN-128K"]))
                .unwrap(),
            ["no-thinking", "yarn-128k"]
        );
        assert_eq!(
            set.applied(&["yarn-128k".to_string()]),
            ["thinking", "yarn-128k"]
        );
        assert_eq!(set.applied(&["no-thinking".to_string()]), ["no-thinking"]);
        assert_eq!(set.applied(&[]), ["thinking"]);
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("thinking")
            .assert_err(&set.check_selection(&PresetSelection::new(["thinking", "THINKING"])));
    }

    #[test]
    fn test_selection_display() {
        assert_eq!(PresetSelection::default().to_string(), "");
        assert!(PresetSelection::default().is_empty());
        let s = PresetSelection::new(["a", "b"]).with_patch(json!({"beam": 1}));
        assert!(!s.is_empty());
        assert_eq!(s.to_string(), "[a,b]{beam: 1}");
        assert_eq!(
            PresetSelection::default().with_patch(json!({})).to_string(),
            "{}"
        );
    }

    #[test]
    fn test_over_a_family() {
        let family = thinking_set();
        let row = PresetSet {
            presets: vec![
                preset("thinking", json!({"thinking": true, "budget": 1024})),
                preset("fast", json!({"beam": 1})),
            ],
            axes: vec![],
            default: vec![],
            ..Default::default()
        };
        let merged = row.over(&family);
        assert_eq!(
            merged.names().collect::<Vec<_>>(),
            ["thinking", "no-thinking", "yarn-128k", "fast"]
        );
        // The row's `thinking` is laid over the family's.
        assert_eq!(
            merged.get("thinking").unwrap().patch,
            json!({"thinking": true, "budget": 1024})
        );
        assert_eq!(merged.axes, family.axes);
        assert_eq!(merged.default, ["thinking"]);
        merged.validate().unwrap();

        let with_default = PresetSet {
            default: vec!["no-thinking".to_string()],
            ..Default::default()
        };
        assert_eq!(with_default.over(&family).default, ["no-thinking"]);
        assert_eq!(PresetSet::default().over(&family), family);
        assert_eq!(family.over(&PresetSet::default()), family);
    }

    /// A same-named preset merges: the row's patch over the family's, the
    /// family's description when the row gives none.
    #[test]
    fn test_over_merges_same_named_presets() {
        let family = thinking_set();
        let row = PresetSet {
            presets: vec![Preset {
                name: "no-thinking".to_string(),
                description: String::new(),
                patch: json!({"sampling": {"temperature": 0.7}}),
            }],
            ..Default::default()
        };
        let merged = row.over(&family);
        let p = merged.get("no-thinking").unwrap();
        assert_eq!(
            p.patch,
            json!({"thinking": false, "sampling": {"temperature": 0.7}})
        );
        assert_eq!(p.description, "the no-thinking preset");
        merged.validate().unwrap();
    }
}
