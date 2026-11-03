//! A pretrained's presets: named, described merge patches over its config.

use std::collections::BTreeSet;

use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
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

    /// A line for a listing and for errors.
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
/// axis replaces the default's member of that axis.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetSet {
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

impl PresetSet {
    /// Whether there are no presets.
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty()
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

    /// Checks the set is well formed.
    ///
    /// Names are unique, non-empty, and in the preset alphabet; every preset
    /// and axis has a description; an axis has two or more members, each a
    /// preset, and a preset is on at most one axis; the default names
    /// presets, each once, at most one per axis, and their patches set
    /// disjoint paths.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), naming the first
    /// rule broken.
    pub fn validate(&self) -> BunsenResult<()> {
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
}
