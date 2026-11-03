//! Resolving a config through presets: [`resolve_presets`].

use serde::{
    Serialize,
    de::DeserializeOwned,
};
use serde_json::Value;

use crate::{
    data::pretrained::{
        PresetSet,
        quoted,
    },
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    support::merge_patch,
};

/// What a lookup asks of a pretrained's presets: the presets it names, and
/// a final patch over the result.
#[derive(Clone, Debug, Default, PartialEq)]
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
}

/// A config resolved through a pretrained's presets, and what resolved it.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPresets<C> {
    /// The config, fully specified.
    pub config: C,

    /// The presets applied, in order: the default's that no selection
    /// displaced, then the selected.
    pub presets: Vec<String>,

    /// The final patch, if the selection had one.
    pub patch: Option<Value>,
}

/// Resolves `base` through `set`'s presets as `selection` asks.
///
/// The layers, in order, each a [merge patch](merge_patch) over the one
/// before:
///
/// 1. `base`: the family's config, serialized;
/// 2. the default presets, except those whose axis `selection` selects a member
///    of;
/// 3. the selected presets;
/// 4. the selection's final patch.
///
/// The selected presets are one layer: two that set the same path are an
/// error, so their order never matters. A layer over the default may set
/// what the default set. Every patch is checked against the config it is
/// applied to ([`validate_paths`](merge_patch::validate_paths)), and the
/// result is deserialized as `C`.
///
/// # Errors
/// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup): a selected preset that
///   `set` lacks (its candidates are `set`'s presets), a preset selected twice,
///   or a patch naming a field the config lacks.
/// - [`Policy`](crate::errors::BunsenErrorKind::Policy): two selected members
///   of one axis, two selected presets setting the same path, or a result that
///   does not deserialize as `C`.
/// - [`Internal`](crate::errors::BunsenErrorKind::Internal): `base` does not
///   serialize.
pub fn resolve_presets<C>(
    base: &C,
    set: &PresetSet,
    selection: &PresetSelection,
) -> BunsenResult<ResolvedPresets<C>>
where
    C: Serialize + DeserializeOwned,
{
    let selected = check_selection(set, selection)?;

    let mut presets: Vec<String> = set
        .default
        .iter()
        .filter(|name| {
            let displaced = set
                .axis_of(name)
                .is_some_and(|axis| selected.iter().any(|s| axis.members.iter().any(|m| m == s)));
            !displaced && !selected.contains(name)
        })
        .cloned()
        .collect();
    presets.extend(selected.iter().cloned());

    let mut value = serde_json::to_value(base)
        .map_err(|e| BunsenError::internal("a config that does not serialize").with_cause(e))?;
    for name in &presets {
        let preset = set.get(name).expect("checked: every applied preset exists");
        apply_layer(&mut value, &preset.patch)
            .map_err(|e| e.context(format!("the preset `{name}`")))?;
    }
    if let Some(patch) = &selection.patch {
        apply_layer(&mut value, patch).map_err(|e| e.context("the lookup's patch"))?;
    }

    let config = serde_json::from_value(value).map_err(|e| {
        BunsenError::policy("presets that resolve to an invalid config").with_cause(e)
    })?;
    Ok(ResolvedPresets {
        config,
        presets,
        patch: selection.patch.clone(),
    })
}

/// The selection's names, checked: each a preset of `set`, named once, at
/// most one per axis, and the lot setting disjoint paths.
fn check_selection(
    set: &PresetSet,
    selection: &PresetSelection,
) -> BunsenResult<Vec<String>> {
    let mut selected: Vec<String> = Vec::new();
    for name in &selection.presets {
        if set.get(name).is_none() {
            return Err(BunsenError::lookup(
                LookupError::missing("preset", name.clone()).with_candidates(set.names()),
            ));
        }
        if selected.contains(name) {
            return Err(BunsenError::lookup(LookupError::duplicate(
                "preset",
                name.clone(),
            )));
        }
        if let Some(axis) = set.axis_of(name)
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
        .map(|name| &set.get(name).expect("checked above").patch)
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

fn apply_layer(
    value: &mut Value,
    patch: &Value,
) -> BunsenResult<()> {
    merge_patch::validate_paths(value, patch)?;
    merge_patch::apply(value, patch);
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde_json::json;

    use super::*;
    use crate::{
        data::pretrained::{
            Preset,
            PresetAxis,
        },
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
    };

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Sampling {
        temperature: f64,
        top_p: f64,
    }

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Toy {
        thinking: bool,
        sampling: Sampling,
        beam: usize,
        yarn: Option<f64>,
    }

    fn base() -> Toy {
        Toy {
            thinking: false,
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
            },
            beam: 1,
            yarn: None,
        }
    }

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

    fn set() -> PresetSet {
        let set = PresetSet {
            presets: vec![
                preset(
                    "thinking",
                    json!({"thinking": true, "sampling": {"temperature": 0.6, "top_p": 0.95}}),
                ),
                preset(
                    "no-thinking",
                    json!({"thinking": false, "sampling": {"temperature": 0.7, "top_p": 0.8}}),
                ),
                preset("yarn-128k", json!({"yarn": 4.0})),
                preset("wide", json!({"beam": 5})),
                preset("narrow", json!({"beam": 2})),
            ],
            axes: vec![PresetAxis {
                name: "thinking-mode".to_string(),
                description: "whether the model thinks before it answers".to_string(),
                members: vec!["thinking".to_string(), "no-thinking".to_string()],
            }],
            default: vec!["thinking".to_string()],
        };
        set.validate().unwrap();
        set
    }

    fn resolved(selection: PresetSelection) -> BunsenResult<ResolvedPresets<Toy>> {
        resolve_presets(&base(), &set(), &selection)
    }

    #[test]
    fn test_no_selection_is_the_default() {
        let r = resolved(PresetSelection::default()).unwrap();
        assert_eq!(r.presets, vec!["thinking"]);
        assert_eq!(
            r.config,
            Toy {
                thinking: true,
                sampling: Sampling {
                    temperature: 0.6,
                    top_p: 0.95
                },
                ..base()
            }
        );
        assert_eq!(r.patch, None);
    }

    #[test]
    fn test_an_axis_member_displaces_the_default() {
        let r = resolved(PresetSelection::new(["no-thinking"])).unwrap();
        assert_eq!(r.presets, vec!["no-thinking"]);
        assert!(!r.config.thinking);
        assert_eq!(r.config.sampling.temperature, 0.7);

        // Selecting the default's own member is the default.
        let r = resolved(PresetSelection::new(["thinking"])).unwrap();
        assert_eq!(r.presets, vec!["thinking"]);
    }

    #[test]
    fn test_presets_off_the_axis_add_to_the_default() {
        let r = resolved(PresetSelection::new(["yarn-128k", "wide"])).unwrap();
        assert_eq!(r.presets, vec!["thinking", "yarn-128k", "wide"]);
        assert_eq!(r.config.yarn, Some(4.0));
        assert_eq!(r.config.beam, 5);
        assert!(r.config.thinking);
    }

    #[test]
    fn test_selected_presets_commute() {
        let a = resolved(PresetSelection::new(["no-thinking", "yarn-128k"])).unwrap();
        let b = resolved(PresetSelection::new(["yarn-128k", "no-thinking"])).unwrap();
        assert_eq!(a.config, b.config);
    }

    #[test]
    fn test_the_patch_is_last() {
        let r = resolved(
            PresetSelection::new(["wide"])
                .with_patch(json!({"beam": 3, "sampling": {"top_p": 0.5}})),
        )
        .unwrap();
        assert_eq!(r.config.beam, 3);
        assert_eq!(r.config.sampling.top_p, 0.5);
        assert_eq!(r.config.sampling.temperature, 0.6);
        assert_eq!(
            r.patch,
            Some(json!({"beam": 3, "sampling": {"top_p": 0.5}}))
        );
    }

    #[test]
    fn test_an_unknown_preset_lists_the_candidates() {
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_eq(
                "no preset \"thinkin\"; there are: thinking, no-thinking, yarn-128k, wide, narrow",
            )
            .has_cause::<LookupError>()
            .assert_err(&resolved(PresetSelection::new(["thinkin"])));
    }

    #[test]
    fn test_a_preset_selected_twice() {
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("wide")
            .has_cause::<LookupError>()
            .assert_err(&resolved(PresetSelection::new(["wide", "wide"])));
    }

    #[test]
    fn test_two_members_of_an_axis() {
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_eq(
                "`thinking` and `no-thinking` are both on the axis `thinking-mode` \
                 (whether the model thinks before it answers): select one",
            )
            .assert_err(&resolved(PresetSelection::new(["thinking", "no-thinking"])));
    }

    #[test]
    fn test_selected_presets_that_set_one_path() {
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_eq("the presets `wide` and `narrow` each set `beam`: select one")
            .assert_err(&resolved(PresetSelection::new(["wide", "narrow"])));
    }

    #[test]
    fn test_a_patch_naming_an_unknown_field() {
        let r = resolved(PresetSelection::default().with_patch(json!({"sampling": {"top_q": 1}})));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("sampling.top_q")
            .has_cause::<LookupError>()
            .assert_err(&r);
        assert!(
            r.unwrap_err().to_string().contains("the lookup's patch"),
            "the error names the layer"
        );
    }

    #[test]
    fn test_a_result_that_does_not_deserialize() {
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("invalid config")
            .assert_err(&resolved(
                PresetSelection::default().with_patch(json!({"beam": "five"})),
            ));
    }

    #[test]
    fn test_an_empty_set_resolves_to_the_base() {
        let r =
            resolve_presets(&base(), &PresetSet::default(), &PresetSelection::default()).unwrap();
        assert_eq!(r.config, base());
        assert!(r.presets.is_empty());
    }
}
