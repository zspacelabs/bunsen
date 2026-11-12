//! Resolving a config through presets: [`resolve_presets`].

use serde::{
    Serialize,
    de::DeserializeOwned,
};
use serde_json::Value;

use crate::{
    data::{
        builds::PretrainedRef,
        catalog::presets::{
            PresetSelection,
            ResolvedPresetSet,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    support::merge_patch,
};

/// A config resolved through a pretrained's presets, and what resolved it.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPresets<C> {
    /// The config, fully specified.
    pub config: C,

    /// The presets applied, in order: the default's that no selection
    /// displaced, then the selected, each named as the set declares it.
    pub presets: Vec<String>,

    /// The final patch, if the selection had one.
    pub patch: Option<Value>,
}

impl<C> ResolvedPresets<C> {
    /// What was applied, as a lookup expression's tail: the applied presets
    /// and the patch ([`PresetSelection`]'s `Display`).
    pub fn applied(&self) -> PresetSelection {
        PresetSelection {
            presets: self.presets.clone(),
            patch: self.patch.clone(),
        }
    }
}

/// Resolves `base` through `set`'s presets as `selection` asks.
///
/// The layers, in order, each a [merge patch](merge_patch) over the one
/// before:
///
/// 1. `base`: the family's config, serialized;
/// 2. the set's [`common`](ResolvedPresetSet::common) patch, which every
///    selection keeps;
/// 3. each axis use's default, unless `selection` selects a member of its axis,
///    and the presets on no axis the set applies by default;
/// 4. the selected presets;
/// 5. the selection's final patch.
///
/// The selected presets are one layer, checked by
/// [`ResolvedPresetSet::check_selection`]: two that set the same path are an error,
/// so their order never matters. A layer over the default may set what the
/// default set. Every patch is checked against the config it is applied to
/// ([`validate_paths_open`](merge_patch::validate_paths_open)), except under
/// the base's maps (its empty objects,
/// [`empty_objects`](merge_patch::empty_objects)), which take any key; and
/// the result is deserialized as `C`.
///
/// # Errors
/// - As [`ResolvedPresetSet::check_selection`].
/// - [`Lookup`](crate::errors::BunsenErrorKind::Lookup): a patch naming a field
///   the config lacks, under a frame naming the preset or the lookup's patch.
/// - [`Policy`](crate::errors::BunsenErrorKind::Policy): a result that does not
///   deserialize as `C`.
/// - [`Internal`](crate::errors::BunsenErrorKind::Internal): `base` does not
///   serialize.
pub fn resolve_presets<C>(
    base: &C,
    set: &ResolvedPresetSet,
    selection: &PresetSelection,
) -> BunsenResult<ResolvedPresets<C>>
where
    C: Serialize + DeserializeOwned,
{
    let selected = set.check_selection(selection)?;
    let presets = set.applied(&selected);

    let mut value = serde_json::to_value(base)
        .map_err(|e| BunsenError::internal("a config that does not serialize").with_cause(e))?;
    let open = merge_patch::empty_objects(&value);
    apply_layer::<C>(&mut value, &set.common, &open).map_err(|e| e.context("the base patch"))?;
    for name in &presets {
        let preset = set.get(name).expect("checked: every applied preset exists");
        apply_layer::<C>(&mut value, &preset.patch, &open)
            .map_err(|e| e.context(format!("the preset `{name}`")))?;
    }
    if let Some(patch) = &selection.patch {
        apply_layer::<C>(&mut value, patch, &open).map_err(|e| e.context("the lookup's patch"))?;
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

/// Resolves `base` through the presets of the row `model` names, as its
/// lookup asked: [`resolve_presets`] over the row's [`ResolvedPresetSet`] and
/// the ref's [`selection`](PretrainedRef::selection).
///
/// `family` stands in for a row with no presets of its own, such as an
/// `hf:` repo's, and for a given resource set, which has no row: the
/// presets of the model family the hook found the files to be (empty when
/// the kit keeps none). A given bundle takes its family's default.
///
/// What a hook that resolves presets calls in its `try_init`.
///
/// # Errors
/// As [`resolve_presets`].
pub fn resolve_ref_presets<C>(
    base: &C,
    model: &PretrainedRef,
    family: &ResolvedPresetSet,
) -> BunsenResult<ResolvedPresets<C>>
where
    C: Serialize + DeserializeOwned,
{
    let none = PresetSelection::default();
    let selection = model.selection().unwrap_or(&none);
    let set = match model.named() {
        Some((_, row)) if !row.presets.is_empty() => &row.presets,
        _ => family,
    };
    resolve_presets(base, set, selection)
}

/// Checks and applies one layer, then normalizes the result through `C`:
/// deserialized and serialized again, so an optional field a layer left
/// unset is `null` again, and a later layer may set it.
fn apply_layer<C>(
    value: &mut Value,
    patch: &Value,
    open: &[merge_patch::PatchPath],
) -> BunsenResult<()>
where
    C: Serialize + DeserializeOwned,
{
    merge_patch::validate_paths_open(value, patch, open)?;
    merge_patch::apply(value, patch);
    let typed: C = serde_json::from_value(value.clone()).map_err(|e| {
        BunsenError::policy("presets that resolve to an invalid config").with_cause(e)
    })?;
    *value = serde_json::to_value(typed)
        .map_err(|e| BunsenError::internal("a config that does not serialize").with_cause(e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde_json::json;

    use super::*;
    use crate::{
        data::catalog::{
            CatalogName,
            document::{
                AxisKind,
                MergeMode,
                PresetEntry,
            },
            presets::{
                AxisInfo,
                ResolvedAxisUse,
            },
        },
        errors::{
            BunsenErrorKind,
            LookupError,
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
    ) -> PresetEntry {
        PresetEntry {
            name: name.to_string(),
            merge: MergeMode::Update,
            description: format!("the {name} preset"),
            patch,
        }
    }

    fn set() -> ResolvedPresetSet {
        let thinking_mode = vec!["thinking".to_string(), "no-thinking".to_string()];
        ResolvedPresetSet {
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
            axes: vec![ResolvedAxisUse {
                axis: AxisInfo {
                    name: CatalogName::root("thinking-mode").unwrap(),
                    kind: AxisKind::Categorical,
                    members: thinking_mode.clone(),
                    description: "whether the model thinks before it answers".to_string(),
                },
                offers: thinking_mode,
                default: "thinking".to_string(),
            }],
            ..Default::default()
        }
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
        let r = resolved(PresetSelection::of(["no-thinking"])).unwrap();
        assert_eq!(r.presets, vec!["no-thinking"]);
        assert!(!r.config.thinking);
        assert_eq!(r.config.sampling.temperature, 0.7);

        // Selecting the default's own member is the default.
        let r = resolved(PresetSelection::of(["thinking"])).unwrap();
        assert_eq!(r.presets, vec!["thinking"]);
    }

    #[test]
    fn test_presets_off_the_axis_add_to_the_default() {
        let r = resolved(PresetSelection::of(["yarn-128k", "wide"])).unwrap();
        assert_eq!(r.presets, vec!["thinking", "yarn-128k", "wide"]);
        assert_eq!(r.config.yarn, Some(4.0));
        assert_eq!(r.config.beam, 5);
        assert!(r.config.thinking);
    }

    #[test]
    fn test_selected_presets_commute() {
        let a = resolved(PresetSelection::of(["no-thinking", "yarn-128k"])).unwrap();
        let b = resolved(PresetSelection::of(["yarn-128k", "no-thinking"])).unwrap();
        assert_eq!(a.config, b.config);
    }

    #[test]
    fn test_the_patch_is_last() {
        let r = resolved(
            PresetSelection::of(["wide"])
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
            .message_contains(
                "no preset `thinkin`; there are: thinking, no-thinking, yarn-128k, wide, narrow",
            )
            .assert_err(&resolved(PresetSelection::of(["thinkin"])));
    }

    #[test]
    fn test_a_preset_selected_twice() {
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("the preset `wide` is selected twice")
            .assert_err(&resolved(PresetSelection::of(["wide", "wide"])));
    }

    #[test]
    fn test_two_members_of_an_axis() {
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains(
                "`thinking` and `no-thinking` are both on the axis `thinking-mode`: select one",
            )
            .assert_err(&resolved(PresetSelection::of(["thinking", "no-thinking"])));
    }

    #[test]
    fn test_selected_presets_that_set_one_path() {
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("the presets `wide` and `narrow` each set `beam`: select one")
            .assert_err(&resolved(PresetSelection::of(["wide", "narrow"])));
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
        let r = resolve_presets(
            &base(),
            &ResolvedPresetSet::default(),
            &PresetSelection::default(),
        )
        .unwrap();
        assert_eq!(r.config, base());
        assert!(r.presets.is_empty());
    }

    /// A row with presets resolves through them; a row with none (an `hf:`
    /// repo's) and a given bundle resolve through the family's.
    #[test]
    fn test_resolve_ref_presets_falls_back_to_the_family() {
        use crate::data::{
            builds::Pretrained,
            resources::ResourceSet,
        };

        let named = |presets: ResolvedPresetSet, selection: PresetSelection| PretrainedRef::Named {
            provider: "hf".to_string(),
            pretrained: Box::new(Pretrained {
                name: "org/repo".to_string(),
                aliases: Vec::new(),
                description: "a repo".to_string(),
                license: None,
                origin: None,
                prefab: None,
                resources: ResourceSet::new("org/repo"),
                presets,
                args: Default::default(),
                tags: Default::default(),
            }),
            selection,
        };
        let family = set();

        let own = ResolvedPresetSet {
            presets: vec![preset("wide", json!({"beam": 9}))],
            ..Default::default()
        };
        let r = resolve_ref_presets(&base(), &named(own, PresetSelection::of(["wide"])), &family)
            .unwrap();
        assert_eq!(r.config.beam, 9);
        assert!(!r.config.thinking, "the row's own set, not the family's");

        let r = resolve_ref_presets(
            &base(),
            &named(
                ResolvedPresetSet::default(),
                PresetSelection::of(["no-thinking"]),
            ),
            &family,
        )
        .unwrap();
        assert_eq!(r.presets, ["no-thinking"]);

        let given = PretrainedRef::Given(ResourceSet::new("files"));
        let r = resolve_ref_presets(&base(), &given, &family).unwrap();
        assert_eq!(r.presets, ["thinking"]);
        let r = resolve_ref_presets(&base(), &given, &ResolvedPresetSet::default()).unwrap();
        assert_eq!(r.config, base());
    }

    /// The set's common patch applies under every selection, the default's and
    /// a displacing one's alike; a preset may set what the common patch set.
    #[test]
    fn test_the_common_patch_is_under_every_selection() {
        let mut set = set();
        set.common = json!({"beam": 3, "yarn": 2.0});
        let r = resolve_presets(&base(), &set, &PresetSelection::default()).unwrap();
        assert_eq!((r.config.beam, r.config.yarn), (3, Some(2.0)));
        let r =
            resolve_presets(&base(), &set, &PresetSelection::of(["no-thinking", "wide"])).unwrap();
        assert_eq!((r.config.beam, r.config.yarn), (5, Some(2.0)));

        set.common = json!({"bean": 3});
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("bean")
            .assert_err(&resolve_presets(&base(), &set, &PresetSelection::default()));
    }
}
