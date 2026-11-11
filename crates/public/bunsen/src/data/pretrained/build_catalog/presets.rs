//! A build's presets: the template-class chain's sets folded into one
//! [`ResolvedPresetSet`], and choosing presets from it.
//!
//! Folding runs root first. At each link a same-named preset meets the
//! inherited one by its `merge` ([`PresetEntry::merge`]): by default its
//! patch merges over the inherited patch; `replace` takes the inherited
//! preset's place, and `deny` refuses one. A same-named axis use replaces the
//! one above it and may only **narrow** what that offered, a non-empty
//! `default` list replaces the one above it, and the common patches merge.
//!
//! A preset whose name is a member of a used axis is on that axis. Every
//! offered member is a preset, and what isn't offered can't be selected:
//! when a link narrows a use, the presets of the members it drops are
//! pruned. A link that itself declares a member it doesn't offer (or that a
//! use above it no longer offers) is an error. An offer of one member is a
//! fixed mode.

use alloc::collections::BTreeSet;

use serde_json::Value;

use crate::{
    data::pretrained::build_catalog::{
        document::{
            AxisKind,
            MergeMode,
            PresetEntry,
            PresetSetEntry,
        },
        error::{
            CatalogError,
            CatalogErrorKind,
            CatalogResult,
        },
        name::{
            CatalogName,
            CatalogRef,
            is_segment,
        },
    },
    support::merge_patch,
};

/// What folding needs to know of a registered axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisInfo {
    /// The axis's full name.
    pub name: CatalogName,
    /// Its kind.
    pub kind: AxisKind,
    /// Its members.
    pub members: Vec<String>,
}

impl AxisInfo {
    /// The positive member of a bool axis: the axis's own segment.
    pub fn positive(&self) -> &str {
        self.name.leaf()
    }

    /// The negation of `member`, on a bool axis.
    pub fn negate(
        &self,
        member: &str,
    ) -> Option<String> {
        if self.kind != AxisKind::Bool {
            return None;
        }
        let positive = self.positive();
        let negative = format!("no-{positive}");
        if member == positive {
            Some(negative)
        } else if member == negative {
            Some(positive.to_string())
        } else {
            None
        }
    }
}

/// One axis use after folding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedAxisUse {
    /// The axis.
    pub axis: AxisInfo,
    /// The members offered.
    pub offers: Vec<String>,
    /// The member applied when a selection names none.
    pub default: String,
}

impl ResolvedAxisUse {
    /// Whether one member is offered: a fixed mode.
    pub fn is_fixed(&self) -> bool {
        self.offers.len() == 1
    }
}

/// One link of a chain: who wrote the set, and the set. The set's axis
/// references resolve from its owner.
#[derive(Clone, Copy, Debug)]
pub struct PresetLink<'a> {
    /// The template class or build that wrote it.
    pub owner: &'a CatalogName,
    /// The set.
    pub set: &'a PresetSetEntry,
}

/// A build's presets, folded along its chain.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPresetSet {
    /// The merged common patch.
    pub common: Value,
    /// The presets.
    pub presets: Vec<PresetEntry>,
    /// The axis uses.
    pub axes: Vec<ResolvedAxisUse>,
    /// Presets on no axis that apply unless selected over.
    pub default: Vec<String>,
}

/// What [`ResolvedPresetSet::select`] asks for: preset names, and a final
/// patch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PresetSelection {
    /// The presets selected.
    pub presets: Vec<String>,
    /// A patch applied after every preset.
    pub patch: Option<Value>,
}

impl PresetSelection {
    /// A selection of `presets`.
    pub fn of<I, S>(presets: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            presets: presets.into_iter().map(Into::into).collect(),
            patch: None,
        }
    }
}

/// A resolved document, and what resolved it.
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedPresets {
    /// The document: the caller's base with every layer applied.
    pub document: Value,
    /// The presets applied, in order.
    pub applied: Vec<String>,
}

impl ResolvedPresetSet {
    /// Folds `links`, root first, resolving each link's axis references
    /// with `axis`.
    ///
    /// # Errors
    /// Every [`CatalogErrorKind::Preset`] or [`CatalogErrorKind::Reference`]
    /// problem found.
    pub fn fold(
        links: &[PresetLink<'_>],
        axis: &dyn Fn(&PresetLink<'_>, &CatalogRef) -> CatalogResult<AxisInfo>,
    ) -> Result<Self, Vec<CatalogError>> {
        let mut errors = Vec::new();
        let mut set = Self {
            common: Value::Object(Default::default()),
            presets: Vec::new(),
            axes: Vec::new(),
            default: Vec::new(),
        };
        // The link that last declared each preset, and last set each use.
        let mut declared_at: Vec<usize> = Vec::new();
        let mut used_at: Vec<usize> = Vec::new();
        for (at, link) in links.iter().enumerate() {
            let owner = link.owner;
            if !link.set.common.is_object() {
                errors.push(preset_error(owner, "the common patch is not an object"));
            } else {
                merge_patch::apply(&mut set.common, &link.set.common);
            }
            let mut seen = BTreeSet::new();
            for preset in &link.set.presets {
                if !seen.insert(preset.name.as_str()) {
                    errors.push(preset_error(
                        owner,
                        format!("the preset `{}` is declared twice", preset.name),
                    ));
                    continue;
                }
                match set.presets.iter().position(|p| p.name == preset.name) {
                    Some(i) => match preset.merge {
                        MergeMode::Update => {
                            let above = &mut set.presets[i];
                            merge_patch::apply(&mut above.patch, &preset.patch);
                            if !preset.description.trim().is_empty() {
                                above.description = preset.description.clone();
                            }
                            declared_at[i] = at;
                        }
                        MergeMode::Replace => {
                            set.presets[i] = PresetEntry {
                                merge: MergeMode::Update,
                                ..preset.clone()
                            };
                            declared_at[i] = at;
                        }
                        MergeMode::Deny => errors.push(preset_error(
                            owner,
                            format!(
                                "the preset `{}` is `$merge: deny`, but the chain above already \
                                 declares it",
                                preset.name
                            ),
                        )),
                    },
                    None => {
                        set.presets.push(PresetEntry {
                            merge: MergeMode::Update,
                            ..preset.clone()
                        });
                        declared_at.push(at);
                    }
                }
            }
            for using in &link.set.axes {
                let info = match axis(link, &using.axis) {
                    Ok(info) => info,
                    Err(e) => {
                        errors.push(e.at(owner));
                        continue;
                    }
                };
                let axis_name = info.name.clone();
                let resolved = ResolvedAxisUse {
                    axis: info,
                    offers: using.offers.clone(),
                    default: using.default.clone(),
                };
                match set.axes.iter().position(|a| a.axis.name == axis_name) {
                    Some(i) => {
                        let above = &mut set.axes[i];
                        let widened: Vec<&String> = resolved
                            .offers
                            .iter()
                            .filter(|o| !above.offers.contains(o))
                            .collect();
                        if !widened.is_empty() {
                            errors.push(preset_error(
                                owner,
                                format!(
                                    "the use of `{}` offers {widened:?}, which its chain doesn't \
                                     (offers {:?}): a use may only narrow",
                                    axis_name, above.offers
                                ),
                            ));
                        }
                        *above = resolved;
                        used_at[i] = at;
                    }
                    None => {
                        set.axes.push(resolved);
                        used_at.push(at);
                    }
                }
            }
            if !link.set.default.is_empty() {
                set.default = link.set.default.clone();
            }
        }
        // Prune the presets of members a use no longer offers, when the use
        // was set below the preset's last declaration.
        let mut keep = Vec::new();
        for (preset, &declared) in set.presets.iter().zip(&declared_at) {
            let dropped_by = set.axes.iter().zip(&used_at).find(|(a, _)| {
                a.axis.members.contains(&preset.name) && !a.offers.contains(&preset.name)
            });
            match dropped_by {
                Some((_, &used)) if used > declared => keep.push(false),
                _ => keep.push(true),
            }
        }
        let mut flags = keep.into_iter();
        set.presets
            .retain(|_| flags.next().expect("one flag per preset"));
        let owner = links
            .last()
            .map(|l| l.owner.clone())
            .unwrap_or_else(|| CatalogName::root("presets").expect("a legal segment"));
        errors.extend(set.check(&owner));
        if errors.is_empty() {
            Ok(set)
        } else {
            Err(errors)
        }
    }

    fn check(
        &self,
        owner: &CatalogName,
    ) -> Vec<CatalogError> {
        let mut errors = Vec::new();
        let err = |m: String| preset_error(owner, m);
        for preset in &self.presets {
            if !is_segment(&preset.name) {
                errors.push(err(format!("`{}` is not a preset name", preset.name)));
            }
            if preset.description.trim().is_empty() {
                errors.push(err(format!(
                    "the preset `{}` has no description",
                    preset.name
                )));
            }
        }
        let mut on_an_axis: Vec<(&str, &CatalogName)> = Vec::new();
        for using in &self.axes {
            let name = &using.axis.name;
            if using.offers.is_empty() {
                errors.push(err(format!("the use of `{name}` offers nothing")));
            }
            let mut seen = BTreeSet::new();
            for offer in &using.offers {
                if !seen.insert(offer) {
                    errors.push(err(format!("the use of `{name}` offers `{offer}` twice")));
                }
                if !using.axis.members.contains(offer) {
                    errors.push(err(format!(
                        "the use of `{name}` offers `{offer}`, which is not one of its members {:?}",
                        using.axis.members
                    )));
                } else if self.get(offer).is_none() {
                    errors.push(err(format!(
                        "the use of `{name}` offers `{offer}`, which is not a preset"
                    )));
                }
            }
            if !using.offers.contains(&using.default) {
                errors.push(err(format!(
                    "the use of `{name}` defaults to `{}`, which it doesn't offer",
                    using.default
                )));
            }
            for member in &using.axis.members {
                if !using.offers.contains(member) && self.get(member).is_some() {
                    errors.push(err(format!(
                        "the preset `{member}` is declared at or below a use of `{name}` that \
                         doesn't offer it"
                    )));
                }
                if let Some((_, other)) = on_an_axis.iter().find(|(m, _)| m == member) {
                    errors.push(err(format!(
                        "`{member}` is a member of both `{other}` and `{name}`, which are both used"
                    )));
                }
                on_an_axis.push((member, name));
            }
        }
        let mut seen = BTreeSet::new();
        for name in &self.default {
            if self.get(name).is_none() {
                errors.push(err(format!(
                    "the default names `{name}`, which is not a preset"
                )));
            }
            if !seen.insert(name) {
                errors.push(err(format!("the default names `{name}` twice")));
            }
            if let Some(using) = self.axis_of(name) {
                errors.push(err(format!(
                    "the default names `{name}`, which is on `{}`: the axis use sets its own default",
                    using.axis.name
                )));
            }
        }
        let defaults = self.default_names();
        let patches: Vec<&Value> = defaults
            .iter()
            .filter_map(|n| self.get(n).map(|p| &p.patch))
            .collect();
        if let Some(conflict) = merge_patch::conflicts(&patches).first() {
            let who: Vec<&str> = conflict
                .patches
                .iter()
                .map(|&i| defaults[i].as_str())
                .collect();
            errors.push(err(format!(
                "the defaults {who:?} each set `{}`",
                conflict.path
            )));
        }
        errors
    }

    /// The preset named `name`.
    pub fn get(
        &self,
        name: &str,
    ) -> Option<&PresetEntry> {
        self.presets.iter().find(|p| p.name == name)
    }

    /// The axis use `name` is a member of, if any.
    pub fn axis_of(
        &self,
        name: &str,
    ) -> Option<&ResolvedAxisUse> {
        self.axes
            .iter()
            .find(|a| a.axis.members.iter().any(|m| m == name))
    }

    /// The use of the axis named `axis`, if the set uses it.
    pub fn axis_use(
        &self,
        axis: &CatalogName,
    ) -> Option<&ResolvedAxisUse> {
        self.axes.iter().find(|a| &a.axis.name == axis)
    }

    /// The presets that apply when nothing is selected: each axis use's
    /// default, then the `default` list.
    pub fn default_names(&self) -> Vec<String> {
        self.axes
            .iter()
            .map(|a| a.default.clone())
            .chain(self.default.iter().cloned())
            .collect()
    }

    /// The negation of a preset on a used bool axis, if it is offered.
    pub fn negate(
        &self,
        preset: &str,
    ) -> Option<String> {
        let using = self.axis_of(preset)?;
        using
            .axis
            .negate(preset)
            .filter(|n| using.offers.contains(n))
    }

    /// The member of the bool axis `axis` that `on` names: the shortcut a
    /// flag such as `--think` / `--no-think` stands for.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] when the set doesn't use the axis,
    /// the axis is not a bool, or the member is not offered (the mode is
    /// fixed the other way).
    pub fn bool_member(
        &self,
        axis: &CatalogName,
        on: bool,
    ) -> CatalogResult<String> {
        let Some(using) = self.axis_use(axis) else {
            return Err(CatalogError::new(
                CatalogErrorKind::Selection,
                format!("the build doesn't use the axis `{axis}`"),
            ));
        };
        if using.axis.kind != AxisKind::Bool {
            return Err(CatalogError::new(
                CatalogErrorKind::Selection,
                format!("`{axis}` is not a bool axis"),
            ));
        }
        let positive = using.axis.positive().to_string();
        let member = if on {
            positive
        } else {
            format!("no-{positive}")
        };
        if using.offers.contains(&member) {
            Ok(member)
        } else {
            Err(not_offered(using, &member))
        }
    }

    /// Checks `selection`'s preset names and returns them as declared.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Selection`] when a name is unknown or not
    /// offered, a preset is named twice, two members of one axis are named,
    /// or two selected presets set the same path.
    pub fn check_selection(
        &self,
        selection: &PresetSelection,
    ) -> CatalogResult<Vec<String>> {
        let mut selected: Vec<String> = Vec::new();
        for name in &selection.presets {
            let preset = self.get(name).or_else(|| {
                self.presets
                    .iter()
                    .find(|p| p.name.eq_ignore_ascii_case(name))
            });
            let Some(preset) = preset else {
                if let Some(using) = self
                    .axes
                    .iter()
                    .find(|a| a.axis.members.iter().any(|m| m == name))
                {
                    return Err(not_offered(using, name));
                }
                let known: Vec<&str> = self.presets.iter().map(|p| p.name.as_str()).collect();
                return Err(CatalogError::new(
                    CatalogErrorKind::Selection,
                    format!("no preset `{name}`; the build has {known:?}"),
                ));
            };
            let name = &preset.name;
            if selected.contains(name) {
                return Err(CatalogError::new(
                    CatalogErrorKind::Selection,
                    format!("the preset `{name}` is selected twice"),
                ));
            }
            if let Some(using) = self.axis_of(name)
                && let Some(other) = selected
                    .iter()
                    .find(|s| using.axis.members.iter().any(|m| m == *s))
            {
                return Err(CatalogError::new(
                    CatalogErrorKind::Selection,
                    format!(
                        "`{other}` and `{name}` are both on the axis `{}`: select one",
                        using.axis.name
                    ),
                ));
            }
            selected.push(name.clone());
        }
        let patches: Vec<&Value> = selected
            .iter()
            .map(|n| &self.get(n).expect("found above").patch)
            .collect();
        if let Some(conflict) = merge_patch::conflicts(&patches).first() {
            let who: Vec<&str> = conflict
                .patches
                .iter()
                .map(|&i| selected[i].as_str())
                .collect();
            return Err(CatalogError::new(
                CatalogErrorKind::Selection,
                format!(
                    "the presets {who:?} each set `{}`: select one",
                    conflict.path
                ),
            ));
        }
        Ok(selected)
    }

    /// The presets applied for `selected`, in order: each axis use's
    /// default unless a member of that axis is selected, the `default`
    /// list less what is selected, then the selected.
    pub fn applied(
        &self,
        selected: &[String],
    ) -> Vec<String> {
        let mut applied: Vec<String> = Vec::new();
        for using in &self.axes {
            let displaced = selected
                .iter()
                .any(|s| using.axis.members.iter().any(|m| m == s));
            if !displaced {
                applied.push(using.default.clone());
            }
        }
        applied.extend(
            self.default
                .iter()
                .filter(|d| !selected.contains(d))
                .cloned(),
        );
        applied.extend(selected.iter().cloned());
        applied
    }

    /// Resolves `base` through the set as `selection` asks: the set's common
    /// patch, the applied presets in order, then the selection's patch.
    ///
    /// # Errors
    /// As [`check_selection`](Self::check_selection).
    pub fn select(
        &self,
        base: Value,
        selection: &PresetSelection,
    ) -> CatalogResult<AppliedPresets> {
        let selected = self.check_selection(selection)?;
        let applied = self.applied(&selected);
        let mut document = base;
        merge_patch::apply(&mut document, &self.common);
        for name in &applied {
            merge_patch::apply(&mut document, &self.get(name).expect("checked").patch);
        }
        if let Some(patch) = &selection.patch {
            merge_patch::apply(&mut document, patch);
        }
        Ok(AppliedPresets { document, applied })
    }
}

fn not_offered(
    using: &ResolvedAxisUse,
    member: &str,
) -> CatalogError {
    let how = if using.is_fixed() {
        format!("fixed at `{}`", using.offers[0])
    } else {
        format!("offers only {:?}", using.offers)
    };
    CatalogError::new(
        CatalogErrorKind::Selection,
        format!(
            "`{member}` is not offered: on this build the axis `{}` is {how}",
            using.axis.name
        ),
    )
}

fn preset_error(
    owner: &CatalogName,
    message: impl Into<String>,
) -> CatalogError {
    CatalogError::new(CatalogErrorKind::Preset, message).at(owner)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::data::pretrained::build_catalog::document::AxisUse;

    fn thinking() -> AxisInfo {
        AxisInfo {
            name: CatalogName::root("thinking").unwrap(),
            kind: AxisKind::Bool,
            members: vec!["thinking".into(), "no-thinking".into()],
        }
    }

    fn lookup(
        _link: &PresetLink<'_>,
        name: &CatalogRef,
    ) -> CatalogResult<AxisInfo> {
        if name.to_string() == "thinking" {
            Ok(thinking())
        } else {
            Err(CatalogError::new(
                CatalogErrorKind::Reference,
                format!("no axis `{name}`"),
            ))
        }
    }

    fn preset(
        name: &str,
        patch: Value,
    ) -> PresetEntry {
        PresetEntry {
            name: name.into(),
            merge: MergeMode::Update,
            description: format!("{name}."),
            patch,
        }
    }

    fn uses(
        offers: &[&str],
        default: &str,
    ) -> AxisUse {
        AxisUse {
            axis: CatalogRef::parse("thinking").unwrap(),
            offers: offers.iter().map(|s| s.to_string()).collect(),
            default: default.into(),
        }
    }

    fn hybrid() -> PresetSetEntry {
        PresetSetEntry {
            common: json!({"generation": {"max_tokens": 512}}),
            presets: vec![
                preset("thinking", json!({"generation": {"temperature": 0.6}})),
                preset(
                    "no-thinking",
                    json!({"generation": {"temperature": 0.7}, "chat": {"think": false}}),
                ),
                preset("greedy", json!({"generation": {"greedy": true}})),
            ],
            axes: vec![uses(&["thinking", "no-thinking"], "thinking")],
            default: vec![],
        }
    }

    fn fold(sets: &[&PresetSetEntry]) -> Result<ResolvedPresetSet, Vec<CatalogError>> {
        let names: Vec<CatalogName> = (0..sets.len())
            .map(|i| CatalogName::root(&format!("link{i}")).unwrap())
            .collect();
        let links: Vec<PresetLink<'_>> = sets
            .iter()
            .zip(&names)
            .map(|(set, owner)| PresetLink { owner, set })
            .collect();
        ResolvedPresetSet::fold(&links, &lookup)
    }

    #[test]
    fn test_defaults_apply_and_selection_displaces_them() {
        let set = fold(&[&hybrid()]).unwrap();
        let none = set.select(json!({}), &PresetSelection::default()).unwrap();
        assert_eq!(none.applied, ["thinking"]);
        assert_eq!(
            none.document,
            json!({"generation": {"max_tokens": 512, "temperature": 0.6}})
        );
        let off = set
            .select(json!({}), &PresetSelection::of(["no-thinking", "greedy"]))
            .unwrap();
        assert_eq!(off.applied, ["no-thinking", "greedy"]);
        assert_eq!(off.document["chat"], json!({"think": false}));
        assert_eq!(off.document["generation"]["greedy"], json!(true));
    }

    #[test]
    fn test_selection_patch_applies_last() {
        let set = fold(&[&hybrid()]).unwrap();
        let sel = PresetSelection {
            presets: vec![],
            patch: Some(json!({"generation": {"temperature": 0.1}})),
        };
        let out = set.select(json!({"keep": 1}), &sel).unwrap();
        assert_eq!(out.document["generation"]["temperature"], json!(0.1));
        assert_eq!(out.document["keep"], json!(1));
    }

    #[test]
    fn test_a_fixed_use_narrows_and_refuses_the_other_member() {
        let fixed = PresetSetEntry {
            axes: vec![uses(&["thinking"], "thinking")],
            ..Default::default()
        };
        // Narrowing below the declaration prunes the dropped member's preset.
        let set = fold(&[&hybrid(), &fixed]).unwrap();
        assert!(set.get("no-thinking").is_none());
        assert!(set.get("greedy").is_some());
        assert!(set.axes[0].is_fixed());
        let err = set
            .check_selection(&PresetSelection::of(["no-thinking"]))
            .unwrap_err();
        assert!(err.message.contains("fixed at `thinking`"), "{err}");
        assert!(set.bool_member(&thinking().name, false).is_err());
        assert_eq!(set.bool_member(&thinking().name, true).unwrap(), "thinking");
        assert_eq!(set.negate("thinking"), None);
    }

    #[test]
    fn test_declaring_an_unoffered_member_is_refused() {
        // The same link declares `no-thinking` and doesn't offer it.
        let mut same_link = hybrid();
        same_link.axes = vec![uses(&["thinking"], "thinking")];
        let errors = fold(&[&same_link]).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("doesn't offer it")),
            "{errors:?}"
        );

        // A link below a narrowing use declares it again.
        let fixed = PresetSetEntry {
            axes: vec![uses(&["thinking"], "thinking")],
            ..Default::default()
        };
        let redeclare = PresetSetEntry {
            presets: vec![preset("no-thinking", json!({}))],
            ..Default::default()
        };
        let errors = fold(&[&hybrid(), &fixed, &redeclare]).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("doesn't offer it")),
            "{errors:?}"
        );
    }

    #[test]
    fn test_widening_is_refused() {
        let narrow = PresetSetEntry {
            presets: vec![preset("thinking", json!({}))],
            axes: vec![uses(&["thinking"], "thinking")],
            ..Default::default()
        };
        let wide = PresetSetEntry {
            presets: vec![preset("no-thinking", json!({}))],
            axes: vec![uses(&["thinking", "no-thinking"], "thinking")],
            ..Default::default()
        };
        let errors = fold(&[&narrow, &wide]).unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("may only narrow")));
    }

    #[test]
    fn test_lower_links_merge_patches_and_move_defaults() {
        let small = PresetSetEntry {
            presets: vec![PresetEntry {
                name: "thinking".into(),
                merge: MergeMode::Update,
                description: String::new(),
                patch: json!({"generation": {"top_p": 0.9}}),
            }],
            axes: vec![uses(&["thinking", "no-thinking"], "no-thinking")],
            ..Default::default()
        };
        let set = fold(&[&hybrid(), &small]).unwrap();
        assert_eq!(set.default_names(), ["no-thinking"]);
        assert_eq!(
            set.get("thinking").unwrap().patch,
            json!({"generation": {"temperature": 0.6, "top_p": 0.9}})
        );
        assert_eq!(set.get("thinking").unwrap().description, "thinking.");
        assert_eq!(set.negate("thinking").as_deref(), Some("no-thinking"));
    }

    fn with_merge(
        merge: MergeMode,
        preset: PresetEntry,
    ) -> PresetEntry {
        PresetEntry { merge, ..preset }
    }

    #[test]
    fn test_a_replacing_preset_takes_the_inherited_ones_place() {
        let lower = PresetSetEntry {
            presets: vec![with_merge(
                MergeMode::Replace,
                preset("thinking", json!({"generation": {"top_p": 0.9}})),
            )],
            ..Default::default()
        };
        let set = fold(&[&hybrid(), &lower]).unwrap();
        let thinking = set.get("thinking").unwrap();
        assert_eq!(thinking.patch, json!({"generation": {"top_p": 0.9}}));
        assert_eq!(thinking.merge, MergeMode::Update);
        // The others are inherited as they were.
        assert_eq!(
            set.get("greedy").unwrap().patch,
            json!({"generation": {"greedy": true}})
        );
    }

    #[test]
    fn test_a_denying_preset_must_be_new() {
        let lower = PresetSetEntry {
            presets: vec![with_merge(
                MergeMode::Deny,
                preset("greedy", json!({"generation": {"beams": 4}})),
            )],
            ..Default::default()
        };
        let errors = fold(&[&hybrid(), &lower]).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("`greedy` is `$merge: deny`")),
            "{errors:?}"
        );
        // A new name is declared, whatever its mode.
        let new = PresetSetEntry {
            presets: vec![
                with_merge(MergeMode::Deny, preset("beam", json!({"beams": 4}))),
                with_merge(MergeMode::Replace, preset("short", json!({"max": 8}))),
            ],
            ..Default::default()
        };
        let set = fold(&[&hybrid(), &new]).unwrap();
        assert_eq!(set.get("beam").unwrap().patch, json!({"beams": 4}));
        assert_eq!(set.get("short").unwrap().patch, json!({"max": 8}));
    }

    #[test]
    fn test_selection_errors() {
        let set = fold(&[&hybrid()]).unwrap();
        let err = |names: &[&str]| {
            set.check_selection(&PresetSelection::of(names.iter().copied()))
                .unwrap_err()
                .message
        };
        assert!(err(&["fast"]).contains("no preset `fast`"));
        assert!(err(&["thinking", "no-thinking"]).contains("both on the axis"));
        assert!(err(&["greedy", "greedy"]).contains("twice"));
        assert_eq!(
            set.check_selection(&PresetSelection::of(["GREEDY"]))
                .unwrap(),
            ["greedy"]
        );
    }

    #[test]
    fn test_bad_uses_are_reported() {
        let bad = PresetSetEntry {
            presets: vec![preset("thinking", json!({}))],
            axes: vec![
                uses(&["thinking", "maybe"], "no-thinking"),
                AxisUse {
                    axis: CatalogRef::parse("nope").unwrap(),
                    offers: vec!["x".into()],
                    default: "x".into(),
                },
            ],
            default: vec!["thinking".into(), "ghost".into()],
            ..Default::default()
        };
        let errors = fold(&[&bad]).unwrap_err();
        let text: Vec<&str> = errors.iter().map(|e| e.message.as_str()).collect();
        assert!(
            text.iter().any(|t| t.contains("not one of its members")),
            "{text:?}"
        );
        assert!(text.iter().any(|t| t.contains("which it doesn't offer")));
        assert!(text.iter().any(|t| t.contains("no axis `nope`")));
        assert!(
            text.iter()
                .any(|t| t.contains("`ghost`, which is not a preset"))
        );
        assert!(text.iter().any(|t| t.contains("sets its own default")));
    }

    #[test]
    fn test_conflicting_defaults_are_reported() {
        let set = PresetSetEntry {
            presets: vec![
                preset("thinking", json!({"t": 1})),
                preset("no-thinking", json!({"t": 2})),
                preset("also", json!({"t": 3})),
            ],
            axes: vec![uses(&["thinking", "no-thinking"], "thinking")],
            default: vec!["also".into()],
            ..Default::default()
        };
        let errors = fold(&[&set]).unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("each set `t`")));
    }
}
