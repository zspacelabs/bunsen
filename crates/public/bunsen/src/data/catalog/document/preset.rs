use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::data::catalog::{
    document::{
        MergeMode,
        support::{
            empty_object,
            is_empty_object,
        },
    },
    name::CatalogRef,
};

/// A preset set as a template class or build writes it, under `$presets`:
/// `$common`, `$axes` (keyed by the axis), `$default`, and the presets,
/// keyed by name.
#[derive(Clone, Debug, PartialEq)]
pub struct PresetSetEntry {
    /// A merge patch every selection keeps.
    pub common: Value,

    /// The presets, in the order written.
    pub presets: Vec<PresetEntry>,

    /// The registered axes used, with what each offers, in the order
    /// written.
    pub axes: Vec<AxisUse>,

    /// Presets on no axis that apply unless displaced.
    pub default: Vec<String>,
}

impl Default for PresetSetEntry {
    fn default() -> Self {
        Self {
            common: empty_object(),
            presets: Vec::new(),
            axes: Vec::new(),
            default: Vec::new(),
        }
    }
}

impl PresetSetEntry {
    /// Whether the set says nothing.
    pub fn is_empty(&self) -> bool {
        is_empty_object(&self.common)
            && self.presets.is_empty()
            && self.axes.is_empty()
            && self.default.is_empty()
    }
}

/// One preset: a name, a description, and a merge patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetEntry {
    /// The preset's name. A member of a used axis is on that axis.
    #[serde(skip)]
    pub name: String,

    /// How it meets a preset of its name inherited down the template-class
    /// chain: [`Update`](MergeMode::Update), the default, merges its patch
    /// over the inherited one; [`Replace`](MergeMode::Replace) takes the
    /// inherited one's place; [`Deny`](MergeMode::Deny) refuses one. With
    /// nothing inherited, the preset is declared, whatever its mode.
    #[serde(
        rename = "$merge",
        default = "MergeMode::update",
        skip_serializing_if = "MergeMode::is_update"
    )]
    pub merge: MergeMode,

    /// What selecting it does, for a person.
    #[serde(
        rename = "$description",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub description: String,

    /// The merge patch it applies.
    #[serde(
        rename = "$patch",
        default = "empty_object",
        skip_serializing_if = "is_empty_object"
    )]
    pub patch: Value,
}

/// A use of a registered axis: the members offered, and the default.
///
/// An offer of one member is a fixed mode. It is written under `$axes`,
/// keyed by the axis: `thinking: {$offers: [..], $default: ..}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisUse {
    /// The axis.
    pub axis: CatalogRef,

    /// The members offered: a non-empty subset of the axis's.
    pub offers: Vec<String>,

    /// The member that applies when a selection names none.
    pub default: String,
}
