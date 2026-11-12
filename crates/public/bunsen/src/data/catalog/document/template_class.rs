use alloc::collections::BTreeMap;

use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::data::catalog::{
    document::{
        MergeMode,
        PresetSetEntry,
    },
    name::CatalogRef,
};

/// The name of the template class every chain ends at.
///
/// A template class with no `$base` inherits it, and so does a build with
/// no `$template_class`. Its `$required_facets` hold for every build.
pub const TEMPLATE_CLASS_ROOT: &str = "template_class_root";

/// A template class: a preamble for builds.
///
/// Its memberships, tag values, presets, dependencies and requirements
/// become part of every build whose chain includes it. Its references
/// resolve where the class is written, not where its builds are. Where it
/// is placed confers no membership.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemplateClassEntry {
    /// The class's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the class shares, for a person.
    #[serde(rename = "$description")]
    pub description: String,

    /// The base class it inherits; [`TEMPLATE_CLASS_ROOT`] when absent.
    #[serde(rename = "$base", default, skip_serializing_if = "Option::is_none")]
    pub base: Option<CatalogRef>,

    /// Memberships every build of the class has.
    #[serde(rename = "$join", default, skip_serializing_if = "Vec::is_empty")]
    pub join: Vec<CatalogRef>,

    /// Tag values, by tag. They override the base class's.
    #[serde(skip)]
    pub tags: BTreeMap<CatalogRef, Value>,

    /// Presets, applied over the base class's.
    #[serde(skip)]
    pub presets: PresetSetEntry,

    /// The contract, by its name in `$contracts`. It overrides the base
    /// class's.
    #[serde(rename = "$contract", default, skip_serializing_if = "Option::is_none")]
    pub contract: Option<CatalogRef>,

    /// The contract's resource slots, bound to resources by reference. They
    /// override the base class's.
    #[serde(
        rename = "$resources",
        default,
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub resources: BTreeMap<String, CatalogRef>,

    /// The contract's argument slots, bound to values. They override
    /// the base class's.
    #[serde(rename = "$args", default, skip_serializing_if = "BTreeMap::is_empty")]
    pub args: BTreeMap<String, Value>,

    /// Facets every build of the class must be in exactly one group of.
    #[serde(
        rename = "$required_facets",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub required_facets: Vec<CatalogRef>,
}
