use alloc::collections::BTreeMap;

use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::data::catalog::{
    document::{
        CatalogEntry,
        Membership,
        MergeMode,
        PresetSetEntry,
    },
    name::CatalogRef,
};

/// A build: something we can build.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildEntry {
    /// The build's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the build is, for a person.
    #[serde(
        rename = "$description",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub description: String,

    /// Further memberships, release memberships among them.
    #[serde(rename = "$join", default, skip_serializing_if = "Vec::is_empty")]
    pub join: Vec<Membership>,

    /// The template class the build inherits;
    /// [`TEMPLATE_CLASS_ROOT`](super::TEMPLATE_CLASS_ROOT) when absent.
    #[serde(
        rename = "$template_class",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub template_class: Option<CatalogRef>,

    /// The checkpoint's license.
    #[serde(rename = "$license", default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Where the checkpoint comes from, for a person.
    #[serde(rename = "$origin", default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,

    /// Tag values, by tag. They override the chain's.
    #[serde(skip)]
    pub tags: BTreeMap<CatalogRef, Value>,

    /// The build's own presets, applied over its chain's.
    #[serde(skip)]
    pub presets: PresetSetEntry,

    /// The contract, by its name in `$contracts`. It overrides the chain's.
    #[serde(rename = "$contract", default, skip_serializing_if = "Option::is_none")]
    pub contract: Option<CatalogRef>,

    /// The contract's resource slots, bound to resources by reference. They
    /// override the chain's.
    #[serde(
        rename = "$resources",
        default,
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub resources: BTreeMap<String, CatalogRef>,

    /// The contract's argument slots, bound to values. They override
    /// the chain's.
    #[serde(rename = "$args", default, skip_serializing_if = "BTreeMap::is_empty")]
    pub args: BTreeMap<String, Value>,

    /// Facets the build must be in exactly one group of, beyond its
    /// chain's.
    #[serde(
        rename = "$required_facets",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub required_facets: Vec<CatalogRef>,

    /// Entries in the build's scope: its own resources, tags and axes.
    #[serde(skip)]
    pub children: Vec<CatalogEntry>,
}
