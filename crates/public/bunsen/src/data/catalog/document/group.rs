use serde::{
    Deserialize,
    Serialize,
};

use crate::data::catalog::{
    document::{
        CatalogEntry,
        MergeMode,
    },
    name::CatalogRef,
};

/// A `namespace` group.
///
/// A **declaring** occurrence (`$merge: deny`, the default) carries the
/// group's content. An **extending** occurrence
/// (`$merge: update`) carries only its `$kind`, optionally the `$facet` it
/// expects the group to have, and children: it opens the group's scope to add
/// children to it, and asserts what it expects of the group.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupEntry {
    /// The group's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the group is, for a person. Required on a declaring occurrence.
    #[serde(
        rename = "$description",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,

    /// The facet the group belongs to. On an extending occurrence, the facet
    /// the declaration must have.
    #[serde(rename = "$facet", default, skip_serializing_if = "Option::is_none")]
    pub facet: Option<CatalogRef>,

    /// Further memberships.
    #[serde(rename = "$join", default, skip_serializing_if = "Vec::is_empty")]
    pub join: Vec<CatalogRef>,

    /// Tags every member build must have a value for.
    #[serde(rename = "$requires", default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<CatalogRef>,

    /// Entries nested in the group's scope.
    #[serde(skip)]
    pub children: Vec<CatalogEntry>,
}
