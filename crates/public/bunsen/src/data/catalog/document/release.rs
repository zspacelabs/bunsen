use serde::{
    Deserialize,
    Serialize,
};

use crate::data::catalog::{
    document::MergeMode,
    name::CatalogRef,
};

/// A `release` group: its members are releases of one sequence.
///
/// A build joins it with a `$released` timestamp ([`Membership::Release`]).
/// It holds no children.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseEntry {
    /// The group's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the sequence is, for a person.
    #[serde(rename = "$description")]
    pub description: String,

    /// Further memberships.
    #[serde(rename = "$join", default, skip_serializing_if = "Vec::is_empty")]
    pub join: Vec<CatalogRef>,
}

/// A build's membership in a group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Membership {
    /// A plain membership: a reference to the group.
    Plain(CatalogRef),
    /// A membership in a `release` group, with its release timestamp.
    Release(ReleaseMembership),
}

impl Membership {
    /// The group joined.
    pub fn group(&self) -> &CatalogRef {
        match self {
            Self::Plain(group) => group,
            Self::Release(r) => &r.group,
        }
    }
}

/// `{$group, $released}`: a release membership.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseMembership {
    /// The `release` group.
    #[serde(rename = "$group")]
    pub group: CatalogRef,

    /// When the build was released, in canonical form (`2025`, `2025-07`,
    /// `2025-07-15`, `2025-07-15T09:30Z`, `2025-07-15T09:30:00Z`; UTC).
    #[serde(rename = "$released")]
    pub released: String,
}
