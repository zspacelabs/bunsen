use serde::{
    Deserialize,
    Serialize,
};

use crate::data::catalog::document::{
    MergeMode,
    TypeExpr,
};

/// A tag declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TagEntry {
    /// The tag's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the tag says, for a person.
    #[serde(rename = "$description")]
    pub description: String,

    /// The values the tag takes.
    #[serde(rename = "$type")]
    pub tag_type: TypeExpr,
}

/// An axis's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisKind {
    /// Two generated members, `X` and `no-X`, for the axis `X`, with a
    /// negation between them.
    Bool,
    /// Declared members, which may grow; no negation.
    Categorical,
}

/// An axis declaration. It is also the declaration of the tag the axis
/// implies on every build that uses it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxisEntry {
    /// The axis's own segment. A bool axis's positive member has this name.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the axis selects, for a person.
    #[serde(rename = "$description")]
    pub description: String,

    /// The axis's kind.
    #[serde(rename = "$axis")]
    pub axis: AxisKind,

    /// A categorical axis's members; a bool axis has none written.
    #[serde(rename = "$members", default, skip_serializing_if = "Vec::is_empty")]
    pub members: Vec<String>,
}

impl AxisEntry {
    /// The axis's members: the declared ones, or `[X, no-X]` for a bool
    /// axis `X`.
    pub fn member_names(&self) -> Vec<String> {
        match self.axis {
            AxisKind::Bool => vec![self.name.clone(), format!("no-{}", self.name)],
            AxisKind::Categorical => self.members.clone(),
        }
    }

    /// The negation of `member` on a bool axis; `None` on a categorical
    /// axis or for a non-member.
    pub fn negate(
        &self,
        member: &str,
    ) -> Option<String> {
        if self.axis != AxisKind::Bool {
            return None;
        }
        let negative = format!("no-{}", self.name);
        if member == self.name {
            Some(negative)
        } else if member == negative {
            Some(self.name.clone())
        } else {
            None
        }
    }
}

/// A facet declaration: a named, exclusive set of groups. A build is in at
/// most one group of each facet.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacetEntry {
    /// The facet's name; facets are top-level only.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the facet sorts groups by, for a person.
    #[serde(rename = "$description")]
    pub description: String,
}
