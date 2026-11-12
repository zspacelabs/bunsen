use serde::{
    Deserialize,
    Serialize,
};

use crate::data::catalog::{
    document::{
        EntryKind,
        MergeMode,
    },
    name::CatalogRef,
};

/// An alias: another name for one entry, which a lookup for that entry's
/// kind resolves through. It may sit anywhere a namespace can hold it, and
/// an update or a replacement retargets it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasEntry {
    /// The alias's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// Why the alias exists, for a person.
    #[serde(
        rename = "$description",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,

    /// The entry it names.
    #[serde(rename = "$ref")]
    pub target: AliasTarget,
}

/// What an alias names: a reference, resolved from where the alias is
/// written, and the kind the entry must have.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasTarget {
    /// The entry, as a reference.
    #[serde(rename = "$path")]
    pub path: CatalogRef,

    /// Its kind; never `alias`.
    #[serde(rename = "$kind")]
    pub kind: EntryKind,
}
