use alloc::collections::BTreeMap;

use serde::{
    Deserialize,
    Serialize,
};

use crate::data::catalog::document::{
    CatalogEntry,
    MergeMode,
    TypeExpr,
    support::is_false,
};

/// A contract: what a loader implements, as the resources and arguments a
/// build binds ([`ContractEntry::resources`], [`ContractEntry::args`]).
///
/// Contracts live in `$contracts`. A contract may hold variants, child
/// contracts that add slots to its own, and `type` entries. A build names
/// one contract that isn't [`abstract_`](Self::abstract_) with `$contract`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractEntry {
    /// The contract's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the loader is, for a person.
    #[serde(rename = "$description")]
    pub description: String,

    /// Whether no build may name it: it holds the slots its variants share.
    #[serde(rename = "$abstract", default, skip_serializing_if = "is_false")]
    pub abstract_: bool,

    /// The resource slots it adds, by slot name.
    #[serde(
        rename = "$resources",
        default,
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub resources: BTreeMap<String, ResourceSlot>,

    /// The argument slots it adds, by slot name.
    #[serde(rename = "$args", default, skip_serializing_if = "BTreeMap::is_empty")]
    pub args: BTreeMap<String, ArgSlot>,

    /// Its variants and its types.
    #[serde(skip)]
    pub children: Vec<CatalogEntry>,
}

/// A contract's resource slot: a resource the loader reads.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceSlot {
    /// What the slot is for, for a person.
    #[serde(
        rename = "$description",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,

    /// The formats it takes, by a resource's `$format`; any, when empty.
    #[serde(rename = "$format", default, skip_serializing_if = "Vec::is_empty")]
    pub formats: Vec<String>,

    /// Whether a build may leave it unbound.
    #[serde(rename = "$optional", default, skip_serializing_if = "is_false")]
    pub optional: bool,
}

/// A contract's argument slot: a value the loader is told.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArgSlot {
    /// What the slot is for, for a person.
    #[serde(
        rename = "$description",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,

    /// What it takes; references resolve from the contract.
    #[serde(rename = "$type")]
    pub type_expr: TypeExpr,

    /// Whether a build may leave it unbound.
    #[serde(rename = "$optional", default, skip_serializing_if = "is_false")]
    pub optional: bool,
}

/// A type: a named, documented type expression.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypeEntry {
    /// The type's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the type's values are, for a person.
    #[serde(rename = "$description")]
    pub description: String,

    /// The expression it names; references resolve from the type.
    #[serde(rename = "$type")]
    pub type_expr: TypeExpr,
}
