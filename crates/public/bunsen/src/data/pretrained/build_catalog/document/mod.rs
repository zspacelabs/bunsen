//! The catalog document: [`PretrainedCatalog`] and its entries, with the
//! keyed form they are read from and written to (YAML or JSON).
//!
//! The document is the namespace tree. **Every key the catalog language
//! defines starts with `$`; a bare key is a name.**
//!
//! ```yaml
//! $description: "Qwen3 chat models."
//! qwen:                                  # an entry, keyed by its segment
//!   $kind: namespace
//!   $merge: update
//!   qwen3-0.6b:                          # a child: a map with `$kind`
//!     $kind: build
//!     $description: "Qwen3-0.6B."
//!     $template_class: qwen3/hybrid
//!     parameters: 600000000              # a tag
//!     checkpoint: {$kind: resource, $file: model.safetensors, $format: safetensors, $sources: []}
//! ```
//!
//! - **An entry** is a map with `$kind`, keyed by its segment. Its name is its
//!   path of keys: nesting is the only placement.
//! - **Bare keys under an entry:** under a namespace or a contract, children;
//!   under a template class, tags; under a build, a child when the value is a
//!   map with `$kind`, and a tag otherwise. Other kinds have none.
//! - **`$contracts`**, at the root of the base catalog only, holds the
//!   contracts and their `type` entries: the interfaces bunsen's loaders
//!   implement. A build names its contract with `$contract`.
//! - **`$presets`** holds `$common`, `$axes` (keyed by the axis), `$default`,
//!   and the presets, keyed by name.
//! - **Foreign documents are data:** a preset's `$patch` and `$common` keep the
//!   model's keys.
//! - **Refused:** a `$` word the kind doesn't define, a child without `$kind`,
//!   a bare key where the kind has none, and a repeated key.
//! - **Order is the file's:** `entries`, children, presets and axis uses keep
//!   the order they are written in.
//!
//! A reference to another entry ([`CatalogRef`]) resolves relative to the
//! entry it is written in, climbing the namespace to the root; a leading `/`
//! resolves from the root only.

use alloc::{
    collections::BTreeMap,
    fmt,
};

use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::data::pretrained::build_catalog::{
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogResult,
    },
    name::CatalogRef,
};

mod keyed;
mod types;

pub use keyed::OrderedNode;
pub use types::*;

/// The name of the template class every chain ends at.
///
/// A template class with no `$base` inherits it, and so does a build with
/// no `$template_class`. Its `$required_facets` hold for every build.
pub const TEMPLATE_CLASS_ROOT: &str = "template_class_root";

/// One catalog document: `$description`, `$contracts` in the base catalog,
/// and the top-level entries keyed by their segments ([module](self)).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PretrainedCatalog {
    /// What the catalog holds, for a person.
    pub description: Option<String>,

    /// The `$contracts` section: contracts and their types, in the order
    /// written. Only the base catalog has one, and it must; `None` when the
    /// document doesn't write it.
    pub contracts: Option<Vec<CatalogEntry>>,

    /// The top-level entries, in the order written.
    pub entries: Vec<CatalogEntry>,
}

impl PretrainedCatalog {
    /// Parses a YAML document.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it isn't a catalog.
    pub fn from_yaml(yaml: &str) -> CatalogResult<Self> {
        Self::from_node(OrderedNode::from_yaml(yaml)?)
    }

    /// Writes the catalog as YAML.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it doesn't serialize.
    pub fn to_yaml(&self) -> CatalogResult<String> {
        serde_saphyr::to_string(self)
            .map_err(|e| CatalogError::new(CatalogErrorKind::Parse, e.to_string()))
    }

    /// Parses a JSON document.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it isn't a catalog.
    pub fn from_json(json: &str) -> CatalogResult<Self> {
        Self::from_node(OrderedNode::from_json(json)?)
    }

    /// Reads the catalog `node` writes.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it isn't a catalog, with the path of
    /// the entry at fault.
    pub fn from_node(node: OrderedNode) -> CatalogResult<Self> {
        keyed::read_catalog(node)
    }

    /// The catalog as a tree: its entries, children, presets and axis uses
    /// in their order, and each entry's `$` fields in their declared order.
    pub fn to_node(&self) -> OrderedNode {
        keyed::write_catalog(self)
    }

    /// Writes the catalog as pretty JSON.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it doesn't serialize.
    pub fn to_json(&self) -> CatalogResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| CatalogError::new(CatalogErrorKind::Parse, e.to_string()))
    }
}

/// The closed set of entry kinds. Each has its own fields and its own
/// rules; a catalog can't add one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntryKind {
    /// A plain group: a naming scope, and a membership.
    Namespace,
    /// What a loader implements: the resources and arguments a build binds.
    /// Contracts live in `$contracts`, and nest as variants.
    Contract,
    /// A group whose members are releases of one sequence, ordered by time.
    Release,
    /// Something we can build.
    Build,
    /// A preamble that builds and other template classes inherit.
    TemplateClass,
    /// A declared tag: a key, a type and a description.
    Tag,
    /// A declared preset axis; it is also its own tag's declaration.
    Axis,
    /// A declared exclusive set of groups.
    Facet,
    /// One artifact.
    Resource,
    /// Another name for an entry of a stated kind, anywhere in the
    /// namespace.
    Alias,
    /// A named, documented type expression.
    Type,
}

impl EntryKind {
    /// Every kind.
    pub const ALL: [EntryKind; 11] = [
        Self::Namespace,
        Self::Contract,
        Self::Release,
        Self::Build,
        Self::TemplateClass,
        Self::Tag,
        Self::Axis,
        Self::Facet,
        Self::Resource,
        Self::Alias,
        Self::Type,
    ];

    /// Whether entries of this kind are groups (a build may be a member).
    pub fn is_group(self) -> bool {
        matches!(self, Self::Namespace | Self::Release)
    }

    /// Whether entries of this kind are groups that may hold children and
    /// be scoped.
    pub fn is_container_group(self) -> bool {
        matches!(self, Self::Namespace)
    }

    /// The kind the document spells `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.as_str() == name)
    }

    /// The kind as the document spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Namespace => "namespace",
            Self::Contract => "contract",
            Self::Release => "release",
            Self::Build => "build",
            Self::TemplateClass => "template_class",
            Self::Tag => "tag",
            Self::Axis => "axis",
            Self::Facet => "facet",
            Self::Resource => "resource",
            Self::Alias => "alias",
            Self::Type => "type",
        }
    }
}

impl Serialize for EntryKind {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for EntryKind {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::from_name(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("`{name}` is not a kind")))
    }
}

impl fmt::Display for EntryKind {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One entry, tagged by its `$kind`. It serializes as its keyed form, without
/// its name: the `$` fields, then its tags, `$presets` and its children.
#[derive(Clone, Debug, PartialEq)]
pub enum CatalogEntry {
    /// A `namespace` group.
    Namespace(GroupEntry),
    /// A contract, in `$contracts`.
    Contract(ContractEntry),
    /// A `release` group.
    Release(ReleaseEntry),
    /// A build.
    Build(BuildEntry),
    /// A template class.
    TemplateClass(TemplateClassEntry),
    /// A tag declaration.
    Tag(TagEntry),
    /// An axis declaration.
    Axis(AxisEntry),
    /// A facet declaration.
    Facet(FacetEntry),
    /// A resource declaration.
    Resource(ResourceEntry),
    /// An alias.
    Alias(AliasEntry),
    /// A type.
    Type(TypeEntry),
}

impl CatalogEntry {
    /// The entry's keyed form, without its name, as a tree.
    pub fn to_node(&self) -> OrderedNode {
        keyed::write_entry(self)
    }

    /// The entry's kind.
    pub fn kind(&self) -> EntryKind {
        match self {
            Self::Namespace(_) => EntryKind::Namespace,
            Self::Contract(_) => EntryKind::Contract,
            Self::Release(_) => EntryKind::Release,
            Self::Build(_) => EntryKind::Build,
            Self::TemplateClass(_) => EntryKind::TemplateClass,
            Self::Tag(_) => EntryKind::Tag,
            Self::Axis(_) => EntryKind::Axis,
            Self::Facet(_) => EntryKind::Facet,
            Self::Resource(_) => EntryKind::Resource,
            Self::Alias(_) => EntryKind::Alias,
            Self::Type(_) => EntryKind::Type,
        }
    }

    /// The entry's own name segment, as written.
    pub fn name(&self) -> &str {
        match self {
            Self::Namespace(e) => &e.name,
            Self::Contract(e) => &e.name,
            Self::Release(e) => &e.name,
            Self::Build(e) => &e.name,
            Self::TemplateClass(e) => &e.name,
            Self::Tag(e) => &e.name,
            Self::Axis(e) => &e.name,
            Self::Facet(e) => &e.name,
            Self::Resource(e) => &e.name,
            Self::Alias(e) => &e.name,
            Self::Type(e) => &e.name,
        }
    }

    /// The nested children, if the kind has any.
    pub fn children(&self) -> &[CatalogEntry] {
        match self {
            Self::Namespace(e) => &e.children,
            Self::Contract(e) => &e.children,
            Self::Build(e) => &e.children,
            _ => &[],
        }
    }

    /// Takes the nested children out, leaving none.
    pub(crate) fn take_children(&mut self) -> Vec<CatalogEntry> {
        match self {
            Self::Namespace(e) => core::mem::take(&mut e.children),
            Self::Contract(e) => core::mem::take(&mut e.children),
            Self::Build(e) => core::mem::take(&mut e.children),
            _ => Vec::new(),
        }
    }

    /// How the occurrence meets an existing declaration of its name.
    pub fn merge(&self) -> MergeMode {
        match self {
            Self::Namespace(e) => e.merge,
            Self::Contract(e) => e.merge,
            Self::Release(e) => e.merge,
            Self::Build(e) => e.merge,
            Self::TemplateClass(e) => e.merge,
            Self::Tag(e) => e.merge,
            Self::Axis(e) => e.merge,
            Self::Facet(e) => e.merge,
            Self::Resource(e) => e.merge,
            Self::Alias(e) => e.merge,
            Self::Type(e) => e.merge,
        }
    }

    /// The group fields, for a `namespace`.
    pub fn as_group(&self) -> Option<&GroupEntry> {
        match self {
            Self::Namespace(e) => Some(e),
            _ => None,
        }
    }
}

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

/// How an occurrence of a name meets an existing declaration of it.
///
/// What `update` means depends on the kind:
///
/// | kind | `update` |
/// |---|---|
/// | `namespace` | adds its children (each child's own `$merge` decides how it meets an existing child), and asserts the facet it expects; it states nothing else |
/// | `resource` | matches every field but `sources`, and puts its new sources first |
/// | `contract` | refused: a contract is the interface a loader is built against, and only the base catalog declares contracts |
/// | every other kind | refused |
///
/// `replace` is the same for every kind: the occurrence's own fields take
/// the place of the declaration's, for every reference to the name,
/// including references from catalogs below it. Children are entries of
/// their own and are untouched. The kind can't change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeMode {
    /// A new declaration: an existing one is a conflict.
    #[default]
    Deny,
    /// Updates an existing declaration, as its kind defines.
    Update,
    /// Substitutes the occurrence's own fields for the declaration's.
    Replace,
}

impl MergeMode {
    /// Whether this is the default, [`Deny`](Self::Deny).
    pub fn is_deny(&self) -> bool {
        *self == Self::Deny
    }

    /// [`Update`](Self::Update): a preset's default.
    pub fn update() -> Self {
        Self::Update
    }

    /// Whether this is [`Update`](Self::Update), a preset's default.
    pub fn is_update(&self) -> bool {
        *self == Self::Update
    }
}

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

fn is_false(b: &bool) -> bool {
    !*b
}

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
    #[serde(rename = "$description")]
    pub description: String,

    /// Further memberships, release memberships among them.
    #[serde(rename = "$join", default, skip_serializing_if = "Vec::is_empty")]
    pub join: Vec<Membership>,

    /// The template class the build inherits; [`TEMPLATE_CLASS_ROOT`] when
    /// absent.
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

/// A resource declaration: one artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceEntry {
    /// The resource's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the artifact is, for a person.
    #[serde(
        rename = "$description",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,

    /// The file name it is stored under.
    #[serde(rename = "$file")]
    pub file: String,

    /// The file's format, as its loader names it.
    #[serde(rename = "$format")]
    pub format: String,

    /// The file's SHA-256, in lowercase hex, when it is pinned.
    #[serde(rename = "$sha256", default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,

    /// The dtype a checkpoint stores, when it is one.
    #[serde(rename = "$dtype", default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Where to fetch it, tried in order.
    #[serde(rename = "$sources")]
    pub sources: Vec<ResourceSource>,
}

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

/// Where a resource can be fetched.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ResourceSource {
    /// A file in a Hugging Face Hub repo, at a revision: `{$hub: {$repo,
    /// $revision}}`.
    #[serde(rename = "$hub")]
    Hub {
        /// `org/name`.
        #[serde(rename = "$repo")]
        repo: String,
        /// A commit.
        #[serde(rename = "$revision")]
        revision: String,
    },
    /// A URL: `{$url: ..}`.
    #[serde(rename = "$url")]
    Url(String),
}

impl fmt::Display for ResourceSource {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Hub { repo, revision } => write!(f, "hub:{repo}@{revision}"),
            Self::Url(url) => f.write_str(url),
        }
    }
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

fn is_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(serde_json::Map::is_empty)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
family: { $kind: facet, $description: "A line of related architectures." }
thinking: { $kind: axis, $axis: bool, $description: "Whether the model thinks first." }
qwen:
  $kind: namespace
  $description: "Qwen."
  qwen3-4b:
    $kind: build
    $description: "Qwen3-4B."
    $join: [qwen/qwen3-line, { $group: qwen/qwen3-releases, $released: "2025-04" }]
    $resources: { config: qwen/qwen3-4b/config }
    parameters: 4000000000
    $presets:
      $axes:
        thinking: { $offers: [thinking], $default: thinking }
      thinking: { $merge: replace, $description: "Think.", $patch: { a: 1 } }
    config:
      $kind: resource
      $file: config.json
      $format: json
      $sources: [{ $hub: { $repo: Qwen/Qwen3-4B, $revision: abc } }, { $url: "https://x/y" }]
"#;

    #[test]
    fn test_yaml_parses_every_kind_written() {
        let doc = PretrainedCatalog::from_yaml(SAMPLE).unwrap();
        let kinds: Vec<EntryKind> = doc.entries.iter().map(CatalogEntry::kind).collect();
        assert_eq!(
            kinds,
            [EntryKind::Facet, EntryKind::Axis, EntryKind::Namespace]
        );
        let CatalogEntry::Build(build) = &doc.entries[2].children()[0] else {
            panic!("a build")
        };
        assert_eq!(build.join.len(), 2);
        assert!(matches!(build.join[1], Membership::Release(_)));
        assert_eq!(build.presets.axes[0].offers, ["thinking"]);
        assert_eq!(build.presets.presets[0].merge, MergeMode::Replace);
        let CatalogEntry::Resource(res) = &build.children[0] else {
            panic!("a resource")
        };
        assert_eq!(res.sources.len(), 2);
    }

    #[test]
    fn test_yaml_and_json_round_trip() {
        let doc = PretrainedCatalog::from_yaml(SAMPLE).unwrap();
        let json = doc.to_json().unwrap();
        assert_eq!(PretrainedCatalog::from_json(&json).unwrap(), doc);
        let yaml = doc.to_yaml().unwrap();
        assert_eq!(PretrainedCatalog::from_yaml(&yaml).unwrap(), doc);
    }

    #[test]
    fn test_unknown_fields_and_kinds_are_refused() {
        let unknown_field = "f: { $kind: facet, $description: \"d\", $extra: 1 }\n";
        assert!(PretrainedCatalog::from_yaml(unknown_field).is_err());
        let unknown_kind = "f: { $kind: family, $description: \"d\" }\n";
        assert!(PretrainedCatalog::from_yaml(unknown_kind).is_err());
    }

    fn refused(yaml: &str) -> String {
        PretrainedCatalog::from_yaml(yaml).unwrap_err().to_string()
    }

    #[test]
    fn test_a_bare_key_is_a_child_or_a_tag_by_kind_and_shape() {
        let doc = PretrainedCatalog::from_yaml(
            r#"
g:
  $kind: namespace
  $description: "G."
  b:
    $kind: build
    $description: "B."
    parameters: 7
    scores: { mmlu: 0.5 }                 # a map without `$kind`: a tag
    weights: { $kind: resource, $file: w, $format: f, $sources: [] }
  t:
    $kind: template_class
    $description: "T."
    language: [en]
"#,
        )
        .unwrap();
        let g = &doc.entries[0];
        assert_eq!(g.children().len(), 2);
        let CatalogEntry::Build(b) = &g.children()[0] else {
            panic!("a build")
        };
        let tags: Vec<String> = b.tags.keys().map(ToString::to_string).collect();
        assert_eq!(tags, ["parameters", "scores"]);
        assert_eq!(b.children.len(), 1);
        assert_eq!(b.children[0].name(), "weights");
        let CatalogEntry::TemplateClass(t) = &g.children()[1] else {
            panic!("a template class")
        };
        assert_eq!(t.tags.len(), 1);
    }

    #[test]
    fn test_the_keyed_form_refuses_what_it_cant_read() {
        assert!(refused("g: { $description: \"No kind.\" }\n").contains("no `$kind`"));
        assert!(
            refused("g:\n  $kind: namespace\n  $description: \"G.\"\n  child: { $file: x }\n")
                .contains("no `$kind`")
        );
        assert!(
            refused("t: { $kind: tag, $type: $bool, $description: \"T.\", extra: 1 }\n")
                .contains("a tag has no bare keys")
        );
        assert!(refused("$entries: []\n").contains("the catalog has no `$entries`"));
        assert!(
            refused("b: { $kind: build, $description: \"B.\", $presets: { $base: {} } }\n")
                .contains("`$presets` has no `$base`")
        );
        let repeated = refused("a: { $kind: facet, $description: \"A.\" }\na: { $kind: facet }\n");
        assert!(
            repeated.contains("repeated") || repeated.contains("uplicate"),
            "{repeated}"
        );
    }

    #[test]
    fn test_the_file_order_is_kept() {
        let doc = PretrainedCatalog::from_yaml(
            r#"
zeta: { $kind: facet, $description: "Z." }
alpha:
  $kind: namespace
  $description: "A."
  y: { $kind: namespace, $description: "Y." }
  b: { $kind: namespace, $description: "B." }
c:
  $kind: template_class
  $description: "C."
  $presets:
    $axes:
      zz: { $offers: [zz], $default: zz }
      aa: { $offers: [aa], $default: aa }
    second: {}
    first: {}
"#,
        )
        .unwrap();
        let names: Vec<&str> = doc.entries.iter().map(CatalogEntry::name).collect();
        assert_eq!(names, ["zeta", "alpha", "c"]);
        let kids: Vec<&str> = doc.entries[1]
            .children()
            .iter()
            .map(CatalogEntry::name)
            .collect();
        assert_eq!(kids, ["y", "b"]);
        let CatalogEntry::TemplateClass(c) = &doc.entries[2] else {
            panic!("a template class")
        };
        let axes: Vec<String> = c.presets.axes.iter().map(|a| a.axis.to_string()).collect();
        assert_eq!(axes, ["zz", "aa"]);
        let presets: Vec<&str> = c.presets.presets.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(presets, ["second", "first"]);
        // Written back, the order holds.
        let again = PretrainedCatalog::from_yaml(&doc.to_yaml().unwrap()).unwrap();
        assert_eq!(again, doc);
        assert!(doc.to_yaml().unwrap().starts_with("zeta:"));
    }

    #[test]
    fn test_an_editor_reads_changes_and_writes_a_tree() {
        let mut node = OrderedNode::from_yaml(SAMPLE).unwrap();
        let OrderedNode::Map(entries) = &mut node else {
            panic!("a map")
        };
        // Move the namespace first; the catalog reads it in the new order.
        let qwen = entries.pop().unwrap();
        entries.insert(0, qwen);
        let doc = PretrainedCatalog::from_node(node.clone()).unwrap();
        assert_eq!(doc.entries[0].name(), "qwen");
        // Written back, the entries keep their order; each entry's fields
        // come in their declared order.
        assert_eq!(PretrainedCatalog::from_node(doc.to_node()).unwrap(), doc);
        assert_eq!(
            doc.entries[0].to_node().get("$kind"),
            Some(&OrderedNode::Scalar(Value::String("namespace".into())))
        );
        assert!(node.get("qwen").is_some_and(OrderedNode::is_entry));
        let yaml = node.to_yaml().unwrap();
        assert_eq!(OrderedNode::from_yaml(&yaml).unwrap(), node);
        assert!(OrderedNode::from_yaml("a: 1\na: 2\n").is_err());
    }

    #[test]
    fn test_bool_axis_members_and_negation() {
        let axis = AxisEntry {
            name: "thinking".into(),
            merge: MergeMode::Deny,
            description: "d".into(),
            axis: AxisKind::Bool,
            members: vec![],
        };
        assert_eq!(axis.member_names(), ["thinking", "no-thinking"]);
        assert_eq!(axis.negate("thinking").as_deref(), Some("no-thinking"));
        assert_eq!(axis.negate("no-thinking").as_deref(), Some("thinking"));
        assert_eq!(axis.negate("other"), None);

        let task = AxisEntry {
            name: "task".into(),
            merge: MergeMode::Deny,
            description: "d".into(),
            axis: AxisKind::Categorical,
            members: vec!["transcribe".into(), "translate".into()],
        };
        assert_eq!(task.negate("transcribe"), None);
    }
}
