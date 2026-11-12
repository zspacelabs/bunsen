use alloc::fmt;

use serde::{
    Deserialize,
    Serialize,
};

use crate::data::catalog::document::{
    AliasEntry,
    AxisEntry,
    BuildEntry,
    ContractEntry,
    FacetEntry,
    GroupEntry,
    OrderedNode,
    ReleaseEntry,
    ResourceEntry,
    TagEntry,
    TemplateClassEntry,
    TypeEntry,
    keyed,
};

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
