//! The catalog document: [`PretrainedCatalog`] and its entries, with the
//! serde form they are read from and written to (YAML or JSON).
//!
//! Every entry names its `kind`. Groups (`namespace`, `contract`) and builds
//! may hold `children`, which is a shortcut for naming the container as the
//! child's `parent`.
//!
//! A reference to another entry ([`CatalogRef`]) resolves relative to the
//! entry it is written in, climbing the namespace to the root; a leading `/`
//! resolves from the root only. A naming `parent` is always read from the
//! root.

use std::{
    collections::BTreeMap,
    fmt,
};

use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::{
    error::{
        CatalogError,
        CatalogErrorKind,
        CatalogResult,
    },
    name::CatalogRef,
};

/// The name of the template class every chain ends at.
///
/// A template class with no `parent` inherits it, and so does a build with
/// no `template_class`. Its `required_facets` hold for every build.
pub const TEMPLATE_CLASS_ROOT: &str = "template_class_root";

/// One catalog document.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PretrainedCatalog {
    /// What the catalog holds, for a person.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The top-level entries.
    #[serde(default)]
    pub entries: Vec<CatalogEntry>,
}

impl PretrainedCatalog {
    /// Parses a YAML document.
    ///
    /// # Errors
    /// [`CatalogErrorKind::Parse`] when it isn't a catalog.
    pub fn from_yaml(yaml: &str) -> CatalogResult<Self> {
        serde_saphyr::from_str(yaml)
            .map_err(|e| CatalogError::new(CatalogErrorKind::Parse, e.to_string()))
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
        serde_json::from_str(json)
            .map_err(|e| CatalogError::new(CatalogErrorKind::Parse, e.to_string()))
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
    /// A group that also states what a loader expects: its dependency keys.
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
}

impl EntryKind {
    /// Whether entries of this kind are groups (a build may be a member).
    pub fn is_group(self) -> bool {
        matches!(self, Self::Namespace | Self::Contract | Self::Release)
    }

    /// Whether entries of this kind may hold `children` and be scoped.
    pub fn is_container_group(self) -> bool {
        matches!(self, Self::Namespace | Self::Contract)
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
        }
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

/// One entry, tagged by its `kind`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatalogEntry {
    /// A `namespace` group.
    Namespace(GroupEntry),
    /// A `contract` group.
    Contract(GroupEntry),
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
}

impl CatalogEntry {
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
        }
    }

    /// The entry's own name segment, as written.
    pub fn name(&self) -> &str {
        match self {
            Self::Namespace(e) | Self::Contract(e) => &e.name,
            Self::Release(e) => &e.name,
            Self::Build(e) => &e.name,
            Self::TemplateClass(e) => &e.name,
            Self::Tag(e) => &e.name,
            Self::Axis(e) => &e.name,
            Self::Facet(e) => &e.name,
            Self::Resource(e) => &e.name,
        }
    }

    /// The naming `parent` the entry states, if it states one.
    pub fn naming_parent(&self) -> Option<&CatalogRef> {
        match self {
            Self::Namespace(e) | Self::Contract(e) => e.parent.as_ref(),
            Self::Release(e) => e.parent.as_ref(),
            Self::Build(e) => e.parent.as_ref(),
            Self::Tag(e) => e.parent.as_ref(),
            Self::Axis(e) => e.parent.as_ref(),
            Self::Resource(e) => e.parent.as_ref(),
            Self::TemplateClass(e) => e.parent.as_ref(),
            Self::Facet(_) => None,
        }
    }

    /// Clears the naming `parent`.
    pub(crate) fn clear_naming_parent(&mut self) {
        match self {
            Self::Namespace(e) | Self::Contract(e) => e.parent = None,
            Self::Release(e) => e.parent = None,
            Self::Build(e) => e.parent = None,
            Self::Tag(e) => e.parent = None,
            Self::Axis(e) => e.parent = None,
            Self::Resource(e) => e.parent = None,
            Self::TemplateClass(e) => e.parent = None,
            Self::Facet(_) => {}
        }
    }

    /// The nested children, if the kind has any.
    pub fn children(&self) -> &[CatalogEntry] {
        match self {
            Self::Namespace(e) | Self::Contract(e) => &e.children,
            Self::Build(e) => &e.children,
            _ => &[],
        }
    }

    /// Takes the nested children out, leaving none.
    pub(crate) fn take_children(&mut self) -> Vec<CatalogEntry> {
        match self {
            Self::Namespace(e) | Self::Contract(e) => std::mem::take(&mut e.children),
            Self::Build(e) => std::mem::take(&mut e.children),
            _ => Vec::new(),
        }
    }

    /// How the occurrence meets an existing declaration of its name.
    pub fn merge(&self) -> MergeMode {
        match self {
            Self::Namespace(e) | Self::Contract(e) => e.merge,
            Self::Release(e) => e.merge,
            Self::Build(e) => e.merge,
            Self::TemplateClass(e) => e.merge,
            Self::Tag(e) => e.merge,
            Self::Axis(e) => e.merge,
            Self::Facet(e) => e.merge,
            Self::Resource(e) => e.merge,
        }
    }

    /// The group fields, for a `namespace` or `contract`.
    pub fn as_group(&self) -> Option<&GroupEntry> {
        match self {
            Self::Namespace(e) | Self::Contract(e) => Some(e),
            _ => None,
        }
    }
}

/// A `namespace` or `contract` group.
///
/// A **declaring** occurrence (`merge: deny`, the default) carries the
/// group's content. On a `namespace`, an **extending** occurrence
/// (`merge: update`) carries only
/// its name, optionally the facet it expects the group to have, and `children`:
/// it opens the group's scope to add children to it, and asserts what it
/// expects of the group.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupEntry {
    /// The group's own segment.
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the group is, for a person. Required on a declaring occurrence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The facet the group belongs to. On an extending occurrence, the facet
    /// the declaration must have.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facet: Option<CatalogRef>,

    /// The naming scope, when the entry isn't nested. Also a membership.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<CatalogRef>,

    /// Further memberships.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub join: Vec<CatalogRef>,

    /// Tags every member build must have a value for.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<CatalogRef>,

    /// On a `contract`: the dependency keys its loader reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dependency_keys: Option<DependencyKeys>,

    /// Entries nested in the group's scope.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<CatalogEntry>,
}

/// How an occurrence of a name meets an existing declaration of it.
///
/// What `update` means depends on the kind:
///
/// | kind | `update` |
/// |---|---|
/// | `namespace` | adds its children (each child's own `merge` decides how it meets an existing child), and asserts the facet it expects; it states nothing else |
/// | `resource` | matches every field but `sources`, and puts its new sources first |
/// | `contract` | refused: a contract is the interface a loader is built against |
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
}

/// The dependency keys a contract's loader reads.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependencyKeys {
    /// Keys every member build must have.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,

    /// Keys a member build may have.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub optional: Vec<String>,
}

/// A `release` group: its members are releases of one sequence.
///
/// A build joins it with a `released` timestamp ([`Membership::Release`]).
/// It holds no children.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseEntry {
    /// The group's own segment.
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the sequence is, for a person.
    pub description: String,

    /// The naming scope, when the entry isn't nested. Also a membership.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<CatalogRef>,

    /// Further memberships.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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

/// `{group, released}`: a release membership.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseMembership {
    /// The `release` group.
    pub group: CatalogRef,

    /// When the build was released, in canonical form (`2025`, `2025-07`,
    /// `2025-07-15`, `2025-07-15T09:30Z`, `2025-07-15T09:30:00Z`; UTC).
    pub released: String,
}

/// A build: something we can build.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildEntry {
    /// The build's own segment.
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the build is, for a person.
    pub description: String,

    /// The naming scope, when the entry isn't nested. Also a membership.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<CatalogRef>,

    /// Further memberships, release memberships among them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub join: Vec<Membership>,

    /// The template class the build inherits; [`TEMPLATE_CLASS_ROOT`] when
    /// absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template_class: Option<CatalogRef>,

    /// The checkpoint's license.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Where the checkpoint comes from, for a person.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,

    /// Tag values, by tag. They override the chain's.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tags: BTreeMap<CatalogRef, Value>,

    /// The build's own presets, applied over its chain's.
    #[serde(default, skip_serializing_if = "PresetSetEntry::is_empty")]
    pub presets: PresetSetEntry,

    /// Dependency keys to resources. They override the chain's.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dependencies: BTreeMap<String, CatalogRef>,

    /// Facets the build must be in exactly one group of, beyond its
    /// chain's.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_facets: Vec<CatalogRef>,

    /// Entries in the build's scope: its own resources, tags and axes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the class shares, for a person.
    pub description: String,

    /// The naming scope, when the entry isn't nested. Unlike a group's or
    /// a build's, it confers no membership.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<CatalogRef>,

    /// The base class it inherits; [`TEMPLATE_CLASS_ROOT`] when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<CatalogRef>,

    /// Memberships every build of the class has.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub join: Vec<CatalogRef>,

    /// Tag values, by tag. They override the base class's.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tags: BTreeMap<CatalogRef, Value>,

    /// Presets, applied over the base class's.
    #[serde(default, skip_serializing_if = "PresetSetEntry::is_empty")]
    pub presets: PresetSetEntry,

    /// Dependency keys to resources. They override the base class's.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dependencies: BTreeMap<String, CatalogRef>,

    /// Facets every build of the class must be in exactly one group of.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_facets: Vec<CatalogRef>,
}

/// A preset set as a template class or build writes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresetSetEntry {
    /// A merge patch every selection keeps.
    #[serde(default = "empty_object", skip_serializing_if = "is_empty_object")]
    pub base: Value,

    /// The presets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub presets: Vec<PresetEntry>,

    /// The registered axes used, with what each offers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub axes: Vec<AxisUse>,

    /// Presets on no axis that apply unless displaced.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub default: Vec<String>,
}

impl Default for PresetSetEntry {
    fn default() -> Self {
        Self {
            base: empty_object(),
            presets: Vec::new(),
            axes: Vec::new(),
            default: Vec::new(),
        }
    }
}

impl PresetSetEntry {
    /// Whether the set says nothing.
    pub fn is_empty(&self) -> bool {
        is_empty_object(&self.base)
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
    pub name: String,

    /// What selecting it does, for a person.
    #[serde(default)]
    pub description: String,

    /// The merge patch it applies.
    #[serde(default = "empty_object", skip_serializing_if = "is_empty_object")]
    pub patch: Value,
}

/// A use of a registered axis: the members offered, and the default.
///
/// An offer of one member is a fixed mode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxisUse {
    /// The axis.
    #[serde(rename = "use")]
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
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the tag says, for a person.
    pub description: String,

    /// The naming scope, when the entry isn't nested. Only members of the
    /// scope may set the tag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<CatalogRef>,

    /// The values the tag takes.
    #[serde(rename = "type")]
    pub tag_type: TagType,
}

/// The values a tag takes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagType {
    /// `true` or `false`.
    Bool,
    /// A whole number, zero or more.
    Count,
    /// Any text.
    Text,
    /// A list of strings.
    TextList,
    /// A list of BCP 47 language tags, `"*"` among them for "others".
    Languages,
    /// An object of numbers, by name.
    Scores,
    /// One of these strings.
    OneOf(Vec<String>),
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
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the axis selects, for a person.
    pub description: String,

    /// The naming scope, when the entry isn't nested. Only members of the
    /// scope may use the axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<CatalogRef>,

    /// The axis's kind.
    pub axis: AxisKind,

    /// A categorical axis's members; a bool axis has none written.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the facet sorts groups by, for a person.
    pub description: String,
}

/// A resource declaration: one artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceEntry {
    /// The resource's own segment.
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the artifact is, for a person.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The naming scope, when the entry isn't nested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<CatalogRef>,

    /// The file name it is stored under.
    pub file: String,

    /// The file's format, as its loader names it.
    pub format: String,

    /// The file's SHA-256, in lowercase hex, when it is pinned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,

    /// The dtype a checkpoint stores, when it is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Where to fetch it, tried in order.
    pub sources: Vec<ResourceSource>,
}

/// Where a resource can be fetched.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ResourceSource {
    /// A file in a Hugging Face Hub repo, at a revision.
    Hub {
        /// `org/name`.
        repo: String,
        /// A commit.
        revision: String,
    },
    /// A URL.
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
entries:
  - kind: facet
    name: family
    description: "A line of related architectures."
  - kind: axis
    name: thinking
    axis: bool
    description: "Whether the model thinks first."
  - kind: namespace
    name: qwen
    description: "Qwen."
    children:
      - kind: build
        name: qwen3-4b
        description: "Qwen3-4B."
        join:
          - qwen/qwen3-line
          - { group: qwen/qwen3-releases, released: "2025-04" }
        tags: { parameters: 4000000000 }
        presets:
          axes:
            - { use: thinking, offers: [thinking], default: thinking }
          presets:
            - { name: thinking, description: "Think.", patch: { a: 1 } }
        dependencies: { config: qwen/qwen3-4b/config }
        children:
          - kind: resource
            name: config
            file: config.json
            format: json
            sources: [{ hub: { repo: Qwen/Qwen3-4B, revision: abc } }, { url: "https://x/y" }]
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
        let unknown_field = "entries:\n  - { kind: facet, name: f, description: d, extra: 1 }\n";
        assert!(PretrainedCatalog::from_yaml(unknown_field).is_err());
        let unknown_kind = "entries:\n  - { kind: family, name: f, description: d }\n";
        assert!(PretrainedCatalog::from_yaml(unknown_kind).is_err());
    }

    #[test]
    fn test_bool_axis_members_and_negation() {
        let axis = AxisEntry {
            name: "thinking".into(),
            merge: MergeMode::Deny,
            description: "d".into(),
            parent: None,
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
            parent: None,
            axis: AxisKind::Categorical,
            members: vec!["transcribe".into(), "translate".into()],
        };
        assert_eq!(task.negate("transcribe"), None);
    }
}
