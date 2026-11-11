//! Errors: one [`CatalogError`] per problem, and [`CatalogErrors`] for a
//! validation pass that reports every problem it finds.

use std::fmt;

/// What sort of rule a [`CatalogError`] breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CatalogErrorKind {
    /// The document doesn't parse as a catalog.
    Parse,
    /// A name or name segment is malformed.
    Name,
    /// An entry is in a place its kind can't be, or carries a field its
    /// kind or position forbids.
    Structure,
    /// Two declarations claim one name.
    Conflict,
    /// A reference names nothing, or names an entry of the wrong kind.
    Reference,
    /// An occurrence's `merge` doesn't fit the declaration it meets.
    Merge,
    /// A facet rule is broken: exclusivity, a requirement, or nesting.
    Facet,
    /// A build is in no contract, or in more than one.
    Contract,
    /// A tag value or tag permission is wrong.
    Tag,
    /// A preset set, an axis use or the template-class chain's narrowing
    /// is wrong.
    Preset,
    /// A release timestamp or release ordering is wrong.
    Release,
    /// A dependency key or the resource it names is wrong.
    Dependency,
    /// A lookup by name matched nothing, or more than one entry.
    Lookup,
    /// A preset selection is wrong for the build it selects on.
    Selection,
}

impl fmt::Display for CatalogErrorKind {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let name = match self {
            Self::Parse => "parse",
            Self::Name => "name",
            Self::Structure => "structure",
            Self::Conflict => "conflict",
            Self::Reference => "reference",
            Self::Merge => "merge",
            Self::Facet => "facet",
            Self::Contract => "contract",
            Self::Tag => "tag",
            Self::Preset => "preset",
            Self::Release => "release",
            Self::Dependency => "dependency",
            Self::Lookup => "lookup",
            Self::Selection => "selection",
        };
        f.write_str(name)
    }
}

/// One problem, with where it was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogError {
    /// The rule broken.
    pub kind: CatalogErrorKind,
    /// What is wrong, for a person.
    pub message: String,
    /// The label of the layer being loaded, when the problem is in one.
    pub layer: Option<String>,
    /// The full name of the entry at fault, when there is one.
    pub entry: Option<String>,
}

impl CatalogError {
    /// A problem of `kind`, not yet placed in a layer or an entry.
    pub fn new(
        kind: CatalogErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            message: message.into(),
            layer: None,
            entry: None,
        }
    }

    /// Places the problem at `entry`.
    pub fn at(
        mut self,
        entry: impl ToString,
    ) -> Self {
        self.entry = Some(entry.to_string());
        self
    }

    /// Places the problem in the layer `layer`, unless it is already
    /// placed.
    pub fn in_layer(
        mut self,
        layer: &str,
    ) -> Self {
        if self.layer.is_none() {
            self.layer = Some(layer.to_string());
        }
        self
    }
}

impl fmt::Display for CatalogError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if let Some(layer) = &self.layer {
            write!(f, "[{layer}] ")?;
        }
        if let Some(entry) = &self.entry {
            write!(f, "`{entry}`: ")?;
        }
        write!(f, "{} ({})", self.message, self.kind)
    }
}

impl std::error::Error for CatalogError {}

/// Every problem a validation pass found, in the order found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CatalogErrors(pub Vec<CatalogError>);

impl CatalogErrors {
    /// Whether any problem of `kind` was found.
    pub fn has(
        &self,
        kind: CatalogErrorKind,
    ) -> bool {
        self.0.iter().any(|e| e.kind == kind)
    }

    /// Whether any problem's message contains `text`.
    pub fn mentions(
        &self,
        text: &str,
    ) -> bool {
        self.0.iter().any(|e| e.message.contains(text))
    }
}

impl From<CatalogError> for CatalogErrors {
    fn from(error: CatalogError) -> Self {
        Self(vec![error])
    }
}

impl fmt::Display for CatalogErrors {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        for (i, error) in self.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{error}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CatalogErrors {}

/// A result with one [`CatalogError`].
pub type CatalogResult<T> = Result<T, CatalogError>;
