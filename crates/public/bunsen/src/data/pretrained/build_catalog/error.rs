//! Errors: every failure is a [`BunsenError`]. A catalog problem carries a
//! [`CatalogProblem`] as its cause: the rule it breaks, and where it was
//! found. A validation pass that finds several reports them together, each a
//! member of a [`Multiple`].

use alloc::fmt;

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    Multiple,
};

/// What sort of rule a [`CatalogProblem`] breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CatalogRule {
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
    /// A contract slot, or what a build binds to it, is wrong.
    Dependency,
    /// A lookup by name matched nothing, or more than one entry.
    Lookup,
    /// A preset selection is wrong for the build it selects on.
    Selection,
}

impl fmt::Display for CatalogRule {
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

/// One catalog problem, with where it was found: the cause a catalog's
/// [`BunsenError`] carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogProblem {
    /// The rule broken.
    pub rule: CatalogRule,
    /// What is wrong, for a person.
    pub message: String,
    /// The label of the layer being loaded, when the problem is in one.
    pub layer: Option<String>,
    /// The full name of the entry at fault, when there is one.
    pub entry: Option<String>,
}

impl CatalogRule {
    /// The kind of the error a problem of this rule is: a lookup that
    /// matched nothing, or one too many, is
    /// [`Lookup`](BunsenErrorKind::Lookup); every other rule is
    /// [`Policy`](BunsenErrorKind::Policy), a catalog or a request read from
    /// outside the program that breaks it.
    pub fn error_kind(self) -> BunsenErrorKind {
        match self {
            Self::Lookup => BunsenErrorKind::Lookup,
            _ => BunsenErrorKind::Policy,
        }
    }
}

impl CatalogProblem {
    /// A problem of `rule`, not yet placed in a layer or an entry.
    pub fn new(
        rule: CatalogRule,
        message: impl Into<String>,
    ) -> Self {
        Self {
            rule,
            message: message.into(),
            layer: None,
            entry: None,
        }
    }

    /// The error that carries it.
    #[track_caller]
    pub fn into_error(self) -> BunsenError {
        BunsenError::from_cause(self.rule.error_kind(), self)
    }
}

impl fmt::Display for CatalogProblem {
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
        write!(f, "{} ({})", self.message, self.rule)
    }
}

impl core::error::Error for CatalogProblem {}

/// The error of a [`CatalogProblem`] of `rule`.
#[track_caller]
pub fn catalog_error(
    rule: CatalogRule,
    message: impl Into<String>,
) -> BunsenError {
    CatalogProblem::new(rule, message).into_error()
}

/// The errors a pass found, as one: the error itself when there is one,
/// and a [`Multiple`] of them otherwise.
///
/// # Panics
/// When `errors` is empty: there is no failure to report.
#[track_caller]
pub fn catalog_errors(errors: Vec<BunsenError>) -> BunsenError {
    match <[BunsenError; 1]>::try_from(errors) {
        Ok([one]) => one,
        Err(errors) => {
            assert!(!errors.is_empty(), "a pass with no problems is no error");
            let summary = format!("{} catalog problems", errors.len());
            let members = errors.into_iter().map(|e| (String::new(), e)).collect();
            let multiple = Multiple::new(summary, members);
            let kind = multiple.kind();
            BunsenError::from_detailed(kind, multiple)
        }
    }
}

/// What a catalog error says: its problems, and where they are.
pub trait CatalogErrorExt: Sized {
    /// The problems it carries: its own, or its members'.
    fn problems(&self) -> Vec<&CatalogProblem>;

    /// The error with each problem placed at `entry`.
    #[must_use]
    fn at(
        self,
        entry: impl fmt::Display,
    ) -> Self;

    /// The error with each problem placed in the layer `layer`, unless it is
    /// already placed.
    #[must_use]
    fn in_layer(
        self,
        layer: &str,
    ) -> Self;

    /// The errors it is made of: its members, or itself.
    fn members(self) -> Vec<BunsenError>;

    /// The first problem's rule.
    fn rule(&self) -> Option<CatalogRule> {
        self.problems().first().map(|p| p.rule)
    }

    /// The first problem's message, without where it was found.
    fn problem_message(&self) -> String;

    /// Whether any problem breaks `rule`.
    fn has(
        &self,
        rule: CatalogRule,
    ) -> bool {
        self.problems().iter().any(|p| p.rule == rule)
    }

    /// Whether any problem's message contains `text`.
    fn mentions(
        &self,
        text: &str,
    ) -> bool {
        self.problems().iter().any(|p| p.message.contains(text))
    }

    /// Each problem on a line of its own, placed.
    fn problem_lines(&self) -> String {
        self.problems()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl CatalogErrorExt for BunsenError {
    fn problems(&self) -> Vec<&CatalogProblem> {
        if let Some(problem) = self.find::<CatalogProblem>() {
            return vec![problem];
        }
        match self.find::<Multiple>() {
            Some(multiple) => multiple
                .members
                .iter()
                .flat_map(|(_, member)| member.problems())
                .collect(),
            None => Vec::new(),
        }
    }

    fn at(
        self,
        entry: impl fmt::Display,
    ) -> Self {
        let entry = entry.to_string();
        self.placed(&|p| p.entry = Some(entry.clone()))
    }

    fn in_layer(
        self,
        layer: &str,
    ) -> Self {
        self.placed(&|p| {
            if p.layer.is_none() {
                p.layer = Some(layer.to_string());
            }
        })
    }

    fn members(self) -> Vec<BunsenError> {
        match self.find::<Multiple>() {
            Some(multiple) if self.find::<CatalogProblem>().is_none() => {
                multiple.members.iter().map(|(_, m)| m.clone()).collect()
            }
            _ => vec![self],
        }
    }

    fn problem_message(&self) -> String {
        self.problems()
            .first()
            .map_or_else(|| self.message().to_string(), |p| p.message.clone())
    }
}

trait Placed {
    fn placed(
        self,
        place: &dyn Fn(&mut CatalogProblem),
    ) -> Self;
}

impl Placed for BunsenError {
    fn placed(
        self,
        place: &dyn Fn(&mut CatalogProblem),
    ) -> Self {
        if let Some(problem) = self.find::<CatalogProblem>() {
            let mut problem = problem.clone();
            place(&mut problem);
            return problem.into_error();
        }
        if let Some(multiple) = self.find::<Multiple>() {
            let members = multiple
                .members
                .iter()
                .map(|(_, member)| member.clone().placed(place))
                .collect();
            return catalog_errors(members);
        }
        self
    }
}

/// A result whose error is a [`BunsenError`], a catalog's carrying its
/// [`CatalogProblem`]s.
pub type CatalogResult<T> = BunsenResult<T>;
