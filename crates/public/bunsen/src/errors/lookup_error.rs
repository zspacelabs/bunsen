//! `LookupError`: a key did not resolve.

use alloc::{
    borrow::Cow,
    string::{
        String,
        ToString,
    },
    sync::Arc,
    vec::Vec,
};
use core::{
    error::Error,
    fmt,
};
use std::path::Path;

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
};

/// How many candidates a [`LookupError`]'s one-line `Display` names; the rest
/// are counted. [`LookupError::candidates`] holds them all.
const MAX_CANDIDATES_SHOWN: usize = 16;

/// Where a [`LookupError`]'s key was looked up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Namespace {
    /// An in-process table, named by what it holds: `"shard set"`,
    /// `"resource"`, `"row entry"`.
    Table(Cow<'static, str>),
    /// A file system; the key is a path.
    Path,
    /// A server; the key is a URL.
    Url,
}

/// Why a [`LookupError`]'s key did not resolve.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LookupProblem {
    /// The key is not there: no such entry, file (`ENOENT`) or URL (HTTP 404,
    /// 410).
    Missing,

    /// No valid credentials were presented (HTTP 401). The key may or may not
    /// exist: the Hugging Face Hub, for one, answers 401 to an anonymous
    /// caller both for a repo that doesn't exist and for a gated or private
    /// one.
    Unauthorized,

    /// Access was refused (HTTP 403, `EACCES`).
    Denied,

    /// The key is bound already, and binding it again is not allowed.
    Duplicate,

    /// The key is used but was never declared.
    Undeclared,

    /// The key is declared, but its optional value is not present.
    Absent,

    /// An index or id past the end of the table.
    OutOfRange {
        /// The number of entries there are.
        len: usize,
    },

    /// A set of keys is not the set expected.
    KeySet {
        /// Keys present but not expected.
        unexpected: Vec<String>,
        /// Keys expected but not present.
        missing: Vec<String>,
    },
}

/// A key did not resolve: the cause of a
/// [`Lookup`](BunsenErrorKind::Lookup) error.
///
/// The key may name an entry in an in-process table, a file, or a URL; the
/// [`namespace`](Self::namespace) says which and the
/// [`problem`](Self::problem) says why. For an in-process table,
/// [`candidates`](Self::candidates) lists the keys there are, for a "did you
/// mean". For a file or URL, the [`source`](Self::source) is the `io::Error`
/// or fetch failure underneath.
///
/// ```
/// use bunsen::errors::{
///     BunsenError,
///     BunsenErrorKind,
///     LookupError,
/// };
///
/// let e = BunsenError::lookup(
///     LookupError::missing("shard set", "large")
///         .with_candidates(["tiny", "small"]),
/// );
/// assert_eq!(e.kind(), BunsenErrorKind::Lookup);
/// assert_eq!(
///     e.to_string(),
///     "no shard set \"large\"; there are: tiny, small"
/// );
/// ```
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LookupError {
    /// Where the key was looked up.
    pub namespace: Namespace,
    /// The key.
    pub key: String,
    /// Why it did not resolve.
    pub problem: LookupProblem,
    /// The keys there are, for an in-process table; may be empty.
    pub candidates: Vec<String>,
    /// The failure underneath, for a file or URL.
    pub source: Option<Arc<dyn Error + Send + Sync + 'static>>,
}

impl LookupError {
    /// A lookup of `key` in `namespace` that failed for `problem`.
    pub fn new(
        namespace: Namespace,
        key: impl Into<String>,
        problem: LookupProblem,
    ) -> Self {
        Self {
            namespace,
            key: key.into(),
            problem,
            candidates: Vec::new(),
            source: None,
        }
    }

    /// No entry `key` in the `table` (named by what it holds).
    pub fn missing(
        table: impl Into<Cow<'static, str>>,
        key: impl Into<String>,
    ) -> Self {
        Self::new(Namespace::Table(table.into()), key, LookupProblem::Missing)
    }

    /// `key` is already bound in the `table`.
    pub fn duplicate(
        table: impl Into<Cow<'static, str>>,
        key: impl Into<String>,
    ) -> Self {
        Self::new(
            Namespace::Table(table.into()),
            key,
            LookupProblem::Duplicate,
        )
    }

    /// `key` is used in the `table` but was never declared.
    pub fn undeclared(
        table: impl Into<Cow<'static, str>>,
        key: impl Into<String>,
    ) -> Self {
        Self::new(
            Namespace::Table(table.into()),
            key,
            LookupProblem::Undeclared,
        )
    }

    /// `key` is declared in the `table`, but its optional value is absent.
    pub fn absent(
        table: impl Into<Cow<'static, str>>,
        key: impl Into<String>,
    ) -> Self {
        Self::new(Namespace::Table(table.into()), key, LookupProblem::Absent)
    }

    /// The index or id `key` is past the end of the `table` of `len`
    /// entries.
    pub fn out_of_range(
        table: impl Into<Cow<'static, str>>,
        key: impl fmt::Display,
        len: usize,
    ) -> Self {
        Self::new(
            Namespace::Table(table.into()),
            key.to_string(),
            LookupProblem::OutOfRange { len },
        )
    }

    /// The file at `path` did not resolve, for `problem`.
    pub fn path(
        path: impl AsRef<Path>,
        problem: LookupProblem,
    ) -> Self {
        Self::new(
            Namespace::Path,
            path.as_ref().display().to_string(),
            problem,
        )
    }

    /// The `url` did not resolve, for `problem`.
    pub fn url(
        url: impl Into<String>,
        problem: LookupProblem,
    ) -> Self {
        Self::new(Namespace::Url, url, problem)
    }

    /// Sets the keys there are.
    pub fn with_candidates<I, S>(
        mut self,
        candidates: I,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.candidates = candidates.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the failure underneath.
    pub fn with_source<E>(
        mut self,
        source: E,
    ) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        self.source = Some(Arc::new(source));
        self
    }
}

impl fmt::Display for LookupError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let key = &self.key;
        match &self.namespace {
            Namespace::Table(table) => {
                match &self.problem {
                    LookupProblem::Missing => write!(f, "no {table} {key:?}")?,
                    LookupProblem::Unauthorized => write!(f, "{table} {key:?}: unauthorized")?,
                    LookupProblem::Denied => write!(f, "{table} {key:?}: access denied")?,
                    LookupProblem::Duplicate => write!(f, "duplicate {table} {key:?}")?,
                    LookupProblem::Undeclared => write!(f, "{table} {key:?} is not declared")?,
                    LookupProblem::Absent => write!(f, "{table} {key:?} is absent")?,
                    LookupProblem::OutOfRange { len } => {
                        write!(f, "{table} {key} is out of range; there are {len}")?
                    }
                    LookupProblem::KeySet {
                        unexpected,
                        missing,
                    } => {
                        write!(f, "{table} {key}: keys do not match")?;
                        if !unexpected.is_empty() {
                            write!(f, "; unexpected: {}", unexpected.join(", "))?;
                        }
                        if !missing.is_empty() {
                            write!(f, "; missing: {}", missing.join(", "))?;
                        }
                    }
                }
                if !self.candidates.is_empty() {
                    let shown = self.candidates.len().min(MAX_CANDIDATES_SHOWN);
                    write!(f, "; there are: {}", self.candidates[..shown].join(", "))?;
                    if shown < self.candidates.len() {
                        write!(f, ", and {} more", self.candidates.len() - shown)?;
                    }
                }
                Ok(())
            }
            Namespace::Path | Namespace::Url => match &self.problem {
                LookupProblem::Missing => write!(f, "{key}: not found"),
                LookupProblem::Unauthorized => write!(f, "{key}: unauthorized"),
                LookupProblem::Denied => write!(f, "{key}: access denied"),
                LookupProblem::Duplicate => write!(f, "{key}: already exists"),
                LookupProblem::Undeclared => write!(f, "{key}: not declared"),
                LookupProblem::Absent => write!(f, "{key}: absent"),
                LookupProblem::OutOfRange { len } => {
                    write!(f, "{key}: out of range; there are {len}")
                }
                LookupProblem::KeySet { .. } => write!(f, "{key}: keys do not match"),
            },
        }
    }
}

impl Error for LookupError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|s| s as &(dyn Error + 'static))
    }
}

impl From<LookupError> for BunsenError {
    #[track_caller]
    fn from(error: LookupError) -> Self {
        BunsenError::from_cause(BunsenErrorKind::Lookup, error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_display() {
        assert_eq!(
            LookupError::missing("resource", "weights").to_string(),
            "no resource \"weights\""
        );
        assert_eq!(
            LookupError::duplicate("provider", "hf").to_string(),
            "duplicate provider \"hf\""
        );
        assert_eq!(
            LookupError::out_of_range("shard", 9, 4).to_string(),
            "shard 9 is out of range; there are 4"
        );
    }

    #[test]
    fn test_many_candidates_are_counted() {
        let names: Vec<String> = (0..20).map(|i| alloc::format!("c{i}")).collect();
        let text = LookupError::missing("language", "xx")
            .with_candidates(names)
            .to_string();
        assert!(text.ends_with("c15, and 4 more"), "{text}");
    }

    #[test]
    fn test_path_display_and_source() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let e = LookupError::path("/a/b", LookupProblem::Missing).with_source(io);
        assert_eq!(e.to_string(), "/a/b: not found");
        assert!(e.source().is_some());
    }
}
