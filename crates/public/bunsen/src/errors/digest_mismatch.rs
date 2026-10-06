//! `DigestMismatch`: bytes are not what was pinned.

use alloc::string::String;
use core::fmt;

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
};

/// Where the bytes of a [`DigestMismatch`] came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DigestOrigin {
    /// Downloaded from `url` just now: the transfer was corrupt, or the
    /// source changed. Another mirror may have the pinned bytes.
    Download {
        /// The URL the bytes came from.
        url: String,
    },
    /// Already on disk: the file changed after it was verified.
    AtRest,
    /// Bundled into the binary.
    Bundled,
}

/// Bytes are not what was pinned: their sha256 is not the expected digest.
///
/// Its kind, through `From`, is
/// [`InvalidResource`](BunsenErrorKind::InvalidResource). A cache with more
/// than one source reads the [`origin`](Self::origin) to try another.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigestMismatch {
    /// What was checked: a path, a key, a URL.
    pub subject: String,
    /// Where the bytes came from.
    pub origin: DigestOrigin,
    /// The pinned digest.
    pub expected: String,
    /// The digest of the bytes.
    pub found: String,
}

impl DigestMismatch {
    /// `subject`'s bytes, from `origin`, hash to `found`, not `expected`.
    pub fn new(
        subject: impl Into<String>,
        origin: DigestOrigin,
        expected: impl Into<String>,
        found: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            origin,
            expected: expected.into(),
            found: found.into(),
        }
    }
}

impl fmt::Display for DigestMismatch {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "{}: sha256 is {}, not the pinned {}",
            self.subject, self.found, self.expected
        )?;
        match &self.origin {
            DigestOrigin::Download { url } => write!(
                f,
                " (downloaded from {url}: the transfer was corrupt, or the source changed)"
            ),
            DigestOrigin::AtRest => f.write_str(" (the file changed after it was verified)"),
            DigestOrigin::Bundled => f.write_str(" (bundled)"),
        }
    }
}

impl core::error::Error for DigestMismatch {}

impl From<DigestMismatch> for BunsenError {
    #[track_caller]
    fn from(error: DigestMismatch) -> Self {
        BunsenError::from_cause(BunsenErrorKind::InvalidResource, error)
    }
}
