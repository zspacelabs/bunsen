//! `BunsenErrorKind`: the reaction class of an error.

use core::fmt;

/// What a generic consumer should do with a
/// [`BunsenError`](crate::errors::BunsenError): the reaction class of a
/// failure.
///
/// The [module docs](crate::errors#which-kind) say which kind fits which
/// failure. [`Illegal`](Self::Illegal) and [`Internal`](Self::Internal) are
/// programming bugs, which a generic consumer escalates or panics on.
/// [`Unavailable`](Self::Unavailable) is the only kind a generic consumer
/// retries. A subsystem with its own recovery strategy reads the error's
/// cause as well, through [`find`](crate::errors::BunsenError::find).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BunsenErrorKind {
    /// A programming bug in the caller: the call should never have happened.
    ///
    /// The caller broke a rule documented by the function it called. If the
    /// type system could express the rule, the call would not have compiled.
    Illegal,

    /// A programming bug in bunsen: one of its own invariants broke.
    Internal,

    /// The operation is misconfigured, and bunsen refuses to proceed.
    ///
    /// A rule set elsewhere rejects the request: a setting, a configuration
    /// read from outside the program, a baseline to verify against. The call
    /// that fails only enforces it.
    Policy,

    /// A key did not resolve: it is missing, unauthorized, denied, a
    /// duplicate, undeclared, absent, or out of range. The key may name an
    /// entry in an in-process table, a file, or a URL; the
    /// [`LookupError`](crate::errors::LookupError) cause says which and why.
    Lookup,

    /// Not reachable now: trying again later may succeed.
    Unavailable,

    /// Something from outside the program is present and reachable, but is
    /// not what was expected.
    ///
    /// A resource is anything the program did not build itself: a file, a
    /// download, a cache entry, a checkpoint, a stream, a host timestamp,
    /// model output. Neither the code nor its configuration is at fault. A
    /// broken *declaration* of a resource, such as a
    /// [`Resource`](crate::data::pretrained::Resource) that fails its own
    /// validation, is not this kind: it is [`Illegal`](Self::Illegal) or
    /// [`Policy`](Self::Policy).
    InvalidResource,

    /// A valid request this code, or this build, does not do: an operation
    /// that is not implemented, or a feature that was not compiled in.
    Unsupported,

    /// The host failed: a full disk, a read-only file system, a thread that
    /// would not spawn, a device or backend error.
    Sys,

    /// Not classified: usually a foreign error passed through as it is.
    ///
    /// A bail-out, not a category. A generic consumer escalates it, and an
    /// `Other` that reaches a log is a prompt to give its site a real kind.
    Other,
}

impl BunsenErrorKind {
    /// Whether the kind is a programming bug: [`Illegal`](Self::Illegal) or
    /// [`Internal`](Self::Internal).
    pub fn is_bug(self) -> bool {
        matches!(self, Self::Illegal | Self::Internal)
    }

    /// Whether a generic consumer should retry: true only for
    /// [`Unavailable`](Self::Unavailable).
    pub fn is_retryable(self) -> bool {
        self == Self::Unavailable
    }

    /// Whether the kind is a runtime failure: every kind but the two bugs.
    pub fn is_runtime(self) -> bool {
        !self.is_bug()
    }

    /// The kind's name, as `Display` prints it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Illegal => "Illegal",
            Self::Internal => "Internal",
            Self::Policy => "Policy",
            Self::Lookup => "Lookup",
            Self::Unavailable => "Unavailable",
            Self::InvalidResource => "InvalidResource",
            Self::Unsupported => "Unsupported",
            Self::Sys => "Sys",
            Self::Other => "Other",
        }
    }
}

impl fmt::Display for BunsenErrorKind {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bugs_and_retryable() {
        assert!(BunsenErrorKind::Illegal.is_bug());
        assert!(BunsenErrorKind::Internal.is_bug());
        assert!(!BunsenErrorKind::Policy.is_bug());
        assert!(BunsenErrorKind::Policy.is_runtime());

        assert!(BunsenErrorKind::Unavailable.is_retryable());
        assert!(!BunsenErrorKind::Lookup.is_retryable());
        assert!(!BunsenErrorKind::Illegal.is_retryable());
    }

    #[test]
    fn test_display() {
        assert_eq!(
            BunsenErrorKind::InvalidResource.to_string(),
            "InvalidResource"
        );
    }
}
