//! `ConstraintError` and `Rule`: a value broke a rule.

use alloc::{
    borrow::Cow,
    string::{
        String,
        ToString,
    },
};
use core::fmt;

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
};

/// A check a value must pass: the rule half of a [`ConstraintError`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rule {
    /// The value must not be zero or empty.
    ZeroOrEmpty,

    /// The value must lie in a range.
    OutOfRange {
        /// The value, as printed.
        value: String,
        /// The range it must lie in, as printed: `"[0.0, 1.0]"`, `"> 0"`.
        range: String,
    },

    /// The value must be a multiple of something.
    NotMultiple {
        /// The value.
        value: usize,
        /// What it must be a multiple of, as printed: `"2"`, `"window_size
        /// (7)"`.
        of: String,
    },

    /// Two values must stand in a relation.
    Relation {
        /// The left side: its name and value.
        lhs: (String, String),
        /// The relation the two must stand in: `"=="`, `"<="`, `"divides"`.
        op: &'static str,
        /// The right side: its name and value.
        rhs: (String, String),
    },

    /// Adjacent stages must chain: one stage's output must be what the next
    /// takes.
    Chain {
        /// The index of the later stage.
        index: usize,
        /// What the earlier stage puts out.
        out: String,
        /// What the later stage takes.
        r#in: String,
    },

    /// A required value is missing.
    Missing,

    /// The value is used in the wrong state: called out of order, written
    /// twice, read after it ended.
    State {
        /// The state it must be in: `"open"`, `"unwritten"`.
        expected: &'static str,
    },

    /// A rule no other variant describes, in words.
    Custom(String),
}

impl fmt::Display for Rule {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::ZeroOrEmpty => f.write_str("must not be zero or empty"),
            Self::OutOfRange { value, range } => {
                write!(f, "{value} is outside {range}")
            }
            Self::NotMultiple { value, of } => {
                write!(f, "{value} is not a multiple of {of}")
            }
            Self::Relation { lhs, op, rhs } => write!(
                f,
                "{} ({}) must be {op} {} ({})",
                lhs.0, lhs.1, rhs.0, rhs.1
            ),
            Self::Chain { index, out, r#in } => write!(
                f,
                "stage {index} takes {}, but the stage before it puts out {}",
                r#in, out
            ),
            Self::Missing => f.write_str("is required"),
            Self::State { expected } => write!(f, "must be {expected}"),
            Self::Custom(text) => f.write_str(text),
        }
    }
}

/// A value broke a [`Rule`]: the report of a failed check on a config, an
/// argument, or a value's state.
///
/// Its kind, through `From`, is [`Illegal`](BunsenErrorKind::Illegal): the
/// rule is documented on the type or function that checks it. Where the value
/// came from outside the program, the boundary that read it re-marks the error
/// [`as_policy`](BunsenError::as_policy).
///
/// ```
/// use bunsen::errors::{
///     BunsenError,
///     BunsenErrorKind,
///     ConstraintError,
///     Rule,
/// };
///
/// let e: BunsenError =
///     ConstraintError::new("SlidingStftConfig", "stride", Rule::ZeroOrEmpty)
///         .into();
/// assert_eq!(e.kind(), BunsenErrorKind::Illegal);
/// assert_eq!(
///     e.to_string(),
///     "SlidingStftConfig.stride: must not be zero or empty"
/// );
/// ```
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintError {
    /// The type or function whose rule broke: `"SlidingStftConfig"`.
    pub owner: &'static str,
    /// The field or argument, as a path: `"stride"`, `"blocks[3].kernel"`.
    /// Empty when the rule is on the owner as a whole.
    pub field: Cow<'static, str>,
    /// The rule it broke.
    pub rule: Rule,
}

impl ConstraintError {
    /// A broken `rule` on `owner`'s `field`.
    pub fn new(
        owner: &'static str,
        field: impl Into<Cow<'static, str>>,
        rule: Rule,
    ) -> Self {
        Self {
            owner,
            field: field.into(),
            rule,
        }
    }

    /// `owner.field` is out of `range`.
    pub fn out_of_range(
        owner: &'static str,
        field: impl Into<Cow<'static, str>>,
        value: impl fmt::Display,
        range: impl fmt::Display,
    ) -> Self {
        Self::new(
            owner,
            field,
            Rule::OutOfRange {
                value: value.to_string(),
                range: range.to_string(),
            },
        )
    }

    /// `owner.field` must not be zero or empty.
    pub fn zero_or_empty(
        owner: &'static str,
        field: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self::new(owner, field, Rule::ZeroOrEmpty)
    }

    /// `owner.field` breaks a rule described in words.
    pub fn custom(
        owner: &'static str,
        field: impl Into<Cow<'static, str>>,
        text: impl fmt::Display,
    ) -> Self {
        Self::new(owner, field, Rule::Custom(text.to_string()))
    }
}

impl fmt::Display for ConstraintError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if self.field.is_empty() {
            write!(f, "{}: {}", self.owner, self.rule)
        } else {
            write!(f, "{}.{}: {}", self.owner, self.field, self.rule)
        }
    }
}

impl core::error::Error for ConstraintError {}

impl From<ConstraintError> for BunsenError {
    #[track_caller]
    fn from(error: ConstraintError) -> Self {
        BunsenError::from_cause(BunsenErrorKind::Illegal, error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        assert_eq!(
            ConstraintError::out_of_range("Relaxation", "omega", 2.5, "(0, 2)").to_string(),
            "Relaxation.omega: 2.5 is outside (0, 2)"
        );
        assert_eq!(
            ConstraintError::new(
                "SlidingStftConfig",
                "",
                Rule::Relation {
                    lhs: ("hop".into(), "512".into()),
                    op: "<=",
                    rhs: ("window".into(), "400".into()),
                }
            )
            .to_string(),
            "SlidingStftConfig: hop (512) must be <= window (400)"
        );
    }

    #[test]
    fn test_kind() {
        let e: BunsenError = ConstraintError::zero_or_empty("X", "y").into();
        assert_eq!(e.kind(), BunsenErrorKind::Illegal);
        assert_eq!(e.find::<ConstraintError>().unwrap().field, "y");
    }
}
