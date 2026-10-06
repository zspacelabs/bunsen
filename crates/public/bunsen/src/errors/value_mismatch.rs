//! `ValueMismatch`: a verification found values that differ.

use alloc::{
    borrow::Cow,
    format,
    string::String,
    vec::Vec,
};
use core::fmt;

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    Detailed,
};

/// A verification found values that differ from what was expected: tensor
/// data against a reference, an audit stream against its baseline.
///
/// Its kind, through `From`, is [`Policy`](BunsenErrorKind::Policy): the
/// expectation was set elsewhere, and the check enforces it. It is
/// [`Detailed`]: the differing values are the error's details.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum ValueMismatch {
    /// The data types differ.
    DType {
        /// The actual dtype, as printed.
        actual: String,
        /// The expected dtype, as printed.
        expected: String,
    },

    /// The quantization schemes differ, or only one side is quantized.
    Quantization {
        /// The actual scheme, as printed.
        actual: String,
        /// The expected scheme, as printed.
        expected: String,
    },

    /// The shapes differ.
    Shape {
        /// The actual shape.
        actual: Vec<usize>,
        /// The expected shape.
        expected: Vec<usize>,
    },

    /// Some values differ.
    Values {
        /// How many values differ.
        count: usize,
        /// How many values were compared.
        total: usize,
        /// The first differences, one line each.
        first: Vec<String>,
    },

    /// Any other difference a verification reports: a summary line, and
    /// details.
    Other {
        /// One line.
        summary: String,
        /// Multi-line evidence, if any.
        details: Option<String>,
    },
}

impl fmt::Display for ValueMismatch {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::DType { actual, expected } => {
                write!(f, "data types differ: {actual} != {expected}")
            }
            Self::Quantization { actual, expected } => {
                write!(f, "quantization differs: {actual} != {expected}")
            }
            Self::Shape { actual, expected } => {
                write!(f, "shapes differ: {actual:?} != {expected:?}")
            }
            Self::Values { count, total, .. } => {
                write!(f, "{count} of {total} values differ")
            }
            Self::Other { summary, .. } => f.write_str(summary),
        }
    }
}

impl core::error::Error for ValueMismatch {}

impl Detailed for ValueMismatch {
    fn details(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::Values { count, first, .. } if !first.is_empty() => {
                let mut text = first.join("\n");
                if *count > first.len() {
                    text.push_str(&format!("\n... and {} more", count - first.len()));
                }
                Some(Cow::Owned(text))
            }
            Self::Other {
                details: Some(details),
                ..
            } => Some(Cow::Borrowed(details)),
            _ => None,
        }
    }
}

impl From<ValueMismatch> for BunsenError {
    #[track_caller]
    fn from(error: ValueMismatch) -> Self {
        BunsenError::from_detailed(BunsenErrorKind::Policy, error)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        string::ToString,
        vec,
    };

    use super::*;

    #[test]
    fn test_values_details() {
        let m = ValueMismatch::Values {
            count: 7,
            total: 100,
            first: vec!["[0]: 1 != 2".to_string(), "[3]: 4 != 5".to_string()],
        };
        let e: BunsenError = m.into();
        assert_eq!(e.kind(), BunsenErrorKind::Policy);
        assert_eq!(e.to_string(), "7 of 100 values differ");
        assert_eq!(
            e.details(),
            Some("[0]: 1 != 2\n[3]: 4 != 5\n... and 5 more")
        );
    }
}
