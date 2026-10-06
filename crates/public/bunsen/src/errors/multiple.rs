//! `Multiple`: several failures reported as one.

use alloc::{
    borrow::Cow,
    string::String,
    vec::Vec,
};
use core::fmt::{
    self,
    Write,
};

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    Detailed,
};

/// Several failures reported as one: the parts of a batch that did not land,
/// the mirrors that all failed, the violations of a contract.
///
/// Each member is a labelled [`BunsenError`], kept whole, so a specialist can
/// read each member's kind and cause. It is [`Detailed`]: one line per member.
///
/// Its kind, through `From`, comes from the members (see
/// [`kind`](Self::kind)).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Multiple {
    /// One line: `"2 of 5 resources did not land"`.
    pub summary: String,
    /// The members, each with a label: a key, a URL, a subject.
    pub members: Vec<(String, BunsenError)>,
}

impl Multiple {
    /// `members` summed up as `summary`.
    pub fn new(
        summary: impl Into<String>,
        members: Vec<(String, BunsenError)>,
    ) -> Self {
        Self {
            summary: summary.into(),
            members,
        }
    }

    /// The kind of the aggregate.
    ///
    /// [`Unavailable`](BunsenErrorKind::Unavailable) only if every member is,
    /// so a generic consumer retries only when a retry can help them all.
    /// Otherwise the most severe member's kind: `Internal`, then `Illegal`,
    /// then the first member's. [`Other`](BunsenErrorKind::Other) with no
    /// members.
    pub fn kind(&self) -> BunsenErrorKind {
        let kinds = || self.members.iter().map(|(_, e)| e.kind());
        if self.members.is_empty() {
            BunsenErrorKind::Other
        } else if kinds().all(|k| k == BunsenErrorKind::Unavailable) {
            BunsenErrorKind::Unavailable
        } else if kinds().any(|k| k == BunsenErrorKind::Internal) {
            BunsenErrorKind::Internal
        } else if kinds().any(|k| k == BunsenErrorKind::Illegal) {
            BunsenErrorKind::Illegal
        } else {
            kinds()
                .find(|k| *k != BunsenErrorKind::Unavailable)
                .unwrap_or(BunsenErrorKind::Other)
        }
    }
}

impl fmt::Display for Multiple {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(&self.summary)
    }
}

impl core::error::Error for Multiple {}

impl Detailed for Multiple {
    fn details(&self) -> Option<Cow<'_, str>> {
        if self.members.is_empty() {
            return None;
        }
        let mut text = String::new();
        for (i, (label, error)) in self.members.iter().enumerate() {
            if i > 0 {
                text.push('\n');
            }
            let _ = write!(text, "{label}: [{}] {error}", error.kind());
        }
        Some(Cow::Owned(text))
    }
}

impl From<Multiple> for BunsenError {
    #[track_caller]
    fn from(error: Multiple) -> Self {
        let kind = error.kind();
        BunsenError::from_detailed(kind, error)
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
    fn test_kind() {
        let m = |es: Vec<BunsenError>| {
            Multiple::new("x", es.into_iter().map(|e| ("k".to_string(), e)).collect()).kind()
        };
        assert_eq!(
            m(vec![
                BunsenError::unavailable("a"),
                BunsenError::unavailable("b")
            ]),
            BunsenErrorKind::Unavailable
        );
        assert_eq!(
            m(vec![
                BunsenError::unavailable("a"),
                BunsenError::invalid_resource("b")
            ]),
            BunsenErrorKind::InvalidResource
        );
        assert_eq!(
            m(vec![BunsenError::policy("a"), BunsenError::illegal("b")]),
            BunsenErrorKind::Illegal
        );
        assert_eq!(m(vec![]), BunsenErrorKind::Other);
    }

    #[test]
    fn test_details() {
        let e: BunsenError = Multiple::new(
            "2 of 2 mirrors failed",
            vec![
                (
                    "https://a".to_string(),
                    BunsenError::unavailable("timed out"),
                ),
                ("https://b".to_string(), BunsenError::unavailable("503")),
            ],
        )
        .into();
        assert_eq!(e.kind(), BunsenErrorKind::Unavailable);
        assert_eq!(
            e.details(),
            Some("https://a: [Unavailable] timed out\nhttps://b: [Unavailable] 503")
        );
    }
}
