//! `ErrorMatcher`: a fluent matcher on `BunsenError`.

use alloc::{
    boxed::Box,
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};
use core::{
    any::type_name,
    error::Error,
    fmt::{
        self,
        Debug,
        Write,
    },
    marker::PhantomData,
};

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    Multiple,
    testing::{
        BoxMatcher,
        Description,
        Matcher,
        text,
        value,
    },
};

/// A matcher on a [`BunsenError`], built fluently from clauses: the kind, the
/// message, the one-line `Display`, a frame, the details, a typed cause, the
/// members of a [`Multiple`].
///
/// Every clause must match. [`check`](Self::check) runs every clause, not just
/// up to the first that misses, and the [`ErrorMismatch`] it returns lists
/// each miss and prints the error's full report. The assert methods panic
/// with that text, at the caller's line.
///
/// ```
/// use bunsen::errors::{
///     BunsenError,
///     BunsenErrorKind,
///     ResultContext,
///     testing::ErrorMatcher,
/// };
///
/// let e =
///     BunsenError::policy("downloads are off").context("fetching \"large\"");
///
/// ErrorMatcher::kind(BunsenErrorKind::Policy)
///     .message_eq("downloads are off")
///     .display_contains("fetching \"large\": ")
///     .assert(&e);
///
/// let miss = ErrorMatcher::kind(BunsenErrorKind::Lookup)
///     .check(&e)
///     .unwrap_err();
/// assert!(miss.to_string().contains("kind was Policy"));
/// ```
#[derive(Default)]
pub struct ErrorMatcher {
    clauses: Vec<BoxMatcher<BunsenError>>,
}

impl ErrorMatcher {
    /// A matcher with no clauses, which matches any error.
    pub fn new() -> Self {
        Self::default()
    }

    /// A matcher for errors of `kind`: `new().with_kind(kind)`.
    pub fn kind(kind: BunsenErrorKind) -> Self {
        Self::new().with_kind(kind)
    }

    fn clause(
        mut self,
        clause: impl Matcher<BunsenError> + 'static,
    ) -> Self {
        self.clauses.push(Box::new(clause));
        self
    }

    /// The kind is `kind`.
    pub fn with_kind(
        self,
        kind: BunsenErrorKind,
    ) -> Self {
        self.with_kind_matching(value::eq(kind))
    }

    /// The kind matches `matcher`, such as
    /// [`value::one_of`](crate::errors::testing::value::one_of).
    pub fn with_kind_matching(
        self,
        matcher: impl Matcher<BunsenErrorKind> + 'static,
    ) -> Self {
        self.clause(KindClause(Box::new(matcher)))
    }

    /// The message, without the frames, matches `matcher`.
    pub fn message(
        self,
        matcher: impl Matcher<str> + 'static,
    ) -> Self {
        self.clause(TextClause {
            part: Part::Message,
            matcher: Box::new(matcher),
        })
    }

    /// The message contains `text`.
    pub fn message_contains(
        self,
        text: &str,
    ) -> Self {
        self.message(text::contains(text))
    }

    /// The message is `text`.
    pub fn message_eq(
        self,
        text: &str,
    ) -> Self {
        self.message(text::eq(text))
    }

    /// The one-line `Display`, frames included, matches `matcher`.
    pub fn display(
        self,
        matcher: impl Matcher<str> + 'static,
    ) -> Self {
        self.clause(TextClause {
            part: Part::Display,
            matcher: Box::new(matcher),
        })
    }

    /// The one-line `Display` contains `text`.
    pub fn display_contains(
        self,
        text: &str,
    ) -> Self {
        self.display(text::contains(text))
    }

    /// The details match `matcher`; an error without details does not match.
    pub fn details(
        self,
        matcher: impl Matcher<str> + 'static,
    ) -> Self {
        self.clause(TextClause {
            part: Part::Details,
            matcher: Box::new(matcher),
        })
    }

    /// The details contain `text`.
    pub fn details_contains(
        self,
        text: &str,
    ) -> Self {
        self.details(text::contains(text))
    }

    /// Some frame's message matches `matcher`.
    pub fn frame(
        self,
        matcher: impl Matcher<str> + 'static,
    ) -> Self {
        self.clause(FrameClause {
            details: false,
            matcher: Box::new(matcher),
        })
    }

    /// Some frame's message contains `text`.
    pub fn frame_contains(
        self,
        text: &str,
    ) -> Self {
        self.frame(text::contains(text))
    }

    /// Some frame's details contain `text`.
    pub fn frame_details_contains(
        self,
        text: &str,
    ) -> Self {
        self.clause(FrameClause {
            details: true,
            matcher: Box::new(text::contains(text)),
        })
    }

    /// The cause chain holds an error of type `T` (see
    /// [`BunsenError::find`]).
    pub fn has_cause<T>(self) -> Self
    where
        T: Error + 'static,
    {
        self.clause(CauseClause::<T> {
            matcher: None,
            _t: PhantomData,
        })
    }

    /// The cause chain holds an error of type `T` that matches `matcher`.
    pub fn cause<T>(
        self,
        matcher: impl Matcher<T> + 'static,
    ) -> Self
    where
        T: Error + 'static,
    {
        self.clause(CauseClause::<T> {
            matcher: Some(Box::new(matcher)),
            _t: PhantomData,
        })
    }

    /// The cause chain holds a [`Multiple`] with a member that matches
    /// `matcher`.
    pub fn any_member(
        self,
        matcher: ErrorMatcher,
    ) -> Self {
        self.clause(MemberClause {
            index: None,
            matcher,
        })
    }

    /// The cause chain holds a [`Multiple`] whose member `index` matches
    /// `matcher`.
    pub fn member(
        self,
        index: usize,
        matcher: ErrorMatcher,
    ) -> Self {
        self.clause(MemberClause {
            index: Some(index),
            matcher,
        })
    }

    /// Runs every clause on `error`.
    ///
    /// # Errors
    /// An [`ErrorMismatch`] listing every clause that missed.
    pub fn check(
        &self,
        error: &BunsenError,
    ) -> Result<(), ErrorMismatch> {
        let misses: Vec<String> = self
            .clauses
            .iter()
            .filter(|c| !c.matches(error))
            .map(|c| {
                let mut d = Description::new();
                c.describe_mismatch(error, &mut d);
                d.text().to_string()
            })
            .collect();
        if misses.is_empty() {
            Ok(())
        } else {
            Err(ErrorMismatch {
                expected: self.to_string(),
                misses,
                actual: format!("{error:#}"),
            })
        }
    }

    /// Panics, at the caller, unless `error` matches.
    #[track_caller]
    pub fn assert(
        &self,
        error: &BunsenError,
    ) {
        if let Err(miss) = self.check(error) {
            panic!("{miss}");
        }
    }

    /// Panics, at the caller, unless `result` is an error that matches.
    #[track_caller]
    pub fn assert_err<T: Debug>(
        &self,
        result: &BunsenResult<T>,
    ) {
        match result {
            Ok(value) => panic!("ErrorMatcher failed\n  expected: {self}\n  actual: Ok({value:?})"),
            Err(error) => self.assert(error),
        }
    }
}

impl Matcher<BunsenError> for ErrorMatcher {
    fn matches(
        &self,
        actual: &BunsenError,
    ) -> bool {
        self.clauses.iter().all(|c| c.matches(actual))
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        d.push("an error");
        for clause in &self.clauses {
            d.push(", ");
            clause.describe(d);
        }
    }

    fn describe_mismatch(
        &self,
        actual: &BunsenError,
        d: &mut Description,
    ) {
        let mut first = true;
        for clause in self.clauses.iter().filter(|c| !c.matches(actual)) {
            if !first {
                d.push("; ");
            }
            first = false;
            clause.describe_mismatch(actual, d);
        }
    }
}

impl fmt::Display for ErrorMatcher {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let mut d = Description::new();
        Matcher::describe(self, &mut d);
        f.write_str(d.text())
    }
}

impl Debug for ErrorMatcher {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "ErrorMatcher({self})")
    }
}

/// What an [`ErrorMatcher`] expected, every clause that missed, and the
/// error's report. `Display` prints all three.
#[derive(Clone, Debug)]
pub struct ErrorMismatch {
    /// The matcher's description.
    pub expected: String,
    /// One line per clause that missed.
    pub misses: Vec<String>,
    /// The error's full report.
    pub actual: String,
}

impl fmt::Display for ErrorMismatch {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        writeln!(f, "ErrorMatcher failed")?;
        writeln!(f, "  expected: {}", self.expected)?;
        for (i, miss) in self.misses.iter().enumerate() {
            let label = if i == 0 {
                "  mismatch: "
            } else {
                "            "
            };
            writeln!(f, "{label}{miss}")?;
        }
        writeln!(f, "  actual:")?;
        for line in self.actual.lines() {
            writeln!(f, "    {line}")?;
        }
        Ok(())
    }
}

impl Error for ErrorMismatch {}

struct KindClause(BoxMatcher<BunsenErrorKind>);

impl Matcher<BunsenError> for KindClause {
    fn matches(
        &self,
        actual: &BunsenError,
    ) -> bool {
        self.0.matches(&actual.kind())
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        d.push("of kind ");
        self.0.describe(d);
    }

    fn describe_mismatch(
        &self,
        actual: &BunsenError,
        d: &mut Description,
    ) {
        let _ = write!(d, "kind was {}", actual.kind());
    }
}

#[derive(Clone, Copy)]
enum Part {
    Message,
    Display,
    Details,
}

impl Part {
    fn name(self) -> &'static str {
        match self {
            Part::Message => "message",
            Part::Display => "display",
            Part::Details => "details",
        }
    }
}

struct TextClause {
    part: Part,
    matcher: BoxMatcher<str>,
}

impl TextClause {
    fn text(
        &self,
        error: &BunsenError,
    ) -> Option<String> {
        match self.part {
            Part::Message => Some(error.message().to_string()),
            Part::Display => Some(error.to_string()),
            Part::Details => error.details().map(ToString::to_string),
        }
    }
}

impl Matcher<BunsenError> for TextClause {
    fn matches(
        &self,
        actual: &BunsenError,
    ) -> bool {
        self.text(actual).is_some_and(|t| self.matcher.matches(&t))
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        let _ = write!(d, "with {} ", self.part.name());
        self.matcher.describe(d);
    }

    fn describe_mismatch(
        &self,
        actual: &BunsenError,
        d: &mut Description,
    ) {
        match self.text(actual) {
            None => {
                let _ = write!(d, "had no {}", self.part.name());
            }
            Some(text) => {
                let _ = write!(d, "{} ", self.part.name());
                self.matcher.describe_mismatch(&text, d);
            }
        }
    }
}

struct FrameClause {
    details: bool,
    matcher: BoxMatcher<str>,
}

impl FrameClause {
    fn texts<'a>(
        &self,
        error: &'a BunsenError,
    ) -> Vec<&'a str> {
        error
            .frames()
            .iter()
            .filter_map(|f| {
                if self.details {
                    f.details()
                } else {
                    Some(f.message())
                }
            })
            .collect()
    }
}

impl Matcher<BunsenError> for FrameClause {
    fn matches(
        &self,
        actual: &BunsenError,
    ) -> bool {
        self.texts(actual).iter().any(|t| self.matcher.matches(t))
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        d.push(if self.details {
            "with a frame's details "
        } else {
            "with a frame "
        });
        self.matcher.describe(d);
    }

    fn describe_mismatch(
        &self,
        actual: &BunsenError,
        d: &mut Description,
    ) {
        let what = if self.details {
            "frame details"
        } else {
            "frames"
        };
        let _ = write!(d, "{what} were {:?}", self.texts(actual));
    }
}

struct CauseClause<T: ?Sized> {
    matcher: Option<BoxMatcher<T>>,
    _t: PhantomData<fn(&T)>,
}

impl<T: Error + 'static> Matcher<BunsenError> for CauseClause<T> {
    fn matches(
        &self,
        actual: &BunsenError,
    ) -> bool {
        match (actual.find::<T>(), &self.matcher) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some(cause), Some(m)) => m.matches(cause),
        }
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        let _ = write!(d, "with a cause {}", type_name::<T>());
        if let Some(m) = &self.matcher {
            d.push(" that is ");
            m.describe(d);
        }
    }

    fn describe_mismatch(
        &self,
        actual: &BunsenError,
        d: &mut Description,
    ) {
        match (actual.find::<T>(), &self.matcher) {
            (None, _) => {
                let _ = write!(d, "no cause of type {} in the chain", type_name::<T>());
            }
            (Some(cause), Some(m)) => {
                let _ = write!(d, "cause {}: ", type_name::<T>());
                m.describe_mismatch(cause, d);
            }
            (Some(_), None) => {}
        }
    }
}

struct MemberClause {
    index: Option<usize>,
    matcher: ErrorMatcher,
}

impl Matcher<BunsenError> for MemberClause {
    fn matches(
        &self,
        actual: &BunsenError,
    ) -> bool {
        let Some(multiple) = actual.find::<Multiple>() else {
            return false;
        };
        match self.index {
            Some(i) => multiple
                .members
                .get(i)
                .is_some_and(|(_, e)| self.matcher.matches(e)),
            None => multiple
                .members
                .iter()
                .any(|(_, e)| self.matcher.matches(e)),
        }
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        match self.index {
            Some(i) => {
                let _ = write!(d, "with member {i} (");
            }
            None => d.push("with a member ("),
        }
        Matcher::describe(&self.matcher, d);
        d.push(")");
    }

    fn describe_mismatch(
        &self,
        actual: &BunsenError,
        d: &mut Description,
    ) {
        match actual.find::<Multiple>() {
            None => d.push("no Multiple in the chain"),
            Some(multiple) => match self.index {
                Some(i) => match multiple.members.get(i) {
                    None => {
                        let _ = write!(d, "no member {i}; there are {}", multiple.members.len());
                    }
                    Some((label, e)) => {
                        let _ = write!(d, "member {i} ({label}): ");
                        Matcher::describe_mismatch(&self.matcher, e, d);
                    }
                },
                None => {
                    let _ = write!(d, "none of {} members matched", multiple.members.len());
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::errors::{
        LookupError,
        LookupProblem,
        ResultContext,
        testing::predicate,
    };

    fn lookup() -> BunsenResult<()> {
        Err(BunsenError::lookup(LookupError::missing(
            "shard set",
            "large",
        )))
        .context_details(|| ("loading the corpus", "sets: tiny, small"))
    }

    #[test]
    fn test_all_clauses_match() {
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_eq("no shard set \"large\"")
            .display_contains("loading the corpus: no shard set")
            .frame_contains("corpus")
            .frame_details_contains("tiny")
            .has_cause::<LookupError>()
            .cause(predicate("missing", |c: &LookupError| {
                c.problem == LookupProblem::Missing
            }))
            .assert_err(&lookup());
    }

    #[test]
    fn test_every_miss_is_listed() {
        let e = lookup().unwrap_err();
        let miss = ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_contains("downloads")
            .details_contains("x")
            .check(&e)
            .unwrap_err();
        assert_eq!(miss.misses.len(), 3);
        assert_eq!(miss.misses[0], "kind was Lookup");
        assert_eq!(miss.misses[1], "message was \"no shard set \\\"large\\\"\"");
        assert_eq!(miss.misses[2], "had no details");
        let text = miss.to_string();
        assert!(text.contains("error[Lookup]: no shard set"), "{text}");
    }

    #[test]
    fn test_members() {
        let e: BunsenError = Multiple::new(
            "2 of 2 failed",
            vec![
                ("a".into(), BunsenError::unavailable("timed out")),
                ("b".into(), BunsenError::invalid_resource("bad digest")),
            ],
        )
        .into();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .member(0, ErrorMatcher::kind(BunsenErrorKind::Unavailable))
            .any_member(ErrorMatcher::new().message_contains("digest"))
            .assert(&e);
        assert!(
            !ErrorMatcher::new()
                .member(5, ErrorMatcher::new())
                .matches(&e)
        );
    }

    #[test]
    #[should_panic(expected = "actual: Ok(3)")]
    fn test_assert_err_on_ok() {
        let r: BunsenResult<i32> = Ok(3);
        ErrorMatcher::new().assert_err(&r);
    }
}
