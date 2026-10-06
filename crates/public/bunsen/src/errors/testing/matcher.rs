//! `Matcher`, `Description` and the combinators.

use alloc::{
    boxed::Box,
    string::String,
    vec::Vec,
};
use core::{
    fmt,
    marker::PhantomData,
};

/// A check on a value of type `T` that can describe itself: hamcrest's
/// `Matcher<T>`.
pub trait Matcher<T: ?Sized>: Send + Sync {
    /// Whether `actual` matches.
    fn matches(
        &self,
        actual: &T,
    ) -> bool;

    /// Writes what the matcher expects: `a string containing "no shard set"`.
    fn describe(
        &self,
        d: &mut Description,
    );

    /// Writes why `actual` does not match: `was "downloads are off"`.
    fn describe_mismatch(
        &self,
        actual: &T,
        d: &mut Description,
    );
}

/// A boxed [`Matcher`].
pub type BoxMatcher<T> = Box<dyn Matcher<T>>;

impl<T: ?Sized> Matcher<T> for Box<dyn Matcher<T>> {
    fn matches(
        &self,
        actual: &T,
    ) -> bool {
        (**self).matches(actual)
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        (**self).describe(d)
    }

    fn describe_mismatch(
        &self,
        actual: &T,
        d: &mut Description,
    ) {
        (**self).describe_mismatch(actual, d)
    }
}

/// The text a [`Matcher`] writes about itself: a string builder that
/// implements `fmt::Write`.
#[derive(Clone, Debug, Default)]
pub struct Description {
    text: String,
}

impl Description {
    /// An empty description.
    pub fn new() -> Self {
        Self::default()
    }

    /// The text so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Appends `text`.
    pub fn push(
        &mut self,
        text: &str,
    ) {
        self.text.push_str(text);
    }

    /// Whether nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

impl fmt::Write for Description {
    fn write_str(
        &mut self,
        s: &str,
    ) -> fmt::Result {
        self.text.push_str(s);
        Ok(())
    }
}

impl fmt::Display for Description {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// The text of `matcher`'s [`describe`](Matcher::describe).
pub fn describe<T: ?Sized>(matcher: &dyn Matcher<T>) -> String {
    let mut d = Description::new();
    matcher.describe(&mut d);
    d.text
}

/// The text of `matcher`'s [`describe_mismatch`](Matcher::describe_mismatch)
/// on `actual`.
pub fn describe_mismatch<T: ?Sized>(
    matcher: &dyn Matcher<T>,
    actual: &T,
) -> String {
    let mut d = Description::new();
    matcher.describe_mismatch(actual, &mut d);
    d.text
}

/// Matches when every matcher matches. See [`all_of`].
pub struct AllOf<T: ?Sized> {
    matchers: Vec<BoxMatcher<T>>,
}

/// Matches when every one of `matchers` matches.
pub fn all_of<T: ?Sized>(matchers: impl IntoIterator<Item = BoxMatcher<T>>) -> AllOf<T> {
    AllOf {
        matchers: matchers.into_iter().collect(),
    }
}

impl<T: ?Sized> Matcher<T> for AllOf<T> {
    fn matches(
        &self,
        actual: &T,
    ) -> bool {
        self.matchers.iter().all(|m| m.matches(actual))
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        for (i, m) in self.matchers.iter().enumerate() {
            if i > 0 {
                d.push(" and ");
            }
            d.push("(");
            m.describe(d);
            d.push(")");
        }
    }

    fn describe_mismatch(
        &self,
        actual: &T,
        d: &mut Description,
    ) {
        let mut first = true;
        for m in self.matchers.iter().filter(|m| !m.matches(actual)) {
            if !first {
                d.push("; ");
            }
            first = false;
            m.describe_mismatch(actual, d);
        }
    }
}

/// Matches when any matcher matches. See [`any_of`].
pub struct AnyOf<T: ?Sized> {
    matchers: Vec<BoxMatcher<T>>,
}

/// Matches when at least one of `matchers` matches.
pub fn any_of<T: ?Sized>(matchers: impl IntoIterator<Item = BoxMatcher<T>>) -> AnyOf<T> {
    AnyOf {
        matchers: matchers.into_iter().collect(),
    }
}

impl<T: ?Sized> Matcher<T> for AnyOf<T> {
    fn matches(
        &self,
        actual: &T,
    ) -> bool {
        self.matchers.iter().any(|m| m.matches(actual))
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        for (i, m) in self.matchers.iter().enumerate() {
            if i > 0 {
                d.push(" or ");
            }
            d.push("(");
            m.describe(d);
            d.push(")");
        }
    }

    fn describe_mismatch(
        &self,
        actual: &T,
        d: &mut Description,
    ) {
        for (i, m) in self.matchers.iter().enumerate() {
            if i > 0 {
                d.push("; ");
            }
            m.describe_mismatch(actual, d);
        }
    }
}

/// Matches when its matcher does not. See [`not`].
pub struct Not<M> {
    matcher: M,
}

/// Matches when `matcher` does not.
pub fn not<M>(matcher: M) -> Not<M> {
    Not { matcher }
}

impl<T: ?Sized, M: Matcher<T>> Matcher<T> for Not<M> {
    fn matches(
        &self,
        actual: &T,
    ) -> bool {
        !self.matcher.matches(actual)
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        d.push("not ");
        self.matcher.describe(d);
    }

    fn describe_mismatch(
        &self,
        _actual: &T,
        d: &mut Description,
    ) {
        d.push("matched ");
        self.matcher.describe(d);
    }
}

/// Matches anything. See [`anything`].
pub struct Anything;

/// Matches anything.
pub fn anything() -> Anything {
    Anything
}

impl<T: ?Sized> Matcher<T> for Anything {
    fn matches(
        &self,
        _actual: &T,
    ) -> bool {
        true
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        d.push("anything");
    }

    fn describe_mismatch(
        &self,
        _actual: &T,
        _d: &mut Description,
    ) {
    }
}

/// Matches what a closure accepts. See [`predicate`].
pub struct Predicate<T: ?Sized, F> {
    description: String,
    f: F,
    _t: PhantomData<fn(&T)>,
}

/// Matches what `f` accepts; `description` says what that is.
///
/// The closure is where a test binds the fields it cares about:
///
/// ```
/// use bunsen::errors::{
///     LookupError,
///     LookupProblem,
///     testing::{
///         Matcher,
///         predicate,
///     },
/// };
///
/// let m = predicate("a missing \"large\"", |c: &LookupError| {
///     c.problem == LookupProblem::Missing && c.key == "large"
/// });
/// assert!(m.matches(&LookupError::missing("shard set", "large")));
/// ```
pub fn predicate<T: ?Sized, F>(
    description: impl Into<String>,
    f: F,
) -> Predicate<T, F>
where
    F: Fn(&T) -> bool + Send + Sync,
{
    Predicate {
        description: description.into(),
        f,
        _t: PhantomData,
    }
}

impl<T: ?Sized, F> Matcher<T> for Predicate<T, F>
where
    F: Fn(&T) -> bool + Send + Sync,
{
    fn matches(
        &self,
        actual: &T,
    ) -> bool {
        (self.f)(actual)
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        d.push(&self.description);
    }

    fn describe_mismatch(
        &self,
        _actual: &T,
        d: &mut Description,
    ) {
        d.push("was not ");
        d.push(&self.description);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::testing::text;

    #[test]
    fn test_combinators() {
        let m = all_of::<str>([
            Box::new(text::contains("shard")) as BoxMatcher<str>,
            Box::new(not(text::contains("tile"))),
        ]);
        assert!(m.matches("no shard set"));
        assert!(!m.matches("no shard tile"));
        assert_eq!(
            describe(&m),
            "(a string containing \"shard\") and (not a string containing \"tile\")"
        );

        let m = any_of::<str>([
            Box::new(text::eq("a")) as BoxMatcher<str>,
            Box::new(text::eq("b")),
        ]);
        assert!(m.matches("b"));
        assert!(!m.matches("c"));

        let p = predicate("even", |n: &usize| n % 2 == 0);
        assert!(p.matches(&4));
        assert_eq!(describe_mismatch(&p, &3), "was not even");
    }
}
