//! # Text matchers
//!
//! [`Matcher`]s on `str`: [`eq`], [`contains`], [`starts_with`],
//! [`ends_with`]. A mismatch prints the actual text, quoted.

use alloc::string::String;
use core::fmt::Write;

use crate::errors::testing::{
    Description,
    Matcher,
};

#[derive(Clone, Copy, Debug)]
enum Op {
    Eq,
    Contains,
    StartsWith,
    EndsWith,
}

/// A text matcher. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct TextMatcher {
    op: Op,
    text: String,
}

/// Matches text equal to `text`.
pub fn eq(text: impl Into<String>) -> TextMatcher {
    TextMatcher {
        op: Op::Eq,
        text: text.into(),
    }
}

/// Matches text that contains `text`.
pub fn contains(text: impl Into<String>) -> TextMatcher {
    TextMatcher {
        op: Op::Contains,
        text: text.into(),
    }
}

/// Matches text that starts with `text`.
pub fn starts_with(text: impl Into<String>) -> TextMatcher {
    TextMatcher {
        op: Op::StartsWith,
        text: text.into(),
    }
}

/// Matches text that ends with `text`.
pub fn ends_with(text: impl Into<String>) -> TextMatcher {
    TextMatcher {
        op: Op::EndsWith,
        text: text.into(),
    }
}

impl Matcher<str> for TextMatcher {
    fn matches(
        &self,
        actual: &str,
    ) -> bool {
        match self.op {
            Op::Eq => actual == self.text,
            Op::Contains => actual.contains(self.text.as_str()),
            Op::StartsWith => actual.starts_with(self.text.as_str()),
            Op::EndsWith => actual.ends_with(self.text.as_str()),
        }
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        let op = match self.op {
            Op::Eq => "equal to",
            Op::Contains => "containing",
            Op::StartsWith => "starting with",
            Op::EndsWith => "ending with",
        };
        let _ = write!(d, "a string {op} {:?}", self.text);
    }

    fn describe_mismatch(
        &self,
        actual: &str,
        d: &mut Description,
    ) {
        let _ = write!(d, "was {actual:?}");
    }
}

impl Matcher<String> for TextMatcher {
    fn matches(
        &self,
        actual: &String,
    ) -> bool {
        Matcher::<str>::matches(self, actual.as_str())
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        Matcher::<str>::describe(self, d)
    }

    fn describe_mismatch(
        &self,
        actual: &String,
        d: &mut Description,
    ) {
        Matcher::<str>::describe_mismatch(self, actual.as_str(), d)
    }
}
