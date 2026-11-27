//! # Matchers
//!
//! Checks that describe themselves, in the style of hamcrest: a [`Matcher`]
//! says whether a value matches, describes what it expects, and describes how
//! a value misses. Tests compose them, and a failure prints what was expected
//! and what missed, not only that an assertion failed.
//!
//! - [`Matcher`], [`BoxMatcher`], [`Description`], [`describe`],
//!   [`describe_mismatch`]: the core.
//! - [`all_of`], [`any_of`], [`not`], [`anything`], [`predicate`]: the
//!   combinators; a `predicate` takes any closure.
//! - [`text`] on strings, [`value`] on values.
//!
//! The matchers of particular things build on these:
//! [`ErrorMatcher`](crate::errors::testing::ErrorMatcher) on a
//! [`BunsenError`](crate::errors::BunsenError) (and `errors::testing`
//! re-exports this module's items).

mod matcher;
#[cfg(test)]
mod tensor;
pub mod text;
pub mod value;

pub use matcher::*;
#[cfg(test)]
pub(crate) use tensor::*;
