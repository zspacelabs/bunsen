//! # Testing errors
//!
//! Matchers that check the parts of a
//! [`BunsenError`](crate::errors::BunsenError) a test cares about, in the style
//! of hamcrest: the kind, the message, a frame, the details, a typed cause. A
//! test pins what it is about and nothing else, and a failure says what was
//! expected, what missed, and prints the error's full report.
//!
//! This module exists with the `testing` feature, and in bunsen's own unit
//! tests.
//!
//! ```
//! use bunsen::errors::{
//!     BunsenError,
//!     BunsenErrorKind,
//!     BunsenResult,
//!     LookupError,
//!     LookupProblem,
//!     ResultContext,
//!     testing::{
//!         ErrorMatcher,
//!         predicate,
//!     },
//! };
//!
//! fn lookup_set(name: &str) -> BunsenResult<()> {
//!     Err(BunsenError::lookup(
//!         LookupError::missing("shard set", name)
//!             .with_candidates(["tiny", "small"]),
//!     ))
//!     .context("loading the corpus")
//! }
//!
//! ErrorMatcher::lookup()
//!     .message_contains("no shard set")
//!     .frame_contains("loading")
//!     .cause(predicate("a missing \"large\"", |c: &LookupError| {
//!         c.problem == LookupProblem::Missing && c.key == "large"
//!     }))
//!     .assert_err(&lookup_set("large"));
//! ```
//!
//! [`ErrorMatcher`] is a fluent builder of clauses, each a [`Matcher`] on one
//! part of the error. The building blocks are plain functions returning
//! matchers: [`text`] on strings, [`value`] on values, and the combinators
//! [`all_of`], [`any_of`], [`not`], [`anything`] and [`predicate`]. A
//! `predicate` takes any closure, which is where a test binds the fields of a
//! typed cause it cares about. They live in
//! [`support::testing::matching`](crate::support::testing::matching), the
//! matchers of every kind, and are re-exported here.
//!
//! A matcher per kind reads as the error's own constructor
//! (`ErrorMatcher::illegal()` for `BunsenError::illegal`), and
//! [`constraint`](ErrorMatcher::constraint) matches a refusal by the
//! [`ConstraintError`](crate::errors::ConstraintError) it carries, so a test
//! need not match the message's words.

mod error_matcher;

pub use error_matcher::*;

// The generic matcher core lives in `support::testing::matching`; it is
// re-exported here, so `errors::testing::{predicate, text, value, ..}` keep
// their paths.
pub use crate::support::testing::matching::*;
