//! # Errors
//!
//! [`BunsenError`] is the crate-wide error type, and [`BunsenResult`] the
//! result that carries it. An error is:
//!
//! - a [kind](BunsenErrorKind): what a generic consumer should do with it;
//! - a one-line message: what went wrong;
//! - optional multi-line details: the evidence, such as a diff or a dump;
//! - the [frames](Frame) of context it gathered on its way up: what the code
//!   was doing;
//! - an optional cause: a typed fact, such as a [`LookupError`], or a foreign
//!   error.
//!
//! Its `Display` is one line, the frames then the message. Its
//! [report](BunsenError::report) (`{:#}`, `Debug`) is the full, multi-line
//! form.
//!
//! # Which kind
//!
//! The kind is chosen by who or what has to change for the operation to
//! succeed. Ask in order:
//!
//! 1. Would fixing the **code** fix it? [`Illegal`](BunsenErrorKind::Illegal):
//!    the caller broke a rule of the API.
//!    [`Internal`](BunsenErrorKind::Internal) when the code at fault is
//!    bunsen's own.
//! 2. Would fixing the **request or its settings** fix it?
//!    [`Policy`](BunsenErrorKind::Policy): the operation is misconfigured, and
//!    bunsen refuses to proceed. Verification failures are `Policy` too: the
//!    expectation was set elsewhere, and the check enforces it.
//! 3. Otherwise the **world** disagrees:
//!    - a key that did not resolve, in a table, a file system or a server:
//!      [`Lookup`](BunsenErrorKind::Lookup);
//!    - not reachable now: [`Unavailable`](BunsenErrorKind::Unavailable);
//!    - present, but not what was expected:
//!      [`InvalidResource`](BunsenErrorKind::InvalidResource).
//! 4. A valid request this code or build does not do is
//!    [`Unsupported`](BunsenErrorKind::Unsupported). The host failing is
//!    [`Sys`](BunsenErrorKind::Sys).
//! 5. If none of these is clear, usually for a foreign error whose meaning
//!    cannot be seen, [`Other`](BunsenErrorKind::Other). It is a bail-out, not
//!    a category.
//!
//! | Kind | A generic consumer | For example |
//! |------|--------------------|-------------|
//! | `Illegal` | escalates, or panics in an `x` twin | a config built in code breaks a rule; a stream used after it ended |
//! | `Internal` | escalates: a bug in bunsen | an invariant of bunsen's own broke |
//! | `Policy` | reports it: fix the settings | a config file breaks a rule; downloads are turned off; values differ from a baseline |
//! | `Lookup` | falls back, or suggests a candidate | no such resource, file or URL; access denied |
//! | `Unavailable` | retries later | a timeout, a server error, a dropped connection |
//! | `InvalidResource` | reports the source | a digest mismatch; a file that is not the format it claims |
//! | `Unsupported` | takes another path | an unimplemented mode; a feature not compiled in |
//! | `Sys` | reports it to the operator | a full disk; a thread that would not spawn |
//! | `Other` | escalates | an unclassified foreign error |
//!
//! **Locality** usually decides between `Illegal` and `Policy`. `Illegal`
//! breaks a rule documented by the function that was called, in its own
//! `# Errors` or `# Panics` section. `Policy` breaks a rule set elsewhere,
//! which the call only enforces. The same check can be either: a config
//! built in code that breaks its own documented rule is `Illegal`, and the
//! same config read from a file is `Policy`. Code that reads values from
//! outside the program re-marks the errors they cause; see
//! [Re-marking](#re-marking).
//!
//! # Building an error
//!
//! Each kind has a constructor that takes a message:
//! [`illegal`](BunsenError::illegal), [`internal`](BunsenError::internal),
//! [`policy`](BunsenError::policy), [`unavailable`](BunsenError::unavailable),
//! [`invalid_resource`](BunsenError::invalid_resource),
//! [`unsupported`](BunsenError::unsupported), [`sys`](BunsenError::sys).
//! [`with_details`](BunsenError::with_details) adds the evidence.
//!
//! A typed cause carries the facts a caller may want to read. The shared
//! cause types convert with `From`, and pick their own kind:
//!
//! | Cause | Kind | Use |
//! |-------|------|-----|
//! | [`ConstraintError`] | `Illegal` | a value broke a [`Rule`]: zero, out of range, not a multiple, two fields that disagree |
//! | [`LookupError`] | `Lookup` | a key did not resolve, with the keys there are |
//! | [`DigestMismatch`] | `InvalidResource` | bytes are not what was pinned |
//! | [`ValueMismatch`] | `Policy` | a verification found values that differ |
//! | [`SlicingError`] | `Illegal` | slices that do not fit a shape |
//! | [`Multiple`] | from its members | several failures as one |
//!
//! A [`ParseError`] has no kind of its own: the site says whether the text
//! was outside data, a request, or the program's own, with
//! [`from_cause`](BunsenError::from_cause). `from_cause` builds an error of
//! any kind from any cause, with the cause's `Display` as the message;
//! [`from_detailed`](BunsenError::from_detailed) does the same for a
//! [`Detailed`] cause, whose details become the error's.
//! [`with_cause`](BunsenError::with_cause) attaches a cause under a message of
//! the error's own.
//!
//! A foreign error is sorted by what it means. An `io::Error` goes through
//! [`sys_at`] (with the path it was about) or [`sys_op`] (without one), which
//! sort it by its `io::ErrorKind`: a missing or forbidden path is a `Lookup`,
//! a timeout is `Unavailable`, and the rest is `Sys`. There is no
//! `From<io::Error>`, because a bare `?` could not say which path failed. A
//! foreign error with no better kind is [`other`](BunsenError::other).
//!
//! ```
//! use bunsen::errors::{
//!     BunsenError,
//!     BunsenErrorKind,
//!     BunsenResult,
//!     ConstraintError,
//!     Rule,
//!     sys_at,
//! };
//!
//! fn check_hop(hop: usize) -> BunsenResult<()> {
//!     if hop == 0 {
//!         return Err(ConstraintError::new(
//!             "StftConfig",
//!             "hop",
//!             Rule::ZeroOrEmpty,
//!         )
//!         .into());
//!     }
//!     Ok(())
//! }
//!
//! fn read_vocab(path: &str) -> BunsenResult<String> {
//!     std::fs::read_to_string(path).map_err(sys_at("read", path))
//! }
//!
//! assert_eq!(check_hop(0).unwrap_err().kind(), BunsenErrorKind::Illegal);
//! assert_eq!(
//!     read_vocab("/no/such/vocab").unwrap_err().kind(),
//!     BunsenErrorKind::Lookup
//! );
//! assert_eq!(
//!     BunsenError::unavailable("the hub timed out").kind(),
//!     BunsenErrorKind::Unavailable
//! );
//! ```
//!
//! # Context and details
//!
//! A layer an error passes through adds a [frame](Frame) saying what it was
//! doing, with [`ResultContext::context`] (or
//! [`with_context`](ResultContext::with_context), to build the text only on
//! error). A frame is narrative: it prints, and nothing should branch on it.
//! [`context_details`](ResultContext::context_details) adds a frame with
//! multi-line details, for evidence a higher layer holds, such as the event
//! an audit was checking. Context never changes the kind, and records the
//! source location of the call.
//!
//! ```
//! use bunsen::errors::{
//!     BunsenError,
//!     BunsenResult,
//!     ResultContext,
//! };
//!
//! fn load_block() -> BunsenResult<()> {
//!     Err(BunsenError::illegal("stride is zero")
//!         .with_details("stride: 0\nkernel: 3"))
//! }
//!
//! let e = load_block()
//!     .context("block 3")
//!     .with_context(|| format!("loading {:?}", "resnet18"))
//!     .unwrap_err();
//!
//! // One line: the frames, outermost first, then the message.
//! assert_eq!(
//!     e.to_string(),
//!     "loading \"resnet18\": block 3: stride is zero"
//! );
//!
//! // The report: kind, message, location, details, frames, causes.
//! let report = format!("{e:#}");
//! assert!(report.starts_with("error[Illegal]: stride is zero"));
//! assert!(report.contains("  | kernel: 3"));
//! assert!(report.contains("  while block 3 ("));
//! ```
//!
//! # Re-marking
//!
//! A layer that knows more than the site that built an error may change its
//! kind, within three rules:
//!
//! - [`as_policy`](ResultContext::as_policy) turns `Illegal` into `Policy`. An
//!   **input boundary** uses it: code that reads a config file, parses a
//!   command line, or loads a checkpoint, and passes the values on. A rule
//!   those values break is the input's fault, not the code's.
//! - [`as_internal`](ResultContext::as_internal) turns `Illegal` into
//!   `Internal`, where bunsen passed values it built itself.
//! - [`as_illegal`](ResultContext::as_illegal) turns any runtime kind into
//!   `Illegal`: a higher level declares that the call should never have
//!   happened, such as a lookup of a name the caller itself supplied.
//!
//! Every other combination passes the error through unchanged: an `Internal`
//! error stays `Internal`. The report lists each re-mark with its location.
//!
//! # Handling errors
//!
//! A **generic** consumer reads only the kind:
//! [`is_retryable`](BunsenErrorKind::is_retryable) is true only for
//! `Unavailable`, and [`is_bug`](BunsenErrorKind::is_bug) for `Illegal` and
//! `Internal`, which it escalates or panics on.
//!
//! A **specialist** with its own recovery strategy reads the kind and a cause
//! type it knows, with [`find`](BunsenError::find), which walks the cause
//! chain. The cache's mirror loop is one: a download whose digest does not
//! match is `InvalidResource` to everyone else, but to the cache its
//! `DigestMismatch` from a download says "try the next mirror". A subsystem
//! that needs such handling defines its own cause type, with a
//! `From<Cause> for BunsenError` that picks the kind from the variant, so
//! kind and cause cannot disagree.
//!
//! Keep the chain intact: add context with `context`, never by formatting an
//! error into a new message, and hold the members of an aggregate as
//! `BunsenError` values in a [`Multiple`]. `find` cannot see through text.
//!
//! # The `try_x` and `x` convention
//!
//! A fallible operation comes as a pair:
//!
//! - `try_x` returns a [`BunsenResult`]. Input that can be wrong (a config, a
//!   file, a user's spec) is reported here, as an error of the kind it is, and
//!   not as a panic.
//! - `x` (or `expect_x`) is its panicking twin, for a caller that cannot
//!   recover or knows the input is good. It calls `try_x` and unwraps with
//!   [`ok_or_panic`](WithOkOrPanic::ok_or_panic), which panics with the error's
//!   report.
//!
//! The pairs in bunsen include:
//!
//! - [`ModuleInit::try_init`] / `init`;
//! - [`ToStructureConfig::try_to_structure`] / `to_structure`;
//! - [`XmlModuleTree::try_select`] / `select`;
//! - [`try_probability`] / [`expect_probability`];
//! - [`try_point_bounds_check`] / [`expect_point_bounds_check`].
//!
//! ```
//! use bunsen::errors::{
//!     BunsenErrorKind,
//!     BunsenResult,
//!     ConstraintError,
//!     WithOkOrPanic,
//! };
//!
//! /// Parses a width, which must be a positive integer.
//! ///
//! /// # Errors
//! /// `Illegal`, with a [`ConstraintError`] cause, for a spec that is not a
//! /// positive integer.
//! fn try_parse_width(spec: &str) -> BunsenResult<usize> {
//!     match spec.parse::<usize>() {
//!         Ok(width) if width > 0 => Ok(width),
//!         _ => Err(ConstraintError::out_of_range(
//!             "width",
//!             "",
//!             format!("{spec:?}"),
//!             "positive integers",
//!         )
//!         .into()),
//!     }
//! }
//!
//! /// Parses a width, which must be a positive integer, or panics.
//! fn parse_width(spec: &str) -> usize {
//!     try_parse_width(spec).ok_or_panic()
//! }
//!
//! assert_eq!(parse_width("640"), 640);
//! assert_eq!(
//!     try_parse_width("0").unwrap_err().kind(),
//!     BunsenErrorKind::Illegal
//! );
//! ```
//!
//! A long-running service, or one that loads and runs graphs dynamically,
//! calls the `try_x` half and contains `Illegal` errors in its run container,
//! logging the report.
//!
//! # Testing errors
//!
//! `errors::testing` (feature `testing`) holds `ErrorMatcher`, which checks
//! only the parts of an error a test cares about: the kind, the message, a
//! frame, the details, a typed cause. It replaces matching on the whole error,
//! which `BunsenError` does not support: it is `Clone`, but not `PartialEq`.
//!
//! [`ModuleInit::try_init`]: crate::burner::module::ModuleInit::try_init
//! [`ToStructureConfig::try_to_structure`]: crate::burner::module::ToStructureConfig::try_to_structure
//! [`XmlModuleTree::try_select`]: crate::burner::module::reflection::XmlModuleTree::try_select
//! [`try_probability`]: crate::support::validators::try_probability
//! [`expect_probability`]: crate::support::validators::expect_probability
//! [`try_point_bounds_check`]: crate::zspace::try_point_bounds_check
//! [`expect_point_bounds_check`]: crate::zspace::expect_point_bounds_check

mod bunsen_error;
mod constraint_error;
mod detailed;
mod digest_mismatch;
mod foreign;
mod frame;
mod kind;
mod lookup_error;
mod multiple;
mod parse_error;
mod report;
mod result_ext;
mod slicing_error;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod value_mismatch;

pub use bunsen_error::*;
pub use constraint_error::*;
pub use detailed::*;
pub use digest_mismatch::*;
pub use foreign::*;
pub use frame::*;
pub use kind::*;
pub use lookup_error::*;
pub use multiple::*;
pub use parse_error::*;
pub use report::*;
pub use result_ext::*;
pub use slicing_error::*;
pub use value_mismatch::*;
