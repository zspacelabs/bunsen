//! # Errors
//!
//! [`BunsenError`] is the crate-wide error type, and [`BunsenResult`] the
//! result that carries it. [`SlicingError`] is the error of a slice bounds
//! check ([`check_slices_bounds`](crate::zspace::check_slices_bounds)); it
//! travels inside a `BunsenError` as [`BunsenError::SliceError`].
//! [`WithOkOrPanic`] turns any of them into a panic, for the callers that want
//! one.
//!
//! # Convention: `try_x` and `x`
//!
//! A fallible operation comes as a pair:
//!
//! - `try_x` returns a [`BunsenResult`]. Input that can be wrong (a config, a
//!   file, a user's spec) is reported here, as [`BunsenError::Invalid`], and
//!   not as a panic.
//! - `x` (or `expect_x`) is its panicking twin, for a caller that cannot
//!   recover or knows the input is good. It calls `try_x` and unwraps with
//!   [`ok_or_panic`](WithOkOrPanic::ok_or_panic), which panics with the error's
//!   message.
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
//!     BunsenError,
//!     BunsenResult,
//!     WithOkOrPanic,
//! };
//!
//! /// Parses a width, which must be positive.
//! fn try_parse_width(spec: &str) -> BunsenResult<usize> {
//!     match spec.parse::<usize>() {
//!         Ok(width) if width > 0 => Ok(width),
//!         _ => Err(BunsenError::Invalid(format!(
//!             "width must be a positive integer: {spec:?}"
//!         ))),
//!     }
//! }
//!
//! /// Parses a width, which must be positive, or panics.
//! fn parse_width(spec: &str) -> usize {
//!     try_parse_width(spec).ok_or_panic()
//! }
//!
//! assert_eq!(parse_width("640"), 640);
//! assert!(matches!(try_parse_width("0"), Err(BunsenError::Invalid(_))));
//! ```
//!
//! # Which variant
//!
//! | Variant | For |
//! |---------|-----|
//! | [`Invalid`](BunsenError::Invalid) | Input that breaks a constraint: a config whose fields disagree, an argument out of range, a malformed spec. The usual error of a `try_x`. |
//! | [`ParseError`](BunsenError::ParseError) | Text that does not parse as the type asked for. |
//! | [`ResourceNotFound`](BunsenError::ResourceNotFound) | Something named that is absent: a file, a cache entry, a pretrained name, a key. |
//! | [`External`](BunsenError::External) | A failure inside a dependency or the outside world: I/O, the network, a codec. Build it with [`BunsenError::external`]. |
//! | [`AssertionError`](BunsenError::AssertionError) | Values that fail a check against an expectation: the [`TensorDataCheckExt`] checks, audit verification. |
//! | [`SliceError`](BunsenError::SliceError) | A slice that does not fit a shape; carries a [`SlicingError`]. |
//! | [`UnsupportedRank`](BunsenError::UnsupportedRank) | An operation asked of a tensor rank it does not support. |
//!
//! `BunsenError` is `Clone` and `PartialEq`, and most `std` errors are
//! neither, so the variants hold messages rather than source errors:
//! [`BunsenError::external`] keeps a foreign error's `Display` text, and drops
//! its type and its `source()` chain.
//!
//! [`ModuleInit::try_init`]: crate::burner::module::ModuleInit::try_init
//! [`ToStructureConfig::try_to_structure`]: crate::burner::module::ToStructureConfig::try_to_structure
//! [`XmlModuleTree::try_select`]: crate::burner::module::reflection::XmlModuleTree::try_select
//! [`try_probability`]: crate::support::validators::try_probability
//! [`expect_probability`]: crate::support::validators::expect_probability
//! [`try_point_bounds_check`]: crate::zspace::try_point_bounds_check
//! [`expect_point_bounds_check`]: crate::zspace::expect_point_bounds_check
//! [`TensorDataCheckExt`]: crate::burner::tensor::TensorDataCheckExt

mod result_ext;

use burn::{
    prelude::Shape,
    tensor::Slice,
};
pub use result_ext::*;

/// The crate-wide error type.
///
/// Every variant but [`SliceError`](Self::SliceError) and
/// [`UnsupportedRank`](Self::UnsupportedRank) is a message, and displays as
/// it. The [module docs](crate::errors#which-variant) say which variant fits
/// which failure.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum BunsenError {
    /// Values failed a check against an expectation.
    #[error("{0}")]
    AssertionError(String),

    /// A named file, entry or key is absent.
    #[error("{0}")]
    ResourceNotFound(String),

    /// Text did not parse as the type asked for.
    #[error("{0}")]
    ParseError(String),

    /// Input broke a constraint: the usual error of a `try_x`.
    #[error("{0}")]
    Invalid(String),

    /// A dependency or the outside world failed; see
    /// [`external`](Self::external).
    #[error("{0}")]
    External(String),

    /// A slice bounds check failed.
    #[error("{0}")]
    SliceError(SlicingError),

    /// The tensor rank is not supported for the requested operation.
    #[error("rank: {rank}:: {msg}")]
    UnsupportedRank {
        /// Message.
        msg: String,

        /// Rank.
        rank: usize,
    },
}

impl BunsenError {
    /// Wraps a foreign error as [`External`](Self::External).
    ///
    /// Keeps only the error's `Display` text; its type and its `source()`
    /// chain are dropped, which keeps `BunsenError` `Clone` and `PartialEq`.
    /// Shaped for `map_err`:
    ///
    /// ```
    /// use bunsen::errors::{
    ///     BunsenError,
    ///     BunsenResult,
    /// };
    ///
    /// fn read(path: &str) -> BunsenResult<String> {
    ///     std::fs::read_to_string(path).map_err(BunsenError::external)
    /// }
    ///
    /// assert!(matches!(
    ///     read("/no/such/file"),
    ///     Err(BunsenError::External(_))
    /// ));
    /// ```
    pub fn external<E>(e: E) -> Self
    where
        E: std::error::Error,
    {
        BunsenError::External(e.to_string())
    }
}

/// The result of a fallible bunsen operation: the `try_x` half of the
/// [convention](crate::errors#convention-try_x-and-x).
pub type BunsenResult<T> = core::result::Result<T, BunsenError>;

/// Why a list of slices does not fit a tensor shape.
///
/// Returned by [`check_slices_bounds`](crate::zspace::check_slices_bounds),
/// and carried as [`BunsenError::SliceError`]. Both variants hold the
/// offending shape and slices.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum SlicingError {
    /// A slice reaches outside its dimension.
    #[error("out of bounds: {msg}\nshape: {shape}\nslices: {slices:?}")]
    OutOfBounds {
        /// Message.
        msg: String,

        /// Shape.
        shape: Shape,

        /// Slices.
        slices: Vec<Slice>,
    },

    /// There are more slices than the shape has dimensions.
    #[error("invalid rank: {msg}\nshape: {shape}\nslices: {slices:?}")]
    InvalidRank {
        /// Message.
        msg: String,

        /// Shape.
        shape: Shape,

        /// Slices.
        slices: Vec<Slice>,
    },
}
