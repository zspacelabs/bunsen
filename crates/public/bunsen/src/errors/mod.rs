//! Common bunsen errors.

mod result_ext;

use burn::{
    prelude::Shape,
    tensor::Slice,
};
#[doc(inline)]
pub use result_ext::*;

/// Common bunsen error type.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BunsenError {
    /// Resource not found.
    #[error("{0}")]
    ResourceNotFound(String),

    /// Parse error.
    #[error("{0}")]
    ParseError(String),

    /// Invalid constraint.
    #[error("{0}")]
    Invalid(String),

    /// Error from an external component.
    #[error("{0}")]
    External(String),

    /// An error occurred while slicing.
    #[error("{0}")]
    SliceError(SlicingError),

    /// Invalid Arguments.
    #[error("{msg}")]
    InvalidArgument {
        /// Message.
        msg: String,
    },

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
    /// Maps an error to an External string error.
    pub fn external<E>(e: E) -> Self
    where
        E: std::error::Error,
    {
        BunsenError::External(e.to_string())
    }
}

/// Result type for bunsen operations.
pub type BunsenResult<T> = core::result::Result<T, BunsenError>;

/// Errors that can occur when checking tensor slices.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SlicingError {
    /// Out of bounds.
    #[error("out of bounds: {msg}\nshape: {shape}\nslices: {slices:?}")]
    OutOfBounds {
        /// Message.
        msg: String,

        /// Shape.
        shape: Shape,

        /// Slices.
        slices: Vec<Slice>,
    },

    /// Invalid rank.
    #[error("out of bounds: {msg}\nshape: {shape}\nslices: {slices:?}")]
    InvalidRank {
        /// Message.
        msg: String,

        /// Shape.
        shape: Shape,

        /// Slices.
        slices: Vec<Slice>,
    },
}
