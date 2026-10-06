//! `SlicingError`: slices that do not fit a shape.

use alloc::{
    borrow::Cow,
    format,
    string::String,
    vec::Vec,
};

use burn::{
    prelude::Shape,
    tensor::Slice,
};

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    Detailed,
};

/// Why a list of slices does not fit a tensor shape.
///
/// Returned by [`check_slices_bounds`](crate::zspace::check_slices_bounds).
/// Both variants hold the offending shape and slices, which are its
/// [`Detailed`] details. Its kind, through `From`, is
/// [`Illegal`](BunsenErrorKind::Illegal).
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum SlicingError {
    /// A slice reaches outside its dimension.
    #[error("slices out of bounds: {msg}")]
    OutOfBounds {
        /// Message.
        msg: String,

        /// Shape.
        shape: Shape,

        /// Slices.
        slices: Vec<Slice>,
    },

    /// There are more slices than the shape has dimensions.
    #[error("slices of the wrong rank: {msg}")]
    InvalidRank {
        /// Message.
        msg: String,

        /// Shape.
        shape: Shape,

        /// Slices.
        slices: Vec<Slice>,
    },
}

impl Detailed for SlicingError {
    fn details(&self) -> Option<Cow<'_, str>> {
        let (Self::OutOfBounds { shape, slices, .. } | Self::InvalidRank { shape, slices, .. }) =
            self;
        Some(Cow::Owned(format!("shape: {shape}\nslices: {slices:?}")))
    }
}

impl From<SlicingError> for BunsenError {
    #[track_caller]
    fn from(error: SlicingError) -> Self {
        BunsenError::from_detailed(BunsenErrorKind::Illegal, error)
    }
}
