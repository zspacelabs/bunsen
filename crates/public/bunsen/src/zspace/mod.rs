//! # Z-space: integer index spaces
//!
//! Z-space is the space of n-dimensional integer coordinate tuples: the index
//! space of a tensor. bunsen uses it to give n-dimensional indices, shapes
//! and slices a precise meaning.
//!
//! It is Manhattan (taxicab) space, plus a **partial order**. A point `a` is
//! at most a point `b` when every coordinate of `a` is at most the matching
//! coordinate of `b`; it is less when, in addition, the two differ. Points
//! that are larger on one axis and smaller on another are incomparable:
//!
//! ```text
//! [1, 2] == [1, 2]
//! [0, 0] <  [0, 1] <  [1, 1]
//! [1, 0] and [0, 1] are incomparable
//! ```
//!
//! The only regions z-space describes are axis-aligned boxes, and the order
//! is chosen to make them simple to state and to test: the half-open box
//! `[start, end)` holds the points `p` with `start[i] <= p[i] < end[i]` on
//! every axis `i`.
//!
//! # What is here
//!
//! - **Points and boxes.** [`zspace_partial_cmp`] is the partial order.
//!   [`try_point_bounds_check`] / [`expect_point_bounds_check`] test a point
//!   against a half-open box, axis by axis; they are a `try_x` / `x` pair of
//!   the [errors convention](crate::errors#convention-try_x-and-x).
//! - **Slices.** [`check_slices_bounds`] checks a list of burn
//!   [`Slice`](burn::tensor::Slice)s against a shape before slicing, and says
//!   what is wrong with a [`SlicingError`](crate::errors::SlicingError).
//!   [`DynTensor`](crate::burner::tensor::dynamic::DynTensor)'s slicing calls
//!   it.
//! - **Raveling.** [`ravel_dims`] and [`ravel_shape`] turn a coordinate tuple
//!   into a row-major offset;
//!   [`TensorDataView`](crate::burner::tensor::TensorDataView) indexes with
//!   them.

mod bounds;
mod check_slices_bounds;
mod ravel_utils;

pub use bounds::*;
pub use check_slices_bounds::*;
pub use ravel_utils::*;
