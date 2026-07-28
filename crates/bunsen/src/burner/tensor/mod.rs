//! # Tensor Operations
//!
//! Extension traits that add utility methods to [`burn::Tensor`], along with
//! indexed view wrappers for [`burn::tensor::TensorData`]. Importing the
//! traits (e.g. via `use bunsen::burner::tensor::*;`) makes the methods
//! available directly on `Tensor` values.
//!
//! ## Extension Traits
//!
//! [`TensorOpExt`] — all tensor kinds (`Float`, `Int`, `Bool`):
//! * [`swap`](TensorOpExt::swap) — exchange the contents of two tensors in
//!   place.
//! * [`release`](TensorOpExt::extract) — take the tensor's value out (e.g. from
//!   a struct field), leaving an empty tensor behind.
//! * [`select_dim`](TensorOpExt::select_dim) — select a single index along a
//!   dimension and squeeze it, reducing the rank by one (e.g. extract one row
//!   or column of a matrix as a vector).
//!
//! [`TensorOrderedOpExt`] — ordered type (Int, Float) tensors:
//! - [`in_range`](TensorOrderedOpExt::in_range) - elementwise `Range<E>` test.
//! - [`in_range_scalar`](TensorOrderedOpExt::in_range_scalar) - elementwise
//!   `Range<E>` test.
//!
//! [`TensorIntOpExt`] — `Int` tensors:
//! * [`square`](TensorIntOpExt::square) — elementwise square.
//!
//! [`TensorBoolOpExt`] — `Bool` tensors:
//! * [`count_dim`](TensorBoolOpExt::count_dim) /
//!   [`count_dims`](TensorBoolOpExt::count_dims) — count `true` elements along
//!   one or more dimensions (negative indexing supported), producing an `Int`
//!   tensor with the aggregated dimensions reduced to size 1.
//!
//! # Example
//! ```rust,no_run
//! use bunsen::burner::tensor::*;
//! use burn::prelude::*;
//!
//! fn row_counts<B: Backend>(
//!     occupied: Tensor<B, 2, Bool>
//! ) -> Tensor<B, 2, Int> {
//!     // Count the `true` cells in each row.
//!     occupied.count_dim(-1)
//! }
//! ```
//!
//! ## `TensorData` Views
//!
//! [`TensorDataIndexView`] and [`TensorDataIndexMutView`] wrap a
//! [`burn::tensor::TensorData`] to provide multi-dimensional element access
//! via `view[&[i, j]]` indexing.

mod data_view;
mod tensor_op_ext;

#[doc(inline)]
pub use data_view::*;
#[doc(inline)]
pub use tensor_op_ext::*;
