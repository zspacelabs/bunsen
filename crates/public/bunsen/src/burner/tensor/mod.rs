//! # Tensor Operations
//!
//! Extension traits that add utility methods to [`burn::Tensor`] and
//! [`burn::tensor::TensorData`], indexed view wrappers for `TensorData`, and
//! type- and rank-erased tensors in [`dynamic`]. Importing the traits (e.g. via
//! `use bunsen::burner::tensor::*;`, or [`crate::prelude`], which re-exports
//! this module) makes the methods available directly on `Tensor` and
//! `TensorData` values.
//!
//! ## `Tensor` Extension Traits
//!
//! [`TensorOpExt`] — all tensor kinds (`Float`, `Int`, `Bool`):
//! * [`swap`](TensorOpExt::swap) — exchange the contents of two tensors in
//!   place.
//! * [`extract`](TensorOpExt::extract) — take the tensor's value out (e.g. from
//!   a struct field), leaving an empty tensor behind.
//! * [`replace_with`](TensorOpExt::replace_with) — replace the value with
//!   `f(value)`; `f` owns the tensor while it runs, so operations inside it may
//!   run in place.
//! * [`select_dim`](TensorOpExt::select_dim) — select a single index along a
//!   dimension and squeeze it, reducing the rank by one (e.g. extract one row
//!   or column of a matrix as a vector).
//! * [`copy_slice`](TensorOpExt::copy_slice) — copy one slice of the tensor
//!   onto another slice of it.
//!
//! [`TensorElemOpExt`] — element-type conversions on the way out:
//! * [`to_data_as`](TensorElemOpExt::to_data_as) /
//!   [`into_data_as`](TensorElemOpExt::into_data_as) — the tensor's data,
//!   converted to an element type `E`.
//! * [`to_data_cast`](TensorElemOpExt::to_data_cast) /
//!   [`into_data_cast`](TensorElemOpExt::into_data_cast) — the tensor's data,
//!   converted to a runtime `DType`.
//!
//! [`TensorOrderedOpExt`] — ordered type (`Int`, `Float`) tensors:
//! * [`in_range`](TensorOrderedOpExt::in_range) — elementwise `[start, end)`
//!   test against `start` and `end` tensors, producing a `Bool` mask.
//! * [`in_range_scalar`](TensorOrderedOpExt::in_range_scalar) — elementwise
//!   test against a scalar `Range<E>`, producing a `Bool` mask.
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
//! ## `TensorData` Extension Traits
//!
//! * [`TensorDataViewExt`] — builds the index views below (`try_index_view`,
//!   `try_index_mut_view`, and their `expect_` forms).
//! * [`TensorDataToVecAsExt`] — dtype conversion: `try_cast`, `try_convert`,
//!   `to_vec_as`, `into_vec_as`.
//! * [`TensorDataCheckExt`] — `TensorData`'s equality assertions as checks that
//!   return a `BunsenResult` instead of panicking.
//!
//! ## `TensorData` Views
//!
//! [`TensorDataView`] and [`TensorDataViewMut`] wrap a
//! [`burn::tensor::TensorData`] to provide multi-dimensional element access
//! via `view[&[i, j]]` indexing.
//!
//! ## Free Functions
//!
//! [`backend_float_dtype`] — the backend's default float `DType`
//! (`B::FloatElem`'s): what an undtyped `Tensor<B, D>` gets, and so the
//! dtype a module's interface speaks when its parameters were loaded at
//! some other precision.
//!
//! ## Dynamic Tensors
//!
//! [`dynamic`] wraps a `Tensor<B, R, K>` of any rank and kind as a
//! [`DynTensor`](dynamic::DynTensor), which can be sliced, cast, and moved
//! between devices without naming its rank or kind, then unwrapped back to a
//! typed tensor.
//! [`DynTensorEnv`](dynamic::DynTensorEnv) is a shared map of named
//! `DynTensor` bindings.

pub mod dynamic;

mod data_view;
mod float_dtype;
mod tensor_data_checks;
mod tensor_op_ext;

pub use data_view::*;
pub use float_dtype::*;
pub use tensor_data_checks::*;
pub use tensor_op_ext::*;
