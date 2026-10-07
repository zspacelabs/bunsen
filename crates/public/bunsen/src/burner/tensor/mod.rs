//! # Tensor Operations
//!
//! Extension traits that add utility methods to [`burn::Tensor`] and
//! [`burn::tensor::TensorData`], and
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
//! * [`replace_with`](TensorOpExt::replace_with) — replace the value with
//!   `f(value)`; `f` owns the tensor while it runs, so operations inside it may
//!   run in place.
//! * [`copy_slice`](TensorOpExt::copy_slice) — copy one slice of the tensor
//!   onto another slice of it.
//!
//! [`TensorOrderedOpExt`] — ordered type (`Int`, `Float`) tensors:
//! * [`in_range`](TensorOrderedOpExt::in_range) — elementwise `[start, end)`
//!   test against `start` and `end` tensors, producing a `Bool` mask.
//! * [`in_range_scalar`](TensorOrderedOpExt::in_range_scalar) — elementwise
//!   test against a scalar `Range<E>`, producing a `Bool` mask.
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
//! fn row_counts(occupied: Tensor<2, Bool>) -> Tensor<2, Int> {
//!     // Count the `true` cells in each row.
//!     occupied.count_dim(-1)
//! }
//! ```
//!
//! ## `TensorData` Extension Traits
//!
//! * [`TensorDataCheckExt`] — `TensorData`'s equality assertions as checks that
//!   return a `BunsenResult` instead of panicking.
//!
//! ## Free Functions
//!
//! [`device_float_dtype`] — a device's default float `DType`: what an undtyped
//! float tensor built on it gets, and so the
//! dtype a module's interface speaks when its parameters were loaded at
//! some other precision.
//!
//! ## Dynamic Tensors
//!
//! [`dynamic`] wraps a `Tensor<R, K>` of any rank and kind as a
//! [`DynTensor`](dynamic::DynTensor), which can be sliced, cast, and moved
//! between devices without naming its rank or kind, then unwrapped back to a
//! typed tensor.
//! [`DynTensorEnv`](dynamic::DynTensorEnv) is a shared map of named
//! `DynTensor` bindings.

pub mod dynamic;

mod float_dtype;
mod tensor_data_checks;
mod tensor_op_ext;

pub use float_dtype::*;
pub use tensor_data_checks::*;
pub use tensor_op_ext::*;
