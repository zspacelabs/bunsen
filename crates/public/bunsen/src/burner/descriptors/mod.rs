//! # Tensor and Parameter Descriptors
//!
//! Serializable, reference-free descriptions of `burn` tensors, parameters,
//! and comparison tolerances. A descriptor is a plain value: it carries a
//! tensor's metadata without its data, its backend, or the `Tensor<R, K>`
//! generics, so code that is not generic over a tensor type can still name,
//! store, and compare tensors.
//!
//! ## How they relate
//!
//! - [`TensorKindDesc`] names a tensor kind (`Bool`, `Float`, `Int`) as a
//!   value; [`TensorKindDesc::for_kind`] maps `burn`'s kind marker types to it.
//! - [`TensorRankType`] is the slot type of a tensor: kind, dtype, and rank,
//!   without a shape.
//! - [`TensorDesc`] is kind, dtype, and the full
//!   [`Shape`](burn::prelude::Shape), which it derefs to;
//!   [`to_rank_type`](TensorDesc::to_rank_type) drops the shape again.
//! - [`ParamDesc<T>`](ParamDesc) pairs a descriptor with the
//!   [`ParamId`](burn::module::ParamId) of the parameter it describes, and
//!   derefs to the descriptor. [`TensorParamDesc`] is `ParamDesc<TensorDesc>`,
//!   built `From` a `&Param<Tensor<R, K>>`.
//! - [`ToleranceDesc`] describes an approximate comparison: a
//!   [`TolerancePolicy`] (which [`Tolerance`](burn::tensor::Tolerance) rule)
//!   plus the float type the rule resolves at.
//! - [`dtype_from_str`] parses a `DType` from its `Display` form, and [`shims`]
//!   holds serde adapters for `burn` types that do not derive serde.
//!
//! ## Who uses them
//!
//! - Reflection:
//!   [`XmlModuleTree`](crate::burner::module::reflection::XmlModuleTree) stores
//!   a `TensorParamDesc` on each parameter's XML node, and its queries return
//!   them. Their `ParamId`s are what an
//!   [`OptimizerGroup`](crate::burner::optim::OptimizerGroup) is built from.
//! - Audit: an approximate-equality checkpoint takes a `TolerancePolicy`, and
//!   the recorded event stores it with its float type as a `ToleranceDesc`, so
//!   a stored baseline is verified under the tolerance it was recorded with.
//! - [`DynTensor`](crate::burner::tensor::dynamic::DynTensor) reports its
//!   `TensorKindDesc`, and converts to a `TensorDesc` or a `TensorRankType`.

pub mod shims;

mod param_desc;
mod parse_dtype;
mod tensor_desc;
mod tensor_kinds;
mod tolerance_desc;

pub use param_desc::*;
pub use parse_dtype::*;
pub use tensor_desc::*;
pub use tensor_kinds::*;
pub use tolerance_desc::*;
