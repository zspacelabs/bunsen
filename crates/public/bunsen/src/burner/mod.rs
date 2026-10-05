//! # `burn`-Adjacent Infrastructure
//!
//! `bunsen::burner` is the infrastructure layer: the pieces that sit
//! *next to* `burn` itself, not on top of its tensor surface. Where
//! [`crate::blocks`] gives you new `Module`s and [`crate::kits`] gives
//! you whole models, `burner` gives you the tooling for working with
//! `burn` modules, optimizers, and records at a level `burn`'s default
//! surface doesn't expose.
//!
//! Most code that uses `bunsen` won't import from `burner` at all. You
//! reach for it when you need to:
//!
//! - **introspect a model** generically &mdash; the reflection layer in
//!   [`module::reflection`] turns a `Module` into a queryable XML document with
//!   an `XPath` query API, for "select every rank-2 weight under the
//!   transformer blocks" problems;
//! - **compose optimizers** &mdash; the `GroupOptimizerAdaptor{N}` family in
//!   [`optim`] mounts multiple optimizers on a single module (e.g. Muon for
//!   matrix parameters, `AdamW` for the rest), each driving a disjoint group of
//!   parameters, with per-group learning-rate selectors;
//! - **build modules from configs** &mdash; [`module::ModuleInit`] is the
//!   Config → Module step that bunsen's configs implement, and
//!   [`module::ToStructureConfig`] lowers a policy config to its structure
//!   config. Both are in [`crate::prelude`];
//! - **carry tensor metadata** in non-generic code paths &mdash;
//!   [`descriptors::TensorParamDesc`] captures the metadata of any
//!   `Param<Tensor<B, R, K>>` (its `ParamId`, `Shape`, rank, dtype, kind)
//!   without carrying the generics that the underlying tensor type does.
//!
//! The reflection and group-optimizer pieces compose: the canonical
//! pattern is to walk a model with `XmlModuleTree`, slice it into
//! parameter groups with `XPath`, and hand the groups to a
//! `GroupOptimizerAdaptor`.
//!
//! ## Map of the module
//!
//! - [`descriptors`] &mdash; serializable, generics-free descriptors of
//!   tensors, parameters, and comparison tolerances: the vocabulary that
//!   reflection, audit, and [`tensor::dynamic`] use to describe tensors.
//! - [`module`] &mdash; module-side helpers: [`ModuleInit`](module::ModuleInit)
//!   and [`ToStructureConfig`](module::ToStructureConfig) (the config
//!   lifecycle), [`HasDType`](module::HasDType),
//!   [`DTypeMapper`](module::DTypeMapper) (cast a module's float parameters to
//!   one `DType`), and (under `features = ["reflection"]`) the XML/XPath
//!   reflection layer.
//! - [`optim`] &mdash; optimizer extensions (under `features = ["train"]`).
//!   Headlined by the `GroupOptimizerAdaptor{N}` family and the
//!   `OptimizerGroup` / `LrSelector` building blocks.
//! - [`record`] &mdash; [`display_record`](record::display_record), a debug
//!   dump of a `burn::record` record's layout without its tensor data.
//! - [`store`] &mdash; helpers for what crosses a store boundary: load/save
//!   mappers that repair `PyTorch` weights
//!   ([`repair_pytorch_strided_weight`](store::repair_pytorch_strided_weight),
//!   [`FixPytorchLoadMappers`](store::FixPytorchLoadMappers)).
//! - [`tensor`] &mdash; tensor helpers that don't fit neatly in [`crate::ops`]:
//!   `Tensor` and `TensorData` extension traits, `TensorData` index views, and
//!   [`tensor::dynamic`]'s type- and rank-erased
//!   [`DynTensor`](tensor::dynamic::DynTensor) with its named-binding
//!   [`DynTensorEnv`](tensor::dynamic::DynTensorEnv).
//! - [`distribution`] &mdash;
//!   [`DistributionDisplayAdapter`](distribution::DistributionDisplayAdapter),
//!   which lets a `burn::tensor::Distribution` field appear in a module or
//!   config display (used by [`NoiseConfig`](crate::ops::noise::NoiseConfig)).
//!
//! ## Tensor Extensions
//!
//! [`tensor`] provides extension traits that add utility methods directly
//! onto `burn::Tensor` values &mdash; in scope after
//! `use bunsen::burner::tensor::*;` or `use bunsen::prelude::*;`:
//!
//! - [`TensorOpExt`](tensor::TensorOpExt) &mdash; all tensor kinds: `swap`
//!   (exchange two tensors in place), `extract` (move a tensor out of a field,
//!   leaving an empty tensor behind), `replace_with` (replace a tensor with a
//!   function of itself, which owns it meanwhile), `select_dim` (select one
//!   index along a dimension and squeeze it, dropping the rank by one), and
//!   `copy_slice` (copy one slice of a tensor onto another).
//! - [`TensorElemOpExt`](tensor::TensorElemOpExt) &mdash; `to_data_as` /
//!   `into_data_as` and `to_data_cast` / `into_data_cast`: read a tensor's data
//!   out converted to an element type or a `DType`.
//! - [`TensorOrderedOpExt`](tensor::TensorOrderedOpExt) &mdash; ordered (`Int`,
//!   `Float`) tensors: [`in_range`](tensor::TensorOrderedOpExt::in_range) and
//!   [`in_range_scalar`](tensor::TensorOrderedOpExt::in_range_scalar) for
//!   elementwise `[start, end)` range checks producing `Bool` masks.
//! - [`TensorIntOpExt`](tensor::TensorIntOpExt) &mdash; `Int` tensors:
//!   `square`.
//! - [`TensorBoolOpExt`](tensor::TensorBoolOpExt) &mdash; `Bool` tensors:
//!   `count_dim` / `count_dims` to count `true` elements along one or more
//!   dimensions (negative indexing supported).
//!
//! [`tensor`] also carries the [`TensorDataView`](tensor::TensorDataView) /
//! [`TensorDataViewMut`](tensor::TensorDataViewMut) wrappers, built by
//! [`TensorDataViewExt`](tensor::TensorDataViewExt), which give
//! `view[&[i, j]]` multi-dimensional element access to a raw `TensorData`.

pub mod descriptors;
pub mod distribution;
pub mod module;
pub mod record;
pub mod store;

#[cfg(feature = "train")]
pub mod optim;
pub mod tensor;
