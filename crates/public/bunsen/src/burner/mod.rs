//! # `burn`-Adjacent Infrastructure
//!
//! `bunsen::burner` is the infrastructure layer: the pieces that sit
//! *next to* `burn` itself, not on top of its tensor surface. Where
//! [`crate::blocks`] gives you new `Module`s and [`crate::kits`] gives
//! you whole models, `burner` gives you the tooling for working with
//! `burn` modules and optimizers at a level `burn`'s default
//! surface doesn't expose.
//!
//! Most code that uses `bunsen` won't import from `burner` at all. You
//! reach for it when you need to:
//!
//! - **introspect a model** generically &mdash; the reflection layer in
//!   [`module::reflection`] turns a `Module` into a queryable XML document with
//!   an `XPath` query API, for "select every rank-2 weight under the
//!   transformer blocks" problems;
//! - **compose optimizers** &mdash; [`optim::GroupOptimizerPlan`] builds a
//!   `burn` `ModuleOptimizer` that mounts multiple optimizers on a single
//!   module (e.g. Muon for matrix parameters, `AdamW` for the rest), each
//!   driving a disjoint group of parameters, and a `ModuleLrScheduler` with
//!   per-group learning-rate selectors;
//! - **build modules from configs** &mdash; [`module::ModuleInit`] is the
//!   Config → Module step that bunsen's configs implement, and
//!   [`module::ToStructureConfig`] lowers a policy config to its structure
//!   config. Both are in [`crate::prelude`];
//! - **carry tensor metadata** in non-generic code paths &mdash;
//!   [`descriptors::TensorParamDesc`] captures the metadata of any
//!   `Param<Tensor<R, K>>` (its `ParamId`, `Shape`, rank, dtype, kind) without
//!   carrying the generics that the underlying tensor type does.
//!
//! The reflection and group-optimizer pieces compose: the canonical
//! pattern is to walk a model with `XmlModuleTree`, slice it into
//! parameter groups with `XPath`, and hand the groups to a
//! `GroupOptimizerPlan`.
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
//!   Headlined by `GroupOptimizerPlan` and the `OptimizerGroup` / `LrSelector`
//!   building blocks.
//! - [`tensor`] &mdash; tensor helpers that don't fit neatly in [`crate::ops`]:
//!   `Tensor` and `TensorData` extension traits, and [`tensor::dynamic`]'s
//!   type- and rank-erased [`DynTensor`](tensor::dynamic::DynTensor) with its
//!   named-binding [`DynTensorEnv`](tensor::dynamic::DynTensorEnv).
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
//!   (exchange two tensors in place), `replace_with` (replace a tensor with a
//!   function of itself, which owns it meanwhile), and `copy_slice` (copy one
//!   slice of a tensor onto another).
//! - [`TensorOrderedOpExt`](tensor::TensorOrderedOpExt) &mdash; ordered (`Int`,
//!   `Float`) tensors: [`in_range`](tensor::TensorOrderedOpExt::in_range) and
//!   [`in_range_scalar`](tensor::TensorOrderedOpExt::in_range_scalar) for
//!   elementwise `[start, end)` range checks producing `Bool` masks.
//! - [`TensorBoolOpExt`](tensor::TensorBoolOpExt) &mdash; `Bool` tensors:
//!   `count_dim` / `count_dims` to count `true` elements along one or more
//!   dimensions (negative indexing supported).

pub mod descriptors;
pub mod distribution;
pub mod module;

#[cfg(feature = "train")]
pub mod optim;
pub mod tensor;
