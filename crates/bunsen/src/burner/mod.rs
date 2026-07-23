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
//!   parameters, with per-group learning- rate selectors;
//! - **carry tensor metadata** in non-generic code paths &mdash;
//!   [`descriptors::TensorParamDesc`] captures the metadata of any
//!   `Param<Tensor<B, R, K>>` (its `ParamId`, `Shape`, rank, dtype, kind)
//!   without carrying the generics that the underlying tensor type does;
//! - **work with `burn::record`** outside what the derive macros provide for
//!   free.
//!
//! The reflection and group-optimizer pieces compose: the canonical
//! pattern is to walk a model with `XmlModuleTree`, slice it into
//! parameter groups with `XPath`, and hand the groups to a
//! `GroupOptimizerAdaptor`.
//!
//! ## Map of the module
//!
//! - [`descriptors`] &mdash; type-erased descriptors for tensors and
//!   parameters. The lingua franca used by the reflection and optimizer
//!   machinery to talk about parameters uniformly.
//! - [`module`] &mdash; module-side helpers: a type-mapper for `Module` field
//!   re-typing, and (under `features = ["reflection"]`) the XML/XPath
//!   reflection layer.
//! - [`optim`] &mdash; optimizer extensions (under `features = ["train"]`).
//!   Headlined by the `GroupOptimizerAdaptor{N}` family and the
//!   `OptimizerGroup` / `LrSelector` building blocks.
//! - [`record`] &mdash; helpers for working with `burn::record` types.
//! - [`tensor`] &mdash; tensor helpers that don't fit neatly in [`crate::ops`]:
//!   `Tensor` extension traits and `TensorData` index views.
//! - [`distribution`] &mdash; distribution-related utilities.
//!
//! ## Tensor Extensions
//!
//! [`tensor`] provides extension traits that add utility methods directly
//! onto `burn::Tensor` values &mdash; in scope after
//! `use bunsen::burner::tensor::*;`:
//!
//! - [`TensorOpExt`](tensor::TensorOpExt) &mdash; all tensor kinds: `swap`
//!   (exchange two tensors in place), `release` (move a tensor out of a field,
//!   leaving an empty tensor behind), and `select_dim` (select one index along
//!   a dimension and squeeze it, dropping the rank by one).
//! - [`TensorIntOpExt`](tensor::TensorIntOpExt) &mdash; `Int` tensors:
//!   `square`, and `bounded_elem` for elementwise `[start, end)` range checks
//!   producing `Bool` masks.
//! - [`TensorBoolOpExt`](tensor::TensorBoolOpExt) &mdash; `Bool` tensors:
//!   `count_dim` / `count_dims` to count `true` elements along one or more
//!   dimensions (negative indexing supported).
//!
//! [`tensor`] also carries the
//! [`TensorDataIndexView`](tensor::TensorDataIndexView)
//! / [`TensorDataIndexMutView`](tensor::TensorDataIndexMutView) wrappers,
//! which give `view[&[i, j]]` multi-dimensional element access to a raw
//! `TensorData`.

pub mod descriptors;
pub mod distribution;
pub mod module;
pub mod record;

#[cfg(feature = "train")]
pub mod optim;
pub mod tensor;
