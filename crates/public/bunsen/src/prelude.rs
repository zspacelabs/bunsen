//! # Bunsen Prelude
//!
//! The traits and types most code that uses bunsen needs in scope, for one
//! glob import:
//!
//! ```
//! use bunsen::prelude::*;
//! ```
//!
//! Glob it in code that builds bunsen modules from configs, calls the tensor
//! extension methods, or checks shape contracts. Most of what it re-exports is
//! traits, which have to be in scope for their methods to resolve. It
//! re-exports:
//!
//! - [`ModuleInit`] and [`ToStructureConfig`], so `config.init(&device)` and
//!   `policy.to_structure()` resolve on bunsen configs;
//! - everything in [`crate::burner::tensor`]: the `Tensor` and `TensorData`
//!   extension traits (`TensorOpExt`, `TensorElemOpExt`, `TensorOrderedOpExt`,
//!   `TensorIntOpExt`, `TensorBoolOpExt`, `TensorDataViewExt`,
//!   `TensorDataToVecAsExt`, `TensorDataCheckExt`), the `TensorDataView` and
//!   `TensorDataViewMut` views, `backend_float_dtype`, and the `dynamic`
//!   submodule;
//! - everything in [`crate::contracts`]: `ShapeContract` and the shape contract
//!   macros, such as `shape_contract!` and `assert_shape_contract!`;
//! - everything in [`crate::errors`]: `BunsenError`, `BunsenResult`,
//!   `SlicingError`, and `WithOkOrPanic`.
//!
//! It re-exports no `Module` types, configs, or backends; import those from
//! where they live. No name in it is also in `burn::prelude`, so the two globs
//! can sit side by side.

#[doc(no_inline)]
pub use crate::burner::module::{
    ModuleInit,
    ToStructureConfig,
};
#[doc(inline)]
pub use crate::burner::tensor::*;
#[doc(inline)]
pub use crate::contracts::*;
#[doc(inline)]
pub use crate::errors::*;
