//! Store-boundary helpers.
//!
//! Utilities for what happens as parameters cross a module-store boundary —
//! the read and write paths, rather than the module's own structure or its
//! compute.
//!
//! `burn` attaches load and save transformations to a
//! [`Param`](burn::module::Param) itself, so they run whenever the parameter
//! is loaded from or saved to a store. [`LinearConfig`](burn::nn::LinearConfig)
//! uses this for [`LinearLayout::Col`](burn::nn::LinearLayout::Col): the store
//! holds the weight in `PyTorch`'s `[d_output, d_input]` orientation, and the
//! mappers transpose it on the way in and out.
//!
//! The helpers here attach mappers to a parameter that has already been built,
//! for the case where the enclosing module has no layout knob:
//!
//! - [`repair_pytorch_strided_weight`] attaches the repair for one 2-D weight
//!   that `burn-store`'s `PyTorch` reader mangles;
//! - [`FixPytorchLoadMappers`] walks a built module tree and attaches that
//!   repair to each weight that needs it. Apply it on the way to a
//!   `PytorchStore` load, and nowhere else.

mod fix_pytorch_load_mappers;
mod param_mappers;

#[doc(inline)]
pub use fix_pytorch_load_mappers::*;
#[doc(inline)]
pub use param_mappers::*;
