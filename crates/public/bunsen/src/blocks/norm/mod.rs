//! # The norm seam
//!
//! [`BunsenNormalizationConfig`] and [`BunsenNormalization`] are bunsen's
//! normalization layers: the kinds bunsen's blocks use
//! ([`Identity`](BunsenNormalization::Identity),
//! [`Batch`](BunsenNormalization::Batch),
//! [`Layer`](BunsenNormalization::Layer), [`Rms`](BunsenNormalization::Rms)),
//! each burn's layer, behind one seam that bunsen owns. A block that takes a
//! norm takes a [`BunsenNormalizationConfig`], so a norm's dtype handling, and
//! later kinds, are bunsen's to change without waiting on burn's releases.
//!
//! - **Half inputs compute in f32.** [`Rms`](BunsenNormalization::Rms)
//!   normalizes an f16 or bf16 input in f32 and narrows it before the gain, as
//!   `transformers` does; burn's `RmsNorm` divides in the input's dtype.
//!   [`Layer`](BunsenNormalization::Layer) normalizes a half input in f32 and
//!   narrows the result; burn's `LayerNorm` widens f16 alone, as bf16 has f32's
//!   range.
//! - **The parameters meet the input in one dtype.** Each casts its parameters
//!   to the dtype it computes in, so a gain kept in f32 never meets a half
//!   input in one op (burn-flex refuses that; on the cubecl backends it is
//!   wrong: [`burner::dtypes`](crate::burner::dtypes)).
//! - **The output is the input's dtype**
//!   ([`ResultDType`](crate::burner::module::ResultDType)).
//!
//! Beside the seam, [`GatedRmsNorm`] is a root-mean-square norm gated by a
//! second input, `w · x̂ · silu(z)` (the gated `DeltaNet`'s output norm): it
//! takes two inputs, so it is a block of its own, not a seam kind.
//!
//! The variants and their configs keep burn's `NormalizationConfig` names,
//! so a config's serde form and a native record's keys are burn's. bunsen's
//! loaders skip enum variants, so a loaded tensor's path names no variant
//! either way.

mod gated_rms_norm;
mod layer_norm;
mod normalization;

pub use gated_rms_norm::*;
pub use layer_norm::*;
pub use normalization::*;
