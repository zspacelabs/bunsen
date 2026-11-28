//! # The shared pretrained tier: what the kits' `pretrained` modules share
//!
//! A kit's `pretrained` holds its catalog facts and builds its construct,
//! through its `interop`'s loaders; what more than one kit's does alike is
//! here.
//!
//! - [`catalog_contracts`]: each kit's build-catalog contract, the signature
//!   its loader implements, gathered for the catalog builder.
//! - [`Probe`]s: what a pretrained's files say of its presets' values
//!   (`generation_config`, `tokenizer_config`, `chat_template`:
//!   [`builtin_probes`]), each reading its file through
//!   [`transformers::interop`](super::transformers::interop)'s mirrors. What
//!   they find accumulates ([`accumulate_probed`]). The catalog builder (`cargo
//!   catalog`) runs them when it builds the built-in catalog, and
//!   [`probe_loaded`] runs them over a checkpoint no catalog build describes.
//! - [`TransformersLmKit`]: the construct hook of a language model whose
//!   pretrained is a `transformers` checkpoint (Qwen3, Qwen3.5, Gemma): the
//!   resource-set check, the kit's interop rungs, the probed presets, and the
//!   [`LmConstruct`](super::construct::LmConstruct) they build.

pub mod catalog_contracts;
mod kinds;
mod probes;
/// Audits a kit's factory and its rows: the shapes every kit's provider
/// tests share.
///
/// This module exists with the `testing` feature, and in bunsen's own unit
/// tests.
#[cfg(all(any(test, feature = "testing"), feature = "store"))]
pub mod testing;
mod transformers_lm;

pub use kinds::*;
pub use probes::*;
pub use transformers_lm::*;
