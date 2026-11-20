//! # The shared pretrained tier: what the kits' `pretrained` modules share
//!
//! A kit's `pretrained` holds its catalog facts and builds its construct,
//! through its `interop`'s loaders; what more than one kit's does alike is
//! here.
//!
//! - [`catalog_contracts`]: each kit's build-catalog contract, the signature
//!   its loader implements, gathered for the catalog builder.
//! - [`TransformersLmKit`]: the construct hook of a language model whose
//!   pretrained is a `transformers` checkpoint (Qwen3, Qwen3.5, Gemma): the
//!   resource-set check, the kit's interop rungs, the probed presets, and the
//!   [`LmConstruct`](super::construct::LmConstruct) they build.

pub mod catalog_contracts;
mod transformers_lm;

pub use transformers_lm::*;
