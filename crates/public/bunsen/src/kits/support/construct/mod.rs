//! # Construction: what a language model's build loads to
//!
//! The kits' shared construct helpers, over the
//! [`builds`](crate::data::builds) pathway (which resolves a name to a build
//! and brings its files local, for the kit's hook to build from):
//!
//! - [`LmConstruct`]: what a language model's pretrained loads to, its model
//!   and its tokenizer and generation files, with the [`LmPolicy`] its presets
//!   resolved to.
//! - With the `store` feature, the `transformers` language-model readers the
//!   kits' hooks share.

mod lm_construct;
#[cfg(feature = "store")]
mod transformers_lm;

pub use lm_construct::*;
#[cfg(feature = "store")]
pub use transformers_lm::*;
