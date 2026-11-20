//! # Construction: what a language model's build loads to
//!
//! - [`LmConstruct`]: what a language model's pretrained loads to, its model
//!   and its tokenizer and generation files, with the [`LmPolicy`] its presets
//!   resolved to. The `transformers` language models' shared hook is
//!   [`pretrained::TransformersLmKit`](super::pretrained::TransformersLmKit).

mod lm_construct;

pub use lm_construct::*;
