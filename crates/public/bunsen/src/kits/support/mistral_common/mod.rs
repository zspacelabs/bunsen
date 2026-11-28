//! # `mistral_common`: Mistral's shared files, for any kit
//!
//! - [`interop`]: `tekken.json`, its vocabulary and an audio model's stream
//!   geometry, as the `mistral_common` package reads them; the tokenizer it
//!   builds is [`TekkenVocabulary`](crate::kits::support::tokens::TekkenVocabulary).

pub mod interop;
