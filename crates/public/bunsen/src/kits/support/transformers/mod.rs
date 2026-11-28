//! # `transformers`: Hugging Face `transformers`, for any kit
//!
//! What bunsen shares about the `transformers` ecosystem, by use:
//!
//! - [`interop`]: what its files mean: a `config.json` read field by field, the
//!   `generation_config.json` and `tokenizer_config.json` the catalog's probes
//!   read, and the Llama line's layer keys.

pub mod interop;
