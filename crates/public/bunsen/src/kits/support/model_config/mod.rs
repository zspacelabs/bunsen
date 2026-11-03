//! # Model config: what a model is configured from
//!
//! The config side of a kit, apart from building: reading the configs a
//! checkpoint ships with into a kit's own. Building a model from its config
//! is [`construct`](super::construct)'s (feature `cache`).
//!
//! - [`HfConfigJson`]: a `transformers` `config.json`, or one section of it,
//!   read field by field, each error naming the field's dotted path. The kits'
//!   `hf:` readers map it onto their configs.

mod hf_config;

pub use hf_config::*;
