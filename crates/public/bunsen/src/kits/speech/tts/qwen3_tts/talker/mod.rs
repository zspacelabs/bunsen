//! Qwen3-TTS's talker: [`Qwen3TtsTalker`], its config
//! ([`Qwen3TtsTalkerConfig`], from a model `config.json`'s `talker_config`),
//! and the loader of the model file's `talker.*` tensors ([`weights`]).

mod config;
pub use config::*;

mod model;
pub use model::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
