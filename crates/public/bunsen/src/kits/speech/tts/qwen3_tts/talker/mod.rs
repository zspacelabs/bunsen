//! Qwen3-TTS's talker: [`Qwen3TtsTalker`] and its config
//! ([`Qwen3TtsTalkerConfig`]). A model `config.json`'s `talker_config` and
//! the model file's `talker.*` tensors are
//! [`interop::qwen_tts`](super::interop::qwen_tts)'s.

mod model;
pub use model::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
