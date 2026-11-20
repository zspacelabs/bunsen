//! Qwen3-TTS's code predictor: a
//! [`DepthDecoder`](crate::blocks::transformers::decoders::depth::DepthDecoder)
//! over a frame's codebooks after the first, and its config
//! ([`Qwen3TtsCodePredictorConfig`]). A model `config.json`'s
//! `talker_config.code_predictor_config` and the model file's
//! `talker.code_predictor.*` tensors are
//! [`interop::qwen_tts`](super::interop::qwen_tts)'s.

mod config;
pub use config::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
