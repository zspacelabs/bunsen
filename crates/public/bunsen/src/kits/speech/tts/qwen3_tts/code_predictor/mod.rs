//! Qwen3-TTS's code predictor: a
//! [`DepthDecoder`](crate::blocks::transformers::decoders::depth::DepthDecoder)
//! over a frame's codebooks after the first, its config from a model
//! `config.json`'s `talker_config.code_predictor_config`
//! ([`qwen3_tts_code_predictor_config`]), and the loader of the model file's
//! `talker.code_predictor.*` tensors ([`weights`]).

mod config;
pub use config::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
