//! Qwen3-TTS's 12 Hz codec decoder: [`Qwen3TtsCodecDecoder`], its config
//! ([`Qwen3TtsCodecDecoderConfig`], from `config.json`'s `decoder_config`,
//! [`Qwen3TtsCodecDecoderJson`]), and the loader of the codec file's
//! `decoder.*` tensors ([`weights`]).

mod config;
pub use config::*;

mod model;
pub use model::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
