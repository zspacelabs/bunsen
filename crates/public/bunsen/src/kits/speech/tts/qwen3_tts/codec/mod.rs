//! Qwen3-TTS's 12 Hz codec decoder: [`Qwen3TtsCodecDecoder`], its policy
//! ([`Qwen3TtsCodecDecoderContractConfig`], from `config.json`'s
//! `decoder_config`, [`Qwen3TtsCodecDecoderJson`]) and the structure config it
//! lowers to ([`Qwen3TtsCodecDecoderStructureConfig`], a config per part), and
//! the loader of the codec file's `decoder.*` tensors ([`weights`]).

mod config;
pub use config::*;

mod model;
pub use model::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
