//! Qwen3-TTS's 12 Hz codec decoder: [`Qwen3TtsCodecDecoder`], its policy
//! ([`Qwen3TtsCodecDecoderContractConfig`]) and the structure config it
//! lowers to ([`Qwen3TtsCodecDecoderStructureConfig`], a config per part).
//! The codec's `config.json` (its `decoder_config`) and the codec file's
//! `decoder.*` tensors are [`interop::qwen_tts`](super::interop::qwen_tts)'s.

mod model;
pub use model::*;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
