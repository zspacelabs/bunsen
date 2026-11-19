//! # Pretrained Qwen3-TTS models
//!
//! [`default_qwen3_tts_factory`] is the index: Qwen's 12 Hz models,
//! `catalog:qwen/qwen3-tts/qwen3-tts-12hz-0.6b-customvoice` (the default),
//! `…-1.7b-customvoice` and `…-1.7b-voicedesign` (Apache-2.0 weights), every
//! file pinned to its digest at a fixed revision: `model.safetensors` (the
//! talker and code predictor in bf16), `config.json`, `generation_config.json`,
//! the tokenizer's `vocab.json`, `merges.txt` and `tokenizer_config.json`, and
//! the codec's `speech_tokenizer/model.safetensors` (f32) and its
//! `config.json`.
//!
//! [`Qwen3TtsConstructConfig`] reads them into a
//! [`Qwen3TtsConstruct`](super::Qwen3TtsConstruct). The kit has no prefab map:
//! the geometry is the `config.json`s'.
//!
//! The whole module needs the `store` and `cache` features, both default.

mod construct;
mod providers;

pub use construct::*;
pub use providers::*;
