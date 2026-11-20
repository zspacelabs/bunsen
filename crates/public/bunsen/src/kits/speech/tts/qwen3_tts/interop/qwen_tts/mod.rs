//! # `qwen_tts`: Qwen3-TTS's own files, as its reference code reads them
//!
//! Qwen3-TTS ships through its original repository's code, `qwen_tts`: a
//! model's `config.json` and `generation_config.json`, its
//! `model.safetensors` (the talker and code predictor), and the 12 Hz
//! codec's `config.json` and `model.safetensors`.
//!
//! - **Mirrors:** [`Qwen3TtsPromptMirror`] (the prompt's ids, languages and
//!   speakers), [`Qwen3TtsModelTypeMirror`] (the kind and size),
//!   [`Qwen3TtsGenerationMirror`] (the release's sampling),
//!   [`Qwen3TtsCodecMirror`] and [`Qwen3TtsCodecDecoderMirror`] (the codec). A
//!   mirror's conversions are its methods
//!   ([`prompt_config`](Qwen3TtsPromptMirror::prompt_config),
//!   [`take_config`](Qwen3TtsGenerationMirror::take_config),
//!   [`codec_decoder_config`](Qwen3TtsCodecDecoderMirror::codec_decoder_config)).
//! - **The geometry:** [`model_config`] of a `config.json`, from its
//!   [`talker_config`] and [`code_predictor_config`] sections, and
//!   [`model_config_from_str`] / [`model_config_from_path`].
//! - **Names:** [`talker_rekey`] (the talker is a Qwen3 under a TTS prefix, its
//!   layers Qwen3's
//!   [`text_layer_rekey`](crate::kits::language::qwen3::interop::transformers::text_layer_rekey)),
//!   [`code_predictor_rekey`], [`codec_decoder_rekey`].
//! - **Loading** (feature `store`): `load_talker_tensors`,
//!   `load_code_predictor_tensors`, `load_codec_decoder_tensors`,
//!   `load_model_tensors`; with `cache`, the parts of a resource set,
//!   `load_model` and `load_codec_decoder`.

mod code_predictor;
mod codec;
mod model;
mod talker;
#[cfg(feature = "store")]
mod weights;

pub use code_predictor::*;
pub use codec::*;
pub use model::*;
pub use talker::*;
#[cfg(feature = "store")]
pub use weights::*;
