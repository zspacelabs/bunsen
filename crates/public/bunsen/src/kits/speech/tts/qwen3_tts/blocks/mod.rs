//! # Qwen3-TTS's blocks
//!
//! The Modules Qwen3-TTS is built from, each with its config (a release's,
//! from its `config.json`, and its tensors, through
//! [`interop::qwen_tts`](super::interop::qwen_tts)).
//!
//! - **Model.** [`Qwen3TtsModel`]: the talker and its code predictor, and the
//!   take loop ([`Qwen3TtsModelConfig`], [`Qwen3TtsTakeConfig`]).
//! - **Talker.** [`Qwen3TtsTalker`] and [`Qwen3TtsTalkerConfig`]: a model
//!   `config.json`'s `talker_config`, and the model file's `talker.*` tensors.
//! - **Code predictor.** [`Qwen3TtsCodePredictorConfig`]: a
//!   [`DepthDecoder`](crate::blocks::transformers::decoders::depth::DepthDecoder)
//!   over a frame's codebooks after the first (`talker_config`'s
//!   `code_predictor_config`, the `talker.code_predictor.*` tensors).
//! - **Codec decoder.** [`Qwen3TtsCodecDecoder`], its policy
//!   ([`Qwen3TtsCodecDecoderContractConfig`]) and the structure config it
//!   lowers to ([`Qwen3TtsCodecDecoderStructureConfig`], a config per part): a
//!   transformer ([`Qwen3TtsCodecTransformer`]), upsampling stages
//!   ([`Qwen3TtsCodecUpsample`]) and a vocoder ([`Qwen3TtsVocoder`], its blocks
//!   and residual units). The codec's `config.json` (its `decoder_config`) and
//!   the codec file's `decoder.*` tensors.

mod code_predictor;
mod codec;
mod model;
mod talker;

pub use code_predictor::*;
pub use codec::*;
pub use model::*;
pub use talker::*;
