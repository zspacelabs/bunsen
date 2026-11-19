//! # Qwen3-TTS
//!
//! Alibaba's Qwen3-TTS text-to-speech models (the 12 Hz line): a talker, a
//! language model over the first of sixteen audio codebooks, with a code
//! predictor for the other fifteen, and a codec that decodes the codes to
//! 24 kHz audio.
//!
//! - [`codec`]: the 12 Hz codec's decoder,
//!   [`Qwen3TtsCodecDecoder`](codec::Qwen3TtsCodecDecoder), codes at 12.5 Hz to
//!   24 kHz audio: a split residual vector quantizer, a causal convolution, a
//!   transformer of windowed causal attention with layer scales, two x2
//!   upsamplers (a transposed convolution and a `ConvNeXt` block each) and a
//!   DAC vocoder of `SnakeBeta` activations; decoded in chunks as upstream
//!   decodes a take.
//!
//! - [`talker`]: the talker, [`Qwen3TtsTalker`](talker::Qwen3TtsTalker): a
//!   Qwen3 decoder over summed embeddings (the codec table's rows, and the text
//!   table's through a projection), its head the next frame's first codebook;
//!   decoded through a KV cache, a row a step.
//! - [`code_predictor`]: the code predictor, a
//!   [`DepthDecoder`](crate::blocks::transformers::decoders::depth::DepthDecoder)
//!   over a frame's fifteen codebooks after the first, from the talker's hidden
//!   state and the first codebook's embedding
//!   ([`qwen3_tts_code_predictor_config`](code_predictor::qwen3_tts_code_predictor_config)).
//! - [`prompt`]: the rows the talker reads before it generates
//!   ([`Qwen3TtsPromptRows::build`](prompt::Qwen3TtsPromptRows::build)): an
//!   instruction, the role, the codec prefix (think tags around a language, a
//!   speaker), the text upfront or interleaved, and the text each generated
//!   frame adds; the languages and speakers by the model's keys
//!   ([`Qwen3TtsPromptConfig`](prompt::Qwen3TtsPromptConfig)).
//! - [`Qwen3TtsModel`]: the talker and its code predictor, and a take
//!   ([`generate_codes`](Qwen3TtsModel::generate_codes)): per frame, codebook 0
//!   from the talker, the rest from the code predictor, the frame's embeddings
//!   and its trailing text the talker's next row, until EOS; each choice by a
//!   [`Qwen3TtsSampler`](sampler::Qwen3TtsSampler) as upstream's `generate`
//!   makes it ([`Qwen3TtsTakeConfig`], from `generation_config.json`).
//!
//! The codec's weights are `Qwen/Qwen3-TTS-Tokenizer-12Hz`'s
//! `model.safetensors`, whose `decoder.*` tensors the decoder reads as they
//! are ([`codec::weights`]). A model's `model.safetensors` holds the talker
//! under `talker.` ([`talker::weights`]), its code predictor under
//! `talker.code_predictor.` ([`code_predictor::weights`]);
//! [`weights::load_qwen3_tts_model_tensors`] loads both.

pub mod code_predictor;
pub mod codec;
pub mod prompt;
pub mod sampler;
pub mod talker;

mod model;
pub use model::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store"))]
mod toy_tests;
