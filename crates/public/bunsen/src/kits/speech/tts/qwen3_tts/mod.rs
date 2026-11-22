//! # Qwen3-TTS
//!
//! Alibaba's Qwen3-TTS text-to-speech models (the 12 Hz line): a talker, a
//! language model over the first of sixteen audio codebooks, with a code
//! predictor for the other fifteen, and a codec that decodes the codes to
//! 24 kHz audio.
//!
//! - [`blocks`]: the Modules it is built from, each with its config.
//!   - The 12 Hz codec's decoder,
//!     [`Qwen3TtsCodecDecoder`](blocks::Qwen3TtsCodecDecoder), codes at 12.5 Hz
//!     to 24 kHz audio: a split residual vector quantizer, a causal
//!     convolution, a transformer of windowed causal attention with layer
//!     scales, two x2 upsamplers (a transposed convolution and a `ConvNeXt`
//!     block each) and a DAC vocoder of `SnakeBeta` activations; decoded in
//!     chunks as upstream decodes a take.
//!   - The talker, [`Qwen3TtsTalker`](blocks::Qwen3TtsTalker): a Qwen3 decoder
//!     over summed embeddings (the codec table's rows, and the text table's
//!     through a projection), its head the next frame's first codebook; decoded
//!     through a KV cache, a row a step.
//!   - The code predictor, a
//!     [`DepthDecoder`](crate::blocks::transformers::decoders::depth::DepthDecoder)
//!     over a frame's fifteen codebooks after the first, from the talker's
//!     hidden state and the first codebook's embedding
//!     ([`Qwen3TtsCodePredictorConfig`](blocks::Qwen3TtsCodePredictorConfig)).
//!   - The model, [`Qwen3TtsModel`](blocks::Qwen3TtsModel): the talker and its
//!     code predictor, and a take
//!     ([`generate_codes`](blocks::Qwen3TtsModel::generate_codes)): per frame,
//!     codebook 0 from the talker, the rest from the code predictor, the
//!     frame's embeddings and its trailing text the talker's next row, until
//!     EOS; each choice by a [`Qwen3TtsSampler`](sampler::Qwen3TtsSampler) as
//!     upstream's `generate` makes it
//!     ([`Qwen3TtsTakeConfig`](blocks::Qwen3TtsTakeConfig), from
//!     `generation_config.json`).
//! - [`prompt`]: the rows the talker reads before it generates
//!   ([`Qwen3TtsPromptRows::build`](prompt::Qwen3TtsPromptRows::build)): an
//!   instruction, the role, the codec prefix (think tags around a language, a
//!   speaker), the text upfront or interleaved, and the text each generated
//!   frame adds; the languages and speakers by the model's keys
//!   ([`Qwen3TtsPromptConfig`](prompt::Qwen3TtsPromptConfig)).
//!
//! - [`driver`]: Qwen3-TTS as a TTS engine,
//!   [`Qwen3TtsEngine`](driver::Qwen3TtsEngine);
//!   [`Qwen3TtsDriver`](driver::Qwen3TtsDriver) is the [TTS
//!   driver](super::driver) over it.
//! - [`Qwen3TtsConstruct`]: a loaded model, its talker and code predictor and
//!   codec with their configs, the prompt's ids, the release's take settings
//!   and the tokenizer's files.
//! - [`pretrained`] (features `store` and `cache`): the named checkpoints,
//!   `qwen/qwen3-tts/qwen3-tts-12hz-{0.6b,1.7b}-customvoice` and
//!   `…-1.7b-voicedesign`, and the factory that loads them into a construct.
//!
//! - [`interop`]: its checkpoints' ecosystems. [`interop::qwen_tts`] reads a
//!   model's `config.json` and `generation_config.json`, its
//!   `model.safetensors` (the talker under `talker.`, its code predictor under
//!   `talker.code_predictor.`), and the 12 Hz codec's `config.json` and
//!   `model.safetensors` (its `decoder.*` tensors).

pub mod blocks;
pub mod prompt;
pub mod sampler;

pub mod interop;

mod construct;
pub use construct::*;

pub mod driver;

#[cfg(all(feature = "store", feature = "cache"))]
pub mod pretrained;

#[cfg(all(test, feature = "store"))]
mod code_predictor_toy_tests;
#[cfg(all(test, feature = "store"))]
mod codec_toy_tests;
#[cfg(all(test, feature = "store"))]
mod talker_toy_tests;
#[cfg(all(test, feature = "store"))]
mod toy_tests;
