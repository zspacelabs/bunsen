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
//!
//! The codec's weights are `Qwen/Qwen3-TTS-Tokenizer-12Hz`'s
//! `model.safetensors`, whose `decoder.*` tensors the decoder reads as they
//! are ([`codec::weights`]). A model's `model.safetensors` holds the talker
//! under `talker.` ([`talker::weights`]), its code predictor under
//! `talker.code_predictor.` ([`code_predictor::weights`]).

pub mod code_predictor;
pub mod codec;
pub mod talker;
