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
//!
//! The codec's weights are `Qwen/Qwen3-TTS-Tokenizer-12Hz`'s
//! `model.safetensors`, whose `decoder.*` tensors the decoder reads as they
//! are ([`codec::weights`]). A model's `model.safetensors` holds the talker
//! under `talker.` ([`talker::weights`]).

pub mod codec;
pub mod talker;
