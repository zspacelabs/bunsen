//! # ZONOS3-Mini
//!
//! Zyphra's ZONOS3-Mini text-to-speech model: a language model over rows
//! of ten audio codebooks and a text token, and a codec that decodes the
//! codes to 44.1 kHz audio.
//!
//! - [`blocks`]: the Modules it is built from, each with its config.
//!   - The language model, [`Zonos3Lm`](blocks::Zonos3Lm). Each row's ids are
//!     summed through their own tables (a speaker row and the condition rows
//!     replaced), run through a stack of gated `DeltaNet` and global attention
//!     layers without a residual stream (a
//!     [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
//!     with [`AttnResRoutes`](crate::blocks::transformers::decoders::attn_res::AttnResRoutes)),
//!     their feed-forwards a dense `SwiGLU` and then latent mixtures of
//!     experts, and read out as every codebook's logits. Decoding is cached,
//!     one row a step, through
//!     [`decode_cache`](blocks::Zonos3Lm::decode_cache).
//!   - The codec decoder, [`Zonos3Codec`](blocks::Zonos3Codec), codes at 15 Hz
//!     to 44.1 kHz audio: the codebooks' embeddings summed into a latent (the
//!     first through a causal `ConvNeXt` head), a trunk of windowed causal
//!     attention stages, each patch-up between them doubling the frame rate,
//!     and a head of inverse STFTs at two FFT sizes, mixed and resampled.
//!   - The speaker encoder,
//!     [`Zonos3SpeakerEncoder`](blocks::Zonos3SpeakerEncoder), a clip to a
//!     speaker embedding: a Kaldi filterbank, `AntSpeaker`'s MECT-B2 stages
//!     (residual convolutions, then a transformer layer with a token soft
//!     mixture of experts over the flattened channels), a mix of the stages,
//!     attentive statistics pooling and a projection.
//! - [`prompt`]: the rows the language model reads before it generates
//!   ([`Zonos3PromptConfig::build`](prompt::Zonos3PromptConfig::build)): a
//!   speaker (an embedding or reference codes), the condition rows, the text
//!   and the silence prefill, and the delay pattern ([`shear`](prompt::shear),
//!   [`unshear`](prompt::unshear)) the audio rows follow. Its text is
//!   normalized by the caller; [`zonos3_text_ids`](prompt::zonos3_text_ids)
//!   tokenizes it.
//! - [`sampler`]: each step's codes from its logits,
//!   [`Zonos3Sampler`](sampler::Zonos3Sampler), as the release samples them
//!   (its settings, [`Zonos3SamplerConfig`](sampler::Zonos3SamplerConfig)).
//! - [`voices`]: the stock voices, each reference codes and a speaker
//!   embedding.
//! - The language model as a
//!   [`TimedDecode`](crate::kits::support::timing::TimedDecode), for a
//!   throughput harness: the prompt's rows prefilled, then a codec frame per
//!   step.
//! - [`Zonos3Construct`]: a loaded model, its language model and codec with
//!   their configs, the prompt layout, the release's engine settings and
//!   languages, the voices and the tokenizer's file.
//! - [`pretrained`] (features `store` and `cache`): the named checkpoint,
//!   `zyphra/zonos3/zonos3-mini`, and the factory that loads it into a
//!   construct.
//!
//! - [`driver`]: ZONOS3 as a TTS engine,
//!   [`Zonos3Engine`](driver::Zonos3Engine), one take decoded whole, as the
//!   release generates it; [`Zonos3Driver`](driver::Zonos3Driver) is the [TTS
//!   driver](super::driver) over it.
//!
//! - [`interop`]: its checkpoint's ecosystem, [`interop::zonos`]: the release's
//!   `config.json` and voices, and its `model.safetensors`, whose `lm.*`
//!   tensors are upstream's release language model as it is, and its `codec.*`
//!   tensors the decoder, weight norm fused.

pub mod blocks;
pub mod prompt;
pub mod sampler;

pub mod voices;

pub mod driver;

mod construct;
mod timed_decode;
pub use construct::*;

pub mod interop;

#[cfg(all(feature = "store", feature = "cache"))]
pub mod pretrained;

#[cfg(all(test, feature = "store"))]
mod codec_toy_tests;
#[cfg(all(test, feature = "store"))]
mod lm_toy_tests;
#[cfg(all(test, feature = "store"))]
mod speaker_toy_tests;
