//! # ZONOS3-Mini
//!
//! Zyphra's ZONOS3-Mini text-to-speech model: a language model over rows
//! of ten audio codebooks and a text token, and a codec that decodes the
//! codes to 44.1 kHz audio.
//!
//! - [`lm`]: the language model, [`Zonos3Lm`](lm::Zonos3Lm). Each row's ids are
//!   summed through their own tables (a speaker row and the condition rows
//!   replaced), run through a stack of gated `DeltaNet` and global attention
//!   layers without a residual stream
//!   ([`AttnResStack`](crate::blocks::transformers::decoders::attn_res::AttnResStack)),
//!   their feed-forwards a dense `SwiGLU` and then latent mixtures of experts,
//!   and read out as every codebook's logits. Decoding is cached, one row a
//!   step, through [`decode_cache`](lm::Zonos3Lm::decode_cache).
//! - [`codec`]: the decoder, [`Zonos3Codec`](codec::Zonos3Codec), codes at 15
//!   Hz to 44.1 kHz audio: the codebooks' embeddings summed into a latent (the
//!   first through a causal `ConvNeXt` head), a trunk of windowed causal
//!   attention stages, each patch-up between them doubling the frame rate, and
//!   a head of inverse STFTs at two FFT sizes, mixed and resampled.
//! - [`prompt`]: the rows the language model reads before it generates
//!   ([`Zonos3PromptConfig::build`](prompt::Zonos3PromptConfig::build)): a
//!   speaker (an embedding or reference codes), the condition rows, the text
//!   and the silence prefill, and the delay pattern ([`shear`](prompt::shear),
//!   [`unshear`](prompt::unshear)) the audio rows follow. Its text is
//!   normalized by the caller; [`zonos3_text_ids`](prompt::zonos3_text_ids)
//!   tokenizes it.
//! - [`voices`] (feature `store`): the stock voices, each reference codes and a
//!   speaker embedding.
//! - [`Zonos3Json`]: the release's `config.json`, from which
//!   [`Zonos3LmConfig::from_json`](lm::Zonos3LmConfig::from_json) and the
//!   codec's and prompt's configs are read.
//! - [`Zonos3Construct`] (feature `store`): a loaded model, its language model
//!   and codec with their configs, the prompt layout, the generation settings,
//!   the voices and the tokenizer's file.
//! - [`pretrained`] (features `store` and `cache`): the named checkpoint,
//!   `zyphra/zonos3/zonos3-mini`, and the factory that loads it into a
//!   construct.
//!
//! The kit has no sampler or driver: a caller samples each step's logits
//! (`config.json`'s `generation` section holds the release's settings), and
//! decodes the unsheared codes with the codec.
//!
//! The weights are `Zyphra/ZONOS3-Mini-hf`'s `model.safetensors`, whose
//! `lm.*` tensors are upstream's release language model as it is
//! ([`lm::weights`]), and its `codec.*` tensors the decoder, weight norm
//! fused ([`codec::weights`]).

pub mod codec;
pub mod lm;
pub mod prompt;

mod config;
pub use config::*;

#[cfg(feature = "store")]
pub mod voices;

#[cfg(feature = "store")]
mod construct;
#[cfg(feature = "store")]
pub use construct::*;

#[cfg(all(feature = "store", feature = "cache"))]
pub mod pretrained;
