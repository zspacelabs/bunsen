//! # Qwen3-ASR
//!
//! Qwen's Qwen3-ASR speech recognizer: a Whisper-style audio encoder over
//! chunked convolutions, whose frames take the place of placeholder tokens
//! in a chat prompt answered by a Qwen3 language model. `Qwen3-ASR-0.6B` is
//! the reference checkpoint; `FermionResearch`'s Phonon-1 builds are
//! re-quantized derivatives with the same architecture.
//!
//! - [`blocks`]: the model, [`Qwen3Asr`], built from a [`Qwen3AsrConfig`]; its
//!   text model is [`kits::gpts::qwen3`](crate::kits::gpts::qwen3).
//! - [`Qwen3AsrFrontEnd`]: the feature extractor, Whisper's.
//! - [`Qwen3AsrGreedyDecoder`]: greedy decoding through a KV cache.
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `Qwen3ASRForConditionalGeneration` state dict.

pub mod blocks;

mod decode;
mod front_end;
pub use decode::*;
pub use front_end::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;

#[doc(inline)]
pub use blocks::{
    Qwen3Asr,
    Qwen3AsrConfig,
};
