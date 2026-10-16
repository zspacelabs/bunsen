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
//! - [`Qwen3AsrPromptConfig`]: the chat prompt around an utterance's audio, and
//!   parsing the reply into a [`Qwen3AsrTranscript`].
//! - [`Qwen3AsrBundle`]: a model with its front end, vocabulary, prompt and
//!   decoder; [`transcribe`](Qwen3AsrBundle::transcribe) one utterance, or
//!   [`transcribe_long`](Qwen3AsrBundle::transcribe_long) any recording in
//!   windows.
//! - [`Qwen3AsrEngine`]: a bundle as the
//!   [`asr_driver`](crate::kits::speech::asr_driver) core's engine; a
//!   [`Qwen3AsrStreamDriver`] streams audio through it, under any
//!   [`EmissionPolicy`](crate::kits::speech::asr_driver::EmissionPolicy).
//! - `read_phonon1_archive` (feature `fermion`): a Phonon-1 archive's weights,
//!   expanded to dense `transformers`-named tensors, and its vocabulary.
//! - [`pretrained`] (features `store_safetensors` and `cache`): the named
//!   checkpoints, `fermion/phonon-1` the default, and the factory that loads
//!   one into a bundle.
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `Qwen3ASRForConditionalGeneration` state dict.

pub mod blocks;

#[cfg(feature = "fermion")]
mod archive;
mod bundle;
mod decode;
mod engine;
mod front_end;
mod prompt;
#[cfg(feature = "fermion")]
pub use archive::*;
pub use bundle::*;
pub use decode::*;
pub use engine::*;
pub use front_end::*;
pub use prompt::*;

#[cfg(all(feature = "store_safetensors", feature = "cache"))]
pub mod pretrained;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;

#[doc(inline)]
pub use blocks::{
    Qwen3Asr,
    Qwen3AsrConfig,
};
