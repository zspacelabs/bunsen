//! # Qwen3-ASR
//!
//! Qwen's Qwen3-ASR speech recognizer: a Whisper-style audio encoder over
//! chunked convolutions, whose frames take the place of placeholder tokens
//! in a chat prompt answered by a Qwen3 language model. `Qwen3-ASR-0.6B` is
//! the reference checkpoint; `FermionResearch`'s Phonon-1 builds are
//! re-quantized derivatives with the same architecture.
//!
//! - [`blocks`]: the model, [`Qwen3Asr`], built from a [`Qwen3AsrConfig`]; its
//!   text model is [`kits::language::qwen3`](crate::kits::language::qwen3).
//! - [`Qwen3AsrFrontEnd`]: the feature extractor, Whisper's.
//! - [`Qwen3AsrGreedyDecoder`]: greedy decoding through a KV cache.
//! - [`Qwen3AsrPromptConfig`]: the chat prompt around an utterance's audio, and
//!   parsing the reply into a [`Qwen3AsrTranscript`].
//! - [`Qwen3AsrConstruct`]: a model with its front end, vocabulary, prompt and
//!   decoder; [`transcribe`](Qwen3AsrConstruct::transcribe) one utterance, or
//!   [`transcribe_long`](Qwen3AsrConstruct::transcribe_long) any recording in
//!   windows.
//! - [`token_frames`](Qwen3AsrConstruct::token_frames): when each reply token
//!   is spoken, from the decoder's attention onto the audio, under the heads
//!   that follow the speech ([`QWEN3_ASR_0_6B_ALIGNMENT_HEADS`]).
//! - [`Qwen3AsrEngine`]: a construct as the
//!   [`asr_driver`](crate::kits::speech::asr::driver) core's engine; a
//!   [`Qwen3AsrStreamDriver`] streams audio through it, under any
//!   [`EmissionPolicy`](crate::kits::speech::asr::driver::EmissionPolicy).
//! - `read_phonon1_archive` (feature `store`): a Phonon-1 archive's weights,
//!   expanded to dense `transformers`-named tensors, and its vocabulary.
//! - [`pretrained`] (features `store` and `cache`): the named checkpoints,
//!   `fermion/phonon-1` the default, and the factory that loads one into a
//!   construct.
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `Qwen3ASRForConditionalGeneration` state dict.

pub mod blocks;

mod alignment;
#[cfg(feature = "store")]
mod archive;
mod construct;
mod decode;
mod engine;
mod front_end;
mod prompt;
pub use alignment::*;
#[cfg(feature = "store")]
pub use archive::*;
pub use construct::*;
pub use decode::*;
pub use engine::*;
pub use front_end::*;
pub use prompt::*;

#[cfg(feature = "store")]
pub mod pretrained;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

#[doc(inline)]
pub use blocks::{
    Qwen3Asr,
    Qwen3AsrConfig,
};
