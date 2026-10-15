//! # Parakeet TDT
//!
//! NVIDIA's Parakeet TDT speech recognizer: a `FastConformer` encoder and a
//! token-and-duration transducer. `parakeet-tdt-0.6b-v3` is the reference
//! checkpoint; `FermionResearch`'s Phonon-2 is a re-quantized derivative with
//! the same architecture.
//!
//! - [`blocks`]: the model, [`ParakeetTdt`], built from a
//!   [`ParakeetTdtConfig`]; its encoder is the shared
//!   [`ConformerEncoder`](crate::blocks::conformer::ConformerEncoder).
//! - [`ParakeetFrontEnd`]: the feature extractor, waveforms to normalized
//!   log-mels, built from a [`ParakeetFrontEndConfig`].
//! - [`TdtGreedyDecoder`]: greedy decoding of the transducer, batched, to
//!   [`TdtHypothesis`] tokens with their frames.
//! - [`ParakeetVocabulary`]: the `SentencePiece` pieces, as a
//!   [`Detokenizer`](crate::kits::tokens::Detokenizer) and as transcripts.
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `ParakeetForTDT` state dict, the names every Parakeet checkpoint format
//!   here is read under.

pub mod blocks;

mod decode;
mod front_end;
mod vocab;
pub use decode::*;
pub use front_end::*;
pub use vocab::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;

#[doc(inline)]
pub use blocks::{
    ParakeetTdt,
    ParakeetTdtConfig,
    ParakeetTdtMeta,
};
