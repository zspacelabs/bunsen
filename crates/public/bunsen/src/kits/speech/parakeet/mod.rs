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
//! - [`read_fermion_container`] (feature `store`): the
//!   `fermion-five-value-parakeet-v1` container `FermionResearch`'s Phonon-2
//!   ships in, expanded to `transformers`-named f32 tensors.
//! - [`ParakeetBundle`]: a loaded model with its front end and vocabulary, and
//!   [`transcribe`](ParakeetBundle::transcribe).
//! - [`pretrained`] (features `store_safetensors` and `cache`): the index of
//!   named checkpoints, `fermion/phonon-2` the default, and the factory that
//!   loads one into a bundle. `read_phonon_archive` (feature `fermion`) reads
//!   Phonon-2's archive.
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `ParakeetForTDT` state dict, the names every Parakeet checkpoint format
//!   here is read under.

pub mod blocks;

#[cfg(feature = "fermion")]
mod archive;
mod bundle;
mod decode;
#[cfg(feature = "store")]
mod fermion;
mod front_end;
mod vocab;
#[cfg(feature = "fermion")]
pub use archive::*;
pub use bundle::*;
pub use decode::*;
#[cfg(feature = "store")]
pub use fermion::*;
pub use front_end::*;
pub use vocab::*;

#[cfg(all(feature = "store_safetensors", feature = "cache"))]
pub mod pretrained;
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
