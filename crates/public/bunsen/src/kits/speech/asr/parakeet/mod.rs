//! # Parakeet TDT
//!
//! NVIDIA's Parakeet TDT speech recognizer: a `FastConformer` encoder and a
//! token-and-duration transducer. `parakeet-tdt-0.6b-v3` is the reference
//! checkpoint; `FermionResearch`'s Phonon-2 is a re-quantized derivative with
//! the same architecture.
//!
//! - [`blocks`]: the model, [`ParakeetTdt`], built from a
//!   [`ParakeetTdtConfig`]; its encoder is the shared
//!   [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder)
//!   a
//!   [`ConformerEncoderConfig`](crate::blocks::transformers::encoders::conformer::ConformerEncoderConfig)
//!   builds.
//! - [`ParakeetFrontEnd`]: the feature extractor, waveforms to normalized
//!   log-mels, built from a [`ParakeetFrontEndConfig`]: `NeMo`'s
//!   [`NemoMelFrontEnd`](crate::kits::speech::nemo::NemoMelFrontEnd) under its
//!   defaults.
//! - [`TdtGreedyDecoder`]: greedy decoding of the transducer, batched, to
//!   [`TdtHypothesis`] tokens with their frames.
//! - [`ParakeetVocabulary`]: the `SentencePiece` pieces, as a
//!   [`Detokenizer`](crate::kits::support::tokens::Detokenizer) and as
//!   transcripts.
//! - [`ParakeetConstruct`]: a loaded model with its front end and vocabulary,
//!   and [`transcribe`](ParakeetConstruct::transcribe).
//! - [`ParakeetEngine`]: a construct as the
//!   [`asr_driver`](crate::kits::speech::asr::driver) core's engine, over
//!   [`ParakeetFeatures`], its front end streamed; a [`ParakeetStreamDriver`]
//!   streams audio through it, under any
//!   [`EmissionPolicy`](crate::kits::speech::asr::driver::EmissionPolicy).
//! - [`pretrained`] (features `store_safetensors` and `cache`): the index of
//!   named checkpoints, `fermion/phonon-2` the default, and the factory that
//!   loads one into a construct. `read_phonon2_archive` (feature `fermion`)
//!   reads Phonon-2's archive through the shared
//!   [`fermion`](crate::data::pretrained::fermion) readers.
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `ParakeetForTDT` state dict, the names every Parakeet checkpoint format
//!   here is read under.

pub mod blocks;

#[cfg(feature = "fermion")]
mod archive;
mod construct;
mod decode;
mod engine;
mod vocab;
#[cfg(feature = "fermion")]
pub use archive::*;
pub use construct::*;
pub use decode::*;
pub use engine::*;
pub use vocab::*;

/// Parakeet's feature extractor: `NeMo`'s mel front end, whose defaults are
/// `parakeet-tdt-0.6b-v3`'s.
pub type ParakeetFrontEnd = crate::kits::speech::nemo::NemoMelFrontEnd;

/// Config for [`ParakeetFrontEnd`]: `NeMo`'s, its defaults
/// `parakeet-tdt-0.6b-v3`'s.
pub type ParakeetFrontEndConfig = crate::kits::speech::nemo::NemoMelFrontEndConfig;

/// Parakeet's front end, streamed.
pub type ParakeetFeatures = crate::kits::speech::nemo::NemoMelFeatures;

pub use crate::kits::speech::nemo::NemoMelFrontEndMeta as ParakeetFrontEndMeta;

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
