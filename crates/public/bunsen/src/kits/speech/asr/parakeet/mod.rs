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
//!   [`NemoMelFrontEnd`](crate::kits::speech::support::nemo::NemoMelFrontEnd)
//!   under its defaults.
//! - [`TdtGreedyDecoder`]: greedy decoding of the transducer, batched, to
//!   [`TdtHypothesis`] tokens with their frames.
//! - [`ParakeetVocabulary`]: the `SentencePiece` pieces, as a
//!   [`Detokenizer`](crate::kits::support::tokens::Detokenizer) and as
//!   transcripts.
//! - [`ParakeetConstruct`]: a loaded model with its front end and vocabulary,
//!   and [`transcribe`](ParakeetConstruct::transcribe).
//! - [`ParakeetEngine`](driver::ParakeetEngine): a construct as the
//!   [`asr_driver`](crate::kits::speech::asr::driver) core's engine, over
//!   [`ParakeetFeatures`], its front end streamed; a
//!   [`ParakeetStreamDriver`](driver::ParakeetStreamDriver) streams audio
//!   through it, under any
//!   [`EmissionPolicy`](crate::kits::speech::asr::driver::EmissionPolicy).
//! - [`pretrained`] (features `store` and `cache`): the index of named
//!   checkpoints, `fermion/phonon-2` the default, and the factory that loads
//!   one into a construct.
//! - [`interop`]: its checkpoints' ecosystems: [`interop::transformers`], a
//!   `ParakeetForTDT`'s `config.json` and state-dict names, the names every
//!   Parakeet checkpoint here is read under; [`interop::phonon`], a Phonon-2
//!   archive's weights and vocabulary.

pub mod blocks;
pub mod driver;

mod construct;
mod decode;
mod vocab;
pub use construct::*;
pub use decode::*;
pub use vocab::*;

/// Parakeet's feature extractor: `NeMo`'s mel front end, whose defaults are
/// `parakeet-tdt-0.6b-v3`'s.
pub type ParakeetFrontEnd = crate::kits::speech::support::nemo::NemoMelFrontEnd;

/// Config for [`ParakeetFrontEnd`]: `NeMo`'s, its defaults
/// `parakeet-tdt-0.6b-v3`'s.
pub type ParakeetFrontEndConfig = crate::kits::speech::support::nemo::NemoMelFrontEndConfig;

/// Parakeet's front end, streamed.
pub type ParakeetFeatures = crate::kits::speech::support::nemo::NemoMelFeatures;

pub use crate::kits::speech::support::nemo::NemoMelFrontEndMeta as ParakeetFrontEndMeta;

#[cfg(feature = "store")]
pub mod pretrained;

pub mod interop;

#[cfg(all(test, feature = "store"))]
mod toy_tests;

#[doc(inline)]
pub use blocks::{
    ParakeetTdt,
    ParakeetTdtConfig,
    ParakeetTdtMeta,
};
