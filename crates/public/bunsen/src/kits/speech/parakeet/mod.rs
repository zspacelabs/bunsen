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
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `ParakeetForTDT` state dict, the names every Parakeet checkpoint format
//!   here is read under.

pub mod blocks;

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
