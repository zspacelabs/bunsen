//! # Sortformer
//!
//! NVIDIA's Sortformer speaker diarizers: audio to per-frame, per-speaker
//! activity, speakers numbered in the order the model first hears them.
//! `Nemotron-3-Diarization` (8 speakers, 10 ms output) is the reference
//! checkpoint.
//!
//! - [`blocks`]: the model, [`Sortformer`], built from a
//!   [`SortformerStructureConfig`] or a policy that lowers to it
//!   ([`NemotronDiarizationConfig`]); its encoder splits as `NeMo` streams it,
//!   into [`pre_encode`](Sortformer::pre_encode) and
//!   [`forward_pre_encoded`](Sortformer::forward_pre_encoded).
//! - [`SortformerStreamingConfig`]: the chunks, the look-ahead, and the
//!   Arrival-Order Speaker Cache's geometry and scoring, with the model cards'
//!   latency presets.
//! - [`SpeakerCache`]: one stream's cache and FIFO between steps.
//! - [`SortformerMelStream`]: the model streamed over mel frames;
//!   [`SortformerStream`]: over samples, through `NeMo`'s mel front end, as a
//!   [`SpeakerActivityStream`](crate::kits::speech::diarization::SpeakerActivityStream).
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `Nemotron3DiarizationForAudioFrameClassification` state dict.
//!
//! Its front end is `NeMo`'s mel front end,
//! [`NemoMelFrontEndConfig::nemotron_diarization`](crate::kits::speech::nemo::NemoMelFrontEndConfig::nemotron_diarization).

pub mod blocks;

mod cache;
mod stream;
mod streaming;
pub use cache::*;
pub use stream::*;
pub use streaming::*;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;

#[doc(inline)]
pub use blocks::{
    NemotronDiarizationConfig,
    Sortformer,
    SortformerMeta,
    SortformerStructureConfig,
};
