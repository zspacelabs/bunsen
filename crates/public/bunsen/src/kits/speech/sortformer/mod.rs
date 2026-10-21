//! # Sortformer
//!
//! NVIDIA's Sortformer speaker diarizers: audio to per-frame, per-speaker
//! activity, speakers numbered in the order the model first hears them.
//! `Nemotron-3-Diarization` (8 speakers, 10 ms output) is the reference
//! checkpoint; the streaming Sortformer v2 and v2.1 (4 speakers, 80 ms
//! output) are built by the same model.
//!
//! - [`blocks`]: the model, [`Sortformer`], built from a
//!   [`SortformerStructureConfig`] or a policy that lowers to it
//!   ([`NemotronDiarizationConfig`], [`SortformerV2Config`]); its encoder
//!   splits as `NeMo` streams it, into [`pre_encode`](Sortformer::pre_encode)
//!   and [`forward_pre_encoded`](Sortformer::forward_pre_encoded).
//! - [`SortformerStreamingConfig`]: the chunks, the look-ahead, and the
//!   Arrival-Order Speaker Cache's geometry and scoring, with the model cards'
//!   latency presets.
//! - [`SpeakerCache`]: one stream's cache and FIFO between steps, its silence
//!   slots filled by a [`SilenceFill`]: the model's learned embedding, or a
//!   running mean of the silent frames.
//! - [`SortformerMelStream`]: the model streamed over mel frames;
//!   [`SortformerStream`]: over samples, through `NeMo`'s mel front end, as a
//!   [`SpeakerActivityStream`](crate::kits::speech::diarization::SpeakerActivityStream).
//! - [`SortformerBundle`]: a loaded model with its front end;
//!   [`diarizer`](SortformerBundle::diarizer) makes it a
//!   [`SpeakerActivityModel`](crate::kits::speech::diarization::SpeakerActivityModel)
//!   under a streaming preset, [`diarize`](SortformerBundle::diarize) runs a
//!   whole recording to speaker turns.
//! - [`pretrained`] (features `store_safetensors` and `cache`): the index of
//!   named checkpoints, `nvidia/nemotron-3-diarization` the default, `hf:`
//!   repos, and the factory that loads one into a bundle.
//! - [`weights`] (feature `store`): loading tensors named as `transformers`'
//!   `Nemotron3DiarizationForAudioFrameClassification` state dict, or as a
//!   Sortformer v2 `.nemo`'s.
//!
//! Its front end is `NeMo`'s mel front end,
//! [`NemoMelFrontEndConfig::nemotron_diarization`](crate::kits::speech::nemo::NemoMelFrontEndConfig::nemotron_diarization).

pub mod blocks;

mod bundle;
mod cache;
mod stream;
mod streaming;
pub use bundle::*;
pub use cache::*;
pub use stream::*;
pub use streaming::*;

#[cfg(all(feature = "store_safetensors", feature = "cache"))]
pub mod pretrained;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;

#[cfg(all(test, feature = "store_safetensors"))]
mod v2_toy_tests;

#[doc(inline)]
pub use blocks::{
    NemotronDiarizationConfig,
    Sortformer,
    SortformerMeta,
    SortformerStructureConfig,
    SortformerV2Config,
};
