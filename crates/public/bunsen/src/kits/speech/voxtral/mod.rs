//! # Voxtral Realtime
//!
//! Mistral's Voxtral Realtime speech recognizer
//! (`mistralai/Voxtral-Mini-4B-Realtime-2602`): a causal audio encoder whose
//! frames are added, one group per 80 ms, to the token embeddings of a
//! Mistral text model that writes the transcript a fixed delay behind the
//! audio. Every part is causal, so the model streams natively: each 80 ms of
//! audio is one decoder step.
//!
//! - [`blocks`]: the model, [`VoxtralRealtime`], built from a
//!   [`VoxtralRealtimeConfig`]: a
//!   [`VoxtralAudioEncoder`](blocks::VoxtralAudioEncoder) and a
//!   [`VoxtralTextModel`](blocks::VoxtralTextModel) on the shared
//!   [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock),
//!   and a [`VoxtralProjector`](blocks::VoxtralProjector) between them.
//! - [`weights`] (feature `store`): loading tensors named as the published
//!   `model.safetensors` names them, and reading its `config.json`.

pub mod blocks;

#[cfg(feature = "store")]
pub mod weights;

#[cfg(all(test, feature = "store_safetensors"))]
mod toy_tests;

#[doc(inline)]
pub use blocks::{
    VoxtralRealtime,
    VoxtralRealtimeConfig,
};
