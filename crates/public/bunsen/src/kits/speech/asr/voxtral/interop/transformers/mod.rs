//! # Voxtral Realtime in `transformers`' names
//!
//! A `VoxtralRealtimeForConditionalGeneration` checkpoint
//! (`mistralai/Voxtral-Mini-4B-Realtime-2602`'s published layout) as bunsen's
//! [`VoxtralRealtime`](crate::kits::speech::asr::voxtral::VoxtralRealtime):
//!
//! - **Config:** [`voxtral_realtime_config`] of a parsed `config.json`, and
//!   [`voxtral_realtime_config_from_str`] /
//!   [`voxtral_realtime_config_from_path`].
//! - **Names:** [`voxtral_realtime_rekey`], over [`audio_layer_rekey`] and
//!   [`text_layer_rekey_voxtral`] (Qwen3's text layers, with Voxtral's
//!   `ada_rms_norm`).
//! - **Loading** (feature `store`): `load_voxtral_realtime_tensors`; with
//!   `cache`, `load_voxtral_realtime`, a resource set's checkpoint.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod config;
mod rekey;

pub use config::*;
pub use rekey::*;

#[cfg(test)]
mod tests;
