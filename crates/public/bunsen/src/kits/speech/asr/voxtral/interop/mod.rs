//! # Voxtral Realtime's interop: its checkpoint's ecosystems
//!
//! - [`transformers`]: a `VoxtralRealtimeForConditionalGeneration` checkpoint,
//!   its `config.json` and state-dict names.
//! - [`mistral_common`]: the stream's layout, from the release's `tekken.json`
//!   (whose vocabulary is the shared
//!   [`TekkenMirror`](crate::kits::support::interop::mistral_common::TekkenMirror)'
//!   s).

pub mod mistral_common;
pub mod transformers;
