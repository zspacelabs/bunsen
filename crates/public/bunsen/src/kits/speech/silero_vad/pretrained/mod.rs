//! # Pretrained Silero VAD models
//!
//! [`default_silero_factory`] is the index: one row,
//! `well-known:silero/vad-16k` (with `store_safetensors`), upstream's
//! `silero_vad_16k.safetensors`: the 16 kHz model, fetched from a pinned
//! revision of `snakers4/silero-vad` (1.2 MB, MIT).
//!
//! [`SileroConstruct`] reads it into a
//! [`SileroVadCollection`](super::SileroVadCollection) of the 16 kHz branch.
//! A resource map of a burnpack (both branches, keyed as `burn-onnx` names
//! the ONNX graph's nodes) reads too. The loaders in [`load`] read files or
//! bytes directly, with no cache.
//!
//! The whole module needs the `store` and `cache` features, both
//! default. The kit has no prefab map.

mod construct;
pub mod load;
mod providers;

pub use construct::*;
pub use providers::*;
