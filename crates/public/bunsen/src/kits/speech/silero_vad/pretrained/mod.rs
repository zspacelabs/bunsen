//! # Pretrained Silero VAD models
//!
//! [`default_silero_factory`] is the index:
//!
//! - `well-known:silero/vad-16k` (with `store_safetensors`): upstream's
//!   `silero_vad_16k.safetensors`, the 16 kHz model, fetched from a pinned
//!   revision of `snakers4/silero-vad` (1.2 MB, MIT);
//! - `bundled:silero/vad` (with `silero-weights`): the burnpack linked into the
//!   binary, both sample-rate branches.
//!
//! [`SileroConstruct`] reads either into a
//! [`SileroVadCollection`](super::SileroVadCollection): the 16 kHz branch
//! from the safetensors, both from the burnpack. The loaders in [`load`]
//! read the files or bytes directly, with no cache, and stay for a binary
//! that wants nothing else.
//!
//! The whole module needs the `store` and `cache` features, both
//! default. The kit has no prefab map.

mod construct;
pub mod load;
mod providers;

#[cfg(feature = "silero-weights")]
pub use bunsen_bundled_silero as bundled;
pub use construct::*;
pub use providers::*;
