//! # Pretrained Silero VAD models
//!
//! [`default_silero_factory`] is the index: with the `silero-weights`
//! feature, one row, `bundled:silero/vad`, the burnpack linked into the
//! binary, and nothing without it. [`SileroConstruct`] reads a burnpack
//! into a [`SileroVadCollection`](super::SileroVadCollection), both
//! branches. The loaders in [`load`] read the bytes directly, with no
//! cache, and stay for a binary that wants nothing else.

mod construct;
pub mod load;
mod providers;

#[cfg(feature = "silero-weights")]
pub use bunsen_bundled_silero as bundled;
pub use construct::*;
pub use providers::*;
