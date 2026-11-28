//! # Silero VAD as upstream's `silero_vad` package saves it
//!
//! `snakers4/silero-vad`'s `silero_vad_16k.safetensors`, the 16 kHz model's
//! `PyTorch` state dict, as bunsen's
//! [`SileroVad`](crate::kits::speech::vad::silero_vad::SileroVad): its names
//! ([`silero_vad_rekey`]).
//!
//! - **Loading** (feature `store`): `load_silero_vad_tensors`, and
//!   `load_silero_vad_16khz_from_path`, a file with no cache.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod rekey;

pub use rekey::*;

#[cfg(test)]
mod tests;
