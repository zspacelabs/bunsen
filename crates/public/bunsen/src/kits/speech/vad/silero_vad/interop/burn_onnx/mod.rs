//! # Silero VAD as `burn-onnx` names upstream's ONNX graph
//!
//! A burnpack of both sample-rate branches of upstream's ONNX graph, keyed as
//! `burn-onnx` names the graph's nodes (`conv1d37`, `linear13`, ...), as
//! bunsen's [`SileroVadCollection`](crate::kits::speech::vad::silero_vad::SileroVadCollection).
//!
//! - **Loading** (feature `store`): the key remappers
//!   (`burn_onnx_16khz_remapper`, `burn_onnx_8khz_remapper`),
//!   `load_silero_vad_from_burnpack` under one, and the files or bytes of a
//!   burnpack read with no cache
//!   (`load_silero_vad_collection_from_burnpack_path`, `..._bytes`, and the
//!   single branches').

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;
