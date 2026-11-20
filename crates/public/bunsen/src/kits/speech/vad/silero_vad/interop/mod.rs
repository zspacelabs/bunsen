//! # Silero VAD's interop: its checkpoints' ecosystems
//!
//! - [`silero_vad`]: upstream's `silero_vad` package, its 16 kHz `PyTorch`
//!   state dict's names.
//! - [`burn_onnx`]: a burnpack of upstream's ONNX graph, its nodes named as
//!   `burn-onnx` names them.

pub mod burn_onnx;
pub mod silero_vad;
