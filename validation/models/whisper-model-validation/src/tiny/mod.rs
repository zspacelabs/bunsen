//! `openai/tiny`, the smallest Whisper: the driver checks CI runs.

use super::*;

/// The stream driver held to the model's own one-shot decode.
#[cfg(all(feature = "checkpoints", feature = "gpu-tests"))]
mod driver;
