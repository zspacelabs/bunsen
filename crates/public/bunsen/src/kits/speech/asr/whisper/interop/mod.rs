//! # Whisper's interop: its checkpoints' ecosystems
//!
//! - [`openai_whisper`]: `OpenAI`'s `.pt`, the state dict its `whisper` package
//!   saves.
//! - [`transformers`]: a `WhisperForConditionalGeneration` checkpoint,
//!   `model.safetensors` or its shards.

pub mod openai_whisper;
pub mod transformers;
