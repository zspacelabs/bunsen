//! # Whisper's interop: its checkpoints' ecosystems
//!
//! - [`openai_whisper`]: `OpenAI`'s `.pt`, the state dict its `whisper` package
//!   saves.
//! - [`transformers`]: a `WhisperForConditionalGeneration` checkpoint,
//!   `model.safetensors` or its shards.
//!
//! Both follow the interop ladder: `whisper_config_from_*` (the geometry
//! from the tensors' shapes, a checkpoint carrying no config the kit reads,
//! and [`WhisperScanDefaults`] for what it does not record),
//! `load_whisper_tensors`, and `load_whisper(loaded, &config, device,
//! policy)`.

pub mod openai_whisper;
pub mod transformers;

mod scan_defaults;

pub use scan_defaults::*;
