//! # Speech to text
//!
//! The ASR kits, and [`driver`], the parts of a speech-to-text stream driver
//! that do not depend on the model: the emission policy, the stream clock,
//! voice-activity regions, speaker attribution, and the engine seam each kit
//! implements.
//!
//! - [`whisper`]: `OpenAI`'s Whisper.
//! - [`parakeet`]: NVIDIA's Parakeet TDT.
//! - [`qwen3_asr`]: Qwen's Qwen3-ASR.
//! - [`voxtral`]: Mistral's Voxtral Realtime.

pub mod driver;
pub mod parakeet;
pub mod qwen3_asr;
pub mod voxtral;
pub mod whisper;
