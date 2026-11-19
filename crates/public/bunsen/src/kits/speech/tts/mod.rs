//! # Text to speech
//!
//! The kits that turn text into audio, and the driver they share:
//!
//! - [`driver`]: the TTS driver, which owns what does not depend on the model;
//!   each kit plugs in as a [`TtsEngine`](driver::TtsEngine).
//! - [`qwen3_tts`]: Alibaba's Qwen3-TTS, a talker and a code predictor over
//!   audio codec codes, and the codec that decodes them.
//! - [`zonos3`]: Zyphra's ZONOS3-Mini, a language model over audio codec codes
//!   and the codec that decodes them.

pub mod driver;
pub mod qwen3_tts;
pub mod zonos3;
