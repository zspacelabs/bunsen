//! # The engine and its driver
//!
//! [`Qwen3TtsEngine`]: Qwen3-TTS's half of the
//! [TTS driver](crate::kits::speech::tts::driver): a chat-wrapped text, a
//! stock speaker or a described voice, and a language to the talker and code
//! predictor's take
//! ([`Qwen3TtsModel::generate_codes`](super::Qwen3TtsModel::generate_codes)),
//! decoded by the codec in chunks. [`Qwen3TtsDriver`] is the driver over it
//! ([`qwen3_tts_driver`]); the text is spoken as typed, as upstream speaks it.

mod engine;

pub use engine::*;

#[cfg(test)]
mod toy_tests;
