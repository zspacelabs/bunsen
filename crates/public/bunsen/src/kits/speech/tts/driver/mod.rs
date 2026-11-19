//! # The TTS driver
//!
//! One driver for every text-to-speech model: [`TtsDriver`] owns what does
//! not depend on the model, and a model plugs in as a [`TtsEngine`].
//!
//! - [`TtsRequest`]: a text, its voice ([`TtsVoiceChoice`]), language,
//!   instruction, seed, length and sampling ([`TtsSamplingOverrides`]).
//! - Request stages ([`TtsRequestStage`]), run in order before the engine: the
//!   text through a normalizer ([`TtsTextStage`]), an instruction dropped on
//!   request ([`StripInstruction`]), or a caller's own. A stage that drops
//!   input says so ([`TtsStageNote`]).
//! - [`TtsEngine`]: a model's capabilities ([`TtsCapabilities`]), voices
//!   ([`TtsVoiceInfo`]), its voice and language resolution, its take and its
//!   render ([`TtsTakeInfo`]).
//! - [`TtsSynthesizer`]: the driver as a trait object, what a tool or a test
//!   holds whatever the model; its output is [`TtsSpeech`].
//!
//! The engines: ZONOS3's
//! ([`Zonos3Engine`](crate::kits::speech::tts::zonos3::driver::Zonos3Engine)).
//!
//! A request's instruction is refused by an engine that takes none unless the
//! driver is told to strip it
//! ([`strip_instruction`](TtsDriverConfig::strip_instruction)): input is
//! dropped only when asked, and then the speech's notes say so.

mod engine;
mod request;
mod stages;
mod tts_driver;

pub use engine::*;
pub use request::*;
pub use stages::*;
pub use tts_driver::*;

#[cfg(test)]
mod tests;
