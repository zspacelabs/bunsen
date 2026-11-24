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
//! - Long text: split into takes at sentence, clause and word boundaries
//!   ([`tts_split_text`]), each take seeded in turn, the takes joined by a
//!   short gap ([`TtsDriverConfig`]).
//! - [`TtsEngine`]: a model's capabilities ([`TtsCapabilities`]), voices
//!   ([`TtsVoiceInfo`]), its voice and language resolution, its take and its
//!   render ([`TtsTakeInfo`]).
//! - Cloned voices, in two levels: a [`TtsVoiceReference`], model-free (its
//!   reference clips at their own rates, its transcript, what was given of the
//!   speaker; saved as a directory, `reference.json` a
//!   [`TtsVoiceReferenceRecord`]; with `audio` and `cache`,
//!   `TtsVoiceReference::collect` makes one from clips, the ingest and annotate
//!   stages, and `read_dir` reads one), from which an engine builds a
//!   [`TtsVoicePrint`] of its own ([`TtsEngine::build_voice`], in one of its
//!   [`TtsCloneMode`]s): tensors, and the [`TtsVoiceIdentity`] of the build
//!   that made it (with `store`, a safetensors file, the identity its only
//!   metadata). A request speaks in one as [`TtsVoiceChoice::Print`].
//! - [`TtsSynthesizer`]: the driver as a trait object, what a tool or a test
//!   holds whatever the model; its output is [`TtsSpeech`].
//! - Streams: a request's speech as events pulled one at a time
//!   ([`TtsDriver::stream`], [`TtsSynthesizer::stream`]: [`TtsSpeechStream`],
//!   [`TtsEvent`]), each take generated a step at a time by its engine
//!   ([`TtsEngine::start_stream`], [`TtsTakeStream`]) and its final frames
//!   decoded in groups ([`TtsChunkPolicy`]). Joined, the stream's audio is the
//!   whole synthesis'.
//!
//! The engines: ZONOS3's
//! ([`Zonos3Engine`](crate::kits::speech::tts::zonos3::driver::Zonos3Engine))
//! and Qwen3-TTS's
//! ([`Qwen3TtsEngine`](crate::kits::speech::tts::qwen3_tts::driver::Qwen3TtsEngine)).
//!
//! A request's instruction is refused by an engine that takes none unless the
//! driver is told to strip it
//! ([`strip_instruction`](TtsDriverConfig::strip_instruction)): input is
//! dropped only when asked, and then the speech's notes say so.

mod engine;
pub mod interop;
mod request;
mod split;
mod stages;
mod stream;
mod tts_driver;
mod voice_print;
mod voice_reference;
#[cfg(all(feature = "audio", feature = "cache"))]
mod voice_reference_dir;

pub use engine::*;
pub use request::*;
pub use split::*;
pub use stages::*;
pub use stream::*;
pub use tts_driver::*;
pub use voice_print::*;
pub use voice_reference::*;
#[cfg(all(feature = "audio", feature = "cache"))]
pub use voice_reference_dir::*;

#[cfg(test)]
mod tests;
