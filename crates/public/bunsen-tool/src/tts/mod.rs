//! Text to speech, as `bunsen-tool tts` runs it: every TTS family's
//! checkpoints ([`TtsFamily`]: ZONOS3-Mini and Qwen3-TTS) through the shared
//! [TTS driver](bunsen::kits::speech::tts::driver).
//!
//! - [`TtsCommandGroup`]: the `tts` command tree, `tts
//!   <run|voices|list|fetch>`.
//! - [`choose_tts_model`]: the checkpoint `--model` and the stage flags choose
//!   among every family's builds ([`tts_selection`]), or one family's
//!   (`--family`).

mod family;
#[cfg(feature = "live")]
mod playback;
mod tts_command_group;
mod tts_select;

pub use family::*;
#[cfg(feature = "live")]
pub use playback::*;
pub use tts_command_group::*;
pub use tts_select::*;
