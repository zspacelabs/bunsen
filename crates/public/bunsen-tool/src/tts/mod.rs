//! Text to speech with ZONOS3-Mini
//! ([`bunsen::kits::speech::tts::zonos3`]), as `bunsen-tool tts` runs it.
//!
//! - [`TtsCommandGroup`]: the `tts` command tree, `tts <run|list|fetch>`.
//! - [`choose_tts_model`]: the checkpoint `--model` and the stage flags choose
//!   among ZONOS3's builds ([`tts_selection`]).

mod tts_command_group;
mod tts_select;

pub use tts_command_group::*;
pub use tts_select::*;
