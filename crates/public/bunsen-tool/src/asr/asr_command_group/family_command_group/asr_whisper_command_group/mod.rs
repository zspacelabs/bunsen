//! `asr family whisper`: Whisper's command group.

mod asr_whisper_inspect;
mod asr_whisper_prefabs;

pub use asr_whisper_inspect::*;
pub use asr_whisper_prefabs::*;

mod command_group;

pub use command_group::*;
