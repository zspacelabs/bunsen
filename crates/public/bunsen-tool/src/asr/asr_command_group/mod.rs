//! `bunsen-tool asr`: the command group.

mod asr_construct_args;
#[cfg(feature = "live")]
mod asr_live;
mod asr_transcribe;
mod diarize_command_group;
mod family_command_group;
mod presets_command_group;

pub use asr_construct_args::*;
#[cfg(feature = "live")]
pub use asr_live::*;
pub use asr_transcribe::*;
pub use diarize_command_group::*;
pub use family_command_group::*;
pub use presets_command_group::*;

#[cfg(test)]
mod tests;

mod command_group;

pub use command_group::*;
