//! `asr diarize`: the command group.

mod asr_diarize_args;
#[cfg(feature = "live")]
mod asr_diarize_live;
mod asr_diarize_run;
mod support;

pub use asr_diarize_args::*;
#[cfg(feature = "live")]
pub use asr_diarize_live::*;
pub use asr_diarize_run::*;
pub use support::load_diarizer;

#[cfg(test)]
mod tests;

mod command_group;

pub use command_group::*;
