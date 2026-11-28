//! `bunsen-tool tts`: the command group.

mod tts_build_voice;
mod tts_clone;
mod tts_collect_voice;
mod tts_fetch;
mod tts_list;
mod tts_run;
mod tts_voices;

pub use tts_build_voice::*;
pub use tts_clone::*;
pub use tts_collect_voice::*;
pub use tts_fetch::*;
pub use tts_list::*;
pub use tts_run::*;
pub use tts_voices::*;

mod command_group;

pub use command_group::*;
