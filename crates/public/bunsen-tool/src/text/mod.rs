//! Text normalization on its own, as `bunsen-tool text` runs it: the
//! normalizers a TTS driver or a word error rate reads text through
//! ([`bunsen::kits::support::text`]).
//!
//! - [`TextCommandGroup`]: the `text` command tree, `text
//!   <normalize|normalizers>`.

mod text_command_group;

pub use text_command_group::*;
