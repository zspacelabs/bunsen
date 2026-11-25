//! Image features on bunsen's image kits ([`bunsen::kits::images`]), as
//! `bunsen-tool embed` computes them: image files to their pooled features,
//! as safetensors.
//!
//! - [`EmbedCommandGroup`]: `embed run`.

mod embed_command_group;

pub use embed_command_group::*;
