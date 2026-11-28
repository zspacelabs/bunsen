//! `bunsen-tool text`: the command group.

mod text_normalize;
mod text_normalizers;

pub use text_normalize::*;
pub use text_normalizers::*;

mod command_group;

pub use command_group::*;
