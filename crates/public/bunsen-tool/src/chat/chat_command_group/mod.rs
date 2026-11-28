//! `bunsen-tool chat`: the command group.

mod chat_fetch;
mod chat_generate_flags;
mod chat_list;
mod chat_repl;
mod chat_run;
mod family_command_group;

pub use chat_fetch::*;
pub use chat_generate_flags::*;
pub use chat_list::*;
pub use chat_repl::*;
pub use chat_run::*;
pub use family_command_group::*;

#[cfg(test)]
mod tests;

mod command_group;

pub use command_group::*;
