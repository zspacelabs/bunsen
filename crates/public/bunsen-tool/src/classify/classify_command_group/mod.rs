//! `bunsen-tool classify`: the command group.

mod classify_fetch;
mod classify_list;
mod classify_run;

pub use classify_fetch::*;
pub use classify_list::*;
pub use classify_run::*;

mod command_group;

pub use command_group::*;
