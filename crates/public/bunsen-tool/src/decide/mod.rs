//! Typed questions about a context, answered with calibrated probabilities by
//! a Laya decision model ([`bunsen::kits::decision::laya`]), as `bunsen-tool
//! decide` runs them.
//!
//! - [`DecideCommandGroup`]: the `decide` command tree, `decide
//!   <run|list|fetch>`.

mod decide_command_group;

pub use decide_command_group::*;
