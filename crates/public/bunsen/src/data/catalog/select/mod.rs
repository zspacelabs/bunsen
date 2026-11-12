//! Selection: a pipeline of stages over weighted rows.
//!
//! - [`query`]: [`SelectQuery`](query::SelectQuery), a kind and a list of
//!   stages, written only in YAML particles.
//! - [`state`]: the [`SelectTable`](state::SelectTable) a query runs on, the
//!   [`SelectEnv`](state::SelectEnv) its tests read entries through, and the
//!   trace.
//! - [`run`]: the query compiled against the stack, and its run.
//! - [`typed`]: language matching, which the tests and `weigh` share.
//! - [`choose`]: composing a query from a context (defaults, named selections),
//!   a tool's preset prefix and a request, and choosing a build.
//!
//! A query only filters, weighs and orders. The caller applies the presets
//! it chose, and filters out the builds they can't apply to with a prefix
//! ([`PresetSelection::prefix_stage`](crate::data::catalog::presets::PresetSelection::prefix_stage)).

pub mod choose;
pub mod query;
pub mod run;
pub mod state;
pub mod typed;
