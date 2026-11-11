//! Selection: choosing a build by membership, typed predicates and presets.
//!
//! - [`core`]: [`Selection`](core::Selection), weighted candidates, and the
//!   steps that derive, weigh, order and cut them.
//! - [`predicate`]: what a `where` entry asks of a declared axis or tag, typed
//!   by its declaration.
//! - [`latest`]: each release group's newest release first, by release time.
//! - [`expr`]: [`Query`](expr::Query), `{select, prefer, order, cutoff,
//!   limit}`, merged as merge patches and compiled against the stack.
//! - [`query`], [`state`] and [`run`]: the selection pipeline. A
//!   [`SelectQuery`](query::SelectQuery) is a kind and a list of stages,
//!   written only in YAML particles; it compiles against a
//!   [`SelectEnv`](state::SelectEnv) and runs on a
//!   [`SelectTable`](state::SelectTable) of weighted rows.
//! - [`choose`]: choosing and ranking
//!   ([`SelectionContext`](choose::SelectionContext)), with names, `--preset`s
//!   and axis flags filling the query.

pub mod choose;
pub mod core;
pub mod expr;
pub mod latest;
pub mod predicate;
pub mod query;
pub mod run;
pub mod state;

#[cfg(test)]
pub(crate) mod testing {
    use crate::{
        document::EntryKind,
        fixtures,
        select::core::{
            Selection,
            SelectionCandidate,
        },
    };

    /// The candidate of the built-in stack's build `leaf`.
    pub fn candidate(leaf: &str) -> SelectionCandidate {
        let stack = fixtures::builtin_stack().unwrap();
        let name = stack.find_build(leaf).unwrap();
        SelectionCandidate::new(stack.resolve_build(&name).unwrap())
    }

    /// A selection of the built-in stack's builds `leaves`, in that order.
    pub fn selection(leaves: &[&str]) -> Selection {
        Selection::new(leaves.iter().map(|l| candidate(l)))
    }

    /// A selection of every built-in build, in declaration order.
    pub fn all_builds() -> Selection {
        let stack = fixtures::builtin_stack().unwrap();
        Selection::new(
            stack
                .of_kind(EntryKind::Build)
                .iter()
                .map(|b| SelectionCandidate::new(stack.resolve_build(b.name()).unwrap())),
        )
    }

    /// The selection's build leaves, in order.
    pub fn shown(selection: &Selection) -> Vec<String> {
        selection
            .names()
            .iter()
            .map(|n| n.leaf().to_string())
            .collect()
    }
}
