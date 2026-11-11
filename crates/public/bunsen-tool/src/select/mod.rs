//! # Selecting a model by its tags
//!
//! What the commands choose their models with, over bunsen's selection core
//! ([`bunsen::data::pretrained::selection`]): a command holds a
//! [`SelectionContext`] (its default selection and the selections it names),
//! builds candidates from its kits' rows, and chooses among them by the
//! command line's `--select`s and `--preset`s ([`SelectionContext::choose`]).
//!
//! - [`SelectArgs`]: `--select NAME|{..}` (`--model`, an alias), repeatable,
//!   and `--patch {..}`, a merge patch over the chosen model's config.
//! - [`SelectionExpr`]: a `--select {..}` object: `name`, `require`, `prefer`,
//!   `latest`, `absent`, `order`, `take`, applied in that order, whatever the
//!   order they are written in. Expressions merge over the command's defaults
//!   as RFC 7386 merge patches ([`merge`]).
//! - Names ([`match_name`]): a row named exactly (its full id, bare name or an
//!   alias) is that row, and the defaults do not apply to it; otherwise a name
//!   is a set of rows: a prefix of their names, else a `groups` tag, else a
//!   `kit` tag (`--select whisper`), which the defaults and the expressions
//!   choose within.
//! - The steps an expression compiles to: [`Require`] and [`Prefer`] (with
//!   [`Want`], what an entry asks of a tag, and [`Absent`], what a candidate
//!   lacking the tag is); [`Latest`], each release sequence's newest first
//!   ([`release_cmp`]); and, for `language`, the graded RFC 4647 match
//!   [`match_language`].

mod args;
mod context;
mod expr;
mod language;
mod names;
mod scorers;
mod version;

pub use args::*;
pub use context::*;
pub use expr::*;
pub use language::*;
pub use names::*;
pub use scorers::*;
pub use version::*;

#[cfg(test)]
pub(crate) mod testing {
    use bunsen::data::pretrained::{
        Pretrained,
        ResourceBundle,
        selection::{
            Selection,
            SelectionCandidate,
        },
    };
    use serde_json::Value;

    /// A candidate of a row called `name`, with `tags`.
    pub fn candidate(
        name: &str,
        tags: Value,
    ) -> SelectionCandidate {
        SelectionCandidate::new(Pretrained {
            name: name.to_string(),
            aliases: Vec::new(),
            description: String::new(),
            license: None,
            origin: None,
            prefab: None,
            resources: ResourceBundle::new(name),
            presets: Default::default(),
            args: Default::default(),
            tags: serde_json::from_value(tags).unwrap(),
        })
    }

    /// The selection's row names, in order.
    pub fn names(selection: &Selection) -> Vec<&str> {
        selection
            .iter()
            .map(|(c, _)| c.pretrained.name.as_str())
            .collect()
    }
}
