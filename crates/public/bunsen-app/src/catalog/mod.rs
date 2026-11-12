//! The build catalogs' commands: dump, browse, show, resolve and select over
//! a base catalog and the extensions the command line adds
//! ([`bunsen::data::catalog`]).
//!
//! ```text
//! catalog tree
//! catalog dump --format json
//! catalog --catalog local.yml show qwen3
//! catalog resolve large-v3 --preset translate
//! catalog select qwen3 --off thinking --limit 1
//! catalog select whisper --weigh '{release: latest}' --desc '$weight' --rank
//! catalog releases release
//! ```
//!
//! The base is the embedding binary's, which it hands
//! [`run`](CatalogCommand::run): bunsen-tool's is the built-in catalog. The
//! extensions (`--catalog`) are imported over it, in the order given.
//!
//! `select` stands in for a tool that chooses builds: it turns its preset
//! flags into the presets it applies and a prefix that keeps only the builds
//! they apply to
//! ([`preset_prefix`](bunsen::data::catalog::select::choose::preset_prefix)),
//! its other flags into stages, in command-line order, and composes them over
//! its defaults
//! ([`SelectionContext::compose`](bunsen::data::catalog::select::choose::SelectionContext::compose)).
//!
//! `select` reads its flags' order from clap's matches, so a tool that
//! embeds [`CatalogCommand`] keeps the [`ArgMatches`](clap::ArgMatches) it
//! parsed, and hands [`run`](CatalogCommand::run) the matches of the command's
//! own level.

#[macro_use]
mod support;

mod command;
mod inspect;
mod resolve;
mod select;
mod tree;

pub use command::{
    CatalogAction,
    CatalogCommand,
    PresetArgs,
};
pub use select::{
    SelectArgs,
    SelectionFlags,
    SelectionStageArgs,
};
pub use tree::{
    AliasView,
    DumpFormat,
};
#[cfg(test)]
mod tests;
