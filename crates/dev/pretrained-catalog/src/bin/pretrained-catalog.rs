//! `pretrained-catalog`: `bunsen-tool catalog`'s commands
//! ([`bunsen_app::catalog`]) over the example catalogs
//! (`build_catalog::fixtures`) rather than the built-in one, without the model
//! kits bunsen-tool builds.
//!
//! ```text
//! pretrained-catalog tree
//! pretrained-catalog --catalog local.yml select qwen3 --off thinking --limit 1
//! ```

use std::process::ExitCode;

use bunsen_app::catalog::CatalogCommand;
use clap::{
    CommandFactory,
    FromArgMatches,
    Parser,
};

/// The example build catalogs: browse them, resolve builds, select them.
///
/// The example catalogs (prelude, commons, qwen3, whisper), with any
/// `--catalog` extensions imported over them in order.
#[derive(Parser, Debug)]
#[command(name = "pretrained-catalog", version)]
struct Cli {
    #[command(flatten)]
    command: CatalogCommand,
}

fn main() -> ExitCode {
    let matches = Cli::command().get_matches();
    let cli = match Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(error) => error.exit(),
    };
    let base = match bunsen::data::pretrained::build_catalog::fixtures::example_stack() {
        Ok(base) => base,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    match cli.command.run(&matches, base) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}", error.message());
            ExitCode::FAILURE
        }
    }
}
