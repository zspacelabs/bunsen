//! `pretrained-catalog`: `bunsen-tool catalog` on its own
//! ([`bunsen_app::catalog`]), without the model kits bunsen-tool builds.
//!
//! ```text
//! pretrained-catalog --examples tree
//! pretrained-catalog --catalog a.yml --catalog b.yml select qwen3 --off thinking --limit 1
//! ```

use std::process::ExitCode;

use bunsen_app::catalog::CatalogCommand;
use clap::{
    CommandFactory,
    FromArgMatches,
    Parser,
};

/// Import build catalogs, and query the state they make.
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
    match cli.command.run(&matches) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{}", error.message());
            ExitCode::FAILURE
        }
    }
}
