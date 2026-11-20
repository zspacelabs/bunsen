//! `source-checks`: layout rules of the repository's source that the
//! compiler can't check.
//!
//! Run from the repository root; CI runs each check in its Clippy job.
//!
//! - `interop`: in `bunsen::kits`, external-format code (upstream mirrors,
//!   rekeys and loaders, `store`-gated code) lives in `interop` modules
//!   ([`interop`]).

pub mod interop;

use std::process::ExitCode;

use clap::{
    Parser,
    Subcommand,
};

/// Checks the source layout that the compiler can't.
#[derive(Debug, Parser)]
#[command(version)]
pub struct Cli {
    /// The check to run.
    #[command(subcommand)]
    pub command: Command,
}

/// A check.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// External-format code in `bunsen::kits` lives in `interop` modules.
    ///
    /// Fails on any finding.
    Interop(interop::InteropArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Interop(args) => interop::run(&args),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("source-checks: {message}");
            ExitCode::from(2)
        }
    }
}
