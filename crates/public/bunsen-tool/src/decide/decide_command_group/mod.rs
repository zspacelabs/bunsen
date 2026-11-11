//! `bunsen-tool decide`: the command group.

mod decide_fetch;
mod decide_list;
mod decide_run;

use bunsen::errors::BunsenResult;
use clap::ArgMatches;
pub use decide_fetch::*;
pub use decide_list::*;
pub use decide_run::*;

/// `bunsen-tool decide`'s commands: typed questions about a context,
/// answered with calibrated probabilities by a Laya decision model. `decide
/// run` reads a request (YAML or JSON: a `context` and its `choices`) and
/// writes one result per choice; `decide list` and `decide fetch` are its
/// checkpoints'.
#[derive(clap::Args, Debug)]
pub struct DecideCommandGroup {
    #[clap(subcommand)]
    command: DecideSubCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum DecideSubCommand {
    /// Answers a request's questions: one result per choice, in order.
    ///
    /// The request is a YAML or JSON document, from a file or stdin, of a
    /// `context` and its `choices`, each a `type` (`choice`, `score` or
    /// `noul`), a `question`, and its `options`.
    Run(Box<DecideRunCommand>),

    /// The checkpoints bunsen knows by name.
    ///
    /// Each one's origin, and each of its files' path in its repo and whether
    /// it is cached.
    List(DecideListCommand),

    /// Brings every file of the named checkpoints into the weights cache.
    Fetch(DecideFetchCommand),
}

impl DecideCommandGroup {
    /// Runs the command; `matches` are the leaf command's own.
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        match &self.command {
            DecideSubCommand::Run(cmd) => cmd.run(matches),
            DecideSubCommand::List(cmd) => cmd.run(),
            DecideSubCommand::Fetch(cmd) => cmd.run(),
        }
    }
}
