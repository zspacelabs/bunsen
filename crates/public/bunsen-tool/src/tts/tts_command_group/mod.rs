//! `bunsen-tool tts`: the command group.

mod tts_fetch;
mod tts_list;
mod tts_run;

use bunsen::errors::BunsenResult;
use clap::ArgMatches;
pub use tts_fetch::*;
pub use tts_list::*;
pub use tts_run::*;

/// `bunsen-tool tts`'s commands: text to speech with ZONOS3-Mini. `tts run`
/// speaks a text in a stock voice to a WAV file; `tts list` and `tts fetch`
/// are its checkpoints'.
#[derive(clap::Args, Debug)]
pub struct TtsCommandGroup {
    #[clap(subcommand)]
    command: TtsSubCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum TtsSubCommand {
    /// Speaks a text in a stock voice, to a WAV file.
    ///
    /// The text, from the arguments or stdin, normalized (numbers, money and
    /// dates spoken), then one take, decoded whole, written as mono 16-bit
    /// audio at 44.1 kHz.
    Run(Box<TtsRunCommand>),

    /// The checkpoints bunsen knows by name.
    ///
    /// Each one's origin, and each of its files' path in its repo and whether
    /// it is cached.
    List(TtsListCommand),

    /// Brings every file of the named checkpoints into the weights cache.
    Fetch(TtsFetchCommand),
}

impl TtsCommandGroup {
    /// Runs the command; `matches` are the leaf command's own.
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        match &self.command {
            TtsSubCommand::Run(cmd) => cmd.run(matches),
            TtsSubCommand::List(cmd) => cmd.run(),
            TtsSubCommand::Fetch(cmd) => cmd.run(),
        }
    }
}
