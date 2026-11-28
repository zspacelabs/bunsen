//! The `text` command group.

use bunsen::errors::BunsenResult;

use super::{
    text_normalize::TextNormalizeCommand,
    text_normalizers::TextNormalizersCommand,
};

/// `bunsen-tool text`'s commands: text normalization on its own. `text
/// normalize` runs normalizers over a text; `text normalizers` lists them.
#[derive(clap::Args, Debug)]
pub struct TextCommandGroup {
    #[clap(subcommand)]
    command: TextSubCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum TextSubCommand {
    /// Normalizes a text: numbers spoken, Unicode folded, punctuation dropped.
    ///
    /// The text, from the arguments, a file or stdin, through each
    /// `--normalize` in order, to stdout; `--explain` shows each stage's
    /// output.
    Normalize(TextNormalizeCommand),

    /// The normalizers `--normalize` names, and whether this build has each.
    Normalizers(TextNormalizersCommand),
}

impl TextCommandGroup {
    /// Runs the command.
    pub fn run(&self) -> BunsenResult<()> {
        match &self.command {
            TextSubCommand::Normalize(cmd) => cmd.run(),
            TextSubCommand::Normalizers(cmd) => cmd.run(),
        }
    }
}
