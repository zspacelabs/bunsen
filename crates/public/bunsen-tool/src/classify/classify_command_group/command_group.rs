//! The `classify` command group.

use bunsen::errors::BunsenResult;

use super::{
    classify_fetch::ClassifyFetchCommand,
    classify_list::ClassifyListCommand,
    classify_run::ClassifyRunCommand,
};

/// `bunsen-tool classify`'s commands: image files to their top classes with
/// a pretrained image classifier (`classify run`); `classify list` and
/// `classify fetch` are its checkpoints'.
#[derive(clap::Args, Debug)]
pub struct ClassifyCommandGroup {
    #[clap(subcommand)]
    command: ClassifySubCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum ClassifySubCommand {
    /// Classifies image files: each one's top classes, most probable first.
    ///
    /// Each image goes through the checkpoint's own eval transform (timm's
    /// `pretrained_cfg`: its size, filter, crop and normalization).
    Run(Box<ClassifyRunCommand>),

    /// The checkpoints bunsen knows by name.
    ///
    /// Each one's origin, and each of its files' path and whether it is
    /// cached.
    List(ClassifyListCommand),

    /// Brings every file of the named checkpoints into the weights cache.
    Fetch(ClassifyFetchCommand),
}

impl ClassifyCommandGroup {
    /// Runs the command.
    pub fn run(&self) -> BunsenResult<()> {
        match &self.command {
            ClassifySubCommand::Run(cmd) => cmd.run(),
            ClassifySubCommand::List(cmd) => cmd.run(),
            ClassifySubCommand::Fetch(cmd) => cmd.run(),
        }
    }
}
