//! `bunsen-tool embed`: the command group.

mod embed_run;

use bunsen::errors::BunsenResult;
pub use embed_run::*;

/// `bunsen-tool embed`'s commands: image files to their pooled features with
/// a pretrained image model (`embed run`).
#[derive(clap::Args, Debug)]
pub struct EmbedCommandGroup {
    #[clap(subcommand)]
    command: EmbedSubCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum EmbedSubCommand {
    /// Image files to their pooled features, as safetensors.
    ///
    /// One `[images, d]` tensor. The features are the model's pre-logits
    /// (timm's `num_features`), each image through the checkpoint's own eval
    /// transform.
    Run(Box<EmbedRunCommand>),
}

impl EmbedCommandGroup {
    /// Runs the command.
    pub fn run(&self) -> BunsenResult<()> {
        match &self.command {
            EmbedSubCommand::Run(cmd) => cmd.run(),
        }
    }
}
