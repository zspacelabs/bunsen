//! `chat family <family>`: one family's command group.

use bunsen::errors::BunsenResult;

use crate::chat::{
    ChatFetchCommand,
    ChatListCommand,
    ChatModelFamily,
    ChatModelFamilyFlags,
    ChatReplCommand,
    ChatRunCommand,
};

/// One family's commands; `F` is its own generation flags.
#[derive(clap::Args, Debug)]
pub struct ChatKitCommandGroup<F: clap::Args> {
    #[command(subcommand)]
    action: ChatKitSubCommand<F>,
}

#[derive(clap::Subcommand, Debug)]
pub enum ChatKitSubCommand<F: clap::Args> {
    /// The family's checkpoints: each one's origin, and each of its files'
    /// path in its repo and whether it is cached.
    List(ChatListCommand),

    /// Brings every file of the family's named checkpoints into the weights
    /// cache.
    Fetch(ChatFetchCommand),

    /// Answers one prompt, from the command line or stdin.
    Run(ChatRunCommand<F>),

    /// Holds a conversation on stdin: `/reset` starts over, `/quit` (or end
    /// of input) leaves.
    Repl(ChatReplCommand<F>),
}

impl<F: ChatModelFamilyFlags> ChatKitCommandGroup<F> {
    /// Runs the command on `family`'s checkpoints.
    pub fn run(
        &self,
        family: ChatModelFamily,
    ) -> BunsenResult<()> {
        match &self.action {
            ChatKitSubCommand::List(cmd) => cmd.run(&[family]),
            ChatKitSubCommand::Fetch(cmd) => cmd.run(Some(family)),
            ChatKitSubCommand::Run(cmd) => cmd.run(Some(family)),
            ChatKitSubCommand::Repl(cmd) => cmd.run(Some(family)),
        }
    }
}
