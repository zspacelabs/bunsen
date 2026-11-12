//! `bunsen-tool chat`: the command group.

mod chat_fetch;
mod chat_generate_flags;
mod chat_list;
mod chat_repl;
mod chat_run;
mod family_command_group;

use bunsen::errors::BunsenResult;
pub use chat_fetch::*;
pub use chat_generate_flags::*;
pub use chat_list::*;
pub use chat_repl::*;
pub use chat_run::*;
use clap::ArgMatches;
pub use family_command_group::*;

use crate::chat::{
    ChatModelFamily,
    ChatThinkFlags,
};

/// `bunsen-tool chat`'s commands: `chat <list|fetch|run|repl>` takes a
/// checkpoint of any family, by its name; `chat family
/// <qwen3|qwen3.5|gemma|nanochat> <list|fetch|run|repl>` runs one family's,
/// its own default by default, with only the flags that apply to it.
#[derive(clap::Args, Debug)]
pub struct ChatCommandGroup {
    #[clap(subcommand)]
    command: ChatSubCommand,
}

#[derive(clap::Subcommand, Debug)]
enum ChatSubCommand {
    /// The checkpoints bunsen knows by name, every family's.
    ///
    /// Each one's origin, and each of its files' path in its repo and whether
    /// it is cached.
    List(ChatListCommand),

    /// Brings every file of the named checkpoints into the weights cache.
    ///
    /// Each name is its own family's.
    Fetch(ChatFetchCommand),

    /// Answers one prompt, from the command line or stdin.
    ///
    /// The checkpoint is the one `--model` names, of whichever family.
    Run(ChatRunCommand<ChatThinkFlags>),

    /// Holds a conversation on stdin: `/reset` starts over, `/quit` leaves.
    ///
    /// The checkpoint is the one `--model` names, of whichever family; end of
    /// input leaves too.
    Repl(ChatReplCommand<ChatThinkFlags>),

    /// The model families, each with its own `list`, `fetch`, `run` and `repl`.
    ///
    /// Each has its own default checkpoint, and only the flags that apply to
    /// it.
    Family(ChatFamilyCommandGroup),
}

impl ChatCommandGroup {
    /// Runs the command; `matches` are the leaf command's own, for the
    /// stage flags' order.
    pub fn run(
        &self,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        match &self.command {
            ChatSubCommand::List(cmd) => cmd.run(&ChatModelFamily::ALL),
            ChatSubCommand::Fetch(cmd) => cmd.run(None),
            ChatSubCommand::Run(cmd) => cmd.run(None, matches),
            ChatSubCommand::Repl(cmd) => cmd.run(None, matches),
            ChatSubCommand::Family(cmd) => cmd.run(matches),
        }
    }
}

#[cfg(test)]
mod tests;
