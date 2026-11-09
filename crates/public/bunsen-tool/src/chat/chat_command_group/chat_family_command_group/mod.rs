//! `chat family`: the command group.

mod chat_kit_command_group;

use bunsen::errors::BunsenResult;
pub use chat_kit_command_group::*;

use crate::chat::{
    ChatModelFamily,
    ChatPlainFlags,
    ChatThinkFlags,
};

/// The model families' commands.
#[derive(clap::Args, Debug)]
pub struct ChatFamilyCommandGroup {
    #[command(subcommand)]
    family: ChatFamilySubCommand,
}

/// The model families, each with its own `list`, `fetch`, `run` and `repl`.
#[derive(clap::Subcommand, Debug)]
pub enum ChatFamilySubCommand {
    /// Qwen's Qwen3: `qwen/qwen3-0.6b` by default, or any `hf:org/repo` in
    /// its layout.
    Qwen3(ChatKitCommandGroup<ChatThinkFlags>),

    /// Qwen's dense Qwen3.5 (its text models): `qwen/qwen3.5-0.8b` by
    /// default, or any `hf:org/repo` in its layout.
    #[command(name = "qwen3.5")]
    Qwen3_5(ChatKitCommandGroup<ChatThinkFlags>),

    /// Google's Gemma, generations 1 to 3: `gemma/gemma-3-1b-it` by
    /// default, each from its `unsloth/` mirror.
    Gemma(ChatKitCommandGroup<ChatPlainFlags>),

    /// Andrej Karpathy's nanochat: community checkpoints of upstream's 2025
    /// and 2026 models, `nanochat-students/rl-d20` by default.
    #[command(name = "nanochat")]
    NanoChat(ChatKitCommandGroup<ChatPlainFlags>),
}

impl ChatFamilyCommandGroup {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.family {
            ChatFamilySubCommand::Qwen3(cmd) => cmd.run(ChatModelFamily::Qwen3),
            ChatFamilySubCommand::Qwen3_5(cmd) => cmd.run(ChatModelFamily::Qwen3_5),
            ChatFamilySubCommand::Gemma(cmd) => cmd.run(ChatModelFamily::Gemma),
            ChatFamilySubCommand::NanoChat(cmd) => cmd.run(ChatModelFamily::NanoChat),
        }
    }
}
