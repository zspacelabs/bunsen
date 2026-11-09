//! `chat repl`.

use bunsen::errors::BunsenResult;

use crate::chat::{
    ChatGenerateFlags,
    ChatModelFamily,
    ChatModelFamilyFlags,
    LoadedChat,
};

/// Holds a conversation on stdin; `F` is the family's own generation flags.
#[derive(clap::Args, Debug)]
pub struct ChatReplCommand<F: clap::Args> {
    #[clap(flatten)]
    flags: ChatGenerateFlags,

    #[clap(flatten)]
    family: F,
}

impl<F: ChatModelFamilyFlags> ChatReplCommand<F> {
    /// Holds a conversation with `family`'s checkpoint, or outside `family`
    /// whichever `--model` names.
    pub fn run(
        &self,
        family: Option<ChatModelFamily>,
    ) -> BunsenResult<()> {
        let (device, chat, model) = self.flags.load(family, &self.family.presets())?;
        match &chat {
            LoadedChat::Qwen3(chat) => self.flags.converse(chat, &self.family, &device, &model)?,
            LoadedChat::Qwen3_5(chat) => {
                self.flags.converse(chat, &self.family, &device, &model)?
            }
            LoadedChat::Gemma(chat) => self.flags.converse(chat, &self.family, &device, &model)?,
            LoadedChat::NanoChat(chat) => {
                self.flags.converse(chat, &self.family, &device, &model)?
            }
        }
        drop(chat);
        device.close()
    }
}
