//! `chat run`.

use std::io::Read;

use bunsen::errors::{
    BunsenResult,
    sys_at,
};

use crate::chat::{
    ChatGenerateFlags,
    ChatModelFamily,
    ChatModelFamilyFlags,
    LoadedChat,
};

/// Answers one prompt; `F` is the family's own generation flags.
#[derive(clap::Args, Debug)]
pub struct ChatRunCommand<F: clap::Args> {
    #[clap(flatten)]
    flags: ChatGenerateFlags,

    #[clap(flatten)]
    family: F,

    /// The prompt; stdin when omitted.
    prompt: Option<String>,
}

impl<F: ChatModelFamilyFlags> ChatRunCommand<F> {
    /// Answers the prompt with `family`'s checkpoint, or outside `family`
    /// whichever `--model` names.
    pub fn run(
        &self,
        family: Option<ChatModelFamily>,
    ) -> BunsenResult<()> {
        let prompt = match &self.prompt {
            Some(prompt) => prompt.clone(),
            None => {
                let mut text = String::new();
                std::io::stdin()
                    .read_to_string(&mut text)
                    .map_err(sys_at("read", "stdin"))?;
                text
            }
        };
        let prompt = prompt.trim_end();
        let (device, chat, model) = self.flags.load(family, &self.family.presets())?;
        match &chat {
            LoadedChat::Qwen3(chat) => {
                self.flags
                    .run_once(chat, &self.family, &device, &model, prompt)?
            }
            LoadedChat::Qwen3_5(chat) => {
                self.flags
                    .run_once(chat, &self.family, &device, &model, prompt)?
            }
            LoadedChat::Gemma(chat) => {
                self.flags
                    .run_once(chat, &self.family, &device, &model, prompt)?
            }
            LoadedChat::NanoChat(chat) => {
                self.flags
                    .run_once(chat, &self.family, &device, &model, prompt)?
            }
        }
        drop(chat);
        device.close()
    }
}
