//! `chat repl`.

use bunsen::errors::BunsenResult;
use clap::ArgMatches;

use crate::{
    chat::{
        ChatGenerateFlags,
        ChatModelFamily,
        ChatModelFamilyFlags,
    },
    support::log_build,
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
    ///
    /// `matches` are the command's own, for the stage flags' order.
    pub fn run(
        &self,
        family: Option<ChatModelFamily>,
        matches: &ArgMatches,
    ) -> BunsenResult<()> {
        let (device, chat, model, build) =
            self.flags.load(family, &self.family.presets(), matches)?;
        log_build(&model, &build);
        eprintln!("# {build}");
        eprintln!("# /quit to exit");
        self.flags.converse(&chat, &self.family, &device, &model)?;
        drop(chat);
        device.close()
    }
}
