//! `asr presets`: the command group.

mod asr_presets_fetch;
mod asr_presets_list;
mod asr_presets_show;

pub use asr_presets_fetch::*;
use asr_presets_list::list;
pub use asr_presets_show::*;
use bunsen::errors::BunsenResult;

/// The presets' commands; `list` when none is named.
#[derive(clap::Args, Debug)]
pub struct AsrPresetsCommandGroup {
    #[command(subcommand)]
    action: Option<AsrPresetsSubCommand>,
}

#[derive(clap::Subcommand, Debug)]
enum AsrPresetsSubCommand {
    /// List the presets: what each is for, and the model each runs.
    ///
    /// Each preset's model is listed for files and for the microphone.
    List,

    /// Show presets: each recipe, as the `bunsen-tool asr family` command it
    /// runs.
    Show(AsrPresetsShowCommand),

    /// Bring every file the named presets run into the weights cache.
    ///
    /// That is both recipes' models, and the VAD where a recipe finds speech
    /// regions.
    Fetch(AsrPresetsFetchCommand),
}

impl AsrPresetsCommandGroup {
    pub fn run(&self) -> BunsenResult<()> {
        match &self.action {
            None | Some(AsrPresetsSubCommand::List) => list(&mut std::io::stdout().lock()),
            Some(AsrPresetsSubCommand::Show(show)) => show.run(),
            Some(AsrPresetsSubCommand::Fetch(fetch)) => fetch.run(),
        }
    }
}
