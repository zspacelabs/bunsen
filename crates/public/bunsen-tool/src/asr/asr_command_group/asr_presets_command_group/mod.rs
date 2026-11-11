//! `asr presets`: the command group.

mod asr_presets_fetch;
mod asr_presets_show;

use std::io::Write;

pub use asr_presets_fetch::*;
pub use asr_presets_show::*;
use bunsen::errors::{
    BunsenResult,
    sys_at,
};

use crate::asr::{
    AsrPreset,
    Mode,
    construct_config,
    presets,
};

/// The presets' commands; `list` when none is named.
#[derive(clap::Args, Debug)]
pub struct AsrPresetsCommandGroup {
    #[command(subcommand)]
    action: Option<AsrPresetsSubCommand>,
}

#[derive(clap::Subcommand, Debug)]
enum AsrPresetsSubCommand {
    /// List the presets: what each is for, and the model each runs for
    /// files and for the microphone.
    List,

    /// Show presets: each recipe, as the `bunsen-tool asr family` command it
    /// runs.
    Show(AsrPresetsShowCommand),

    /// Bring every file the named presets run into the weights cache: both
    /// recipes' models, and the VAD where a recipe finds speech regions.
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

/// A preset's name, and its aliases when it has any.
fn title(preset: &AsrPreset) -> String {
    match preset.aliases.as_slice() {
        [] => preset.name.clone(),
        aliases => format!("{} ({})", preset.name, aliases.join(", ")),
    }
}

/// Writes each preset, and what each of its recipes runs; then the axes
/// their needs are on.
fn list(out: &mut impl Write) -> BunsenResult<()> {
    let w = |e| sys_at("write to", "stdout")(e);
    for (i, preset) in presets().iter().enumerate() {
        if i > 0 {
            writeln!(out).map_err(w)?;
        }
        writeln!(out, "{}", title(preset)).map_err(w)?;
        writeln!(out, "  {}", preset.description).map_err(w)?;
        for mode in [Mode::Offline, Mode::Live] {
            let resolved = preset.resolve(mode)?;
            writeln!(
                out,
                "  {:<11} {} {}",
                format!("{}:", mode.command()),
                resolved.family.name(),
                resolved.model
            )
            .map_err(w)?;
        }
    }
    // The axes the presets' needs are on, and what each value means.
    writeln!(out).map_err(w)?;
    writeln!(out, "The axes (`--speech`, `--task`, `--quality`):").map_err(w)?;
    for axis in &construct_config().axes {
        writeln!(out, "  {}: {}", axis.name, axis.description).map_err(w)?;
        for member in &axis.members {
            writeln!(out, "    {:<11} {}", member.name, member.description).map_err(w)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every preset is listed, with both recipes.
    #[test]
    fn test_list() {
        let mut out = Vec::new();
        list(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("en (default)\n"), "{text}");
        for preset in presets() {
            for mode in [Mode::Offline, Mode::Live] {
                assert!(
                    text.contains(&preset.resolve(mode).unwrap().model),
                    "{text}"
                );
            }
        }
        assert!(
            text.contains("  live:       qwen3-asr qwen/qwen3-asr-0.6b\n"),
            "{text}"
        );
        assert!(
            text.contains(
                "  transcribe: whisper openai/whisper/large/large-v3[translate]{beam_size: 5}\n"
            ),
            "{text}"
        );
    }
}
