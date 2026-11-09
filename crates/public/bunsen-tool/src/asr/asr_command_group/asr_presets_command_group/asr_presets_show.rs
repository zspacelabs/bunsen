//! `asr presets show`.

use std::io::Write;

use bunsen::errors::{
    BunsenResult,
    sys_at,
};

use super::title;
use crate::asr::{
    AsrPreset,
    Mode,
    lookup,
    presets,
    shell_line,
};

/// Shows presets: each recipe, as the `bunsen-tool asr family` command it
/// runs.
#[derive(clap::Args, Debug)]
pub struct AsrPresetsShowCommand {
    /// Presets, by name or alias; every preset when none is named.
    names: Vec<String>,
}

impl AsrPresetsShowCommand {
    pub fn run(&self) -> BunsenResult<()> {
        let presets: Vec<&AsrPreset> = if self.names.is_empty() {
            presets().iter().collect()
        } else {
            self.names
                .iter()
                .map(|n| lookup(n))
                .collect::<BunsenResult<_>>()?
        };
        let mut out = std::io::stdout().lock();
        let w = |e| sys_at("write to", "stdout")(e);
        for (i, preset) in presets.into_iter().enumerate() {
            if i > 0 {
                writeln!(out).map_err(w)?;
            }
            writeln!(out, "{}", title(preset)).map_err(w)?;
            writeln!(out, "  {}", preset.description).map_err(w)?;
            for mode in [Mode::Offline, Mode::Live] {
                let resolved = preset.resolve(mode)?;
                writeln!(
                    out,
                    "  {}: {}",
                    mode.command(),
                    shell_line(&resolved.command_line())
                )
                .map_err(w)?;
            }
        }
        Ok(())
    }
}
