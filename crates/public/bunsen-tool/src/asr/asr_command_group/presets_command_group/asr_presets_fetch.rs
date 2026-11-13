//! `asr presets fetch`.

use bunsen::{
    errors::BunsenResult,
    kits::speech::vad::silero_vad::pretrained::default_silero_factory,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::asr::{
    AsrFamily,
    AsrKit,
    Emission,
    Mode,
    construct_config,
    fetch_with,
    lookup,
};

/// Brings every file the named presets run into the weights cache, and
/// prints where each one is: a deployment that runs offline fills its cache
/// this way ahead of time.
#[derive(clap::Args, Debug)]
pub struct AsrPresetsFetchCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Re-hash each pinned file once it is local, the cached ones included.
    #[arg(long)]
    verify: bool,

    /// Presets, by name or alias (`presets` lists them).
    #[arg(required = true)]
    names: Vec<String>,
}

impl AsrPresetsFetchCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        let mut models: Vec<(AsrFamily, String)> = Vec::new();
        let mut vad = false;
        for name in &self.names {
            let preset = lookup(name)?;
            for mode in [Mode::Offline, Mode::Live] {
                let recipe = preset.recipe(mode)?;
                let model = (recipe.engine.family, recipe.engine.model);
                if !models.contains(&model) {
                    models.push(model);
                }
                vad |= recipe.emission != Emission::Offline;
            }
        }
        let mut out = std::io::stdout().lock();
        for (family, model) in models {
            AsrKit::Asr(family).fetch(&[model], &cache, self.verify, &mut out)?;
        }
        if vad {
            fetch_with(
                default_silero_factory()?,
                &[construct_config().vad.clone()],
                &cache,
                self.verify,
                &mut out,
            )?;
        }
        Ok(())
    }
}
