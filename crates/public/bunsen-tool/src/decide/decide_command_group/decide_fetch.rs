//! `decide fetch`.

use bunsen::{
    errors::BunsenResult,
    kits::decision::laya::pretrained::default_laya_factory,
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LogArgs,
        LogPolicy,
    },
};
use log::LevelFilter;

/// Fetches checkpoints into the weights cache.
#[derive(clap::Args, Debug)]
pub struct DecideFetchCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Checkpoint names, as `--model` takes them (`list` shows them).
    #[arg(required = true)]
    names: Vec<String>,
}

impl DecideFetchCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        let factory = default_laya_factory()?;
        for name in &self.names {
            let model = factory.resolve(name, &cache)?;
            let map = model.plan(&cache)?;
            let loaded = cache.load(&map)?;
            println!("{}:", model.id());
            for (key, part) in loaded.iter() {
                println!("  {key}: {}", part.path.display());
            }
        }
        Ok(())
    }
}
