//! `classify fetch`.

use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LogArgs,
        LogPolicy,
    },
};
use log::LevelFilter;

use crate::classify::image_factory;

/// Fetches checkpoints into the weights cache.
#[derive(clap::Args, Debug)]
pub struct ClassifyFetchCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Checkpoint names, as `--model` takes them (`list` shows them).
    #[arg(required = true)]
    names: Vec<String>,
}

impl ClassifyFetchCommand {
    /// Fetches the checkpoints.
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        let factory = image_factory()?;
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
