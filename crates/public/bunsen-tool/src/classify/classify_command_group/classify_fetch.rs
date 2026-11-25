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

use crate::classify::fetch_image_model;

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
        for name in &self.names {
            let (id, loaded) = fetch_image_model(name, &cache)?;
            println!("{id}:");
            for (key, part) in loaded.iter() {
                println!("  {key}: {}", part.path.display());
            }
        }
        Ok(())
    }
}
