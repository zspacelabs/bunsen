//! `classify list`.

use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LogArgs,
        LogPolicy,
    },
};
use log::LevelFilter;

use crate::{
    classify::{
        DEFAULT_IMAGE_MODEL,
        image_factory,
    },
    support::{
        print_model_row,
        print_providers,
    },
};

/// Lists the checkpoints.
#[derive(clap::Args, Debug)]
pub struct ClassifyListCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Also print each file's first source, where it is fetched from (as
    /// `url: ...`).
    #[arg(long)]
    urls: bool,
}

impl ClassifyListCommand {
    /// Prints the checkpoints.
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        println!("cache: {}", cache.cache_dir().display());
        println!();
        println!("# image checkpoints (the default: {DEFAULT_IMAGE_MODEL})");
        let factory = image_factory()?;
        print_providers(factory.providers());
        for id in factory.ids() {
            let (_, row) = factory.lookup(&id)?;
            print_model_row(&cache, &id, &row, DEFAULT_IMAGE_MODEL, self.urls);
        }
        Ok(())
    }
}
