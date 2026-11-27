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
    classify::DEFAULT_IMAGE_MODEL,
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
        for kit in bunsen::kits::images::classification::pretrained::image_kit_factories()? {
            println!();
            println!("## {}", kit.kit());
            print_providers(&kit.providers());
            for (id, row) in &kit.rows()? {
                print_model_row(&cache, id, row, DEFAULT_IMAGE_MODEL, self.urls);
            }
        }
        Ok(())
    }
}
