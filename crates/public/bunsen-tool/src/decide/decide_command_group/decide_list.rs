//! `decide list`.

use bunsen::{
    errors::BunsenResult,
    kits::decision::laya::pretrained::{
        DEFAULT_LAYA_MODEL,
        default_laya_factory,
    },
};
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LogArgs,
        LogPolicy,
    },
};
use log::LevelFilter;

use crate::support::print_model_row;

/// Lists the checkpoints.
#[derive(clap::Args, Debug)]
pub struct DecideListCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Also print each file's first source, where it is fetched from (as
    /// `url: ...`).
    #[arg(long)]
    urls: bool,
}

impl DecideListCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        println!("cache: {}", cache.cache_dir().display());
        println!();
        println!("# laya checkpoints (the default: {DEFAULT_LAYA_MODEL})");
        let factory = default_laya_factory()?;
        for id in factory.ids() {
            let (_, row) = factory.lookup(&id)?;
            print_model_row(&cache, &id, &row, DEFAULT_LAYA_MODEL, self.urls);
        }
        Ok(())
    }
}
