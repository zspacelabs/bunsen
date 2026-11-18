//! `tts list`.

use bunsen::{
    errors::BunsenResult,
    kits::speech::tts::zonos3::pretrained::{
        DEFAULT_ZONOS3_MODEL,
        default_zonos3_factory,
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

use crate::support::{
    print_model_row,
    print_providers,
};

/// Lists the checkpoints.
#[derive(clap::Args, Debug)]
pub struct TtsListCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Also print each file's first source, where it is fetched from (as
    /// `url: ...`).
    #[arg(long)]
    urls: bool,
}

impl TtsListCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        println!("cache: {}", cache.cache_dir().display());
        println!();
        println!("# zonos3 checkpoints (the default: {DEFAULT_ZONOS3_MODEL})");
        let factory = default_zonos3_factory()?;
        print_providers(factory.providers());
        for id in factory.ids() {
            let (_, row) = factory.lookup(&id)?;
            print_model_row(&cache, &id, &row, DEFAULT_ZONOS3_MODEL, self.urls);
        }
        Ok(())
    }
}
