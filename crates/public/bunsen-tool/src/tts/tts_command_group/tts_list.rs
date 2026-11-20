//! `tts list`.

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
    support::{
        print_model_row,
        print_providers,
    },
    tts::TtsFamily,
};

/// Lists every TTS family's checkpoints.
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
        for (i, family) in TtsFamily::ALL.into_iter().enumerate() {
            if i > 0 {
                println!();
            }
            let default = family.default_model();
            println!("# {} checkpoints (the default: {default})", family.name());
            print_providers(&family.providers()?);
            for (_, row) in family.rows()? {
                print_model_row(&cache, &row.name.to_string(), &row, default, self.urls);
            }
        }
        Ok(())
    }
}
