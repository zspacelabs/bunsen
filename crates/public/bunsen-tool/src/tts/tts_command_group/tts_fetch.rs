//! `tts fetch`.

use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LogArgs,
        LogPolicy,
    },
};
use log::LevelFilter;

use crate::tts::{
    TtsFamily,
    tts_family_of,
};

/// Fetches checkpoints into the weights cache.
#[derive(clap::Args, Debug)]
pub struct TtsFetchCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// The family the names are of.
    ///
    /// The family that lists each, when omitted. Needed for a provider ref
    /// (`hf:org/repo`).
    #[arg(long, value_enum)]
    family: Option<TtsFamily>,

    /// Checkpoint names, as `--model` takes them (`list` shows them).
    #[arg(required = true)]
    names: Vec<String>,
}

impl TtsFetchCommand {
    pub fn run(&self) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Info, LevelFilter::Info))?;
        let cache = self.cache.init()?;
        for name in &self.names {
            let family = match self.family {
                Some(family) => family,
                None => tts_family_of(name)?,
            };
            let (id, parts) = family.fetch(name, &cache)?;
            println!("{id}:");
            for (key, path) in parts {
                println!("  {key}: {}", path.display());
            }
        }
        Ok(())
    }
}
