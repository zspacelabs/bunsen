//! `chat fetch`.

use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::chat::{
    ChatModelFamily,
    choose_checkpoint,
};

/// Fetches checkpoints into the weights cache.
#[derive(clap::Args, Debug)]
pub struct ChatFetchCommand {
    #[clap(flatten)]
    logging: LogArgs,

    #[clap(flatten)]
    cache: WeightsCacheArgs,

    /// Checkpoints, each as `--select` takes one (`list` shows the names).
    #[arg(required = true)]
    names: Vec<String>,
}

impl ChatFetchCommand {
    /// Fetches each name: `family`'s, or outside `family` its own family's.
    pub fn run(
        &self,
        family: Option<ChatModelFamily>,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        for name in &self.names {
            let (family, name) = choose_checkpoint(family, std::slice::from_ref(name), &[])?;
            family.fetch(&name, &cache)?;
        }
        Ok(())
    }
}
