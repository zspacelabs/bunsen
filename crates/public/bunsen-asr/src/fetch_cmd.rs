//! `bunsen-asr <family> fetch`.
use bunsen::errors::BunsenResult;
use bunsen_app::{
    cache::WeightsCacheArgs,
    logging::{
        LevelFilter,
        LogArgs,
        LogPolicy,
    },
};

use crate::family::Family;

/// Brings every file of the named models into the weights cache, fetching
/// what is not local, and prints where each one is: a deployment that
/// runs offline fills its cache this way ahead of time.
#[derive(clap::Args, Debug)]
pub struct FetchCmd {
    #[clap(flatten)]
    pub logging: LogArgs,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    /// Re-hash each pinned file once it is local, the cached ones included.
    #[arg(long)]
    pub verify: bool,

    /// Model names, as `--model` takes them (`list` shows them).
    #[arg(required = true)]
    pub names: Vec<String>,
}

impl FetchCmd {
    pub fn run(
        &self,
        family: Family,
    ) -> BunsenResult<()> {
        self.logging
            .init(LogPolicy::new(LevelFilter::Warn, LevelFilter::Warn))?;
        let cache = self.cache.init()?;
        family.fetch(
            &self.names,
            &cache,
            self.verify,
            &mut std::io::stdout().lock(),
        )
    }
}
