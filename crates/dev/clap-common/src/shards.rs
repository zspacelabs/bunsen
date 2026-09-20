//! Shard-set arguments: which shards, where they sit, and how to fetch them.

use std::path::PathBuf;

use bunsen::{
    data::{
        cache::{
            BunsenDiskCache,
            FetchPolicy,
            OnFailure,
        },
        shards::{
            ShardId,
            ShardSet,
            ShardSetDescriptor,
        },
    },
    errors::BunsenResult,
};
use burn::tensor::Slice;

/// Shard selection and fetch-policy arg group.
///
/// # Example
///
/// ```rust,ignore
/// pub struct Args {
///    # ...
///
///    #[clap(flatten)]
///    pub shards: ShardArgs,
/// }
///
/// fn main() -> anyhow::Result<()> {
///    let args = Args::parse();
///    let cache = BunsenDiskCache::default();
///    let paths = args.shards.fetch_paths(
///        &cache,
///        NANOCHAT_SHARD_SETS.expect_lookup("fineweb-edu-100b-shuffle"),
///    )?;
///    ...
/// }
/// ```
#[derive(clap::Args, Debug, Clone)]
pub struct ShardArgs {
    /// Shards to load: slices, comma separated (`0`, `..8`, `3..5,10`).
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    pub shards: Vec<Slice>,

    /// Directory the shard files sit in directly. Without it, the set lives
    /// under bunsen's data directory, at `shards/<name>/`.
    #[arg(long)]
    pub dataset_dir: Option<PathBuf>,

    /// Shards fetched at once.
    #[arg(long, default_value_t = 8)]
    pub parallel: usize,

    /// Extra attempts per shard after the first.
    #[arg(long, default_value_t = 0)]
    pub retries: u32,

    /// Keep fetching after a shard fails, and report at the end.
    #[arg(long)]
    pub keep_going: bool,
}

impl ShardArgs {
    /// The fetch policy the flags spell.
    pub fn policy(&self) -> FetchPolicy {
        FetchPolicy::default()
            .with_parallel(self.parallel)
            .with_retries(self.retries)
            .with_on_failure(if self.keep_going {
                OnFailure::Continue
            } else {
                OnFailure::Stop
            })
    }

    /// `desc` bound to `--dataset-dir`, or under the cache's data directory.
    pub fn shard_set<'c>(
        &self,
        cache: &'c BunsenDiskCache,
        desc: ShardSetDescriptor,
    ) -> ShardSet<'c> {
        match &self.dataset_dir {
            Some(dir) => ShardSet::at_dir(cache, desc, dir.clone()),
            None => ShardSet::in_cache(cache, desc),
        }
    }

    /// The ids `--shards` names within `desc`.
    ///
    /// # Errors
    /// As [`ShardSetDescriptor::select`].
    pub fn select(
        &self,
        desc: &ShardSetDescriptor,
    ) -> BunsenResult<Vec<ShardId>> {
        desc.select(&self.shards)
    }

    /// Selects the shards, brings them in under the policy, logs the report,
    /// and returns their paths in id order.
    ///
    /// # Errors
    /// A bad selection, or a shard that did not land: the error names each.
    pub fn fetch_paths(
        &self,
        cache: &BunsenDiskCache,
        desc: ShardSetDescriptor,
    ) -> BunsenResult<Vec<PathBuf>> {
        let ids = self.select(&desc)?;
        let set = self.shard_set(cache, desc);
        log::info!(
            "{}: fetching {} shards into {}",
            set.descriptor().name,
            ids.len(),
            set.root().display()
        );
        let report = set.fetch_many(&ids, &self.policy())?;
        log::info!("{}: {report}", set.descriptor().name);
        report.paths()
    }
}
