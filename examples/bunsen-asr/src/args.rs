//! Arguments every command shares.
use std::path::PathBuf;

use bunsen::{
    data::{
        cache::BunsenDiskCacheOptions,
        pretrained::{
            PretrainedCache,
            PretrainedCacheOptions,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
    },
};
use bunsen_app::device::BackendArgs;
use burn::tensor::Device;

/// Where fetched weights live, and whether fetching is allowed.
#[derive(clap::Args, Debug)]
pub struct WeightsCacheArgs {
    /// Directory for fetched weights; `$BUNSEN_CACHE_DIR`, then the
    /// platform's cache directory, when omitted.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    /// Never reach the network: a model not already local is an error.
    #[arg(long)]
    pub offline: bool,
}

impl WeightsCacheArgs {
    /// Opens the cache.
    pub fn init(&self) -> BunsenResult<PretrainedCache> {
        let options = PretrainedCacheOptions::default()
            .with_disk(BunsenDiskCacheOptions::default().with_cache_dir(self.cache_dir.clone()))
            .with_offline(self.offline);
        // The options come from the command line.
        PretrainedCache::new(options).as_policy()
    }
}

/// The device to run on.
#[derive(clap::Args, Debug)]
pub struct DeviceArgs {
    #[command(flatten)]
    backend: BackendArgs,
}

impl DeviceArgs {
    /// The device `--device` names.
    ///
    /// # Errors
    /// [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported) when that
    /// backend is not compiled in.
    pub fn device(&self) -> BunsenResult<Device> {
        self.backend.init().map_err(BunsenError::unsupported)
    }
}
