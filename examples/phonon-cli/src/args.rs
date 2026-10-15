//! Arguments shared by the commands.
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

use crate::transcriber::{
    DEFAULT_MODEL,
    Transcriber,
};

/// Where fetched weights live, and whether fetching is allowed.
#[derive(clap::Args, Debug)]
pub struct WeightsCacheArgs {
    /// Directory for fetched weights; `$BUNSEN_CACHE_DIR`, then the
    /// platform's cache directory, when omitted.
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Never reach the network: a model not already local is an error.
    #[arg(long)]
    offline: bool,
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

/// Which model, on which device.
#[derive(clap::Args, Debug)]
pub struct ModelArgs {
    /// The model, by name: `models list` shows them. Phonon-2
    /// (`fermion/phonon-2`) and Phonon-1 (`fermion/phonon-1`, `-micro`,
    /// `-big`) are `FermionResearch`'s.
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,

    #[clap(flatten)]
    pub cache: WeightsCacheArgs,

    /// The device to run the model on.
    #[command(flatten)]
    device: BackendArgs,
}

impl ModelArgs {
    /// The device `--device` names.
    ///
    /// # Errors
    /// [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported) when that
    /// backend is not compiled in.
    pub fn device(&self) -> BunsenResult<Device> {
        self.device.init().map_err(BunsenError::unsupported)
    }

    /// Loads `--model` through the cache, from whichever kit knows it.
    pub fn load(
        &self,
        device: &Device,
    ) -> BunsenResult<Transcriber> {
        let cache = self.cache.init()?;
        Transcriber::load(&self.model, &cache, device)
    }
}
