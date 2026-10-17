//! Arguments every command shares; the weights cache's are
//! [`bunsen_app::cache::WeightsCacheArgs`].
use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use bunsen_app::device::BackendArgs;
use burn::tensor::Device;

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
