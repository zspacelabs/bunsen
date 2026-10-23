//! Arguments every command shares, but the weights cache's.
use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use bunsen_app::device::{
    BackendArgs,
    DeviceSession,
};

/// The device to run on.
#[derive(clap::Args, Debug)]
pub struct DeviceArgs {
    #[command(flatten)]
    backend: BackendArgs,
}

impl DeviceArgs {
    /// The device `--device` names, as a [`DeviceSession`]: a command opens
    /// it first, so it goes last, and the process exits with the device idle.
    ///
    /// # Errors
    /// [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported) when that
    /// backend is not compiled in.
    pub fn open(&self) -> BunsenResult<DeviceSession> {
        self.backend.open().map_err(BunsenError::unsupported)
    }
}
