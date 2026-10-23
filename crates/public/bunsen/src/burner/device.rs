//! # Device Lifetime
//!
//! A [`DeviceSession`] is a [`Device`] that is idle when it goes away: on
//! drop, or [`close`](DeviceSession::close), it waits for the device to
//! finish the work queued on it.
//!
//! A program should hold one for as long as it uses a device, and make it
//! before anything that puts work on the device, so that it is the last to
//! go. Freeing a model's tensors queues work too, and on the wgpu backends
//! (Vulkan, wgpu) that work runs on the runtime's own device thread. If the
//! process exits with that thread still busy, the driver's exit handlers can
//! tear the driver down beneath it: NVIDIA's Vulkan driver then crashes the
//! process (`SIGSEGV`) after its output is written. Waiting first keeps the
//! exit clean on every backend.
//!
//! ```rust,no_run
//! use bunsen::{
//!     burner::device::DeviceSession,
//!     errors::BunsenResult,
//! };
//! use burn::tensor::Device;
//!
//! fn run() -> BunsenResult<()> {
//!     // First, so the last to drop.
//!     let device = DeviceSession::new(Device::default());
//!     // ... build models on `&device`, run them ...
//!     device.close()
//! }
//! ```

use std::ops::Deref;

use burn::tensor::{
    Device,
    TensorCreationOptions,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A [`Device`] that waits for its queued work to finish when dropped.
///
/// Dereferences to the device, so `&session` passes where a `&Device` is
/// taken, and converts to [`TensorCreationOptions`] as a `&Device` does, so
/// tensor constructors take it too. Dropping it waits for the device
/// ([`Device::sync`]) and logs a failure as a warning; [`close`](Self::close)
/// waits and returns the failure instead. See the [module docs](self) for why.
#[derive(Debug)]
pub struct DeviceSession {
    /// `None` once closed.
    device: Option<Device>,
}

impl DeviceSession {
    /// A session over `device`.
    pub fn new(device: Device) -> Self {
        Self {
            device: Some(device),
        }
    }

    /// The device.
    pub fn device(&self) -> &Device {
        self.device
            .as_ref()
            .expect("an open session has its device")
    }

    /// Waits for the device to finish its queued work, and ends the session.
    ///
    /// # Errors
    /// [`Sys`](crate::errors::BunsenErrorKind::Sys), with the backend's
    /// `ExecutionError` as its cause, when queued work failed.
    pub fn close(mut self) -> BunsenResult<()> {
        let device = self.device.take().expect("an open session has its device");
        sync(&device)
    }
}

impl Deref for DeviceSession {
    type Target = Device;

    fn deref(&self) -> &Device {
        self.device()
    }
}

impl From<Device> for DeviceSession {
    fn from(device: Device) -> Self {
        Self::new(device)
    }
}

impl From<&DeviceSession> for TensorCreationOptions {
    fn from(session: &DeviceSession) -> Self {
        session.device().into()
    }
}

impl Drop for DeviceSession {
    fn drop(&mut self) {
        if let Some(device) = self.device.take()
            && let Err(error) = sync(&device)
        {
            log::warn!("{error}");
        }
    }
}

/// Waits for `device` to finish its queued work.
fn sync(device: &Device) -> BunsenResult<()> {
    device
        .sync()
        .map_err(|e| BunsenError::sys("the device failed to finish its queued work").with_cause(e))
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;

    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_close_waits_for_queued_work() {
        let device = DeviceSession::new(cpu_device());
        let x = Tensor::<2>::ones([4, 4], &device);
        let y = x.clone().matmul(x);
        drop(y);
        device.close().unwrap();
    }

    #[test]
    fn test_drop_waits_without_panicking() {
        let device: DeviceSession = cpu_device().into();
        let _x = Tensor::<1>::zeros([8], &device);
    }

    #[test]
    fn test_derefs_to_its_device() {
        let device = DeviceSession::new(cpu_device());
        assert_eq!(&*device, &cpu_device());
        assert_eq!(device.device(), &cpu_device());
    }
}
