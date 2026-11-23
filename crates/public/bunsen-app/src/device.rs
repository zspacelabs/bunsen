//! Run-time device selection for the example binaries.
//!
//! - [`BackendArgs`]: `--device` and `--device-index`, the backend to run on,
//!   and `--cpus`, where the process's threads run
//!   ([`placement`](crate::placement)). A backend is selectable only when the
//!   binary was built with its feature (`cuda`, `metal`, `vulkan`, `wgpu`),
//!   forwarded to this crate; `flex`, the CPU, is always there. `auto` takes
//!   the first compiled-in accelerator in that order, else the CPU.
//! - [`DeviceArgs`]: the backend, built with the program's [`DevicePrefs`] (its
//!   int dtype, autodiff for training) by [`DeviceArgs::init`].
//!
//! The device's default float dtype stays f32: a program names the dtypes it
//! wants where it makes or loads tensors ([`precision`](crate::precision):
//! `--init-dtype`, `--min-precision`), never through the device.
//!
//! # Exiting with the device idle
//!
//! A program should hold its device as a [`DeviceSession`], made before
//! anything that puts work on it: [`BackendArgs::open`] and
//! [`DeviceArgs::open`] build one. It waits for the device's queued work when
//! it goes away, so the process exits with the device idle; on NVIDIA's
//! Vulkan driver, exiting with the wgpu runtime's device thread still busy
//! crashes the process after its output is written.

pub use bunsen::burner::device::DeviceSession;
use bunsen::errors::{
    BunsenError,
    BunsenResult,
};
use burn::tensor::{
    DType,
    Device,
    DeviceConfig,
    DeviceError,
    FloatDType,
    IntDType,
};

use crate::placement::Placement;

/// A backend to run on.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The first compiled-in accelerator: `cuda`, `metal`, `vulkan`, then
    /// `wgpu`; the CPU when there is none.
    #[default]
    Auto,

    /// CUDA.
    #[cfg(feature = "cuda")]
    Cuda,

    /// Metal, through wgpu.
    #[cfg(feature = "metal")]
    Metal,

    /// Vulkan, through wgpu.
    #[cfg(feature = "vulkan")]
    Vulkan,

    /// wgpu, on whichever graphics API it settles on.
    #[cfg(feature = "wgpu")]
    Wgpu,

    /// The CPU backend.
    Flex,
}

/// Backend selection arguments, for `#[command(flatten)]`.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct BackendArgs {
    /// The backend to run on.
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    pub device: DeviceChoice,

    /// Which device of the backend; the backend's default when omitted.
    #[arg(long)]
    pub device_index: Option<usize>,

    /// The CPUs the process runs on: `auto`, `all`, or a list (`0-3,64-67`).
    ///
    /// `auto` pins an accelerator's process to the CPUs sharing one L3 cache,
    /// where the host thread and the device's runtime thread hand work back and
    /// forth fastest; the CPU backend is not pinned. `all` never pins. Linux
    /// only; elsewhere `auto` does nothing.
    #[arg(long, default_value_t = Placement::Auto)]
    pub cpus: Placement,
}

impl BackendArgs {
    /// The backend `auto` resolves to in this build.
    pub fn auto_choice() -> DeviceChoice {
        let accelerators: &[DeviceChoice] = &[
            #[cfg(feature = "cuda")]
            DeviceChoice::Cuda,
            #[cfg(feature = "metal")]
            DeviceChoice::Metal,
            #[cfg(feature = "vulkan")]
            DeviceChoice::Vulkan,
            #[cfg(feature = "wgpu")]
            DeviceChoice::Wgpu,
        ];
        accelerators.first().copied().unwrap_or(DeviceChoice::Flex)
    }

    /// The selected backend, with `auto` resolved.
    pub fn choice(&self) -> DeviceChoice {
        match self.device {
            DeviceChoice::Auto => Self::auto_choice(),
            choice => choice,
        }
    }

    /// Builds the selected device, with the backend's default dtypes, after
    /// placing the process's threads (`--cpus`, [`Placement::apply`]).
    ///
    /// Only compiled-in backends are [`DeviceChoice`] variants, so this does
    /// not fail; it returns a `Result` for the backends that someday might.
    ///
    /// # Errors
    /// As [`Placement::apply`]; a backend that someday can fail to start
    /// reports [`Unsupported`](bunsen::errors::BunsenErrorKind::Unsupported).
    pub fn init(&self) -> BunsenResult<Device> {
        #[allow(unused_variables)]
        let index = self.device_index;
        self.cpus.apply(self.choice() != DeviceChoice::Flex)?;
        Ok(match self.choice() {
            DeviceChoice::Auto => unreachable!("resolved by `choice`"),
            DeviceChoice::Flex => Device::flex(),
            #[cfg(feature = "cuda")]
            DeviceChoice::Cuda => Device::cuda(index.unwrap_or(0)),
            #[cfg(feature = "metal")]
            DeviceChoice::Metal => Device::metal(wgpu_kind(index)),
            #[cfg(feature = "vulkan")]
            DeviceChoice::Vulkan => Device::vulkan(wgpu_kind(index)),
            #[cfg(feature = "wgpu")]
            DeviceChoice::Wgpu => Device::wgpu(wgpu_kind(index)),
        })
    }

    /// [`init`](Self::init)'s device, as a [`DeviceSession`]: hold it for as
    /// long as the device is used, made before anything that uses it.
    ///
    /// # Errors
    /// As [`init`](Self::init).
    pub fn open(&self) -> BunsenResult<DeviceSession> {
        self.init().map(DeviceSession::new)
    }
}

/// The wgpu device kind for an index: the default adapter, or that discrete
/// GPU.
#[cfg(any(feature = "metal", feature = "vulkan", feature = "wgpu"))]
fn wgpu_kind(index: Option<usize>) -> burn::tensor::DeviceKind {
    match index {
        None => burn::tensor::DeviceKind::DefaultDevice,
        Some(i) => burn::tensor::DeviceKind::DiscreteGpu(i),
    }
}

/// What a program wants of its device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DevicePrefs {
    /// Int dtypes in order of preference; the first the device supports is
    /// the default. Empty keeps the backend's default.
    pub ints: Vec<IntDType>,

    /// Whether the device records gradients, for training.
    pub autodiff: bool,
}

impl DevicePrefs {
    /// The backend's default int dtype; no autodiff.
    pub fn new() -> Self {
        Self::default()
    }

    /// Preferences for a training run: as [`new`](Self::new), with autodiff.
    pub fn training() -> Self {
        Self::new().with_autodiff(true)
    }

    /// Sets the int dtypes, in order of preference.
    pub fn with_ints(
        mut self,
        ints: impl IntoIterator<Item = IntDType>,
    ) -> Self {
        self.ints = ints.into_iter().collect();
        self
    }

    /// Sets whether the device records gradients.
    pub fn with_autodiff(
        mut self,
        autodiff: bool,
    ) -> Self {
        self.autodiff = autodiff;
        self
    }
}

/// Device arguments, for `#[command(flatten)]`: the backend, built with the
/// program's [`DevicePrefs`].
#[derive(clap::Args, Debug, Clone, Default)]
pub struct DeviceArgs {
    /// The backend.
    #[command(flatten)]
    pub backend: BackendArgs,
}

impl DeviceArgs {
    /// Builds the device the flags select: its default int dtype the first
    /// of `prefs`' the device supports (the backend's without one), its
    /// default float f32, and autodiff when `prefs` asks for it.
    ///
    /// A device's default dtypes can be set once per process, before its
    /// first tensor. A second `init` of the same device succeeds when it asks
    /// for the dtypes the device already has.
    ///
    /// # Errors
    /// [`Illegal`](bunsen::errors::BunsenErrorKind::Illegal) when the device
    /// already has other dtypes.
    pub fn init(
        &self,
        prefs: &DevicePrefs,
    ) -> BunsenResult<Device> {
        let mut device = self.backend.init()?;
        let mut config = DeviceConfig::default().float_dtype(FloatDType::F32);
        if let Some(int) = prefs
            .ints
            .iter()
            .find(|dtype| device.supports_dtype(DType::from(**dtype)))
        {
            config = config.int_dtype(*int);
        }
        let (float, int) = (config.float_dtype, config.int_dtype);

        match device.configure(config) {
            Ok(()) => {}
            Err(DeviceError::AlreadyInitialized { .. })
                if Some(device.settings().float_dtype) == float
                    && int.is_none_or(|int| device.settings().int_dtype == int) => {}
            Err(e) => return Err(BunsenError::illegal(e.to_string())),
        }
        let device = if prefs.autodiff {
            device.autodiff()
        } else {
            device
        };
        log::info!("{}", describe(&device));
        Ok(device)
    }

    /// [`init`](Self::init)'s device, as a [`DeviceSession`]: hold it for as
    /// long as the device is used, made before anything that uses it.
    ///
    /// # Errors
    /// As [`init`](Self::init).
    pub fn open(
        &self,
        prefs: &DevicePrefs,
    ) -> BunsenResult<DeviceSession> {
        self.init(prefs).map(DeviceSession::new)
    }
}

/// A one-line description of a device and its default dtypes, for logs.
pub fn describe(device: &Device) -> String {
    let settings = device.settings();
    format!(
        "device {device:?}: float {:?}, int {:?}{}",
        settings.float_dtype,
        settings.int_dtype,
        if device.is_autodiff() {
            ", autodiff"
        } else {
            ""
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU device: f32 floats, the first preferred int it supports,
    /// autodiff when asked.
    #[test]
    fn test_init() {
        let args = DeviceArgs {
            backend: BackendArgs {
                device: DeviceChoice::Flex,
                device_index: None,
                cpus: Placement::Auto,
            },
        };
        let device = args
            .init(&DevicePrefs::new().with_ints([IntDType::I32]))
            .unwrap();
        assert_eq!(device.settings().float_dtype, FloatDType::F32);
        assert_eq!(device.settings().int_dtype, IntDType::I32);
        assert!(!device.is_autodiff());
        let device = args
            .init(&DevicePrefs::training().with_ints([IntDType::I32]))
            .unwrap();
        assert!(device.is_autodiff());
    }
}
