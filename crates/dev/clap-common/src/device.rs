//! Run-time device selection for the example binaries.
//!
//! [`DeviceArgs`] adds `--device` and `--device-index` to a clap app and
//! builds the [`Device`] they name. A backend is selectable only when the
//! binary was built with its feature (`cuda`, `metal`, `vulkan`, `wgpu`),
//! forwarded to this crate; `flex`, the CPU, is always there. `auto` takes
//! the first compiled-in accelerator in that order, else the CPU.

use burn::tensor::Device;

/// A backend to run on.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The first compiled-in accelerator: `cuda`, `metal`, `vulkan`, then
    /// `wgpu`; the CPU when there is none.
    #[default]
    Auto,

    /// CUDA.
    Cuda,

    /// Metal, through wgpu.
    Metal,

    /// Vulkan, through wgpu.
    Vulkan,

    /// wgpu, on whichever graphics API it settles on.
    Wgpu,

    /// The CPU backend.
    Flex,
}

/// Device selection arguments, for `#[command(flatten)]`.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct DeviceArgs {
    /// The backend to run on.
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    pub device: DeviceChoice,

    /// Which device of the backend; the backend's default when omitted.
    #[arg(long)]
    pub device_index: Option<usize>,
}

impl DeviceArgs {
    /// The backend `auto` resolves to in this build.
    pub fn auto_choice() -> DeviceChoice {
        if cfg!(feature = "cuda") {
            DeviceChoice::Cuda
        } else if cfg!(feature = "metal") {
            DeviceChoice::Metal
        } else if cfg!(feature = "vulkan") {
            DeviceChoice::Vulkan
        } else if cfg!(feature = "wgpu") {
            DeviceChoice::Wgpu
        } else {
            DeviceChoice::Flex
        }
    }

    /// The selected backend, with `auto` resolved.
    pub fn choice(&self) -> DeviceChoice {
        match self.device {
            DeviceChoice::Auto => Self::auto_choice(),
            choice => choice,
        }
    }

    /// Builds the selected device.
    ///
    /// # Errors
    /// When the selected backend's feature is not compiled in.
    pub fn init(&self) -> Result<Device, String> {
        let choice = self.choice();
        #[allow(unused_variables)]
        let index = self.device_index;
        let not_built = |name: &str| {
            Err(format!(
                "the `{name}` backend is not compiled in; rebuild with `--features {name}`"
            ))
        };
        match choice {
            DeviceChoice::Auto => unreachable!("resolved above"),
            DeviceChoice::Flex => Ok(Device::flex()),
            DeviceChoice::Cuda => {
                #[cfg(feature = "cuda")]
                return Ok(Device::cuda(index.unwrap_or(0)));
                #[cfg(not(feature = "cuda"))]
                not_built("cuda")
            }
            DeviceChoice::Metal => {
                #[cfg(feature = "metal")]
                return Ok(Device::metal(wgpu_kind(index)));
                #[cfg(not(feature = "metal"))]
                not_built("metal")
            }
            DeviceChoice::Vulkan => {
                #[cfg(feature = "vulkan")]
                return Ok(Device::vulkan(wgpu_kind(index)));
                #[cfg(not(feature = "vulkan"))]
                not_built("vulkan")
            }
            DeviceChoice::Wgpu => {
                #[cfg(feature = "wgpu")]
                return Ok(Device::wgpu(wgpu_kind(index)));
                #[cfg(not(feature = "wgpu"))]
                not_built("wgpu")
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flex_always_builds() {
        let args = DeviceArgs {
            device: DeviceChoice::Flex,
            device_index: None,
        };
        assert!(!args.init().unwrap().is_autodiff());
    }
}
