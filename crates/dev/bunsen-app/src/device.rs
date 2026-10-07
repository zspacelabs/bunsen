//! Run-time device and precision selection for the example binaries.
//!
//! Two argument groups, which a binary stacks as it needs:
//!
//! - [`BackendArgs`]: `--device` and `--device-index`, the backend to run on. A
//!   backend is selectable only when the binary was built with its feature
//!   (`cuda`, `metal`, `vulkan`, `wgpu`), forwarded to this crate; `flex`, the
//!   CPU, is always there. `auto` takes the first compiled-in accelerator in
//!   that order, else the CPU.
//! - [`PrecisionArgs`]: `--precision`, `--float-dtype` and `--int-dtype`, the
//!   dtypes the device's tensors default to.
//!
//! [`DeviceArgs`] flattens both, and [`DeviceArgs::init`] builds the device
//! from them and the program's [`DevicePrefs`].
//!
//! # Precision
//!
//! "Half precision" is two questions. *Which* 16-bit float: `bf16` keeps
//! `f32`'s range and trains without loss scaling; `f16` has more mantissa and
//! less range. And *whether it pays*: a backend can support a dtype without
//! running it fast (the CPU runs both halves, slowly). So a program does not
//! name a dtype; it names a [`Precision`], an intent, and this module resolves
//! it per backend:
//!
//! | intent                    | cuda | metal          | vulkan, wgpu | flex (CPU) |
//! | ------------------------- | ---- | -------------- | ------------ | ---------- |
//! | [`Precision::Full`]       | f32  | f32            | f32          | f32        |
//! | [`Precision::BF16Half`]   | bf16 | bf16, else f32 | f32          | f32        |
//! | [`Precision::AnyHalf`]    | f16  | f16, else f32  | f16          | f32        |
//!
//! Each half falls back to `f32` where the device does not support it
//! ([`Device::supports_dtype`]). `--float-dtype` overrides the table with an
//! exact dtype, and is an error where the device does not support it.

use burn::tensor::{
    DType,
    Device,
    DeviceConfig,
    DeviceError,
    FloatDType,
    IntDType,
};

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

/// Backend selection arguments, for `#[command(flatten)]`.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct BackendArgs {
    /// The backend to run on.
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    pub device: DeviceChoice,

    /// Which device of the backend; the backend's default when omitted.
    #[arg(long)]
    pub device_index: Option<usize>,
}

impl BackendArgs {
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

    /// Builds the selected device, with the backend's default dtypes.
    ///
    /// # Errors
    /// When the selected backend's feature is not compiled in.
    pub fn init(&self) -> Result<Device, String> {
        #[allow(unused_variables)]
        let index = self.device_index;
        let not_built = |name: &str| {
            Err(format!(
                "the `{name}` backend is not compiled in; rebuild with `--features {name}`"
            ))
        };
        match self.choice() {
            DeviceChoice::Auto => unreachable!("resolved by `choice`"),
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

/// A precision intent: what a program wants of its floats, resolved per
/// backend. See the [module docs](self#precision).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// `f32` everywhere.
    #[default]
    Full,

    /// `bf16` where the backend runs it well, else `f32`: half precision with
    /// `f32`'s range, for training.
    BF16Half,

    /// `f16` where the backend runs it well, else `f32`: any 16-bit float,
    /// for values that fit `f16`'s range.
    AnyHalf,
}

/// `--precision`: which of the program's intents to use.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrecisionChoice {
    /// The program's default intent.
    #[default]
    Auto,

    /// `f32`.
    Full,

    /// The program's half-precision intent.
    Half,
}

/// `--float-dtype`: an exact float dtype.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatChoice {
    /// 64-bit float.
    F64,

    /// 32-bit float.
    F32,

    /// 16-bit float.
    F16,

    /// 16-bit brain float.
    Bf16,
}

impl From<FloatChoice> for FloatDType {
    fn from(choice: FloatChoice) -> Self {
        match choice {
            FloatChoice::F64 => FloatDType::F64,
            FloatChoice::F32 => FloatDType::F32,
            FloatChoice::F16 => FloatDType::F16,
            FloatChoice::Bf16 => FloatDType::BF16,
        }
    }
}

/// `--int-dtype`: an exact int dtype.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntChoice {
    /// 64-bit int.
    I64,

    /// 32-bit int.
    I32,

    /// 16-bit int.
    I16,

    /// 8-bit int.
    I8,
}

impl From<IntChoice> for IntDType {
    fn from(choice: IntChoice) -> Self {
        match choice {
            IntChoice::I64 => IntDType::I64,
            IntChoice::I32 => IntDType::I32,
            IntChoice::I16 => IntDType::I16,
            IntChoice::I8 => IntDType::I8,
        }
    }
}

/// Precision arguments, for `#[command(flatten)]`.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct PrecisionArgs {
    /// The float precision: `auto` is the program's choice, `full` is f32,
    /// and `half` is the program's half precision (bf16 or f16, per backend;
    /// f32 where the backend does not run it well).
    #[arg(long, value_enum, default_value_t = PrecisionChoice::Auto)]
    pub precision: PrecisionChoice,

    /// An exact default float dtype, overriding `--precision`; an error where
    /// the device does not support it.
    #[arg(long, value_enum)]
    pub float_dtype: Option<FloatChoice>,

    /// An exact default int dtype, overriding the program's choice; an error
    /// where the device does not support it.
    #[arg(long, value_enum)]
    pub int_dtype: Option<IntChoice>,
}

/// What a program wants of its device, before the flags have their say.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DevicePrefs {
    /// The intent `--precision auto` uses.
    pub precision: Precision,

    /// The intent `--precision half` uses.
    pub half: Precision,

    /// Int dtypes in order of preference; the first the device supports is
    /// the default. Empty keeps the backend's default.
    pub ints: Vec<IntDType>,

    /// Whether the device records gradients, for training.
    pub autodiff: bool,
}

impl DevicePrefs {
    /// Full precision; half means bf16; no autodiff.
    pub fn new() -> Self {
        Self {
            precision: Precision::Full,
            half: Precision::BF16Half,
            ints: Vec::new(),
            autodiff: false,
        }
    }

    /// Preferences for a training run: as [`new`](Self::new), with autodiff.
    pub fn training() -> Self {
        Self::new().with_autodiff(true)
    }

    /// Sets the intent `--precision auto` uses.
    pub fn with_precision(
        mut self,
        precision: Precision,
    ) -> Self {
        self.precision = precision;
        self
    }

    /// Sets the intent `--precision half` uses.
    pub fn with_half(
        mut self,
        half: Precision,
    ) -> Self {
        self.half = half;
        self
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

/// The float dtype an intent resolves to on a backend, given what the device
/// supports.
pub fn resolve_precision(
    precision: Precision,
    backend: DeviceChoice,
    supports: impl Fn(DType) -> bool,
) -> FloatDType {
    use DeviceChoice::*;
    use FloatDType::*;

    let wanted = match (precision, backend) {
        (Precision::Full, _) | (_, Flex) => F32,
        (Precision::BF16Half, Cuda | Metal) => BF16,
        (Precision::BF16Half, Vulkan | Wgpu) => F32,
        (Precision::AnyHalf, Cuda | Metal | Vulkan | Wgpu) => F16,
        (_, Auto) => unreachable!("resolved before the table"),
    };
    if supports(wanted.into()) { wanted } else { F32 }
}

/// The device config the flags and preferences resolve to on a backend.
///
/// # Errors
/// When an exact `--float-dtype` or `--int-dtype` is one the device does not
/// support.
pub fn resolve_config(
    args: &PrecisionArgs,
    prefs: &DevicePrefs,
    backend: DeviceChoice,
    supports: impl Fn(DType) -> bool,
) -> Result<DeviceConfig, String> {
    let float = match args.float_dtype {
        Some(choice) => {
            let dtype = FloatDType::from(choice);
            if !supports(dtype.into()) {
                return Err(format!("the {backend:?} device does not support {dtype:?}"));
            }
            dtype
        }
        None => {
            let intent = match args.precision {
                PrecisionChoice::Auto => prefs.precision,
                PrecisionChoice::Full => Precision::Full,
                PrecisionChoice::Half => prefs.half,
            };
            resolve_precision(intent, backend, &supports)
        }
    };

    let mut config = DeviceConfig::default().float_dtype(float);
    match args.int_dtype {
        Some(choice) => {
            let dtype = IntDType::from(choice);
            if !supports(dtype.into()) {
                return Err(format!("the {backend:?} device does not support {dtype:?}"));
            }
            config = config.int_dtype(dtype);
        }
        None => {
            if let Some(dtype) = prefs.ints.iter().find(|d| supports((**d).into())) {
                config = config.int_dtype(*dtype);
            }
        }
    }
    Ok(config)
}

/// Device and precision arguments, for `#[command(flatten)]`.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct DeviceArgs {
    /// The backend.
    #[command(flatten)]
    pub backend: BackendArgs,

    /// The precision.
    #[command(flatten)]
    pub precision: PrecisionArgs,
}

impl DeviceArgs {
    /// Builds the device the flags select, configured with the default dtypes
    /// the flags and `prefs` resolve to, and with autodiff when `prefs` asks
    /// for it.
    ///
    /// A device's default dtypes can be set once per process, before its
    /// first tensor. A second `init` of the same device succeeds when it asks
    /// for the dtypes the device already has.
    ///
    /// # Errors
    /// When the backend is not compiled in, an exact dtype flag names a dtype
    /// the device does not support, or the device already has other dtypes.
    pub fn init(
        &self,
        prefs: &DevicePrefs,
    ) -> Result<Device, String> {
        let mut device = self.backend.init()?;
        let config = resolve_config(&self.precision, prefs, self.backend.choice(), |dtype| {
            device.supports_dtype(dtype)
        })?;
        let (float, int) = (config.float_dtype, config.int_dtype);

        match device.configure(config) {
            Ok(()) => {}
            Err(DeviceError::AlreadyInitialized { .. })
                if Some(device.settings().float_dtype) == float
                    && int.is_none_or(|int| device.settings().int_dtype == int) => {}
            Err(e) => return Err(e.to_string()),
        }

        let device = if prefs.autodiff {
            device.autodiff()
        } else {
            device
        };
        log::info!("{}", describe(&device));
        Ok(device)
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

    /// Supports everything but bf16: the wgpu and vulkan answer on an RTX 3090.
    fn no_bf16(dtype: DType) -> bool {
        dtype != DType::BF16
    }

    fn all(_: DType) -> bool {
        true
    }

    #[test]
    fn test_the_precision_table() {
        use DeviceChoice::*;
        use FloatDType::*;
        use Precision::*;

        let cases = [
            (Full, Cuda, F32),
            (Full, Flex, F32),
            (BF16Half, Cuda, BF16),
            (BF16Half, Metal, BF16),
            (BF16Half, Vulkan, F32),
            (BF16Half, Wgpu, F32),
            (BF16Half, Flex, F32),
            (AnyHalf, Cuda, F16),
            (AnyHalf, Metal, F16),
            (AnyHalf, Vulkan, F16),
            (AnyHalf, Wgpu, F16),
            (AnyHalf, Flex, F32),
        ];
        for (precision, backend, expected) in cases {
            assert_eq!(
                resolve_precision(precision, backend, all),
                expected,
                "{precision:?} on {backend:?}"
            );
        }
    }

    /// A half the device does not support falls back to f32.
    #[test]
    fn test_unsupported_half_falls_back() {
        assert_eq!(
            resolve_precision(Precision::BF16Half, DeviceChoice::Metal, no_bf16),
            FloatDType::F32
        );
    }

    #[test]
    fn test_flags_choose_the_intent() {
        let prefs = DevicePrefs::new()
            .with_precision(Precision::Full)
            .with_half(Precision::AnyHalf);
        let float = |precision| {
            let args = PrecisionArgs {
                precision,
                ..Default::default()
            };
            resolve_config(&args, &prefs, DeviceChoice::Cuda, all)
                .unwrap()
                .float_dtype
        };
        assert_eq!(float(PrecisionChoice::Auto), Some(FloatDType::F32));
        assert_eq!(float(PrecisionChoice::Full), Some(FloatDType::F32));
        assert_eq!(float(PrecisionChoice::Half), Some(FloatDType::F16));
    }

    #[test]
    fn test_an_exact_dtype_overrides_and_must_be_supported() {
        let prefs = DevicePrefs::new();
        let args = PrecisionArgs {
            precision: PrecisionChoice::Full,
            float_dtype: Some(FloatChoice::F16),
            int_dtype: None,
        };
        let config = resolve_config(&args, &prefs, DeviceChoice::Flex, all).unwrap();
        assert_eq!(config.float_dtype, Some(FloatDType::F16));

        let args = PrecisionArgs {
            float_dtype: Some(FloatChoice::Bf16),
            ..args
        };
        let err = resolve_config(&args, &prefs, DeviceChoice::Wgpu, no_bf16).unwrap_err();
        assert!(err.contains("BF16"), "{err}");
    }

    #[test]
    fn test_int_preferences_take_the_first_supported() {
        let prefs = DevicePrefs::new().with_ints([IntDType::I8, IntDType::I32]);
        let no_i8 = |dtype: DType| dtype != DType::I8;
        let int = |supports: &dyn Fn(DType) -> bool| {
            resolve_config(
                &PrecisionArgs::default(),
                &prefs,
                DeviceChoice::Cuda,
                supports,
            )
            .unwrap()
            .int_dtype
        };
        assert_eq!(int(&all), Some(IntDType::I8));
        assert_eq!(int(&no_i8), Some(IntDType::I32));
        assert_eq!(
            resolve_config(
                &PrecisionArgs::default(),
                &DevicePrefs::new(),
                DeviceChoice::Cuda,
                all
            )
            .unwrap()
            .int_dtype,
            None
        );
    }

    /// The CPU device builds, configured and with autodiff, and a second
    /// `init` asking for the same dtypes succeeds.
    #[test]
    fn test_init_on_the_cpu() {
        let args = DeviceArgs {
            backend: BackendArgs {
                device: DeviceChoice::Flex,
                device_index: None,
            },
            precision: PrecisionArgs::default(),
        };
        let prefs = DevicePrefs::training().with_half(Precision::AnyHalf);

        let device = args.init(&prefs).unwrap();
        assert!(device.is_autodiff());
        assert_eq!(device.settings().float_dtype, FloatDType::F32);
        assert!(args.init(&prefs).is_ok());
    }
}
