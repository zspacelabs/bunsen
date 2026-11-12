//! The load call's dtype policy: [`LoadPolicy`].

use burn::{
    module::Module,
    tensor::{
        Device,
        FloatDType,
    },
};

use crate::{
    burner::{
        dtypes::{
            DTypeUsage,
            dtype_usage,
        },
        module::{
            DTypeMapper,
            module_float_dtype,
        },
        tensor::float_width,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// How a load chooses the float dtype its parameters are placed in: a plain
/// part of the load call
/// ([`BuildFactory::load`](crate::data::builds::BuildFactory::load),
/// [`DeferredBuild::load`](crate::data::builds::DeferredBuild::load),
/// [`ConstructInit::try_init`](crate::data::builds::ConstructInit::try_init)),
/// made by intent at the call site, never guessed by the loader.
///
/// A load only ever widens: a tensor goes in the dtype it was stored in, or
/// f32, or f64, the first of those the policy, the device and the model's
/// hook all allow ([`resolve`](Self::resolve)). bf16 and f16 never become
/// each other. Narrowing or quantizing is an offline conversion that writes
/// new weights, and so a new row, never a load's choice.
///
/// The intents:
/// - [`performance`](Self::performance): a half dtype only where the device
///   accelerates it (tensor cores); f32 elsewhere. The default a program should
///   reach for.
/// - [`as_stored`](Self::as_stored): a half dtype wherever the device computes
///   in it at all: a bf16 check on the CPU, or a model that must fit whatever
///   the speed.
/// - [`at_least`](Self::at_least): never narrower than a floor: the f32
///   validation suites, which hold a model to an f32 reference.
///
/// It has no `Default` and no serde form: a call site says what it wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadPolicy {
    /// The narrowest dtype a tensor may be placed in, if any.
    at_least: Option<FloatDType>,

    /// What the device must do with a half dtype for it to be used:
    /// [`Accelerated`](DTypeUsage::Accelerated) or
    /// [`Arithmetic`](DTypeUsage::Arithmetic). f32 and f64 need arithmetic
    /// alone.
    half_needs: DTypeUsage,
}

impl LoadPolicy {
    /// A half dtype only where the device accelerates it; f32 elsewhere.
    pub fn performance() -> Self {
        Self {
            at_least: None,
            half_needs: DTypeUsage::Accelerated,
        }
    }

    /// The stored dtype wherever the device computes in it.
    pub fn as_stored() -> Self {
        Self {
            at_least: None,
            half_needs: DTypeUsage::Arithmetic,
        }
    }

    /// Never narrower than `dtype`: [`as_stored`](Self::as_stored) above it.
    pub fn at_least(dtype: FloatDType) -> Self {
        Self {
            at_least: Some(dtype),
            half_needs: DTypeUsage::Arithmetic,
        }
    }

    /// The floor, if any.
    pub fn floor(&self) -> Option<FloatDType> {
        self.at_least
    }

    /// The dtype a tensor stored in `stored` is placed in on `device`: the
    /// first of `stored`, f32 and f64, from the policy's floor up, that the
    /// device supports as the policy asks and the hook `accepts`.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) when none
    /// is: the device, the policy and the hook leave nothing to load into.
    pub fn resolve(
        &self,
        stored: FloatDType,
        device: &Device,
        accepts: impl Fn(FloatDType) -> bool,
    ) -> BunsenResult<FloatDType> {
        let candidates = [stored, FloatDType::F32, FloatDType::F64];
        let floor = self.at_least.map_or(0, float_width);
        let mut tried = Vec::new();
        for dtype in candidates {
            if float_width(dtype) < floor || tried.contains(&dtype) {
                continue;
            }
            tried.push(dtype);
            if self.device_allows(dtype, device) && accepts(dtype) {
                return Ok(dtype);
            }
        }
        Err(BunsenError::unsupported(format!(
            "no dtype to load {stored:?} into on {device:?} under {self:?}: tried {tried:?}"
        )))
    }

    /// `module`, loaded at its files' precision, placed by the policy: its
    /// parameters' dtype [resolved](Self::resolve), and the module cast to
    /// the result when it differs. For a hook whose reader loads at the
    /// stored dtype; a module with no float parameter is as it is.
    ///
    /// # Errors
    /// As [`resolve`](Self::resolve).
    pub fn place<M: Module>(
        &self,
        module: M,
        device: &Device,
        accepts: impl Fn(FloatDType) -> bool,
    ) -> BunsenResult<M> {
        let Some(stored) = module_float_dtype(&module) else {
            return Ok(module);
        };
        let dtype = self.resolve(stored, device, accepts)?;
        Ok(if dtype == stored {
            module
        } else {
            module.map(&mut DTypeMapper::new(dtype.into()))
        })
    }

    /// Whether `device` supports `dtype` as the policy asks.
    fn device_allows(
        &self,
        dtype: FloatDType,
        device: &Device,
    ) -> bool {
        let needs = if float_width(dtype) == 16 {
            self.half_needs
        } else {
            DTypeUsage::Arithmetic
        };
        let usage = dtype_usage(device, dtype);
        usage.contains(needs) && usage.contains(DTypeUsage::Arithmetic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    fn any(_: FloatDType) -> bool {
        true
    }

    /// On the CPU, which computes in bf16 but accelerates nothing:
    /// `performance` widens bf16 to f32, `as_stored` keeps it, and a floor
    /// widens to it; an f32 tensor is never narrowed.
    #[test]
    fn test_resolve_on_the_cpu() {
        let device = cpu_device();
        let bf16 = FloatDType::BF16;
        assert_eq!(
            LoadPolicy::performance()
                .resolve(bf16, &device, any)
                .unwrap(),
            FloatDType::F32
        );
        assert_eq!(
            LoadPolicy::as_stored().resolve(bf16, &device, any).unwrap(),
            bf16
        );
        assert_eq!(
            LoadPolicy::at_least(FloatDType::F32)
                .resolve(bf16, &device, any)
                .unwrap(),
            FloatDType::F32
        );
        assert_eq!(
            LoadPolicy::at_least(FloatDType::F64)
                .resolve(bf16, &device, any)
                .unwrap(),
            FloatDType::F64
        );
        assert_eq!(
            LoadPolicy::as_stored()
                .resolve(FloatDType::F32, &device, any)
                .unwrap(),
            FloatDType::F32
        );
    }

    /// A hook that does not take the stored dtype forces widening; bf16
    /// never becomes f16; one that takes nothing the walk reaches fails.
    #[test]
    fn test_resolve_under_the_hook() {
        let device = cpu_device();
        let no_bf16 = |d: FloatDType| d != FloatDType::BF16;
        assert_eq!(
            LoadPolicy::as_stored()
                .resolve(FloatDType::BF16, &device, no_bf16)
                .unwrap(),
            FloatDType::F32
        );
        let only_f16 = |d: FloatDType| d == FloatDType::F16;
        assert!(
            LoadPolicy::as_stored()
                .resolve(FloatDType::BF16, &device, only_f16)
                .is_err()
        );
        assert_eq!(
            LoadPolicy::as_stored()
                .resolve(FloatDType::F16, &device, only_f16)
                .unwrap(),
            FloatDType::F16
        );
    }

    /// `place` keeps a module the policy keeps, and widens one it widens.
    #[test]
    fn test_place() {
        use burn::nn::LinearConfig;
        let device = cpu_device();
        let half = LinearConfig::new(2, 2)
            .init(&device)
            .map(&mut DTypeMapper::new(FloatDType::BF16.into()));
        let kept = LoadPolicy::as_stored()
            .place(half.clone(), &device, any)
            .unwrap();
        assert_eq!(module_float_dtype(&kept), Some(FloatDType::BF16));
        let widened = LoadPolicy::performance().place(half, &device, any).unwrap();
        assert_eq!(module_float_dtype(&widened), Some(FloatDType::F32));
    }

    /// On an accelerator, `performance` keeps a half dtype the device
    /// accelerates (bf16 on CUDA), and widens one it does not (wgpu has no
    /// bf16). Without an accelerator feature, there is nothing to check.
    #[test]
    fn test_resolve_on_the_accelerator() {
        let device = crate::support::testing::performance_device();
        if format!("{device:?}").contains("Flex") {
            return;
        }
        let bf16 = FloatDType::BF16;
        let got = LoadPolicy::performance()
            .resolve(bf16, &device, any)
            .unwrap();
        let accelerated = dtype_usage(&device, bf16).contains(DTypeUsage::Accelerated);
        assert_eq!(got == bf16, accelerated, "{device:?}: {got:?}");
    }
}
