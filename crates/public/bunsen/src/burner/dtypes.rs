//! # Dtypes on a device
//!
//! What a device does with each float dtype, as burn reports it
//! ([`DTypeUsage`]): whether it can store it, compute in it, and run it on
//! accelerated paths (tensor cores). [`float_dtype_usages`] lists every
//! [`FloatDType`] for a device, which `bunsen-tool devices` prints per backend.
//!
//! burn does not check the dtypes of a float op's operands: two tensors of
//! different float dtypes meet without a cast. The tests here pin what each
//! backend then does, since a missed cast in bunsen's blocks looks like
//! that, not like an error.

use burn::tensor::{
    DType,
    Device,
    FloatDType,
};
pub use burn_backend::{
    DTypeUsage,
    DTypeUsageSet,
};

/// Every [`FloatDType`], widest first.
pub const FLOAT_DTYPES: [FloatDType; 5] = [
    FloatDType::F64,
    FloatDType::F32,
    FloatDType::Flex32,
    FloatDType::F16,
    FloatDType::BF16,
];

/// What `device` does with `dtype`: its storage, arithmetic and accelerated
/// support.
pub fn dtype_usage(
    device: &Device,
    dtype: impl Into<DType>,
) -> DTypeUsageSet {
    use burn_backend::Backend;
    burn_dispatch::Dispatch::dtype_usage(device.as_dispatch(), dtype.into())
}

/// [`dtype_usage`] of each of [`FLOAT_DTYPES`] on `device`, in that order.
pub fn float_dtype_usages(device: &Device) -> Vec<(FloatDType, DTypeUsageSet)> {
    FLOAT_DTYPES
        .iter()
        .map(|&dtype| (dtype, dtype_usage(device, dtype)))
        .collect()
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;

    use super::*;
    use crate::support::testing::{
        cpu_device,
        performance_device,
    };

    /// What one op on an f32 and a bf16 operand gave: the result's dtype and
    /// values, or the panic.
    #[derive(Debug, Clone, PartialEq)]
    enum Mixed {
        Ran { dtype: DType, values: Vec<f32> },
        Panicked(String),
    }

    fn run(op: impl FnOnce() -> Tensor<2> + core::panic::UnwindSafe) -> Mixed {
        match std::panic::catch_unwind(op) {
            Ok(t) => Mixed::Ran {
                dtype: t.dtype(),
                values: t
                    .cast(DType::F32)
                    .into_data()
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .unwrap(),
            },
            Err(e) => Mixed::Panicked(
                e.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default(),
            ),
        }
    }

    /// The narrowest float `device` computes in, after f32: bf16, else f16.
    fn half(device: &Device) -> Option<FloatDType> {
        [FloatDType::BF16, FloatDType::F16]
            .into_iter()
            .find(|&d| dtype_usage(device, d).contains(DTypeUsage::Arithmetic))
    }

    /// Exact in bf16 and f16, so a right answer is exactly the f32 one.
    const VALUES: [[f32; 3]; 2] = [[0.5, 1.25, -2.0], [3.0, -0.75, 4.5]];

    /// Each op, with an f32 and a `half` operand in each order, on `device`:
    /// its name, whether the f32 operand is on the left, the right answer,
    /// and what it gave.
    fn probe(
        device: &Device,
        half: FloatDType,
    ) -> Vec<(&'static str, bool, Vec<f32>, Mixed)> {
        let f32_t = || Tensor::<2>::from_data(VALUES, (device, DType::F32));
        let half_t = || Tensor::<2>::from_data(VALUES, (device, DType::from(half)));
        let flat: Vec<f32> = VALUES.iter().flatten().copied().collect();
        let sum: Vec<f32> = flat.iter().map(|x| x + x).collect();
        let product: Vec<f32> = flat.iter().map(|x| x * x).collect();
        let gram: Vec<f32> = (0..2)
            .flat_map(|i| (0..2).map(move |j| (0..3).map(|k| VALUES[i][k] * VALUES[j][k]).sum()))
            .collect();
        vec![
            ("f32 + half", true, sum.clone(), run(|| f32_t() + half_t())),
            ("half + f32", false, sum, run(|| half_t() + f32_t())),
            (
                "f32 * half",
                true,
                product.clone(),
                run(|| f32_t() * half_t()),
            ),
            ("half * f32", false, product, run(|| half_t() * f32_t())),
            (
                "f32 @ half^T",
                true,
                gram.clone(),
                run(|| f32_t().matmul(half_t().transpose())),
            ),
            (
                "half @ f32^T",
                false,
                gram,
                run(|| half_t().matmul(f32_t().transpose())),
            ),
        ]
    }

    fn is_flex(device: &Device) -> bool {
        format!("{device:?}").contains("Flex")
    }

    /// burn-flex computes in f64, f32, f16 and bf16, and accelerates none.
    #[test]
    fn test_flex_dtype_usage() {
        let device = cpu_device();
        for (dtype, usage) in float_dtype_usages(&device) {
            assert!(!usage.contains(DTypeUsage::Accelerated), "{dtype:?}");
            let computes = dtype != FloatDType::Flex32;
            assert_eq!(
                usage.contains(DTypeUsage::Arithmetic),
                computes,
                "{dtype:?}"
            );
        }
    }

    /// burn-flex refuses an op on mixed float dtypes: each panics with a
    /// dtype mismatch.
    #[test]
    fn test_flex_refuses_mixed_dtypes() {
        for (op, _, _, got) in probe(&cpu_device(), FloatDType::BF16) {
            match got {
                Mixed::Panicked(message) => {
                    assert!(message.contains("dtype mismatch"), "{op}: {message}")
                }
                ran => panic!("{op}: {ran:?}"),
            }
        }
    }

    /// A cubecl backend (CUDA, wgpu, Vulkan, Metal) does not refuse an op on
    /// mixed float dtypes: it runs it in the left operand's dtype, reading
    /// the other's bits as that dtype, and the answer is wrong. So bunsen's
    /// blocks cast before they combine. This pins today's burn; if it starts
    /// refusing or promoting, revisit the seams that rely on it.
    ///
    /// The half operand is the narrowest dtype the device computes in: bf16,
    /// or f16 on wgpu, which has no bf16. Without an accelerator feature,
    /// there is nothing to check.
    #[test]
    fn test_accelerators_run_mixed_dtypes_wrong() {
        let device = performance_device();
        if is_flex(&device) {
            return;
        }
        let half = half(&device).expect("an accelerator computes in a half dtype");
        for (op, f32_left, want, got) in probe(&device, half) {
            let Mixed::Ran { dtype, values } = got else {
                panic!("{op} on {half:?}: {got:?}");
            };
            let left = if f32_left {
                DType::F32
            } else {
                DType::from(half)
            };
            assert_eq!(dtype, left, "{op} on {half:?}: the left operand's dtype");
            let off = values
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                off > 1e-2,
                "{op} on {half:?}: right ({values:?}); revisit the seams"
            );
        }
    }
}
