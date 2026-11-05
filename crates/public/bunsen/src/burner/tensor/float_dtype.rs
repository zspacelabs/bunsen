use burn::{
    Tensor,
    module::Param,
    tensor::{
        DType,
        Device,
        FloatDType,
    },
};

/// `dtype`, widened to at least `at_least`: `at_least` when `dtype` is a
/// narrower float, else `dtype` as it is. It never narrows.
///
/// Floats are ordered by width: f16 and bf16 (16 bits), then f32 and
/// `Flex32` (32), then f64. Two floats of one width are neither wider than
/// the other, so bf16 widened to at least f16 stays bf16. A non-float
/// `dtype` is left as it is.
pub fn widen_to(
    dtype: DType,
    at_least: FloatDType,
) -> DType {
    let width = |d: DType| match d {
        DType::F64 => Some(64),
        DType::F32 | DType::Flex32 => Some(32),
        DType::F16 | DType::BF16 => Some(16),
        _ => None,
    };
    let target = DType::from(at_least);
    match (width(dtype), width(target)) {
        (Some(have), Some(want)) if have < want => target,
        _ => dtype,
    }
}

/// The dtype a computation on `dtype` values runs in, for the operations
/// that widen half precision: f32 for f16 and bf16; any other, its own
/// (`widen_to(dtype, F32)`).
///
/// The norms compute in it (`transformers` normalizes a half input in f32),
/// then narrow the result back.
pub fn widen_half(dtype: DType) -> DType {
    widen_to(dtype, FloatDType::F32)
}

/// `value` rounded to `dtype`'s precision: to bf16 or f16 when `dtype` is
/// one, to f32 for any other dtype but f64, which keeps it as it is.
///
/// What a scalar becomes when a model keeps it in its table's dtype, as
/// `transformers` rounds Gemma's embedding scale.
pub fn round_scalar_to_dtype(
    value: f64,
    dtype: DType,
) -> f64 {
    match dtype {
        DType::F64 => value,
        DType::BF16 => super::bf16_round(value as f32) as f64,
        DType::F16 => burn::tensor::f16::from_f64(value).to_f64(),
        _ => value as f32 as f64,
    }
}

/// The device's default float [`DType`].
///
/// This is what a float tensor built on `device` without a dtype of its own
/// gets — `Tensor::random`, `Tensor::zeros`, `Tensor::from_data` — so it is
/// the dtype an operation between a model's output and a freshly-built tensor
/// needs both sides to be in.
///
/// A module whose parameters were loaded at some other precision computes in
/// *that* precision; this is the dtype its **interface** speaks. Casting a
/// result to it at the module boundary lets the caller stay in the device's
/// float without knowing what the checkpoint shipped as.
pub fn device_float_dtype(device: &Device) -> DType {
    device.settings().float_dtype.into()
}

/// The float [`DType`] a parameter holds, **without initializing it**.
///
/// An initialized parameter answers with its tensor's dtype. A lazy one
/// (a module just built from its config, before anything reads it) will
/// initialize in its device's default float dtype, which is what it answers
/// with; reading [`Param::val`] to ask would instead materialize its random
/// initialization, a whole tensor allocated only to be replaced, which is
/// what a loader about to overwrite it must not do.
pub fn param_float_dtype<const D: usize>(param: &Param<Tensor<D>>) -> DType {
    if param.is_initialized() {
        param.val().dtype()
    } else {
        device_float_dtype(&param.lazy_device())
    }
}

#[cfg(test)]
mod tests {
    /// A narrower float widens to the target; one as wide or wider, and any
    /// non-float, stays.
    #[test]
    fn test_widen_to() {
        assert_eq!(widen_to(DType::BF16, FloatDType::F32), DType::F32);
        assert_eq!(widen_to(DType::F16, FloatDType::F64), DType::F64);
        assert_eq!(widen_to(DType::F32, FloatDType::F64), DType::F64);
        assert_eq!(widen_to(DType::F64, FloatDType::F32), DType::F64);
        assert_eq!(widen_to(DType::F32, FloatDType::F16), DType::F32);
        assert_eq!(widen_to(DType::BF16, FloatDType::F16), DType::BF16);
        assert_eq!(widen_to(DType::Flex32, FloatDType::F32), DType::Flex32);
        assert_eq!(widen_to(DType::I32, FloatDType::F32), DType::I32);
    }

    /// Half widens to f32; every other dtype stays.
    #[test]
    fn test_widen_half() {
        assert_eq!(widen_half(DType::BF16), DType::F32);
        assert_eq!(widen_half(DType::F16), DType::F32);
        assert_eq!(widen_half(DType::F32), DType::F32);
        assert_eq!(widen_half(DType::F64), DType::F64);
    }

    /// A scalar is rounded to its dtype's precision, as `transformers` rounds
    /// Gemma's embedding scale; f64 keeps it.
    #[test]
    fn test_round_scalar_to_dtype() {
        let s = 1152f64.sqrt();
        assert_eq!(round_scalar_to_dtype(s, DType::BF16), 34.0);
        assert_eq!(round_scalar_to_dtype(3072f64.sqrt(), DType::BF16), 55.5);
        assert_eq!(round_scalar_to_dtype(s, DType::F64), s);
        assert_eq!(round_scalar_to_dtype(s, DType::F32), s as f32 as f64);
        assert_eq!(
            round_scalar_to_dtype(s, DType::F16),
            burn::tensor::f16::from_f64(s).to_f64()
        );
    }

    use burn::{
        nn::LinearConfig,
        tensor::DType,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    /// The helper agrees with what an undtyped tensor actually gets: that
    /// equality is the whole point of it.
    #[test]
    fn test_matches_a_default_tensor() {
        let device = cpu_device();

        let t: burn::Tensor<1> = burn::Tensor::zeros([2], &device);
        assert_eq!(device_float_dtype(&device), t.dtype());
        assert_eq!(device_float_dtype(&device), DType::F32);
    }

    /// A lazy parameter answers without initializing; an initialized one
    /// answers with its own dtype, which a cast changes.
    #[test]
    fn test_param_float_dtype() {
        let device = cpu_device();
        let linear = LinearConfig::new(3, 2).init(&device);
        assert!(!linear.weight.is_initialized());
        assert_eq!(param_float_dtype(&linear.weight), DType::F32);
        assert!(
            !linear.weight.is_initialized(),
            "asking did not initialize it"
        );

        let half = Param::from_tensor(linear.weight.val().cast(DType::F16));
        assert_eq!(param_float_dtype(&half), DType::F16);
    }
}
