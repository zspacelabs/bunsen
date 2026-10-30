use burn::{
    Tensor,
    module::Param,
    tensor::{
        DType,
        Device,
    },
};

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
