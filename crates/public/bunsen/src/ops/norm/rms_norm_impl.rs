use burn::{
    Tensor,
    tensor::DType::F32,
};

/// Options for root-mean-square norm.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct RmsNormOptions {
    /// Epsilon value for numerical stability.
    pub eps: f32,
}

impl Default for RmsNormOptions {
    fn default() -> Self {
        Self { eps: 1e-5 }
    }
}

impl RmsNormOptions {
    /// Sets epsilon value.
    pub fn with_eps(
        mut self,
        eps: f32,
    ) -> Self {
        self.eps = eps;
        self
    }

    /// Applies root-mean-square norm.
    pub fn norm<const R: usize>(
        &self,
        x: Tensor<R>,
    ) -> Tensor<R> {
        rms_norm(x, self)
    }
}

/// Applies root-mean-square norm.
pub fn rms_norm<const R: usize>(
    x: Tensor<R>,
    options: &RmsNormOptions,
) -> Tensor<R> {
    let eps: f32 = options.eps;
    let dtype = x.dtype();

    let rms = x
        .clone()
        .cast(F32)
        .square()
        .mean_dim(-1)
        .add_scalar(eps)
        .sqrt()
        .cast(dtype);

    x / rms
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        tensor::Distribution,
    };
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        performance_device,
    };

    #[test]
    #[serial]
    fn test_rms_norm() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let x: Tensor<3> = Tensor::random([2, 3, 4], Distribution::Default, &device);
        let options = RmsNormOptions::default();

        let y = rms_norm(x.clone(), &options);

        let x_rms = x
            .clone()
            .square()
            .mean_dim(-1)
            .add_scalar(options.eps)
            .sqrt();
        let expected = x / x_rms;

        y.to_data().assert_eq(&expected.to_data(), true);
    }
}
