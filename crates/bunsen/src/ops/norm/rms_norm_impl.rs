use burn::{
    Tensor,
    prelude::Backend,
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
    /// Set epsilon value.
    pub fn with_eps(
        mut self,
        eps: f32,
    ) -> Self {
        self.eps = eps;
        self
    }

    /// Apply root-mean-square norm.
    pub fn norm<B: Backend, const R: usize>(
        &self,
        x: Tensor<B, R>,
    ) -> Tensor<B, R> {
        rms_norm(x, self)
    }
}

/// Apply root-mean-square norm.
pub fn rms_norm<B: Backend, const R: usize>(
    x: Tensor<B, R>,
    options: &RmsNormOptions,
) -> Tensor<B, R> {
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

    use super::*;
    use crate::testing::PerfTestBackend;

    #[test]
    fn test_rms_norm() {
        type B = PerfTestBackend;
        let device = Default::default();

        let x: Tensor<B, 3> = Tensor::random([2, 3, 4], Distribution::Default, &device);
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
