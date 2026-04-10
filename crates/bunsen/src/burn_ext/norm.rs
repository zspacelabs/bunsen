//! # Norm Extensions
use burn::{
    Tensor,
    prelude::Backend,
    tensor::DType::F32,
};

/// Apply root-mean-square norm.
pub fn rms_norm<B: Backend, const R: usize>(x: Tensor<B, R>) -> Tensor<B, R> {
    let eps: f32 = 1e-7;
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
        backend::Wgpu,
        tensor::Distribution,
    };

    use super::*;

    #[test]
    fn test_rms_norm() {
        type B = Wgpu;
        let device = Default::default();

        let x: Tensor<B, 3> = Tensor::random([2, 3, 4], Distribution::Default, &device);
        let y = rms_norm(x.clone());

        let x_rms = x.clone().square().mean_dim(-1).add_scalar(1e-7).sqrt();
        let expected = x / x_rms;

        y.to_data().assert_eq(&expected.to_data(), true);
    }
}
