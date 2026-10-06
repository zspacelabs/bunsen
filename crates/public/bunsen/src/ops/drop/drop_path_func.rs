//! Stochastic depth, as a function.

use burn::{
    prelude::{
        Backend,
        Tensor,
    },
    tensor::Distribution,
};

use crate::support::validators;

/// Stochastic depth: zeros whole batch rows at random.
///
/// Each row of the leading (batch) axis is kept or zeroed as a unit, with
/// probability `drop_prob` of being zeroed; applied to a residual branch, this
/// skips the branch for that sample. Returns `x` unchanged unless `training`
/// is set and `drop_prob > 0`.
///
/// The [`DropPath`] module wraps this, passing `training` from whether
/// autodiff is enabled.
///
/// Paper: Deep Networks with Stochastic Depth (Huang et al., 2016,
/// <https://arxiv.org/abs/1603.09382>).
///
/// # Arguments
///
/// * `x`: a `[batch, ...]` input tensor.
/// * `drop_prob`: Probability of dropping a path.
/// * `training`: Whether the model is in training mode.
/// * `scale_by_keep`: Whether to scale the output by `1 / (1 - drop_prob)`
///
/// # Returns
///
/// * Output tensor with the same shape as the input tensor.
///
/// # Panics
///
/// If `drop_prob` is not in `[0, 1]`.
///
/// [`DropPath`]: crate::blocks::images::drop::drop_path::DropPath
#[must_use]
pub fn drop_path<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
) -> Tensor<B, D> {
    _drop_path_sample(
        x,
        drop_prob,
        training,
        scale_by_keep,
        |shape, keep_prob, device| {
            Tensor::<B, D>::random(shape, Distribution::Bernoulli(keep_prob), device)
        },
    )
}

/// Internal implementation of `DropPath`.
///
/// Deferred to a separate function to allow for testing sampling.
///
/// # Arguments
///
/// * `x`: Input tensor.
/// * `drop_prob`: Probability of dropping a path.
/// * `training`: Whether the model is in training mode.
/// * `scale_by_keep`: Whether to scale the output by `1 / (1 - drop_prob)`
/// * `sample`: Sampling function to generate the random tensor.
///
/// # Returns
///
/// * Output tensor with the same shape as the input tensor.
#[inline(always)]
#[must_use]
fn _drop_path_sample<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
    sample: fn([usize; D], f64, &B::Device) -> Tensor<B, D>,
) -> Tensor<B, D> {
    validators::expect_probability(drop_prob);

    if !training || drop_prob == 0.0 {
        return x;
    }

    let keep_prob = 1.0 - drop_prob;

    let mut shape = [1; D];
    shape[0] = x.dims()[0];

    let random_tensor = sample(shape, keep_prob, &x.device());

    let random_tensor = if keep_prob > 0.0 && scale_by_keep {
        random_tensor.div_scalar(keep_prob)
    } else {
        random_tensor
    };

    x * random_tensor
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::Tensor,
        tensor::Distribution,
    };

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        cpu_device,
    };

    #[test]
    fn test_drop_path_wrapper() {
        type B = CpuBackend;
        let device = cpu_device();

        let n = 3;
        let shape = [n, 2, 4];

        let x = Tensor::<B, 3>::random(shape, Distribution::Uniform(0.0, 1.0), &device);

        // No-op case: not training and drop_prob = 0.0
        let training = false;
        let drop_prob = 0.0;
        let scale_by_keep = false;
        let res = drop_path(x.clone(), drop_prob, training, scale_by_keep);
        assert_eq!(res.dims(), x.dims());
    }

    #[test]
    fn test_drop_path_sample() {
        type B = CpuBackend;
        let device = cpu_device();

        let n = 3;
        let shape = [n, 2, 4];

        let x = Tensor::<B, 3>::random(shape, Distribution::Uniform(0.0, 1.0), &device);

        // No-op case: not training and drop_prob = 0.0
        let training = false;
        let drop_prob = 0.0;
        let scale_by_keep = false;
        let res = _drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                assert_eq!(shape, [3, 1, 1]);
                assert_eq!(keep_prob, 1.0);
                Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(&x.to_data(), true);

        // No-op case: training, but drop_prob = 0.0
        let training = true;
        let drop_prob = 0.0;
        let scale_by_keep = false;
        let res = _drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                assert_eq!(shape, [3, 1, 1]);
                assert_eq!(keep_prob, 1.0);
                Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(&x.to_data(), true);

        // Training, but no scaling
        let training = true;
        let drop_prob = 0.5;
        let scale_by_keep = false;
        let res = _drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                assert_eq!(shape, [3, 1, 1]);
                assert_eq!(keep_prob, 0.5);
                Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(
            &(x.clone() * Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], &device))
                .to_data(),
            true,
        );

        // Training, with scaling
        let training = true;
        let drop_prob = 0.5;
        let keep_prob = 1.0 - drop_prob;
        let scale_by_keep = true;
        let res = _drop_path_sample(
            x.clone(),
            drop_prob,
            training,
            scale_by_keep,
            |shape, keep_prob, device| {
                assert_eq!(shape, [3, 1, 1]);
                assert_eq!(keep_prob, 0.5);
                Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], device)
            },
        );
        res.to_data().assert_eq(
            &(x.clone() * Tensor::<B, 3>::from_data([[[1.0]], [[0.0]], [[1.0]]], &device))
                .div_scalar(keep_prob)
                .to_data(),
            true,
        );
    }
}
