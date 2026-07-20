//! [`Tensor`] ops.

use burn::prelude::{
    Backend,
    Tensor,
};

/// Creates a 1D tensor `[for i in 0..n | start + i * step]`.
///
/// # Arguments
/// * `num`: the number a points in the result tensor.
/// * `start`: the start value.
/// * `step`: the step size.
///
/// # Returns
/// `[num]` sized vector.
pub fn float_arange_start_step(
    num: usize,
    start: f64,
    step: Option<f64>,
) -> Vec<f64> {
    let step = step.unwrap_or(1.0);
    (0..num)
        .into_iter()
        .map(|n| start + n as f64 * step)
        .collect()
}

/// Creates a vector with evenly spaced floating point values.
///
/// This function generates a vector with `num` values starting from `start`,
/// ending at `end`, and evenly spaced.
///
/// # Arguments
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (inclusive).
/// - `num`: The number of points to generate in the range.
///
/// # Returns
///
/// A vector containing the generated floating point values.
pub fn float_vec_linspace(
    start: f64,
    end: f64,
    num: usize,
) -> Vec<f64> {
    assert!(num > 0, "Number of points must be positive");

    if num == 1 {
        return vec![start];
    }

    let step = (end - start) / (num as f64 - 1.0);
    float_arange_start_step(num - 1, start, Some(step))
}

/// Creates a 1D tensor `[for i in 0..n | start + i * step]`.
///
/// # Arguments
/// * `num`: the number a points in the result tensor.
/// * `start`: the start value.
/// * `step`: the step size.
/// * `device`; the tensor device to allocate on.
///
/// # Returns
/// `[num]` sized tensor.
pub fn arange_start_step<B: Backend>(
    num: usize,
    start: f64,
    step: Option<f64>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let x = Tensor::arange(0..num as i64, device).float();

    let x = match step {
        None => x,
        Some(step) => x.mul_scalar(step),
    };

    x.add_scalar(start)
}

/// Creates a 1D tensor with evenly spaced floating point values.
///
/// This function generates a tensor with `num` values starting from `start`,
/// ending at `end`, and evenly spaced.
///
/// # Arguments
///
/// * `start`: The starting value of the range.
/// * `end`: The end value of the range (inclusive).
/// * `num`: The number of points to generate in the range.
/// * `device`; the tensor device to allocate on.
///
/// # Returns
///
/// A 1D tensor containing the generated floating point values.
pub fn float_linspace<B: Backend>(
    start: f64,
    end: f64,
    num: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    assert!(num > 0, "Number of points must be positive");

    if num == 1 {
        return Tensor::full([1], start, device);
    }

    let step = (end - start) / (num as f64 - 1.0);
    arange_start_step(num - 1, start, Some(step), device)
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::{
            Tolerance,
            backend::BackendTypes,
        },
    };

    use super::*;
    use crate::support::testing::CpuBackend;
    type B = CpuBackend;

    #[test]
    fn test_arange_start_step() {
        let device = Default::default();
        let start: f64 = -3.0;

        let actual = arange_start_step::<B>(5, start, Some(-1.0), &device);

        actual
            .to_data()
            .assert_eq(&TensorData::from([-3.0, -2.0, -1.0, 0.0, 1.0]), false);
    }

    #[test]
    fn test_linspace_int_step() {
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();
        let start: f64 = 0.0;
        let end: f64 = 1.0;
        let num: usize = 5;

        let actual = float_linspace::<B>(start, end, num, &device);

        actual.to_data().assert_approx_eq::<F>(
            &TensorData::from([0.0, 0.25, 0.5, 0.75, 1.0]),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_float_vec_linspace_neg_float_step() {
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();
        let start: f64 = 1.0;
        let end: f64 = -0.2;
        let num: usize = 5;

        let actual = float_linspace::<B>(start, end, num, &device);

        actual.to_data().assert_approx_eq::<F>(
            &TensorData::from([1.0, 0.7, 0.4, 0.1, -0.2]),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_float_vec_linspace_n1() {
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();
        let start: f64 = 0.0;
        let end: f64 = 1.0;
        let num: usize = 1;

        let actual = float_linspace::<B>(start, end, num, &device);

        actual
            .to_data()
            .assert_approx_eq::<F>(&TensorData::from([0.0]), Tolerance::default());
    }
}
