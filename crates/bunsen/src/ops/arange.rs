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
pub fn vec_arange_start_step(
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
pub fn vec_linspace(
    start: f64,
    end: f64,
    num: usize,
) -> Vec<f64> {
    assert!(num > 0, "Number of points must be positive");

    if num == 1 {
        return vec![start];
    }

    let step = (end - start) / (num as f64 - 1.0);
    vec_arange_start_step(num, start, Some(step))
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
pub fn tensor_arange_start_step<B: Backend>(
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
pub fn tensor_linspace<B: Backend>(
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
    tensor_arange_start_step(num, start, Some(step), device)
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
    use crate::support::testing::{
        CpuBackend,
        assert_close_to_vec,
    };
    type B = CpuBackend;

    #[test]
    fn test_arange_start_step() {
        let device = Default::default();

        let num = 5;

        // Pos step
        for step in [None, Some(1.0)] {
            let start: f64 = -3.0;

            let expected = vec![-3.0, -2.0, -1.0, 0.0, 1.0];

            let vec_actual = vec_arange_start_step(num, start, step);
            let tensor_actual = tensor_arange_start_step::<B>(num, start, step, &device);

            assert_close_to_vec(&vec_actual, &expected, 0.0001);
            tensor_actual
                .to_data()
                .assert_eq(&TensorData::from(expected.as_slice()), false);
        }

        // Neg step
        {
            let step = Some(-1.0);

            let start: f64 = 3.0;

            let expected = vec![3.0, 2.0, 1.0, 0.0, -1.0];

            let vec_actual = vec_arange_start_step(num, start, step);
            let tensor_actual = tensor_arange_start_step::<B>(num, start, step, &device);

            assert_close_to_vec(&vec_actual, &expected, 0.0001);

            tensor_actual
                .to_data()
                .assert_eq(&TensorData::from(expected.as_slice()), false);
        }
    }

    #[test]
    fn test_arange_linspace() {
        let device = Default::default();

        let num = 5;

        for expected in [
            vec![3.0, 2.0, 1.0, 0.0, -1.0],
            vec![-3.0, -2.0, -1.0, 0.0, 1.0],
        ] {
            let num = expected.len();
            let start = expected[0];
            let end = expected[num - 1];

            let vec_actual = vec_linspace(start, end, num);
            let tensor_actual = tensor_linspace::<B>(start, end, num, &device);

            assert_close_to_vec(&vec_actual, &expected, 0.0001);

            tensor_actual
                .to_data()
                .assert_eq(&TensorData::from(expected.as_slice()), false);
        }

        // Pos.
        {
            let start: f64 = -3.0;
            let end = 1.0;

            let expected = vec![-3.0, -2.0, -1.0, 0.0, 1.0];

            let vec_actual = vec_linspace(start, end, num);
            let tensor_actual = tensor_linspace::<B>(start, end, num, &device);

            assert_close_to_vec(&vec_actual, &expected, 0.0001);
            tensor_actual
                .to_data()
                .assert_eq(&TensorData::from(expected.as_slice()), false);
        }

        // Neg step
        {
            let start: f64 = 3.0;
            let end = -1.0;

            let expected = vec![3.0, 2.0, 1.0, 0.0, -1.0];

            let vec_actual = vec_linspace(start, end, num);
            let tensor_actual = tensor_linspace::<B>(start, end, num, &device);

            assert_close_to_vec(&vec_actual, &expected, 0.0001);
            tensor_actual
                .to_data()
                .assert_eq(&TensorData::from(expected.as_slice()), false);
        }
    }

    #[test]
    fn test_linspace_int_step() {
        type F = <B as BackendTypes>::FloatElem;

        let device = Default::default();
        let start: f64 = 0.0;
        let end: f64 = 1.0;
        let num: usize = 5;

        let actual = tensor_linspace::<B>(start, end, num, &device);

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

        let actual = tensor_linspace::<B>(start, end, num, &device);

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

        let actual = tensor_linspace::<B>(start, end, num, &device);

        actual
            .to_data()
            .assert_approx_eq::<F>(&TensorData::from([0.0]), Tolerance::default());
    }
}
