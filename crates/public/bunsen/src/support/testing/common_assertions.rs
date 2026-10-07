use std::fmt::Debug;

use burn::{
    prelude::{
        Tensor,
        TensorData,
    },
    tensor::Tolerance,
};
use num_traits::float::Float;

/// Asserts that two host slices of floats are close, element by element.
///
/// The bound is absolute: each pair must satisfy `|a - e| <= tolerance`. For
/// tensors, use [`assert_tensors_close`] or [`assert_tensor_close_to_vec`],
/// which take a relative-and-absolute burn [`Tolerance`].
///
/// Non-finite values follow burn's [`TensorData::assert_approx_eq`]: a NaN
/// matches only a NaN at the same position, and an infinity matches only the
/// same infinity.
///
/// # Panics
///
/// Panics, printing both slices, if the lengths differ or any pair differs
/// by more than `tolerance`, including a NaN on one side only.
pub fn assert_close_to_vec<T>(
    actual: &[T],
    expected: &[T],
    tolerance: T,
) where
    T: Float + std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy + Debug,
{
    let mut pass = actual.len() == expected.len();
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        if !pass {
            break;
        }
        // `a == e` also matches equal infinities, whose difference is NaN.
        if a == e || (a.is_nan() && e.is_nan()) {
            continue;
        }
        // Not `> tolerance`: that is false for a NaN difference.
        let within = (a - e).abs() <= tolerance;
        if !within {
            pass = false;
            break;
        }
    }
    if !pass {
        panic!("Expected (+/- {tolerance:?}):\n{expected:?}\nActual:\n{actual:?}");
    }
}

/// Asserts that a tensor matches a row-major host buffer.
///
/// The comparison runs through [`TensorData::assert_approx_eq`], so a mismatch
/// reports the differing shape, or the relative and absolute error, rather than
/// a bare element index.
///
/// # Panics
///
/// Panics if `expected` does not hold exactly `actual.dims()` elements, or if
/// any pair differs by more than `tolerance`.
pub fn assert_tensor_close_to_vec<const D: usize>(
    actual: &Tensor<D>,
    expected: &[f64],
    tolerance: Tolerance<f32>,
) {
    let expected = TensorData::new(expected.to_vec(), actual.dims()).convert::<f32>();
    actual
        .to_data_as::<f32>()
        .assert_approx_eq::<f32>(&expected, tolerance);
}

/// Asserts that two tensors of the same shape are approximately equal.
///
/// Both sides are read back as `f32`, so tensors that differ only in dtype
/// still compare.
///
/// # Panics
///
/// Panics if the shapes differ, or if any pair of values differs by more than
/// `tolerance`.
pub fn assert_tensors_close<const D: usize>(
    actual: &Tensor<D>,
    expected: &Tensor<D>,
    tolerance: Tolerance<f32>,
) {
    actual
        .to_data_as::<f32>()
        .assert_approx_eq::<f32>(&expected.to_data_as::<f32>(), tolerance);
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::{
            Tensor,
            TensorData,
        },
        tensor::{
            Device,
            Tolerance,
        },
    };
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        performance_device,
    };

    /// A `[2, 2]` tensor holding `values` in row-major order.
    fn square(
        values: [f64; 4],
        device: &Device,
    ) -> Tensor<2> {
        Tensor::from_data(TensorData::new(values.to_vec(), [2, 2]), device)
    }

    #[test]
    fn test_assert_close_to_vec() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.01);

        let actual = vec![1.0, 2.0, 3.1];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.2);
    }

    #[test]
    #[should_panic]
    fn test_assert_close_to_vec_bad_values() {
        let actual = vec![1.0, 2.0, 3.0];
        let expected = vec![1.0, 2.0, 3.5];
        assert_close_to_vec(&actual, &expected, 0.01);
    }

    #[test]
    #[should_panic]
    fn test_assert_close_to_vec_different_lengths() {
        let actual = vec![1.0, 2.0];
        let expected = vec![1.0, 2.0, 3.0];
        assert_close_to_vec(&actual, &expected, 0.01);
    }

    #[test]
    #[should_panic(expected = "Expected (+/- 0.1)")]
    fn test_assert_close_to_vec_nan_actual() {
        assert_close_to_vec(&[1.0, f32::NAN], &[1.0, 1.0], 0.1);
    }

    #[test]
    #[should_panic(expected = "Expected (+/- 0.1)")]
    fn test_assert_close_to_vec_nan_expected() {
        assert_close_to_vec(&[1.0, 1.0], &[1.0, f64::NAN], 0.1);
    }

    #[test]
    fn test_assert_close_to_vec_nan_matches_nan() {
        assert_close_to_vec(&[1.0, f32::NAN], &[1.0, f32::NAN], 0.1);
    }

    #[test]
    fn test_assert_close_to_vec_infinity_matches_same_infinity() {
        assert_close_to_vec(
            &[f64::INFINITY, f64::NEG_INFINITY],
            &[f64::INFINITY, f64::NEG_INFINITY],
            0.1,
        );
    }

    #[test]
    #[should_panic(expected = "Expected (+/- 0.1)")]
    fn test_assert_close_to_vec_infinity_sign_mismatch() {
        assert_close_to_vec(&[f64::INFINITY], &[f64::NEG_INFINITY], 0.1);
    }

    #[test]
    #[serial]
    fn test_assert_tensor_close_to_vec() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let t = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensor_close_to_vec(&t, &[1.0, 2.0, 3.0, 4.0], Tolerance::default());
    }

    #[test]
    #[serial]
    #[should_panic]
    fn test_assert_tensor_close_to_vec_bad_values() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let t = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensor_close_to_vec(&t, &[1.0, 2.0, 3.0, 9.0], Tolerance::default());
    }

    #[test]
    #[serial]
    fn test_assert_tensors_close() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let a = square([1.0, 2.0, 3.0, 4.0], &device);
        let b = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensors_close(&a, &b, Tolerance::default());
    }

    #[test]
    #[serial]
    #[should_panic]
    fn test_assert_tensors_close_bad_values() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let a = square([1.0, 2.0, 3.0, 4.0], &device);
        let b = square([1.0, 2.0, 3.0, 9.0], &device);
        assert_tensors_close(&a, &b, Tolerance::default());
    }

    #[test]
    #[serial]
    #[should_panic(expected = "Tensors are not approx eq")]
    fn test_assert_tensor_close_to_vec_nan() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let t = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensor_close_to_vec(&t, &[1.0, 2.0, 3.0, f64::NAN], Tolerance::default());
    }

    #[test]
    #[serial]
    #[should_panic(expected = "Tensors are not approx eq")]
    fn test_assert_tensors_close_nan() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let a = square([1.0, 2.0, 3.0, f64::NAN], &device);
        let b = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensors_close(&a, &b, Tolerance::default());
    }
}
