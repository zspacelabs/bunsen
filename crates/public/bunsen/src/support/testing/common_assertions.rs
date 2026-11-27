use alloc::vec::Vec;
use core::fmt::Debug;

use burn::{
    prelude::{
        Bool,
        Int,
        Tensor,
        TensorData,
    },
    tensor::Tolerance,
};
use num_traits::float::Float;

use super::DeviceTolerance;

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
    T: Float + core::ops::Sub<Output = T> + core::ops::Add<Output = T> + Copy + Debug,
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

/// Asserts that two scalars are close: `|actual - expected| <= tolerance`,
/// in `f64`.
///
/// The scalar twin of [`assert_close_to_vec`], with its rules for non-finite
/// values: a NaN matches only a NaN, an infinity only the same infinity.
///
/// # Panics
///
/// Panics, printing both values, their difference and the tolerance, if they
/// differ by more than `tolerance`, including a NaN on one side only.
#[track_caller]
pub fn assert_scalar_close(
    actual: impl Into<f64>,
    expected: impl Into<f64>,
    tolerance: impl Into<f64>,
) {
    let (a, e, tolerance) = (actual.into(), expected.into(), tolerance.into());
    // `a == e` also matches equal infinities, whose difference is NaN; not
    // `> tolerance`, which is false for a NaN difference.
    if !(a == e || (a.is_nan() && e.is_nan()) || (a - e).abs() <= tolerance) {
        panic!(
            "{a} is not within {tolerance} of {e}: they differ by {}",
            (a - e).abs()
        );
    }
}

/// Asserts that a tensor matches a row-major host buffer.
///
/// `tolerance` is a [`Tolerance`] or a [`NamedTolerance`](super::NamedTolerance),
/// resolved on `actual`'s device.
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
    tolerance: impl DeviceTolerance,
) {
    let tolerance = tolerance.on(&actual.device());
    let expected = TensorData::new(expected.to_vec(), actual.dims()).convert::<f32>();
    actual
        .to_data_as::<f32>()
        .assert_approx_eq::<f32>(&expected, tolerance);
}

/// Asserts that two tensors of the same shape are approximately equal.
///
/// `tolerance` is a [`Tolerance`] or a [`NamedTolerance`](super::NamedTolerance),
/// resolved on `actual`'s device.
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
    tolerance: impl DeviceTolerance,
) {
    let tolerance = tolerance.on(&actual.device());
    actual
        .to_data_as::<f32>()
        .assert_approx_eq::<f32>(&expected.to_data_as::<f32>(), tolerance);
}

/// A tensor's values on the host, as `E`, row-major.
///
/// # Panics
///
/// Panics if the values do not convert to `E`.
pub fn tensor_to_vec_as<E: burn::tensor::Element, const D: usize>(tensor: &Tensor<D>) -> Vec<E> {
    tensor
        .to_data()
        .try_into_vec_as::<E>()
        .expect("the tensor's values, as the element type")
}

/// An integer tensor's values on the host, as `i64`, row-major.
///
/// # Panics
///
/// Panics if the values do not convert to `i64`.
pub fn int_tensor_to_vec<const D: usize>(tensor: &Tensor<D, Int>) -> Vec<i64> {
    tensor
        .to_data()
        .try_into_vec_as::<i64>()
        .expect("the tensor's values, as i64")
}

/// A boolean tensor's values on the host, row-major.
///
/// # Panics
///
/// Panics if the values do not read back as `bool`.
pub fn bool_tensor_to_vec<const D: usize>(tensor: &Tensor<D, Bool>) -> Vec<bool> {
    tensor
        .to_data()
        .try_into_vec::<bool>()
        .expect("the tensor's values, as bool")
}

/// The cosine of the angle between `a` and `b`, accumulated in `f64`; NaN
/// when either is all zeros.
///
/// # Panics
///
/// Panics if the lengths differ.
pub fn cosine_similarity<T: Copy + Into<f64>>(
    a: &[T],
    b: &[T],
) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine of vectors of different lengths");
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x.into() * y.into()).sum();
    let norm = |v: &[T]| v.iter().map(|&x| x.into().powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
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

    /// The readback is row-major and converts; the cosine is 1 for a
    /// vector and its multiple, -1 against its negation, 0 at a right angle.
    #[test]
    fn test_tensor_to_vec_as_and_cosine() {
        let device = crate::support::testing::cpu_device();
        let t = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        assert_eq!(tensor_to_vec_as::<f64, 2>(&t), [1.0, 2.0, 3.0, 4.0]);
        let a = [1.0f32, 2.0, 3.0];
        assert_scalar_close(cosine_similarity(&a, &[2.0, 4.0, 6.0]), 1.0, 1e-12);
        assert_scalar_close(cosine_similarity(&a, &[-1.0, -2.0, -3.0]), -1.0, 1e-12);
        assert_eq!(cosine_similarity(&[1.0f64, 0.0], &[0.0, 1.0]), 0.0);
    }

    /// The integer and boolean readbacks are row-major.
    #[test]
    fn test_int_and_bool_readbacks() {
        let device = crate::support::testing::cpu_device();
        let ids = Tensor::<2, Int>::from_data([[1, -2], [3, 4]], &device);
        assert_eq!(int_tensor_to_vec(&ids), [1, -2, 3, 4]);
        assert_eq!(
            bool_tensor_to_vec(&ids.greater_elem(1)),
            [false, false, true, true]
        );
    }

    /// Within the tolerance, at its edge, and the non-finite rules.
    #[test]
    fn test_assert_scalar_close() {
        assert_scalar_close(1.0, 1.05, 0.1);
        assert_scalar_close(1.0f32, 1.25f32, 0.25);
        assert_scalar_close(f64::NAN, f64::NAN, 0.0);
        assert_scalar_close(f64::INFINITY, f64::INFINITY, 0.0);
    }

    #[test]
    #[should_panic(expected = "1 is not within 0.1 of 1.5: they differ by 0.5")]
    fn test_assert_scalar_close_far() {
        assert_scalar_close(1.0, 1.5, 0.1);
    }

    #[test]
    #[should_panic(expected = "is not within")]
    fn test_assert_scalar_close_nan_one_side() {
        assert_scalar_close(f64::NAN, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "is not within")]
    fn test_assert_scalar_close_infinity_sign() {
        assert_scalar_close(f64::INFINITY, f64::NEG_INFINITY, 1.0);
    }
}
