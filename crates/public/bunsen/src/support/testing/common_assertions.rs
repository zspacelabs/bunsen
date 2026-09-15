use std::{
    any::{
        Any,
        TypeId,
    },
    fmt::Debug,
    sync::LazyLock,
};

use burn::{
    module::Param,
    prelude::{
        Backend,
        Tensor,
        TensorData,
    },
    tensor::{
        Tolerance,
        backend::DeviceOps,
    },
};
use dashmap::DashMap;
use num_traits::float::Float;

use crate::burner::tensor::TensorElemOpExt;

/// Asserts that two vectors of floating-point numbers are close to each other
/// within a given tolerance.
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
        if (a - e).abs() > tolerance {
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
pub fn assert_tensor_close_to_vec<B, const D: usize>(
    actual: &Tensor<B, D>,
    expected: &[f64],
    tolerance: Tolerance<B::FloatElem>,
) where
    B: Backend,
    B::FloatElem: Float,
{
    let expected = TensorData::new(expected.to_vec(), actual.dims()).convert::<B::FloatElem>();
    actual
        .to_data_as::<B::FloatElem>()
        .assert_approx_eq::<B::FloatElem>(&expected, tolerance);
}

/// Asserts that two tensors of the same shape are approximately equal.
///
/// Both sides are read back at the backend's float element type, so tensors
/// that differ only in dtype still compare.
///
/// # Panics
///
/// Panics if the shapes differ, or if any pair of values differs by more than
/// `tolerance`.
pub fn assert_tensors_close<B, const D: usize>(
    actual: &Tensor<B, D>,
    expected: &Tensor<B, D>,
    tolerance: Tolerance<B::FloatElem>,
) where
    B: Backend,
    B::FloatElem: Float,
{
    actual
        .to_data_as::<B::FloatElem>()
        .assert_approx_eq::<B::FloatElem>(&expected.to_data_as::<B::FloatElem>(), tolerance);
}

/// Applies a parameter's **load**-path mapping to a tensor.
///
/// A [`Param`] can carry transformations that run only as it crosses a store
/// boundary — see [`repair_pytorch_strided_weight`] — which makes them
/// invisible from the outside. This exposes the load side, so a test can
/// assert *which* mappings a module attached, and to which parameters.
///
/// [`repair_pytorch_strided_weight`]:
///     crate::burner::store::repair_pytorch_strided_weight
pub fn param_load_mapping<B, const D: usize>(
    param: &Param<Tensor<B, D>>,
    tensor: Tensor<B, D>,
) -> Tensor<B, D>
where
    B: Backend,
{
    param.clone().consume().2.on_load(tensor)
}

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    use burn::{
        prelude::{
            Device,
            Tensor,
            TensorData,
        },
        tensor::Tolerance,
    };
    use serial_test::serial;

    use super::*;

    type B = PerformanceBackend;

    /// A `[2, 2]` tensor holding `values` in row-major order.
    fn square(
        values: [f64; 4],
        device: &Device<B>,
    ) -> Tensor<B, 2> {
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
    fn test_assert_tensor_close_to_vec() {
        let device = default_device();
        let t = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensor_close_to_vec(&t, &[1.0, 2.0, 3.0, 4.0], Tolerance::default());
    }

    #[test]
    #[should_panic]
    fn test_assert_tensor_close_to_vec_bad_values() {
        let device = default_device();
        let t = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensor_close_to_vec(&t, &[1.0, 2.0, 3.0, 9.0], Tolerance::default());
    }

    #[test]
    fn test_assert_tensors_close() {
        let device = default_device();
        let a = square([1.0, 2.0, 3.0, 4.0], &device);
        let b = square([1.0, 2.0, 3.0, 4.0], &device);
        assert_tensors_close(&a, &b, Tolerance::default());
    }

    #[test]
    #[should_panic]
    fn test_assert_tensors_close_bad_values() {
        let device = default_device();
        let a = square([1.0, 2.0, 3.0, 4.0], &device);
        let b = square([1.0, 2.0, 3.0, 9.0], &device);
        assert_tensors_close(&a, &b, Tolerance::default());
    }
}
