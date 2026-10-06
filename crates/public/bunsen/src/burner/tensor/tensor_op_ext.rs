use std::ops::Range;

use burn::{
    Tensor,
    prelude::{
        ElementConversion,
        SliceArg,
    },
    tensor::{
        AsIndex,
        Bool,
        Int,
        kind::{
            Basic,
            Ordered,
        },
    },
};

/// Operation Extensions for `Tensor<D, K>`.
pub trait TensorOpExt<const D: usize, K>
where
    K: Basic,
{
    /// Swap this tensor with another.
    fn swap(
        &mut self,
        other: &mut Self,
    );

    /// Replace the current value with `f(current)`.
    ///
    /// `f` receives the tensor by value as its sole owner, so operations
    /// inside it may run in place. The receiver holds `Tensor::empty([0; D])`
    /// while `f` runs, and it stays mutably borrowed for the whole call, so a
    /// closure that calls a method on the struct owning the receiver does not
    /// compile: the placeholder cannot be read back that way.
    fn replace_with<F>(
        &mut self,
        f: F,
    ) where
        Self: Sized,
        F: FnOnce(Self) -> Self;

    /// Copy an internal slice from `from` to `to`.
    fn copy_slice<S1: SliceArg, S2: SliceArg>(
        self,
        to: S1,
        from: S2,
    ) -> Tensor<D, K>;
}

impl<const D: usize, K> TensorOpExt<D, K> for Tensor<D, K>
where
    K: Basic,
{
    fn swap(
        &mut self,
        other: &mut Self,
    ) {
        core::mem::swap(self, other);
    }

    fn replace_with<F>(
        &mut self,
        f: F,
    ) where
        Self: Sized,
        F: FnOnce(Self) -> Self,
    {
        let current = self.extract();
        *self = f(current);
    }

    fn copy_slice<S1: SliceArg, S2: SliceArg>(
        self,
        to: S1,
        from: S2,
    ) -> Tensor<D, K> {
        let tmp = self.clone().slice(from);
        self.slice_assign(to, tmp)
    }
}

/// Tensor Extension trait for ordered operations.
pub trait TensorOrderedOpExt<const D: usize, K>
where
    K: Ordered,
{
    /// Elementwise check if the value is in the `[start, end)` range.
    fn in_range_scalar<E: ElementConversion>(
        self,
        range: Range<E>,
    ) -> Tensor<D, Bool>;

    /// Elementwise check if the value is in the `[start, end)` range.
    fn in_range(
        self,
        start: Tensor<D, K>,
        end: Tensor<D, K>,
    ) -> Tensor<D, Bool>;
}

impl<const D: usize, K> TensorOrderedOpExt<D, K> for Tensor<D, K>
where
    K: Ordered,
{
    fn in_range_scalar<E: ElementConversion>(
        self,
        range: Range<E>,
    ) -> Tensor<D, Bool> {
        self.clone()
            .greater_equal_scalar(range.start)
            .bool_and(self.lower_scalar(range.end))
    }

    fn in_range(
        self,
        start: Tensor<D, K>,
        end: Tensor<D, K>,
    ) -> Tensor<D, Bool> {
        assert_eq!(self.shape(), start.shape());
        assert_eq!(self.shape(), end.shape());
        self.clone().greater_equal(start).bool_and(self.lower(end))
    }
}

/// Operation Extensions for `Tensor<D, Bool>`.
pub trait TensorBoolOpExt<const D: usize> {
    /// Aggregate a count of all true elements along the given *dimension* or
    /// *axis* in the tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements;
    ///   supports negative indexing.
    fn count_dim<I: AsIndex>(
        self,
        dim: I,
    ) -> Tensor<D, Int>;

    /// Aggregate a count of all true elements along the given *axes* in the
    /// tensor.
    ///
    /// # Arguments
    ///
    /// * `dims` - the dimensions to aggregate; supports negative indexing.
    ///
    /// # Returns
    ///
    /// The returned tensor will have the same rank,
    /// but the aggregated dimensions will have size 1.
    fn count_dims<I: AsIndex>(
        self,
        dims: &[I],
    ) -> Tensor<D, Int>;
}

impl<const D: usize> TensorBoolOpExt<D> for Tensor<D, Bool> {
    fn count_dim<I: AsIndex>(
        self,
        dim: I,
    ) -> Tensor<D, Int> {
        self.int().sum_dim(dim)
    }

    fn count_dims<I: AsIndex>(
        self,
        dims: &[I],
    ) -> Tensor<D, Int> {
        self.int().sum_dims(dims)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Tensor,
        TensorData,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_release_swap() {
        let device = cpu_device();
        let mut tensor: Tensor<1> =
            Tensor::<1>::from_data(TensorData::from([0.0, 1.0, 2.0, 3.0]), &device);
        assert_eq!(tensor.dims(), [4]);

        let mut old: Tensor<1> = tensor.extract();
        assert_eq!(tensor.dims(), [0]);
        assert_eq!(old.dims(), [4]);

        tensor.swap(&mut old);
        assert_eq!(tensor.dims(), [4]);
        assert_eq!(old.dims(), [0]);
    }

    #[test]
    fn test_replace_with() {
        let device = cpu_device();
        let mut tensor: Tensor<1> =
            Tensor::<1>::from_data(TensorData::from([0.0, 1.0, 2.0, 3.0]), &device);

        tensor.replace_with(|current| {
            // `current` is the sole owner of the handle.
            assert_eq!(current.dims(), [4]);
            current.mul_scalar(2.0)
        });

        tensor
            .to_data()
            .assert_eq(&TensorData::from([0.0, 2.0, 4.0, 6.0]), false);
    }

    #[test]
    fn test_in_range_scalar() {
        let device = cpu_device();
        let x: Tensor<1, Int> = Tensor::from_data([0, 1, 2, 3], &device);

        let b = x.in_range_scalar(1..3);

        b.to_data()
            .assert_eq(&TensorData::from([false, true, true, false]), false);
    }

    #[test]
    fn test_in_range() {
        let device = cpu_device();
        let x: Tensor<1, Int> = Tensor::from_data([0, 0, 0, 0], &device);

        let start: Tensor<1, Int> = Tensor::from_data([-1, 0, 0, 3], &device);
        let end: Tensor<1, Int> = Tensor::from_data([0, 0, 2, 3], &device);

        let b = x.in_range(start, end);

        b.to_data()
            .assert_eq(&TensorData::from([false, false, true, false]), false);
    }

    #[test]
    fn test_bool_count_dim() {
        let device = cpu_device();
        let x: Tensor<2, Bool> =
            Tensor::from_data([[true, true, false], [true, false, false]], &device);

        x.clone()
            .count_dim(0)
            .squeeze_dim::<1>(0)
            .to_data()
            .assert_eq(&TensorData::from([2, 1, 0]), false);
        x.clone()
            .count_dim(1)
            .squeeze_dim::<1>(1)
            .to_data()
            .assert_eq(&TensorData::from([2, 1]), false);

        x.clone()
            .count_dims(&[0])
            .squeeze_dim::<1>(0)
            .to_data()
            .assert_eq(&TensorData::from([2, 1, 0]), false);
        x.clone()
            .count_dims(&[0, 1])
            .squeeze_dim::<1>(0)
            .to_data()
            .assert_eq(&TensorData::from([3]), false);
    }
}
