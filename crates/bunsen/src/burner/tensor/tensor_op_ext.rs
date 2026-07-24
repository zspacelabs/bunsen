use std::ops::Range;

use burn::{
    Tensor,
    prelude::{
        Backend,
        ElementConversion,
    },
    tensor::{
        AsIndex,
        BasicOps,
        Bool,
        Float,
        Int,
    },
};

/// Operation Extensions for `Tensor<B, D, K>`.
pub trait TensorOpExt<B, const D: usize, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    /// Swap this tensor with another.
    /// Backport of: <https://github.com/tracel-ai/burn/pull/5207>
    fn swap(
        &mut self,
        other: &mut Self,
    );

    /// Drop the current value, and replace it with `Tensor::empty([0; D])`.
    /// Backport of: <https://github.com/tracel-ai/burn/pull/5207>
    ///
    /// Returns the previous value.
    fn release(&mut self) -> Self;

    /// Select (and Squeeze) a dimension.
    fn select_dim<const D2: usize, I: AsIndex>(
        self,
        dim: usize,
        index: I,
    ) -> Tensor<B, D2, K>;
}

impl<B, const D: usize, K> TensorOpExt<B, D, K> for Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    fn swap(
        &mut self,
        other: &mut Self,
    ) {
        core::mem::swap(self, other);
    }

    fn release(&mut self) -> Self {
        let mut z = Tensor::empty([0; D], &self.device());
        self.swap(&mut z);
        z
    }

    fn select_dim<const D2: usize, I: AsIndex>(
        self,
        dim: usize,
        index: I,
    ) -> Tensor<B, D2, K> {
        let index = index.as_index();
        self.slice_dim(dim, index..index + 1).squeeze_dim::<D2>(dim)
    }
}

/// Tensor Extension trait for ordered operations.
pub trait TensorOrderedOpExt<B, const D: usize, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    /// Elementwise check if the value is in the `[start, end)` range.
    fn in_range_scalar<E: ElementConversion>(
        self,
        range: Range<E>,
    ) -> Tensor<B, D, Bool>;

    /// Elementwise check if the value is in the `[start, end)` range.
    fn in_range(
        self,
        start: Tensor<B, D, K>,
        end: Tensor<B, D, K>,
    ) -> Tensor<B, D, Bool>;
}

// Impls duplicated because burn doesn't expose Ordered<B>
impl<B, const D: usize> TensorOrderedOpExt<B, D, Float> for Tensor<B, D>
where
    B: Backend,
{
    fn in_range_scalar<E: ElementConversion>(
        self,
        range: Range<E>,
    ) -> Tensor<B, D, Bool> {
        self.clone()
            .greater_equal_elem(range.start)
            .bool_and(self.lower_elem(range.end))
    }

    fn in_range(
        self,
        start: Tensor<B, D>,
        end: Tensor<B, D>,
    ) -> Tensor<B, D, Bool> {
        assert_eq!(self.shape(), start.shape());
        assert_eq!(self.shape(), end.shape());
        self.clone().greater_equal(start).bool_and(self.lower(end))
    }
}

// Impls duplicated because burn doesn't expose Ordered<B>
impl<B, const D: usize> TensorOrderedOpExt<B, D, Int> for Tensor<B, D, Int>
where
    B: Backend,
{
    fn in_range_scalar<E: ElementConversion>(
        self,
        range: Range<E>,
    ) -> Tensor<B, D, Bool> {
        self.clone()
            .greater_equal_elem(range.start)
            .bool_and(self.lower_elem(range.end))
    }

    fn in_range(
        self,
        start: Tensor<B, D, Int>,
        end: Tensor<B, D, Int>,
    ) -> Tensor<B, D, Bool> {
        assert_eq!(self.shape(), start.shape());
        assert_eq!(self.shape(), end.shape());
        self.clone().greater_equal(start).bool_and(self.lower(end))
    }
}

/// Operation Extensions for `Tensor<B, D, Int>`.
pub trait TensorIntOpExt<B, const D: usize>
where
    B: Backend,
{
    /// Returns the square of the tensor.
    /// Backport of: <https://github.com/tracel-ai/burn/pull/5224>
    fn square(self) -> Self;
}

impl<B, const D: usize> TensorIntOpExt<B, D> for Tensor<B, D, Int>
where
    B: Backend,
{
    fn square(self) -> Self {
        self.powi_scalar(2)
    }
}

/// Operation Extensions for `Tensor<B, D, Bool>`.
pub trait TensorBoolOpExt<B, const D: usize>
where
    B: Backend,
{
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
    ) -> Tensor<B, D, Int>;

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
    ) -> Tensor<B, D, Int>;
}

impl<B, const D: usize> TensorBoolOpExt<B, D> for Tensor<B, D, Bool>
where
    B: Backend,
{
    fn count_dim<I: AsIndex>(
        self,
        dim: I,
    ) -> Tensor<B, D, Int> {
        self.int().sum_dim(dim)
    }

    fn count_dims<I: AsIndex>(
        self,
        dims: &[I],
    ) -> Tensor<B, D, Int> {
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
    use crate::support::testing::CpuBackend;
    type B = CpuBackend;

    #[test]
    fn test_release_swap() {
        let device = Default::default();
        let mut tensor: Tensor<B, 1> =
            Tensor::<B, 1>::from_data(TensorData::from([0.0, 1.0, 2.0, 3.0]), &device);
        assert_eq!(tensor.dims(), [4]);

        let mut old: Tensor<B, 1> = tensor.release();
        assert_eq!(tensor.dims(), [0]);
        assert_eq!(old.dims(), [4]);

        tensor.swap(&mut old);
        assert_eq!(tensor.dims(), [4]);
        assert_eq!(old.dims(), [0]);
    }

    #[test]
    fn test_select_dim() {
        let device = Default::default();
        let tensor: Tensor<B, 2> =
            Tensor::from_data(TensorData::from([[0.0, 1.0], [2.0, 3.0]]), &device);

        let r1: Tensor<B, 1> = tensor.clone().select_dim(0, 1);
        r1.to_data().assert_eq(&TensorData::from([2.0, 3.0]), false);

        let c1: Tensor<B, 1> = tensor.clone().select_dim(1, 1);
        c1.to_data().assert_eq(&TensorData::from([1.0, 3.0]), false);
    }

    #[test]
    fn test_in_range_scalar() {
        let device = Default::default();
        let x: Tensor<B, 1, Int> = Tensor::from_data([0, 1, 2, 3], &device);

        let b = x.in_range_scalar(1..3);

        b.to_data()
            .assert_eq(&TensorData::from([false, true, true, false]), false);
    }

    #[test]
    fn test_in_range() {
        let device = Default::default();
        let x: Tensor<B, 1, Int> = Tensor::from_data([0, 0, 0, 0], &device);

        let start: Tensor<B, 1, Int> = Tensor::from_data([-1, 0, 0, 3], &device);
        let end: Tensor<B, 1, Int> = Tensor::from_data([0, 0, 2, 3], &device);

        let b = x.in_range(start, end);

        b.to_data()
            .assert_eq(&TensorData::from([false, false, true, false]), false);
    }

    #[test]
    fn test_int_square() {
        let device = Default::default();
        let x: Tensor<B, 1, Int> = Tensor::from_data([0, 1, 2, 3], &device);

        x.square()
            .to_data()
            .assert_eq(&TensorData::from([0, 1, 4, 9]), false);
    }

    #[test]
    fn test_bool_count_dim() {
        let device = Default::default();
        let x: Tensor<B, 2, Bool> =
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
