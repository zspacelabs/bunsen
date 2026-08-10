use std::ops::Range;

use burn::{
    Tensor,
    prelude::{
        Backend,
        ElementConversion,
        TensorData,
    },
    tensor::{
        AsIndex,
        BasicOps,
        Bool,
        DType,
        DataError,
        Element,
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
    fn swap(
        &mut self,
        other: &mut Self,
    );

    /// Drop the current value, and replace it with `Tensor::empty([0; D])`.
    /// Backport of: <https://github.com/tracel-ai/burn/pull/5207>
    ///
    /// Returns the previous value.
    fn extract(&mut self) -> Self;

    /// Select (and Squeeze) a dimension.
    fn select_dim<const D2: usize>(
        self,
        dim: impl AsIndex,
        index: impl AsIndex,
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

    fn extract(&mut self) -> Self {
        let mut z = Tensor::empty([0; D], &self.device());
        self.swap(&mut z);
        z
    }

    fn select_dim<const D2: usize>(
        self,
        dim: impl AsIndex,
        index: impl AsIndex,
    ) -> Tensor<B, D2, K> {
        let dim = dim.expect_dim_index(D);
        let index = index.as_index();
        self.slice_dim(dim, index).squeeze_dim::<D2>(dim)
    }
}

/// [`Tensor`] element-type aware extension operations.
pub trait TensorElemOpExt<B, const D: usize, K>
where
    B: Backend,
    K: BasicOps<B>,
    K::Elem: Element,
{
    /// Copies the current `Tensor` into a `TensorData`; converts the dtype.
    ///
    /// By contract, this will yield the same result as
    /// `tensor.to_data().convert::<E>()`.
    ///
    /// The conversion is a no-op if the dtype is the same as the current dtype.
    fn to_data_as<E: Element>(&self) -> TensorData;

    /// Copies the current `Tensor` into `TensorData`; converts the dtype.
    ///
    /// By contract, this will yield the same result as
    /// `tensor.to_data().convert_dtype(dtype)`.
    ///
    /// The conversion is a no-op if the dtype is the same as the current dtype.
    fn to_data_dtype(
        &self,
        dtype: DType,
    ) -> TensorData;

    /// Converts the current `Tensor` into `TensorData`; converts the dtype.
    ///
    /// By contract, this will yield the same result as
    /// `tensor.into_data().convert::<E>(dtype)`.
    ///
    /// The conversion is a no-op if the dtype is the same as the current dtype.
    fn into_data_as<E: Element>(self) -> TensorData;

    /// Converts the current `Tensor` into `TensorData`; converts the dtype.
    ///
    /// By contract, this will yield the same result as
    /// `tensor.into_data().convert_dtype(dtype)`.
    ///
    /// The conversion is a no-op if the dtype is the same as the current dtype.
    fn into_data_dtype(
        self,
        dtype: DType,
    ) -> TensorData;
}

impl<B, const D: usize, K> TensorElemOpExt<B, D, K> for Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    K::Elem: Element,
{
    fn to_data_as<E: Element>(&self) -> TensorData {
        self.to_data_dtype(E::dtype())
    }

    fn to_data_dtype(
        &self,
        dtype: DType,
    ) -> TensorData {
        self.to_data().convert_dtype(dtype)
    }

    fn into_data_as<E: Element>(self) -> TensorData {
        self.into_data_dtype(E::dtype())
    }

    fn into_data_dtype(
        self,
        dtype: DType,
    ) -> TensorData {
        self.into_data().convert_dtype(dtype)
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

/// Extension trait for `TensorData` that provides additional methods.
pub trait TensorDataToVecAsExt {
    /// Cast the data to a new dtype.
    ///
    /// TODO: Implement proper error handling in `TensorData`.
    ///
    /// # Returns
    /// Ok(data) on success, (Currently) panics on failure.
    fn try_cast(
        self,
        dtype: DType,
    ) -> Result<TensorData, DataError>;

    /// Convert the data to a new dtype.
    ///
    /// TODO: Implement proper error handling in `TensorData`.
    ///
    /// By contract, this is equivalent to:
    /// `data.try_cast(E::dtype())`
    ///
    /// # Returns
    /// Ok(data) on success, (Currently) panics on failure.
    fn try_convert<E: Element>(self) -> Result<TensorData, DataError>;

    /// Copy and convert the data to a [`Vec<E>`].
    ///
    /// By contract, this is equivalent to:
    /// `data.clone().into_vec_as::<E>()`
    ///
    /// Particular conversions may provide more efficient implementations.
    ///
    /// # Returns
    /// `Ok(vec)` on success, or an error if the conversion fails.
    fn to_vec_as<E: Element>(&self) -> Result<Vec<E>, DataError>;

    /// Convert the data to [`Vec<E>`].
    ///
    /// By contract, this is equivalent to:
    /// `data.try_convert::<E>()?.to_vec::<E>()`
    ///
    /// Particular conversions may provide more efficient implementations.
    ///
    /// # Returns
    /// `Ok(vec)` on success, or an error if the conversion fails.
    fn into_vec_as<E: Element>(self) -> Result<Vec<E>, DataError>;
}

impl TensorDataToVecAsExt for TensorData {
    fn try_cast(
        self,
        dtype: DType,
    ) -> Result<TensorData, DataError> {
        Ok(self.convert_dtype(dtype))
    }

    fn try_convert<E: Element>(self) -> Result<TensorData, DataError> {
        self.try_cast(E::dtype())
    }

    fn to_vec_as<E: Element>(&self) -> Result<Vec<E>, DataError> {
        self.clone().into_vec_as::<E>()
    }

    fn into_vec_as<E: Element>(self) -> Result<Vec<E>, DataError> {
        self.try_convert::<E>()?.to_vec::<E>()
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

        let mut old: Tensor<B, 1> = tensor.extract();
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

    #[test]
    fn test_to_vec_as() {
        let data = TensorData::from([0.0f32, 1.0, 2.5]);

        // Same-dtype copy.
        assert_eq!(data.to_vec_as::<f32>().unwrap(), vec![0.0f32, 1.0, 2.5]);

        // Widening cast (different element size).
        assert_eq!(data.to_vec_as::<f64>().unwrap(), vec![0.0f64, 1.0, 2.5]);

        // Float to int cast (same element size) truncates.
        assert_eq!(data.to_vec_as::<i32>().unwrap(), vec![0i32, 1, 2]);

        // The source data is borrowed, not consumed.
        data.assert_eq(&TensorData::from([0.0f32, 1.0, 2.5]), true);
    }

    #[test]
    fn test_into_vec_as() {
        let data = TensorData::from([0i32, 1, 2, 3]);

        // Same-dtype conversion.
        assert_eq!(
            data.clone().into_vec_as::<i32>().unwrap(),
            vec![0i32, 1, 2, 3]
        );

        // Int to float cast.
        assert_eq!(
            data.clone().into_vec_as::<f32>().unwrap(),
            vec![0.0f32, 1.0, 2.0, 3.0]
        );

        // Narrowing int cast.
        assert_eq!(data.into_vec_as::<u8>().unwrap(), vec![0u8, 1, 2, 3]);
    }
}
