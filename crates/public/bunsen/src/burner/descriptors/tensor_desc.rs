use std::{
    fmt::Debug,
    ops::Deref,
};

use burn::{
    Tensor,
    prelude::{
        Backend,
        Shape,
    },
    tensor::{
        BasicOps,
        DType,
    },
};

use crate::burner::descriptors::{
    ParamDesc,
    ParamKindBinding,
    TensorKindDesc,
};

/// This is meta-descriptor for kind/type + rank slot for [`Tensor`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TensorRankDesc {
    /// Kind of the tensor.
    kind: TensorKindDesc,

    /// Data type of the tensor.
    dtype: DType,

    /// Rank of the tensor.
    rank: usize,
}

impl<B, const R: usize, K> From<&Tensor<B, R, K>> for TensorRankDesc
where
    B: Backend,
    K: BasicOps<B>,
    K: ParamKindBinding,
{
    fn from(param: &Tensor<B, R, K>) -> Self {
        Self {
            kind: TensorKindDesc::for_kind::<K>(),
            dtype: param.dtype(),
            rank: param.rank(),
        }
    }
}

impl TensorRankDesc {
    /// Creates a new `TensorDesc`.
    pub fn new(
        kind: TensorKindDesc,
        dtype: DType,
        rank: usize,
    ) -> Self {
        Self { kind, dtype, rank }
    }

    /// The [`TensorKindDesc`] kind wrapper.
    pub fn kind(&self) -> TensorKindDesc {
        self.kind
    }

    /// The [`DType`] dtype wrapper.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The estimated size of the tensor.
    /// This ignores alignment, padding, and metadata.
    pub fn size_estimate(
        &self,
        num_elements: usize,
    ) -> usize {
        num_elements * self.dtype.size()
    }

    /// Creates a new [`TensorDesc`] from the given shape.
    pub fn to_desc(
        &self,
        shape: impl Into<Shape>,
    ) -> TensorDesc {
        let shape = shape.into();
        assert_eq!(shape.rank(), self.rank);
        TensorDesc {
            kind: self.kind,
            dtype: self.dtype,
            shape,
        }
    }
}

/// This is meta-descriptor for a [`Tensor`].
///
/// This also delegates the [`Shape`] api through [`Deref`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TensorDesc {
    /// Kind of the tensor.
    kind: TensorKindDesc,

    /// Data type of the tensor.
    dtype: DType,

    /// [`Shape`] of the tensor.
    shape: Shape,
}

impl<B, const R: usize, K> From<&Tensor<B, R, K>> for TensorDesc
where
    B: Backend,
    K: BasicOps<B>,
    K: ParamKindBinding,
{
    fn from(param: &Tensor<B, R, K>) -> Self {
        Self {
            kind: TensorKindDesc::for_kind::<K>(),
            dtype: param.dtype(),
            shape: param.shape(),
        }
    }
}

impl Deref for TensorDesc {
    type Target = Shape;

    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}

impl TensorDesc {
    /// Creates a new `TensorDesc`.
    pub fn new(
        kind: TensorKindDesc,
        dtype: DType,
        shape: Shape,
    ) -> Self {
        Self { kind, dtype, shape }
    }

    /// The [`TensorKindDesc`] kind wrapper.
    pub fn kind(&self) -> TensorKindDesc {
        self.kind
    }

    /// The dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The estimated size of the tensor.
    /// This ignores alignment, padding, and metadata.
    pub fn size_estimate(&self) -> usize {
        self.num_elements() * self.dtype.size()
    }
}

/// A type alias for a [`ParamDesc`] of a [`TensorDesc`].
pub type TensorParamDesc = ParamDesc<TensorDesc>;

#[cfg(test)]
#[allow(unused)]
mod tests {
    use burn::prelude::{
        Bool,
        Float,
        Int,
    };

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        PerformanceBackend,
        backend_device,
        default_device,
    };

    #[test]
    fn test_is_send() {
        fn is_send<T: Send>(obj: &T) {}
        is_send(&TensorDesc {
            kind: TensorKindDesc::Float,
            dtype: DType::F32,
            shape: Shape::new([2, 3]),
        });
    }

    #[test]
    fn test_is_sync() {
        fn is_sync<T: Sync>(obj: &T) {}
        is_sync(&TensorDesc {
            kind: TensorKindDesc::Float,
            dtype: DType::F32,
            shape: Shape::new([2, 3]),
        });
    }

    #[test]
    fn test_tensor_rank_desc() {
        type B = PerformanceBackend;
        let device = backend_device::<B>();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        {
            // Float
            let tensor: Tensor<B, 2> = Tensor::ones([2, 3], &device);
            let dtype = tensor.dtype();

            let rank_desc: TensorRankDesc = TensorRankDesc::from(tensor);
            assert_eq!(rank_desc.kind(), TensorKindDesc::Float);
            assert_eq!(rank_desc.dtype(), dtype);
            assert_eq!(rank_desc.rank(), 2);
            assert_eq!(rank_desc.size_estimate(6), dtype.size() * 2 * 3);

            let desc = rank_desc.to_desc(Shape::new([2, 3]));
            assert_eq!(desc.kind(), TensorKindDesc::Float);
            assert_eq!(desc.dtype(), dtype);
            assert_eq!(desc.shape(), &Shape::new([2, 3]));
            assert_eq!(desc.rank(), 2);
            assert_eq!(desc.size_estimate(), dtype.size() * 2 * 3);
        }

        {
            // Int
            let tensor: Tensor<B, 2, Int> = Tensor::ones([2, 3], &device);
            let dtype = tensor.dtype();

            let rank_desc: TensorRankDesc = TensorRankDesc::from(tensor);
            assert_eq!(rank_desc.kind(), TensorKindDesc::Int);
            assert_eq!(rank_desc.dtype(), dtype);
            assert_eq!(rank_desc.rank(), 2);
            assert_eq!(rank_desc.size_estimate(6), dtype.size() * 2 * 3);

            let desc = rank_desc.to_desc(Shape::new([2, 3]));
            assert_eq!(desc.kind(), TensorKindDesc::Int);
            assert_eq!(desc.dtype(), dtype);
            assert_eq!(desc.shape(), &Shape::new([2, 3]));
            assert_eq!(desc.rank(), 2);
            assert_eq!(desc.size_estimate(), dtype.size() * 2 * 3);
        }

        {
            // Bool
            let tensor: Tensor<B, 2, Bool> = Tensor::zeros([2, 3], &device);
            let dtype = tensor.dtype();

            let rank_desc: TensorRankDesc = TensorRankDesc::from(tensor);
            assert_eq!(rank_desc.kind, TensorKindDesc::Bool);
            assert_eq!(rank_desc.dtype, dtype);
            assert_eq!(rank_desc.rank(), 2);
            assert_eq!(rank_desc.size_estimate(6), dtype.size() * 2 * 3);

            let desc = rank_desc.to_desc(Shape::new([2, 3]));
            assert_eq!(desc.kind, TensorKindDesc::Float);
            assert_eq!(desc.dtype, dtype);
            assert_eq!(desc.shape, Shape::new([2, 3]));
            assert_eq!(desc.rank(), 2);
            assert_eq!(desc.size_estimate(), dtype.size() * 2 * 3);
        }
    }
}
