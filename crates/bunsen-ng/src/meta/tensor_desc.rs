use std::fmt::Debug;

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

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    meta::tensor_kinds::{
        ParamKindBinding,
        TensorKindDesc,
    },
    modules::ParamDesc,
};

/// Description af a Tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDesc {
    kind: TensorKindDesc,
    dtype: DType,
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

pub fn dtype_from_str(dtype: &str) -> BunsenResult<DType> {
    Ok(match dtype {
        "F64" => DType::F64,
        "F32" => DType::F32,
        "Flex32" => DType::Flex32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "I64" => DType::I64,
        "I32" => DType::I32,
        "I16" => DType::I16,
        "I8" => DType::I8,
        "U64" => DType::U64,
        "U32" => DType::U32,
        "U16" => DType::U16,
        "U8" => DType::U8,
        "Bool" => DType::Bool,
        _ => return Err(BunsenError::External(format!("Invalid dtype: {}", dtype))),
    })
}

impl TensorDesc {
    /// Create a new `TensorDesc`.
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

    /// The rank of the shape.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// The number of elements in the shape.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// The estimated size of the tensor.
    /// This ignores alignment, padding, and metadata.
    pub fn size_estimate(&self) -> usize {
        self.dtype.size() * self.num_elements()
    }
}

/// A type alias for a [`ParamDesc`] of a [`TensorDesc`].
pub type TensorParamDesc = ParamDesc<TensorDesc>;

#[cfg(test)]
#[allow(unused)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_desc() {
        type B = burn::backend::Cuda;
        let device = Default::default();

        {
            // Float
            let tensor: Tensor<B, 2> = Tensor::ones([2, 3], &device);

            let desc = TensorDesc::from(&tensor);

            assert_eq!(desc.kind, TensorKindDesc::Float);
            assert_eq!(desc.dtype, DType::F32);
            assert_eq!(desc.shape, Shape::new([2, 3]));

            assert_eq!(desc.rank(), 2);
            assert_eq!(desc.size_estimate(), DType::F32.size() * 2 * 3);
        }

        {
            // Bool
            let tensor: Tensor<B, 2, burn::tensor::Bool> = Tensor::zeros([2, 3], &device);

            let desc = TensorDesc::from(&tensor);

            assert_eq!(desc.kind, TensorKindDesc::Bool);
            assert_eq!(desc.dtype, tensor.dtype());
            assert_eq!(desc.shape, Shape::new([2, 3]));

            assert_eq!(desc.rank(), 2);
            assert_eq!(desc.size_estimate(), tensor.dtype().size() * 2 * 3);
        }

        {
            // Int
            let tensor: Tensor<B, 2, burn::tensor::Int> = Tensor::zeros([2, 3], &device);

            let desc = TensorDesc::from(&tensor);

            assert_eq!(desc.kind, TensorKindDesc::Int);
            assert_eq!(desc.dtype, tensor.dtype());
            assert_eq!(desc.shape, Shape::new([2, 3]));

            assert_eq!(desc.rank(), 2);
            assert_eq!(desc.size_estimate(), tensor.dtype().size() * 2 * 3);
        }
    }
}
