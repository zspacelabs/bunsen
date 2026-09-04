use std::any::Any;

use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
        Float,
        Int,
    },
    tensor::{
        DType,
        TensorKind,
    },
};
use serde::{
    Deserialize,
    Serialize,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KindError {
    pub msg: String,
}

/// A flag indicating the tensor kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KindFlag {
    Float,
    Int,
    Bool,
}

impl KindFlag {
    /// Returns the kind of the given tensor.
    pub fn kind<B: Backend, const R: usize, K: TensorKind<B> + 'static>(
        tensor: &Tensor<B, R, K>
    ) -> Result<Self, KindError> {
        let any: &dyn Any = tensor;

        if any.downcast_ref::<Tensor<B, R, Float>>().is_some() {
            Ok(Self::Float)
        } else if any.downcast_ref::<Tensor<B, R, Int>>().is_some() {
            Ok(Self::Int)
        } else if any.downcast_ref::<Tensor<B, R, Bool>>().is_some() {
            Ok(Self::Bool)
        } else {
            Err(KindError {
                msg: format!("Unsupported tensor kind: {:?}", K::name()),
            })
        }
    }
}

impl From<DType> for KindFlag {
    fn from(val: DType) -> Self {
        if val.is_float() {
            KindFlag::Float
        } else if val.is_int() {
            KindFlag::Int
        } else {
            KindFlag::Bool
        }
    }
}

#[cfg(test)]
mod tests {
    use bunsen::support::testing::PerformanceBackend;

    use super::*;

    type B = PerformanceBackend;

    #[test]
    fn test_kind() {
        let device = Default::default();

        assert_eq!(
            KindFlag::kind(&Tensor::<B, 2, Float>::ones([2, 3], &device)).unwrap(),
            KindFlag::Float
        );
        assert_eq!(
            KindFlag::kind(&Tensor::<B, 2, Int>::ones([2, 3], &device)).unwrap(),
            KindFlag::Int
        );
        assert_eq!(
            KindFlag::kind(&Tensor::<B, 2, Bool>::ones([2, 3], &device)).unwrap(),
            KindFlag::Bool
        );
    }
}
