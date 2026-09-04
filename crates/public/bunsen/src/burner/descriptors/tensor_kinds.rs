use burn::{
    Tensor,
    prelude,
    prelude::Backend,
    tensor::{
        DType,
        TensorKind,
    },
};
use strum;

/// A meta-descriptor for [`burn::tensor::TensorKind`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, strum::EnumString, strum::Display,
)]
#[non_exhaustive]
pub enum TensorKindDesc {
    /// A Bool Tensor
    /// Equivalent to [`burn::tensor::Bool`].
    Bool,

    /// A Float Tensor
    /// Equivalent to [`burn::tensor::Float`].
    Float,

    /// An Int Tensor
    /// Equivalent to [`burn::tensor::Int`].
    Int,
}

impl TensorKindDesc {
    /// Returns the [`TensorKindDesc`] for a `burner`
    /// [`burn::tensor::TensorKind`].
    pub const fn for_kind<K: ParamKindBinding>() -> Self {
        K::KIND
    }

    /// Returns the kind of the given tensor.
    pub fn kind<B, const R: usize, K>(_tensor: &Tensor<B, R, K>) -> Self
    where
        B: Backend,
        K: TensorKind<B> + ParamKindBinding,
    {
        Self::for_kind::<K>()
    }
}

impl From<DType> for TensorKindDesc {
    fn from(dtype: DType) -> Self {
        if dtype.is_float() {
            TensorKindDesc::Float
        } else if dtype.is_int() {
            TensorKindDesc::Int
        } else if dtype.is_bool() {
            TensorKindDesc::Bool
        } else {
            panic!("Unsupported dtype: {dtype:?}")
        }
    }
}

/// A trait that binds a `burner` Tensor Kind to a `ParamKind`.
pub trait ParamKindBinding {
    /// The [`TensorKindDesc`] kind wrapper.
    const KIND: TensorKindDesc;
}

impl ParamKindBinding for prelude::Bool {
    const KIND: TensorKindDesc = TensorKindDesc::Bool;
}

impl ParamKindBinding for prelude::Float {
    const KIND: TensorKindDesc = TensorKindDesc::Float;
}

impl ParamKindBinding for prelude::Int {
    const KIND: TensorKindDesc = TensorKindDesc::Int;
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::{
            Bool,
            Float,
            Int,
            Tensor,
        },
        tensor::{
            BoolStore,
            DType,
        },
    };

    use crate::{
        burner::descriptors::TensorKindDesc,
        support::testing::CpuBackend,
    };

    type B = CpuBackend;

    #[test]
    fn test_tensor_kinds() {
        let device = Default::default();
        assert_eq!(TensorKindDesc::for_kind::<Bool>(), TensorKindDesc::Bool);
        assert_eq!(
            TensorKindDesc::kind(&Tensor::<B, 1, Bool>::zeros(&[1], &device)),
            TensorKindDesc::Bool
        );

        assert_eq!(TensorKindDesc::for_kind::<Float>(), TensorKindDesc::Float);
        assert_eq!(
            TensorKindDesc::kind(&Tensor::<B, 1, Float>::zeros(&[1], &device)),
            TensorKindDesc::Float
        );

        assert_eq!(TensorKindDesc::for_kind::<Int>(), TensorKindDesc::Int);
        assert_eq!(
            TensorKindDesc::kind(&Tensor::<B, 1, Int>::zeros(&[1], &device)),
            TensorKindDesc::Int
        );
    }

    #[test]
    fn test_from_dtype() {
        assert_eq!(
            TensorKindDesc::from(DType::Bool(BoolStore::Native)),
            TensorKindDesc::Bool
        );
        assert_eq!(TensorKindDesc::from(DType::F64), TensorKindDesc::Float);
        assert_eq!(TensorKindDesc::from(DType::I64), TensorKindDesc::Int);
    }
}
