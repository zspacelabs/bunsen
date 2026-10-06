use burn::{
    Tensor,
    tensor::{
        DType,
        kind::{
            Kind,
            TensorKind,
        },
    },
};
use strum;

/// A meta-descriptor for [`burn::tensor::kind::TensorKind`].
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    strum::EnumString,
    strum::Display,
    serde::Serialize,
    serde::Deserialize,
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
    /// Returns the [`TensorKindDesc`] for a [`burn::tensor::kind::TensorKind`].
    pub const fn for_kind<K: TensorKind>() -> Self {
        Self::of(K::KIND)
    }

    /// Returns the [`TensorKindDesc`] for a [`Kind`].
    pub const fn of(kind: Kind) -> Self {
        match kind {
            Kind::Bool => TensorKindDesc::Bool,
            Kind::Float => TensorKindDesc::Float,
            Kind::Int => TensorKindDesc::Int,
        }
    }

    /// Returns the kind of the given tensor.
    pub fn kind<const R: usize, K>(_tensor: &Tensor<R, K>) -> Self
    where
        K: TensorKind + burn::tensor::kind::Basic,
    {
        Self::for_kind::<K>()
    }
}

impl From<Kind> for TensorKindDesc {
    fn from(kind: Kind) -> Self {
        Self::of(kind)
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
        support::testing::cpu_device,
    };

    #[test]
    fn test_tensor_kinds() {
        let device = cpu_device();
        assert_eq!(TensorKindDesc::for_kind::<Bool>(), TensorKindDesc::Bool);
        assert_eq!(
            TensorKindDesc::kind(&Tensor::<1, Bool>::zeros(&[1], &device)),
            TensorKindDesc::Bool
        );

        assert_eq!(TensorKindDesc::for_kind::<Float>(), TensorKindDesc::Float);
        assert_eq!(
            TensorKindDesc::kind(&Tensor::<1, Float>::zeros(&[1], &device)),
            TensorKindDesc::Float
        );

        assert_eq!(TensorKindDesc::for_kind::<Int>(), TensorKindDesc::Int);
        assert_eq!(
            TensorKindDesc::kind(&Tensor::<1, Int>::zeros(&[1], &device)),
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
