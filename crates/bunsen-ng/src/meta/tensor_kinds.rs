use burn::prelude;

/// Encodes a description af [`burn::tensor::TensorKind`].
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
    pub const fn for_kind<K: ParamKindBinding>() -> Self {
        K::KIND
    }
}

/// A trait that binds a burn Tensor Kind to a `ParamKind`.
pub trait ParamKindBinding {
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
    use burn::tensor;

    use crate::meta::TensorKindDesc;

    #[test]
    fn test_tensor_kinds() {
        assert_eq!(
            TensorKindDesc::for_kind::<tensor::Bool>(),
            TensorKindDesc::Bool
        );
        assert_eq!(
            TensorKindDesc::for_kind::<tensor::Float>(),
            TensorKindDesc::Float
        );
        assert_eq!(
            TensorKindDesc::for_kind::<tensor::Int>(),
            TensorKindDesc::Int
        );
    }
}
