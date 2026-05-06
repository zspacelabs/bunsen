use std::{
    fmt::Debug,
    ops::Deref,
};

use burn::module::{
    Param,
    ParamId,
    Parameter,
};

/// This is meta-description type of a burn [`burn::module::Param`].
///
/// This type acts as [`AsRef<T>`], [`Deref<T>`].
///
/// Currently, this will always be `Param<Tensor<_, _, _>>`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDesc<T>
where
    T: Debug + Clone + Send + PartialEq,
{
    param_id: ParamId,
    data: T,
}

impl<T> Deref for ParamDesc<T>
where
    T: Debug + Clone + Send + PartialEq,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> AsRef<T> for ParamDesc<T>
where
    T: Debug + Clone + Send + PartialEq,
{
    fn as_ref(&self) -> &T {
        &self.data
    }
}

impl<T> ParamDesc<T>
where
    T: Debug + Clone + Send + PartialEq,
{
    pub fn new(
        param_id: ParamId,
        param: T,
    ) -> Self {
        Self {
            param_id,
            data: param,
        }
    }

    /// Get the [`ParamId`].
    pub fn param_id(&self) -> ParamId {
        self.param_id
    }
}

impl<T, D> From<&Param<T>> for ParamDesc<D>
where
    T: Parameter,
    D: for<'a> From<&'a T> + Debug + Clone + Send + PartialEq + 'static,
{
    fn from(param: &Param<T>) -> Self {
        Self::new(param.id, D::from(&param.val()))
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use burn::{
        nn::LinearConfig,
        prelude::Shape,
        tensor::DType,
    };

    use super::*;
    use crate::meta::{
        TensorDesc,
        TensorKindDesc,
    };

    #[test]
    #[cfg(feature = "cuda")]
    fn test_from_param() {
        type B = burn::backend::Cuda;
        let device = Default::default();

        let linear = LinearConfig::new(2, 3).init::<B>(&device);

        let param = linear.weight;

        let param_desc: ParamDesc<TensorDesc> = (&param).into();

        assert_eq!(param_desc.param_id(), param.id);
        assert_eq!(param_desc.kind(), TensorKindDesc::Float);
        assert_eq!(param_desc.dtype(), DType::F32);
        assert_eq!(param_desc.shape(), &Shape::new([2, 3]));

        assert_eq!(param_desc.rank(), 2);
        assert_eq!(param_desc.size_estimate(), DType::F32.size() * 2 * 3);
    }
}
