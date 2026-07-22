use burn::{
    Tensor,
    prelude::Backend,
    tensor::BasicOps,
};

use crate::burner::tensor::TensorReleaseExt;

/// Tensor Extension trait for `Tensor::inplace_result`.
pub trait TensorInplaceResultExt<B, const D: usize, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    /// Value-returning variant of `Tensor::inplace`.
    ///
    /// Executes an operation on the tensor and modifies its value.
    ///
    /// # Notes
    ///
    /// This won't necessarily reuse the same tensor data/buffer, but it should
    /// if there is no other reference pointing to the same tensor.
    ///
    /// Wrapping operations with inplace is not an optimization, it's mainly
    /// there if you want to mutate a tensor by using owned operations. A
    /// plausible usage would be to update the weights of a mutable model
    /// reference.
    fn inplace_result<V, F>(
        &mut self,
        func: F,
    ) -> V
    where
        F: FnOnce(Tensor<B, D, K>) -> (Tensor<B, D, K>, V);
}

impl<B, const D: usize, K> TensorInplaceResultExt<B, D, K> for Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    fn inplace_result<V, F>(
        &mut self,
        func: F,
    ) -> V
    where
        F: FnOnce(Tensor<B, D, K>) -> (Tensor<B, D, K>, V),
    {
        let (mut z, v) = func(self.release());
        self.swap(&mut z);
        v
    }
}
