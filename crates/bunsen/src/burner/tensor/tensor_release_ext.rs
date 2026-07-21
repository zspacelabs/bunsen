use burn::{
    Tensor,
    prelude::Backend,
    tensor::BasicOps,
};

/// Extension trait for `Tensor` that allows dropping the current value.
///
/// Backport of: <https://github.com/tracel-ai/burn/pull/5207>
pub trait TensorReleaseExt<B, const D: usize, K>
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
    ///
    /// Returns the previous value.
    fn release(&mut self) -> Self;
}

impl<B, const D: usize, K> TensorReleaseExt<B, D, K> for Tensor<B, D, K>
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
}
