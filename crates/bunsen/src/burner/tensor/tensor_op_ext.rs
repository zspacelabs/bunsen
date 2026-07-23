use burn::{
    Tensor,
    prelude::Backend,
    tensor::{
        BasicOps,
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
    /// Backport of: <https://github.com/tracel-ai/burn/pull/5207>
    fn swap(
        &mut self,
        other: &mut Self,
    );

    /// Drop the current value, and replace it with `Tensor::empty([0; D])`.
    /// Backport of: <https://github.com/tracel-ai/burn/pull/5207>
    ///
    /// Returns the previous value.
    fn release(&mut self) -> Self;
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

    fn release(&mut self) -> Self {
        let mut z = Tensor::empty([0; D], &self.device());
        self.swap(&mut z);
        z
    }
}

/// Operation Extensions for `Tensor<B, D, Int>`.
pub trait TensorIntOpExt<B, const D: usize>
where
    B: Backend,
{
    /// Returns the square of the tensor.
    /// Backport of: <https://github.com/tracel-ai/burn/pull/5224>
    fn square(self) -> Tensor<B, D, Int>;
}

impl<B, const D: usize> TensorIntOpExt<B, D> for Tensor<B, D, Int>
where
    B: Backend,
{
    fn square(self) -> Tensor<B, D, Int> {
        self.powi_scalar(2)
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
        let mut tensor: Tensor<B, 1> =
            Tensor::<B, 1>::from_data(TensorData::from([0.0, 1.0, 2.0, 3.0]), &Default::default());
        assert_eq!(tensor.dims(), [4]);

        let mut old: Tensor<B, 1> = tensor.release();
        assert_eq!(tensor.dims(), [0]);
        assert_eq!(old.dims(), [4]);

        tensor.swap(&mut old);
        assert_eq!(tensor.dims(), [4]);
        assert_eq!(old.dims(), [0]);
    }

    #[test]
    fn test_square() {
        let device = Default::default();
        let x: Tensor<B, 1, Int> = Tensor::from_data([0, 1, 2, 3], &device);

        x.square()
            .to_data()
            .assert_eq(&TensorData::from([0, 1, 4, 9]), false);
    }
}
