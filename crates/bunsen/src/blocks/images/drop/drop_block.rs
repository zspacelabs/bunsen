//! # `DropBlock` Layers
//!
//! Based upon [DropBlock (Ghiasi et al., 2018)](https://arxiv.org/pdf/1810.12890.pdf);
//! inspired also by the `python-image-models` implementation.

use burn::{
    config::Config,
    module::Module,
    prelude::{
        Backend,
        Tensor,
    },
};

#[doc(inline)]
pub use crate::ops::drop::DropBlockOptions;
use crate::ops::drop::drop_block_2d;

/// Config for [`DropBlock2dConfig`] module.
#[derive(Config, Debug)]
pub struct DropBlock2dConfig {
    /// The options for the drop block algorithm.
    #[config(default = "DropBlockOptions::default()")]
    pub options: DropBlockOptions,
}

impl From<DropBlockOptions> for DropBlock2dConfig {
    fn from(options: DropBlockOptions) -> Self {
        Self { options }
    }
}

impl DropBlock2dConfig {
    /// Initialize a [`DropBlock2d`] module.
    pub fn init(&self) -> DropBlock2d {
        DropBlock2d {
            options: self.options.clone(),
        }
    }
}

/// `DropBlock2d`
///
/// A module that applies drop block (when training).
///
/// Based upon [DropBlock (Ghiasi et al., 2018)](https://arxiv.org/pdf/1810.12890.pdf);
/// inspired also by the `python-image-models` implementation.
#[derive(Module, Clone, Debug)]
pub struct DropBlock2d {
    /// The options for the drop block algorithm.
    pub options: DropBlockOptions,
}

impl DropBlock2d {
    /// When training, applies `drop_block_2d` to the tensor;
    /// otherwise, is a no-op pass-through.
    ///
    /// # Arguments
    ///
    /// - `tensor` - the input.
    ///
    /// # Returns
    ///
    /// A tensor of the same shape, type, and device.
    pub fn forward<B: Backend>(
        &self,
        tensor: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        if B::ad_enabled(&tensor.device()) {
            drop_block_2d(tensor.clone(), &self.options)
        } else {
            tensor
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Autodiff;

    use super::*;
    use crate::support::testing::CpuBackend;

    #[test]
    fn test_module_inference() {
        type B = CpuBackend;
        let device = Default::default();

        let config = DropBlock2dConfig::new();

        let module = config.init();

        let batch_size = 2;
        let channels = 3;
        let height = 100;
        let width = height;
        let shape = [batch_size, channels, height, width];

        let tensor: Tensor<B, 4> = Tensor::ones(shape, &device);

        assert_eq!(B::ad_enabled(&tensor.device()), false);
        let result = module.forward(tensor.clone());

        // Not under training; so a no-op.
        result.to_data().assert_eq(&tensor.to_data(), false);
    }

    #[test]
    fn test_module_training() {
        type I = CpuBackend;
        type B = Autodiff<I>;
        let device = Default::default();

        let drop_prob = 0.1;

        let config = DropBlock2dConfig::new().with_options(
            DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_kernel([2, 3]),
        );

        let module = config.init();

        let batch_size = 2;
        let channels = 3;
        let height = 100;
        let width = height;
        let shape = [batch_size, channels, height, width];

        let tensor: Tensor<B, 4> = Tensor::ones(shape, &device);

        assert_eq!(B::ad_enabled(&tensor.device()), true);
        let drop = module.forward(tensor.clone());

        // Count all 1.0; which are the non-dropped values.
        let numel = drop.shape().num_elements();

        // They've all been rescaled upwards.
        let keep_count = drop.clone().greater_elem(1.0).int().sum().into_scalar() as usize;
        let drop_count = numel - keep_count;
        let drop_ratio = drop_count as f64 / numel as f64;
        assert!((drop_ratio - drop_prob).abs() < 0.15);

        let total = drop.sum().into_scalar() as f64;
        let norm = total / numel as f64;
        assert!((norm - 1.0).abs() < 0.01);
    }
}
