//! [`DropPath`]: stochastic depth as a module.
//!
//! Paper: Deep Networks with Stochastic Depth (Huang et al., 2016,
//! <https://arxiv.org/abs/1603.09382>).
//!
//! Inspired by the python implementation from the timm library:
//! <https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py>
//!
//! The tensor operation is
//! [`ops::drop::drop_path`](crate::ops::drop::drop_path); [`DropPath`] holds
//! its settings and decides when it runs.

use burn::{
    config::Config,
    module::Module,
    prelude::{
        Backend,
        Tensor,
    },
};

use crate::{
    ops::drop::drop_path,
    support::validators,
};

/// Common introspection interface for `DropPath` module.
pub trait DropPathMeta {
    /// Returns the drop probability.
    fn drop_prob(&self) -> f64;

    /// Returns the keep probability, which is `1.0 - drop_prob`.
    fn keep_prob(&self) -> f64 {
        1.0 - self.drop_prob()
    }

    /// Returns whether the output is scaled by `1 / (1 - drop_prob)`.
    fn scale_by_keep(&self) -> bool;
}

/// Configuration for the [`DropPath`] module.
#[derive(Config, Debug)]
pub struct DropPathConfig {
    /// Probability of dropping a path.
    #[config(default = 0.0)]
    pub drop_prob: f64,

    /// Whether to scale the output by `1 / (1 - drop_prob)`.
    #[config(default = true)]
    pub scale_by_keep: bool,
}

impl DropPathMeta for DropPathConfig {
    fn drop_prob(&self) -> f64 {
        self.drop_prob
    }

    fn scale_by_keep(&self) -> bool {
        self.scale_by_keep
    }
}

impl DropPathConfig {
    /// Initializes a new `DropPath` module.
    #[inline(always)]
    #[must_use]
    pub fn init(&self) -> DropPath {
        DropPath {
            drop_prob: validators::expect_probability(self.drop_prob),
            scale_by_keep: self.scale_by_keep,
        }
    }
}

/// The `DropPath` module.
///
/// Burn Module that implements the `DropPath` (Stochastic Depth)
/// regularization.
///
/// Built by [`DropPathConfig`].
#[derive(Module, Clone, Debug)]
pub struct DropPath {
    /// Probability of dropping a path.
    pub drop_prob: f64,

    /// Whether to scale the output by `1 / (1 - drop_prob)`.
    pub scale_by_keep: bool,
}

impl DropPathMeta for DropPath {
    fn drop_prob(&self) -> f64 {
        self.drop_prob
    }

    fn scale_by_keep(&self) -> bool {
        self.scale_by_keep
    }
}

impl DropPath {
    /// Applies `drop_path` pass on the input tensor.
    #[must_use]
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let training = B::ad_enabled(&input.device());
        drop_path(input, self.drop_prob, training, self.scale_by_keep)
    }

    /// Applies an inner function under conditional stochastic
    /// residual/depth-skip connection.
    ///
    /// This is used for stochastic depth in the transformer block.
    ///
    /// # Arguments
    ///
    /// * `x` - `[B, D]` input tensor.
    /// * `f` - Function to apply on the input tensor.
    ///
    /// # Returns
    ///
    /// The result of the function application, with a stochastic skip
    /// connection applied.
    #[inline]
    #[must_use]
    pub fn with_skip<B: Backend, const D: usize, F>(
        &self,
        x: Tensor<B, D>,
        f: F,
    ) -> Tensor<B, D>
    where
        F: FnOnce(Tensor<B, D>) -> Tensor<B, D>,
    {
        x.clone() + self.forward(f(x))
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::Tensor,
        tensor::Distribution,
    };

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        cpu_device,
    };

    #[test]
    fn test_drop_path() {
        type B = CpuBackend;
        let device = cpu_device();
        let drop_prob = 0.5;
        let scale_by_keep = true;

        let config = DropPathConfig {
            drop_prob,
            scale_by_keep,
        };

        let module = config.init();

        let input = Tensor::<B, 4>::random([2, 3, 4, 5], Distribution::Uniform(0.0, 1.0), &device);
        let output = module.forward(input.clone());

        assert_eq!(input.dims(), output.dims());
    }

    #[test]
    fn test_droppath_module() {
        type B = CpuBackend;
        let drop_prob = 0.2;
        let config = DropPathConfig::new().with_drop_prob(drop_prob);

        assert_eq!(config.drop_prob(), 0.2);
        assert_eq!(config.keep_prob(), 1.0 - drop_prob);
        assert!(config.scale_by_keep());

        let module = config.init();
        assert_eq!(module.drop_prob(), 0.2);
        assert_eq!(module.keep_prob(), 1.0 - drop_prob);
        assert!(module.scale_by_keep());

        let device = cpu_device();
        let shape = [2, 3, 4];
        let x = Tensor::<B, 3>::random(shape, Distribution::Uniform(0.0, 1.0), &device);

        // TODO(crutcher): work out how to enable/disable training mode in
        // tests.
        let output = module.forward(x.clone());
        assert_eq!(x.dims(), output.dims());
    }
}
