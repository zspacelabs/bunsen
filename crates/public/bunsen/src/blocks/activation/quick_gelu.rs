//! [`QuickGelu`]: CLIP's sigmoid approximation of `GELU`.

use burn::{
    module::Module,
    prelude::Tensor,
};

/// The sigmoid's gain in `x · sigmoid(1.702 x)`.
pub const QUICK_GELU_GAIN: f64 = 1.702;

/// `x · sigmoid(1.702 x)`, elementwise: `OpenAI` CLIP's approximation of
/// `GELU` (timm's `QuickGELU`), a layer with no parameters, the activation
/// seam's [`QuickGelu`](super::BunsenActivationConfig::QuickGelu) kind.
#[derive(Module, Debug, Default)]
pub struct QuickGelu;

impl QuickGelu {
    /// The layer.
    pub fn new() -> Self {
        Self
    }

    /// `input · sigmoid(1.702 input)`, elementwise; in `input`'s dtype.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        let gate = burn::tensor::activation::sigmoid(input.clone().mul_scalar(QUICK_GELU_GAIN));
        input * gate
    }
}
