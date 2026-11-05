//! [`LayerNorm`], the norm seam's layer norm.

use burn::{
    module::Param,
    nn::{
        Initializer,
        LayerNormConfig,
    },
    prelude::*,
};

use super::normalization::compute_dtype;

/// A layer norm, [`Norm::Layer`](super::Norm::Layer)'s: burn's `LayerNorm`'s
/// parameters (`gamma`, and `beta` with a bias) and epsilon, under bunsen's
/// forward.
///
/// It is named `LayerNorm`, as burn's is, because burn-store's adapters
/// recognize a layer norm by its type's name (`Struct:LayerNorm`): the
/// `PyTorch` adapter maps a checkpoint's `weight` and `bias` onto `gamma` and
/// `beta`, and the half-precision adapter can leave it out.
#[derive(Module, Debug)]
pub struct LayerNorm {
    /// The gain, `[d_model]`.
    pub gamma: Param<Tensor<1>>,

    /// The bias, `[d_model]`, when the config asks for one.
    pub beta: Option<Param<Tensor<1>>>,

    /// Added to the variance.
    pub epsilon: f64,
}

impl LayerNorm {
    /// Ones for the gain and zeros for the bias, as burn's `LayerNorm`.
    pub fn init(
        config: &LayerNormConfig,
        device: &Device,
    ) -> Self {
        Self {
            gamma: Initializer::Ones.init([config.d_model], device),
            beta: config
                .bias
                .then(|| Initializer::Zeros.init([config.d_model], device)),
            epsilon: config.epsilon,
        }
    }

    /// `input` normalized over its last axis: in f32 when it is half, its
    /// parameters cast to match, narrowed back to `input`'s dtype.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        let dtype = input.dtype();
        let compute = compute_dtype(dtype);
        burn::tensor::module::layer_norm(
            input.cast(compute),
            self.gamma.val().cast(compute),
            self.beta.as_ref().map(|b| b.val().cast(compute)),
            self.epsilon,
        )
        .cast(dtype)
    }
}
