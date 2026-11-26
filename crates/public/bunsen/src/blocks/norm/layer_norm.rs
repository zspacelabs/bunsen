//! [`LayerNorm`], the norm seam's layer norm.

use burn::{
    module::Param,
    nn::{
        Initializer,
        LayerNormConfig,
    },
    prelude::*,
};

use crate::{
    burner::{
        module::ModuleInit,
        tensor::Widening,
    },
    errors::BunsenResult,
};

/// A [`LayerNorm`]'s config: burn's `LayerNormConfig`'s fields and defaults,
/// so the serde form is burn's.
///
/// bunsen's own type, so that `init` builds bunsen's [`LayerNorm`]: burn's
/// config has an inherent `init` that builds burn's.
#[derive(Config, Debug)]
pub struct BunsenLayerNormConfig {
    /// The size of the input features.
    pub d_model: usize,

    /// Added to the variance.
    #[config(default = 1e-5)]
    pub epsilon: f64,

    /// Whether the norm has a bias (`beta`).
    #[config(default = true)]
    pub bias: bool,
}

impl From<LayerNormConfig> for BunsenLayerNormConfig {
    fn from(config: LayerNormConfig) -> Self {
        Self {
            d_model: config.d_model,
            epsilon: config.epsilon,
            bias: config.bias,
        }
    }
}

/// Ones for the gain and zeros for the bias, as burn's `LayerNorm`.
impl ModuleInit<LayerNorm> for BunsenLayerNormConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<LayerNorm> {
        Ok(LayerNorm {
            gamma: Initializer::Ones.init([self.d_model], device),
            beta: self
                .bias
                .then(|| Initializer::Zeros.init([self.d_model], device)),
            epsilon: self.epsilon,
        })
    }
}

/// A layer norm,
/// [`BunsenNormalization::Layer`](super::BunsenNormalization::Layer)'s: burn's
/// `LayerNorm`'s parameters (`gamma`, and `beta` with a bias) and epsilon,
/// under bunsen's forward.
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
    /// `input` normalized over its last axis: in f32 when it is half, its
    /// parameters cast to match, narrowed back to `input`'s dtype.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        // The norm's features are the input's last axis.
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            [..., "features"],
            &input.dims(),
            &[("features", self.gamma.dims()[0])]
        );
        Widening::F32.around(input.dtype(), |w| {
            burn::tensor::module::layer_norm(
                w.widen(input),
                w.widen(self.gamma.val()),
                self.beta.as_ref().map(|b| w.widen(b.val())),
                self.epsilon,
            )
        })
    }
}
