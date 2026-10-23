//! The Conformer's feed-forward module.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    tensor::{
        Device,
        activation::silu,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`ConformerFeedForward`] and [`ConformerFeedForwardConfig`].
pub trait ConformerFeedForwardMeta {
    /// The input and output width.
    fn d_model(&self) -> usize;

    /// The hidden width.
    fn d_ff(&self) -> usize;
}

/// Config for [`ConformerFeedForward`].
///
/// Implements [`ConformerFeedForwardMeta`].
#[derive(Config, Debug)]
pub struct ConformerFeedForwardConfig {
    /// The input and output width.
    pub d_model: usize,

    /// The hidden width; the Conformer's is `4 * d_model`.
    pub d_ff: usize,

    /// Whether the two projections carry a bias.
    #[config(default = "true")]
    pub bias: bool,
}

impl ConformerFeedForwardMeta for ConformerFeedForwardConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn d_ff(&self) -> usize {
        self.d_ff
    }
}

impl ModuleInit<ConformerFeedForward> for ConformerFeedForwardConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConformerFeedForward> {
        Ok(ConformerFeedForward {
            linear1: LinearConfig::new(self.d_model, self.d_ff)
                .with_bias(self.bias)
                .init(device),
            linear2: LinearConfig::new(self.d_ff, self.d_model)
                .with_bias(self.bias)
                .init(device),
        })
    }
}

/// The Conformer's feed-forward module: expand, `SiLU` (Swish), project back.
///
/// The Conformer block holds two of these, macaron style, around its
/// attention and convolution modules (see
/// [`ConformerBlock`](super::ConformerBlock)). Its activation is fixed:
/// burn's `ActivationConfig` has no `SiLU`, which is why this is not a
/// [`Mlp`](crate::blocks::transformers::mlp::Mlp).
///
/// Built by [`ConformerFeedForwardConfig`].
/// Implements [`ConformerFeedForwardMeta`].
#[derive(Module, Debug)]
pub struct ConformerFeedForward {
    /// `d_model -> d_ff`.
    pub linear1: Linear,

    /// `d_ff -> d_model`.
    pub linear2: Linear,
}

impl ConformerFeedForwardMeta for ConformerFeedForward {
    fn d_model(&self) -> usize {
        self.linear1.weight.dims()[0]
    }

    fn d_ff(&self) -> usize {
        self.linear1.weight.dims()[1]
    }
}

impl ConformerFeedForward {
    /// Applies the module.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_model]`.
    ///
    /// # Returns
    /// `[batch, time, d_model]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        let [batch, time] = crate::contracts::unpack_shape_contract!(
            ["batch", "time", "d_model"],
            &x.dims(),
            &["batch", "time"],
            &[("d_model", self.d_model())],
        );

        let y = self.linear2.forward(silu(self.linear1.forward(x)));

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time", "d_model"],
            &y.dims(),
            &[
                ("batch", batch),
                ("time", time),
                ("d_model", self.d_model())
            ],
        );

        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = ConformerFeedForwardConfig::new(6, 10).with_bias(false);
        let module: ConformerFeedForward = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.d_ff(), config.d_ff());
        assert!(module.linear1.bias.is_none());

        let y = module.forward(Tensor::zeros([2, 3, 6], &cpu_device()));
        assert_eq!(y.dims(), [2, 3, 6]);
    }
}
