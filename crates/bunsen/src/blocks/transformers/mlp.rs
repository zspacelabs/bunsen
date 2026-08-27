//! GPT Block MLP

use burn::{
    config::Config,
    module::Module,
    nn::{
        LayerNorm,
        Linear,
        LinearConfig,
        activation::{
            Activation,
            ActivationConfig,
        },
    },
    prelude::{
        Backend,
        Tensor,
    },
};

use crate::{
    burner::module::{
        ModuleInit,
        repair_pytorch_strided_weight,
    },
    errors::BunsenResult,
};

/// Computes layer normalized mlp.
///
/// # Arguments
/// * `layer_norm` - `LayerNorm`.
/// * `mlp` - `Mlp`.
/// * `x` - `[batch, seq_len, n_states]` input.
///
/// # Returns
/// `[batch, seq_len, n_states]`
pub fn layer_norm_mlp<B: Backend>(
    layer_norm: &LayerNorm<B>,
    mlp: &Mlp<B>,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {
    mlp.forward(layer_norm.forward(x))
}

/// Common meta for [`Mlp`] and [`MlpConfig`].
pub trait MlpMeta {
    /// Returns the size of the input and output.
    fn n_embed(&self) -> usize;

    /// Returns the post-activation exponent.
    fn act_exponent(&self) -> Option<f64>;
}

/// Config for [`Mlp`].
#[derive(Config, Debug)]
pub struct MlpConfig {
    /// Embedding Size.
    pub n_embed: usize,

    /// Internal Expansion Factor.
    #[config(default = "4")]
    pub expansion_factor: usize,

    /// Activation Config.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,

    /// Post-Activation Exponent.
    #[config(default = "None")]
    pub act_exponent: Option<f64>,

    /// Do the projections have a bias?
    ///
    /// Whisper's MLP projections do; a GPT-style block generally does not.
    #[config(default = "false")]
    pub bias: bool,

    /// Repair the projection weights against `burn-store`'s stride-blind
    /// `PyTorch` read.
    ///
    /// Set this when loading a checkpoint whose `Linear` weights are stored as
    /// column-major views — every `OpenAI` Whisper checkpoint is. See
    /// [`repair_pytorch_strided_weight`].
    #[config(default = "false")]
    pub repair_strided_weights: bool,
}

impl MlpMeta for MlpConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }

    fn act_exponent(&self) -> Option<f64> {
        self.act_exponent
    }
}

impl MlpConfig {
    /// Returns the size of the hidden layer.
    pub fn hidden_size(&self) -> usize {
        self.n_embed * self.expansion_factor
    }
}

impl<B: Backend> ModuleInit<B, Mlp<B>> for MlpConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<Mlp<B>> {
        let mut linear1 = LinearConfig::new(self.n_embed(), self.hidden_size())
            .with_bias(self.bias)
            .init(device);
        let mut linear2 = LinearConfig::new(self.hidden_size(), self.n_embed())
            .with_bias(self.bias)
            .init(device);

        if self.repair_strided_weights {
            linear1.weight = repair_pytorch_strided_weight(linear1.weight);
            linear2.weight = repair_pytorch_strided_weight(linear2.weight);
        }

        Ok(Mlp {
            linear1,
            act: self.activation.init(device),
            act_exponent: self.act_exponent,
            linear2,
        })
    }
}

/// GPT Block MLP Module
///
/// The position-wise feed-forward block of a transformer: an up-projection to
/// an expanded hidden size, an activation (optionally raised to a configured
/// exponent), and a down-projection back to the embedding size. Construct via
/// [`MlpConfig`] and `.init(device)`, then call [`forward`](Mlp::forward) on a
/// `[batch, time, embed]` input.
///
/// Built by [`MlpConfig`].
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    /// Feed Forward Layer.
    pub linear1: Linear<B>,

    /// Activation.
    pub act: Activation<B>,

    /// Post-Activation Exponent.
    pub act_exponent: Option<f64>,

    /// Output Projection.
    pub linear2: Linear<B>,
}

impl<B: Backend> MlpMeta for Mlp<B> {
    fn n_embed(&self) -> usize {
        self.linear1.weight.dims()[0]
    }

    fn act_exponent(&self) -> Option<f64> {
        self.act_exponent
    }
}

impl<B: Backend> Mlp<B> {
    /// MLP Forward Pass.
    ///
    /// # Arguments
    /// - `x`: a `[batch, time, embed]` input.
    ///
    /// # Returns
    /// a `[batch, time, embed]` result.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        #[cfg(any(debug_assertions, test))]
        let [batch, time] = crate::contracts::unpack_shape_contract!(
            ["batch", "time", "embed"],
            &x,
            &["batch", "time"],
            &[("embed", self.n_embed())]
        );

        let x = self.linear1.forward(x);
        let x = self.act.forward(x);

        let x = match self.act_exponent {
            Some(exp) => x.powf_scalar(exp),
            None => x,
        };

        let x = self.linear2.forward(x);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time", "embed"],
            &x,
            &[("batch", batch), ("time", time), ("embed", self.n_embed())]
        );

        x
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::PerformanceBackend,
    };

    #[test]
    fn test_mlp_config() {
        let cfg = MlpConfig::new(3);

        assert_eq!(cfg.n_embed, 3);
        assert_eq!(cfg.expansion_factor, 4);

        assert_eq!(cfg.n_embed(), 3);
        assert_eq!(cfg.hidden_size(), 3 * 4);
    }

    #[test]
    #[serial]
    fn test_mlp() {
        type B = PerformanceBackend;
        let device = Default::default();

        for activation in [ActivationConfig::Relu, ActivationConfig::Gelu] {
            for ef in [4, 3] {
                let b = 2;
                let t = 3;
                let n_embed = 10;

                let cfg = MlpConfig::new(n_embed)
                    .with_expansion_factor(ef)
                    .with_activation(activation.clone())
                    .with_act_exponent(Some(2.0));

                let mlp: Mlp<B> = cfg.init(&device);

                assert_eq!(mlp.n_embed(), n_embed);

                let input = Tensor::random([b, t, n_embed], Distribution::Default, &device);
                let output = mlp.forward(input.clone());

                let x = input;
                assert_shape_contract!(
                    ["batch", "time", "embed"],
                    &x.dims(),
                    &[("batch", b), ("time", t), ("embed", n_embed)]
                );

                let x = mlp.linear1.forward(x);
                assert_shape_contract!(
                    ["batch", "time", "hidden"],
                    &x.dims(),
                    &[("batch", b), ("time", t), ("hidden", ef * n_embed)]
                );

                let x = mlp.act.forward(x);
                assert_shape_contract!(
                    ["batch", "time", "hidden"],
                    &x.dims(),
                    &[("batch", b), ("time", t), ("hidden", ef * n_embed)]
                );

                let x = x.clone() * x;
                let x = mlp.linear2.forward(x);
                assert_shape_contract!(
                    ["batch", "time", "embed"],
                    &x.dims(),
                    &[("batch", b), ("time", t), ("embed", n_embed)]
                );

                output.to_data().assert_eq(&x.to_data(), true);
            }
        }
    }
}
