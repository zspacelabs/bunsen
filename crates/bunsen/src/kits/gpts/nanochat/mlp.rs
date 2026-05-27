//! GPT Block MLP

use burn::{
    config::Config,
    module::Module,
    nn::{
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

use crate::contracts::{
    assert_shape_contract_periodically,
    unpack_shape_contract,
};

/// Common meta for [`NanoChatGptMlp`] and [`NanoGptMlpConfig`].
pub trait NanoChatGptMlpMeta {
    /// Return the size of the input and output.
    fn n_embed(&self) -> usize;
}

/// Config for [`NanoChatGptMlp`].
#[derive(Config, Debug)]
pub struct NanoGptMlpConfig {
    /// Embedding Size.
    pub n_embed: usize,

    /// Internal Expansion Factor.
    #[config(default = "4")]
    pub expansion_factor: usize,

    /// Activation Config.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,
}

impl NanoChatGptMlpMeta for NanoGptMlpConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }
}

impl NanoGptMlpConfig {
    /// Initialize the module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> NanoChatGptMlp<B> {
        NanoChatGptMlp {
            c_fc: LinearConfig::new(self.n_embed(), self.hidden_size())
                .with_bias(false)
                .init(device),
            act: self.activation.init(device),
            c_proj: LinearConfig::new(self.hidden_size(), self.n_embed())
                .with_bias(false)
                .init(device),
        }
    }

    /// Return the size of the hidden layer.
    pub fn hidden_size(&self) -> usize {
        self.n_embed * self.expansion_factor
    }
}

/// GPT Block MLP Module
#[derive(Module, Debug)]
pub struct NanoChatGptMlp<B: Backend> {
    /// Feed Forward Layer.
    pub c_fc: Linear<B>,

    /// Activation.
    pub act: Activation<B>,

    /// Output Projection.
    pub c_proj: Linear<B>,
}

impl<B: Backend> NanoChatGptMlpMeta for NanoChatGptMlp<B> {
    fn n_embed(&self) -> usize {
        self.c_fc.weight.dims()[0]
    }
}

impl<B: Backend> NanoChatGptMlp<B> {
    /// MLP Forward Pass.
    ///
    /// # Arguments
    /// - `x`: a ``[batch, time, embed]`` input.
    ///
    /// # Returns
    /// a ``[batch, time, embed]`` result.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, time] = unpack_shape_contract!(
            ["batch", "time", "embed"],
            &x,
            &["batch", "time"],
            &[("embed", self.n_embed())]
        );

        let x = self.c_fc.forward(x);
        let x = self.act.forward(x);
        let x = x.square();
        let x = self.c_proj.forward(x);

        assert_shape_contract_periodically!(
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
        let cfg = NanoGptMlpConfig::new(3);

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

                let cfg = NanoGptMlpConfig::new(n_embed)
                    .with_expansion_factor(ef)
                    .with_activation(activation.clone());

                let mlp: NanoChatGptMlp<B> = cfg.init(&device);

                assert_eq!(mlp.n_embed(), n_embed);

                let input = Tensor::random([b, t, n_embed], Distribution::Default, &device);
                let output = mlp.forward(input.clone());

                let x = input;
                assert_shape_contract!(
                    ["batch", "time", "embed"],
                    &x.dims(),
                    &[("batch", b), ("time", t), ("embed", n_embed)]
                );

                let x = mlp.c_fc.forward(x);
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
                let x = mlp.c_proj.forward(x);
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
