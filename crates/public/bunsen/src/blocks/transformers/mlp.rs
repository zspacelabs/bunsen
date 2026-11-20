//! Transformer feed-forwards: [`TransformerMlp`], and
//! [`TransformerFeedForward`], a plain, gated or sparse feed-forward chosen by
//! config; the gated form is
//! [`BunsenGatedMlp`].

use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::{
        activation::{
            BunsenActivation,
            BunsenActivationConfig,
        },
        mlp::{
            BunsenGatedMlp,
            BunsenGatedMlpConfig,
            BunsenGatedMlpMeta,
        },
        transformers::{
            moe::sparse::{
                SparseMoe,
                SparseMoeConfig,
            },
            sublayer::SublayerBranch,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`TransformerMlp`] and [`TransformerMlpConfig`].
pub trait TransformerMlpMeta {
    /// Returns the size of the input and output.
    fn n_embed(&self) -> usize;
}

/// Config for [`TransformerMlp`].
#[derive(Config, Debug)]
pub struct TransformerMlpConfig {
    /// Embedding Size.
    pub n_embed: usize,

    /// Internal Expansion Factor.
    #[config(default = "4")]
    pub expansion_factor: usize,

    /// The activation: nanochat's `relu(x)²` is
    /// `Chain([Relu, Pow(2)])`.
    #[config(default = "BunsenActivationConfig::Relu")]
    pub activation: BunsenActivationConfig,

    /// Whether the projections carry a bias term.
    #[config(default = "false")]
    pub bias: bool,
}

impl TransformerMlpMeta for TransformerMlpConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }
}

impl TransformerMlpConfig {
    /// Returns the size of the hidden layer.
    pub fn hidden_size(&self) -> usize {
        self.n_embed * self.expansion_factor
    }
}

impl ModuleInit<TransformerMlp> for TransformerMlpConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<TransformerMlp> {
        let linear1 = LinearConfig::new(self.n_embed(), self.hidden_size())
            .with_bias(self.bias)
            .init(device);
        let linear2 = LinearConfig::new(self.hidden_size(), self.n_embed())
            .with_bias(self.bias)
            .init(device);

        Ok(TransformerMlp {
            linear1,
            act: self.activation.init(device),
            linear2,
        })
    }
}

/// GPT Block MLP Module
///
/// The position-wise feed-forward block of a transformer: an up-projection to
/// an expanded hidden size, an activation, and a down-projection back to the
/// embedding size. Construct via [`TransformerMlpConfig`] and `.init(device)`,
/// then call [`forward`](TransformerMlp::forward) on a `[batch, time, embed]`
/// input.
///
/// Built by [`TransformerMlpConfig`].
#[derive(Module, Debug)]
pub struct TransformerMlp {
    /// Feed Forward Layer.
    pub linear1: Linear,

    /// Activation.
    pub act: BunsenActivation,

    /// Output Projection.
    pub linear2: Linear,
}

impl TransformerMlpMeta for TransformerMlp {
    fn n_embed(&self) -> usize {
        self.linear1.weight.dims()[0]
    }
}

impl TransformerMlp {
    /// MLP Forward Pass.
    ///
    /// # Arguments
    /// - `x`: a `[batch, time, embed]` input.
    ///
    /// # Returns
    /// a `[batch, time, embed]` result.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        let [batch, time] = crate::contracts::unpack_shape_contract!(
            ["batch", "time", "embed"],
            &x,
            &["batch", "time"],
            &[("embed", self.n_embed())]
        );

        let x = self.linear1.forward(x);
        let x = self.act.forward(x);

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

/// Common meta for [`TransformerFeedForward`] and
/// [`TransformerFeedForwardConfig`].
pub trait TransformerFeedForwardMeta {
    /// The input and output width.
    fn n_embed(&self) -> usize;
}

/// Config for [`TransformerFeedForward`]: which position-wise feed-forward, and
/// its config.
///
/// Implements [`TransformerFeedForwardMeta`].
#[derive(Config, Debug)]
pub enum TransformerFeedForwardConfig {
    /// [`TransformerMlp`]: expand, activate, project back.
    Mlp(TransformerMlpConfig),

    /// [`BunsenGatedMlp`]: the gated feed-forward (`SwiGLU`, `GeGLU`).
    Gated(BunsenGatedMlpConfig),

    /// [`SparseMoe`]: a sparse mixture of experts.
    Moe(SparseMoeConfig),
}

impl From<TransformerMlpConfig> for TransformerFeedForwardConfig {
    fn from(config: TransformerMlpConfig) -> Self {
        Self::Mlp(config)
    }
}

impl From<BunsenGatedMlpConfig> for TransformerFeedForwardConfig {
    fn from(config: BunsenGatedMlpConfig) -> Self {
        Self::Gated(config)
    }
}

impl From<SparseMoeConfig> for TransformerFeedForwardConfig {
    fn from(config: SparseMoeConfig) -> Self {
        Self::Moe(config)
    }
}

impl TransformerFeedForwardMeta for TransformerFeedForwardConfig {
    fn n_embed(&self) -> usize {
        match self {
            Self::Mlp(c) => TransformerMlpMeta::n_embed(c),
            Self::Gated(c) => BunsenGatedMlpMeta::n_embed(c),
            Self::Moe(c) => c.n_embed(),
        }
    }
}

impl ModuleInit<TransformerFeedForward> for TransformerFeedForwardConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<TransformerFeedForward> {
        Ok(match self {
            Self::Mlp(c) => TransformerFeedForward::Mlp(c.try_init(device)?),
            Self::Gated(c) => TransformerFeedForward::Gated(c.try_init(device)?),
            Self::Moe(c) => TransformerFeedForward::Moe(c.try_init(device)?),
        })
    }
}

/// A transformer layer's position-wise feed-forward, whichever its config
/// chose: the generic sublayer a layer holds, as
/// [`BunsenNormalization`](crate::blocks::norm::BunsenNormalization) is for a
/// norm. The variant is not part of its tensors' paths for the loaders bunsen
/// uses.
///
/// Built by [`TransformerFeedForwardConfig`].
/// Implements [`TransformerFeedForwardMeta`].
#[derive(Module, Debug)]
// One per layer: the variants' size gap costs nothing worth a box.
#[allow(clippy::large_enum_variant)]
pub enum TransformerFeedForward {
    /// [`TransformerMlp`].
    Mlp(TransformerMlp),

    /// [`BunsenGatedMlp`].
    Gated(BunsenGatedMlp),

    /// [`SparseMoe`].
    Moe(SparseMoe),
}

impl TransformerFeedForwardMeta for TransformerFeedForward {
    fn n_embed(&self) -> usize {
        match self {
            Self::Mlp(m) => TransformerMlpMeta::n_embed(m),
            Self::Gated(m) => BunsenGatedMlpMeta::n_embed(m),
            Self::Moe(m) => m.n_embed(),
        }
    }
}

impl TransformerFeedForward {
    /// The feed-forward, position by position.
    ///
    /// # Arguments
    /// * `x` - `[batch, seq_len, n_embed]`.
    ///
    /// # Returns
    /// `[batch, seq_len, n_embed]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        match self {
            Self::Mlp(m) => m.forward(x),
            Self::Gated(m) => m.forward(x),
            Self::Moe(m) => m.forward(x),
        }
    }
}

/// Takes only the stream.
impl SublayerBranch<(), 3> for TransformerMlp {
    type Output = Tensor<3>;

    fn call_branch(
        &self,
        z: Tensor<3>,
        (): (),
    ) -> Tensor<3> {
        self.forward(z)
    }
}

/// Takes only the stream.
impl SublayerBranch<(), 3> for BunsenGatedMlp {
    type Output = Tensor<3>;

    fn call_branch(
        &self,
        z: Tensor<3>,
        (): (),
    ) -> Tensor<3> {
        self.forward(z)
    }
}

/// Takes only the stream.
impl SublayerBranch<(), 3> for TransformerFeedForward {
    type Output = Tensor<3>;

    fn call_branch(
        &self,
        z: Tensor<3>,
        (): (),
    ) -> Tensor<3> {
        self.forward(z)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::{
            DeviceMemoryGuard,
            performance_device,
        },
    };

    /// A mixture of experts is a feed-forward like the others: built from
    /// its variant, as wide as its router, and its forward the layer's own.
    #[test]
    fn test_moe_variant() {
        use crate::blocks::transformers::moe::{
            experts::ExpertStackConfig,
            router::TopKRouterConfig,
        };
        let device = crate::support::testing::cpu_device();
        let config: TransformerFeedForwardConfig = SparseMoeConfig::new(
            TopKRouterConfig::new(8, 4, 2),
            ExpertStackConfig::new(4, 4, 6),
        )
        .with_latent(Some(4))
        .into();
        assert_eq!(config.n_embed(), 8);
        let ff: TransformerFeedForward = config.init(&device);
        assert_eq!(ff.n_embed(), 8);
        let TransformerFeedForward::Moe(moe) = &ff else {
            panic!("a mixture of experts");
        };
        let x = crate::support::testing::seeded_tensor::<3>(
            4,
            [2, 3, 8],
            Distribution::Default,
            &device,
        );
        assert_eq!(
            ff.forward(x.clone()).into_data(),
            moe.forward(x).into_data()
        );
    }

    #[test]
    fn test_mlp_config() {
        let cfg = TransformerMlpConfig::new(3);

        assert_eq!(cfg.n_embed, 3);
        assert_eq!(cfg.expansion_factor, 4);

        assert_eq!(cfg.n_embed(), 3);
        assert_eq!(cfg.hidden_size(), 3 * 4);
    }

    #[test]
    #[serial]
    fn test_mlp() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        for activation in [BunsenActivationConfig::Relu, BunsenActivationConfig::Gelu] {
            for ef in [4, 3] {
                let b = 2;
                let t = 3;
                let n_embed = 10;

                let cfg = TransformerMlpConfig::new(n_embed)
                    .with_expansion_factor(ef)
                    .with_activation(BunsenActivationConfig::Chain(vec![
                        activation.clone(),
                        BunsenActivationConfig::Pow(2.0),
                    ]));

                let mlp: TransformerMlp = cfg.init(&device);

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

                let x = activation.init(&device).forward(x);
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
