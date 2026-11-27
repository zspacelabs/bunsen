//! Transformer feed-forwards: [`TransformerFeedForward`], a plain, gated or
//! sparse feed-forward chosen by config, the sublayer a layer holds; its
//! plain form a [`BunsenMlp`] (a transformer's usual one,
//! [`BunsenMlpConfig::expansion`]), its gated form a [`BunsenGatedMlp`].

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::{
        mlp::{
            BunsenGatedMlp,
            BunsenGatedMlpConfig,
            BunsenGatedMlpMeta,
            BunsenMlp,
            BunsenMlpConfig,
            BunsenMlpMeta,
        },
        transformers::moe::sparse::{
            SparseMoe,
            SparseMoeConfig,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

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
    /// [`BunsenMlp`]: expand, activate, project back, as wide out as in
    /// ([`BunsenMlpConfig::expansion`]).
    Mlp(BunsenMlpConfig),

    /// [`BunsenGatedMlp`]: the gated feed-forward (`SwiGLU`, `GeGLU`).
    Gated(BunsenGatedMlpConfig),

    /// [`SparseMoe`]: a sparse mixture of experts.
    Moe(SparseMoeConfig),
}

impl From<BunsenMlpConfig> for TransformerFeedForwardConfig {
    fn from(config: BunsenMlpConfig) -> Self {
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
            Self::Mlp(c) => c.d_input(),
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
            Self::Mlp(c) => {
                if c.d_output() != c.d_input() {
                    return Err(crate::errors::BunsenError::illegal(format!(
                        "a transformer feed-forward is as wide out as in, not {} to {}",
                        c.d_input(),
                        c.d_output()
                    )));
                }
                TransformerFeedForward::Mlp(c.try_init(device)?)
            }
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
    /// [`BunsenMlp`].
    Mlp(BunsenMlp),

    /// [`BunsenGatedMlp`].
    Gated(BunsenGatedMlp),

    /// [`SparseMoe`].
    Moe(SparseMoe),
}

impl TransformerFeedForwardMeta for TransformerFeedForward {
    fn n_embed(&self) -> usize {
        match self {
            Self::Mlp(m) => m.d_input(),
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

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::{
        blocks::activation::BunsenActivationConfig,
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

    /// A transformer's plain feed-forward: `n_embed * factor` hidden, as wide
    /// out as in, no bias.
    #[test]
    fn test_mlp_config() {
        let cfg = BunsenMlpConfig::expansion(3, 4);
        assert_eq!((cfg.d_input(), cfg.d_hidden(), cfg.d_output()), (3, 12, 3));
        assert!(!cfg.bias);
    }

    /// A transformer's plain feed-forward is as wide out as in: one that
    /// resizes is refused.
    #[test]
    fn test_mlp_variant() {
        let device = crate::support::testing::cpu_device();
        let config: TransformerFeedForwardConfig = BunsenMlpConfig::expansion(4, 4).into();
        let TransformerFeedForwardConfig::Mlp(structure) = &config else {
            panic!("a plain feed-forward");
        };
        assert_eq!(structure.d_hidden(), 16);
        let ff: TransformerFeedForward = config.init(&device);
        assert_eq!(ff.n_embed(), 4);
        let resize: TransformerFeedForwardConfig = BunsenMlpConfig::new(4, 8, 2).into();
        crate::errors::testing::ErrorMatcher::illegal()
            .message_contains("as wide out as in")
            .assert_err(&ModuleInit::<TransformerFeedForward>::try_init(
                &resize, &device,
            ));
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

                let cfg = BunsenMlpConfig::expansion(n_embed, ef).with_activation(
                    BunsenActivationConfig::Chain(vec![
                        activation.clone(),
                        BunsenActivationConfig::Pow(2.0),
                    ]),
                );

                let mlp: BunsenMlp = cfg.init(&device);

                assert_eq!(mlp.d_input(), n_embed);

                let input: Tensor<3> =
                    Tensor::random([b, t, n_embed], Distribution::Default, &device);
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
