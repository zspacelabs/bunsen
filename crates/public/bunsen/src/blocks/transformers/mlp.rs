//! Transformer feed-forwards: [`Mlp`], [`GatedMlp`], and [`FeedForward`],
//! either one chosen by config.

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
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    burner::module::ModuleInit,
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
pub fn layer_norm_mlp(
    layer_norm: &LayerNorm,
    mlp: &Mlp,
    x: Tensor<3>,
) -> Tensor<3> {
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

    /// Whether the projections carry a bias term.
    #[config(default = "false")]
    pub bias: bool,
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

impl ModuleInit<Mlp> for MlpConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Mlp> {
        let linear1 = LinearConfig::new(self.n_embed(), self.hidden_size())
            .with_bias(self.bias)
            .init(device);
        let linear2 = LinearConfig::new(self.hidden_size(), self.n_embed())
            .with_bias(self.bias)
            .init(device);

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
pub struct Mlp {
    /// Feed Forward Layer.
    pub linear1: Linear,

    /// Activation.
    pub act: Activation,

    /// Post-Activation Exponent.
    pub act_exponent: Option<f64>,

    /// Output Projection.
    pub linear2: Linear,
}

impl MlpMeta for Mlp {
    fn n_embed(&self) -> usize {
        self.linear1.weight.dims()[0]
    }

    fn act_exponent(&self) -> Option<f64> {
        self.act_exponent
    }
}

impl Mlp {
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

/// Common meta for [`GatedMlp`] and [`GatedMlpConfig`].
pub trait GatedMlpMeta {
    /// The input and output width.
    fn n_embed(&self) -> usize;

    /// The hidden width.
    fn d_ff(&self) -> usize;
}

/// Config for [`GatedMlp`].
///
/// Implements [`GatedMlpMeta`].
#[derive(Config, Debug)]
pub struct GatedMlpConfig {
    /// The input and output width.
    pub n_embed: usize,

    /// The hidden width.
    pub d_ff: usize,

    /// Whether the three projections carry a bias.
    #[config(default = "false")]
    pub bias: bool,
}

impl GatedMlpMeta for GatedMlpConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }

    fn d_ff(&self) -> usize {
        self.d_ff
    }
}

impl ModuleInit<GatedMlp> for GatedMlpConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<GatedMlp> {
        let linear = |i, o| LinearConfig::new(i, o).with_bias(self.bias).init(device);
        Ok(GatedMlp {
            gate_proj: linear(self.n_embed, self.d_ff),
            up_proj: linear(self.n_embed, self.d_ff),
            down_proj: linear(self.d_ff, self.n_embed),
        })
    }
}

/// The `SwiGLU` feed-forward block (Shazeer 2020): `down(silu(gate(x)) *
/// up(x))`, the feed-forward of Llama, Qwen and their kin.
///
/// The fields carry the names `transformers` gives them.
///
/// Built by [`GatedMlpConfig`].
/// Implements [`GatedMlpMeta`].
#[derive(Module, Debug)]
pub struct GatedMlp {
    /// `n_embed -> d_ff`, the gate.
    pub gate_proj: Linear,

    /// `n_embed -> d_ff`.
    pub up_proj: Linear,

    /// `d_ff -> n_embed`.
    pub down_proj: Linear,
}

impl GatedMlpMeta for GatedMlp {
    fn n_embed(&self) -> usize {
        self.gate_proj.weight.dims()[0]
    }

    fn d_ff(&self) -> usize {
        self.gate_proj.weight.dims()[1]
    }
}

impl GatedMlp {
    /// Applies the block.
    ///
    /// # Arguments
    /// * `x` - `[batch, seq, n_embed]`.
    ///
    /// # Returns
    /// `[batch, seq, n_embed]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "n_embed"],
            &x.dims(),
            &["batch", "seq"],
            &[("n_embed", self.n_embed())]
        );
        let gate = burn::tensor::activation::silu(self.gate_proj.forward(x.clone()));
        let y = self.down_proj.forward(gate * self.up_proj.forward(x));
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "n_embed"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("n_embed", self.n_embed())]
        );
        y
    }
}

/// Common meta for [`FeedForward`] and [`FeedForwardConfig`].
pub trait FeedForwardMeta {
    /// The input and output width.
    fn n_embed(&self) -> usize;
}

/// Config for [`FeedForward`]: which position-wise feed-forward, and its
/// config.
///
/// Implements [`FeedForwardMeta`].
#[derive(Config, Debug)]
pub enum FeedForwardConfig {
    /// [`Mlp`]: expand, activate, project back.
    Mlp(MlpConfig),

    /// [`GatedMlp`]: the `SwiGLU` feed-forward.
    Gated(GatedMlpConfig),
}

impl From<MlpConfig> for FeedForwardConfig {
    fn from(config: MlpConfig) -> Self {
        Self::Mlp(config)
    }
}

impl From<GatedMlpConfig> for FeedForwardConfig {
    fn from(config: GatedMlpConfig) -> Self {
        Self::Gated(config)
    }
}

impl FeedForwardMeta for FeedForwardConfig {
    fn n_embed(&self) -> usize {
        match self {
            Self::Mlp(c) => MlpMeta::n_embed(c),
            Self::Gated(c) => GatedMlpMeta::n_embed(c),
        }
    }
}

impl ModuleInit<FeedForward> for FeedForwardConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<FeedForward> {
        Ok(match self {
            Self::Mlp(c) => FeedForward::Mlp(c.try_init(device)?),
            Self::Gated(c) => FeedForward::Gated(c.try_init(device)?),
        })
    }
}

/// A transformer layer's position-wise feed-forward, whichever its config
/// chose: the generic sublayer a layer holds, as
/// [`Normalization`](burn::nn::norm::Normalization) is for a norm. The variant
/// is not part of its tensors' paths for the loaders bunsen uses.
///
/// Built by [`FeedForwardConfig`].
/// Implements [`FeedForwardMeta`].
#[derive(Module, Debug)]
pub enum FeedForward {
    /// [`Mlp`].
    Mlp(Mlp),

    /// [`GatedMlp`].
    Gated(GatedMlp),
}

impl FeedForwardMeta for FeedForward {
    fn n_embed(&self) -> usize {
        match self {
            Self::Mlp(m) => MlpMeta::n_embed(m),
            Self::Gated(m) => GatedMlpMeta::n_embed(m),
        }
    }
}

impl FeedForward {
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
        }
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
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        for activation in [ActivationConfig::Relu, ActivationConfig::Gelu] {
            for ef in [4, 3] {
                let b = 2;
                let t = 3;
                let n_embed = 10;

                let cfg = MlpConfig::new(n_embed)
                    .with_expansion_factor(ef)
                    .with_activation(activation.clone())
                    .with_act_exponent(Some(2.0));

                let mlp: Mlp = cfg.init(&device);

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

    #[test]
    fn test_gated_mlp() {
        use crate::support::testing::cpu_device;
        let config = GatedMlpConfig::new(6, 10).with_bias(true);
        let mlp: GatedMlp = config.init(&cpu_device());
        assert_eq!(mlp.n_embed(), config.n_embed());
        assert_eq!(mlp.d_ff(), config.d_ff());
        assert!(mlp.up_proj.bias.is_some());
        // silu(0) = 0, so a zero input gives the down projection's bias.
        let y = mlp.forward(Tensor::zeros([2, 3, 6], &cpu_device()));
        assert_eq!(y.dims(), [2, 3, 6]);
    }
}
