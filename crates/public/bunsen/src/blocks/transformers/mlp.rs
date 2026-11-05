//! Transformer feed-forwards: [`TransformerMlp`], [`TransformerGatedMlp`], and
//! [`TransformerFeedForward`], either one chosen by config.

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
        norm::BunsenNormalization,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Computes a normalized MLP: `mlp(norm(x))`.
///
/// # Arguments
/// * `norm` - the sublayer's [`BunsenNormalization`].
/// * `mlp` - `TransformerMlp`.
/// * `x` - `[batch, seq_len, n_states]` input.
///
/// # Returns
/// `[batch, seq_len, n_states]`
pub fn norm_mlp(
    norm: &BunsenNormalization,
    mlp: &TransformerMlp,
    x: Tensor<3>,
) -> Tensor<3> {
    mlp.forward(norm.forward(x))
}

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

/// Common meta for [`TransformerGatedMlp`] and [`TransformerGatedMlpConfig`].
pub trait TransformerGatedMlpMeta {
    /// The input and output width.
    fn n_embed(&self) -> usize;

    /// The hidden width.
    fn d_ff(&self) -> usize;
}

/// Config for [`TransformerGatedMlp`].
///
/// Implements [`TransformerGatedMlpMeta`].
#[derive(Config, Debug)]
pub struct TransformerGatedMlpConfig {
    /// The input and output width.
    pub n_embed: usize,

    /// The hidden width.
    pub d_ff: usize,

    /// Whether the three projections carry a bias.
    #[config(default = "false")]
    pub bias: bool,

    /// Whether the down projection carries a bias; `None` follows `bias`.
    ///
    /// Voxtral Realtime's audio encoder biases only its down projection
    /// (`Some(true)` without `bias`).
    #[config(default = "None")]
    pub down_bias: Option<bool>,

    /// The gate's activation: `Silu` for a `SwiGLU`, `GeluApproximate` for
    /// Gemma's `GeGLU`.
    #[config(default = "BunsenActivationConfig::Silu")]
    pub activation: BunsenActivationConfig,
}

impl TransformerGatedMlpConfig {
    /// Whether the down projection carries a bias: `down_bias`, else `bias`.
    pub fn down_bias(&self) -> bool {
        self.down_bias.unwrap_or(self.bias)
    }
}

impl TransformerGatedMlpMeta for TransformerGatedMlpConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }

    fn d_ff(&self) -> usize {
        self.d_ff
    }
}

impl ModuleInit<TransformerGatedMlp> for TransformerGatedMlpConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<TransformerGatedMlp> {
        let linear = |i, o, bias| LinearConfig::new(i, o).with_bias(bias).init(device);
        Ok(TransformerGatedMlp {
            gate_proj: linear(self.n_embed, self.d_ff, self.bias),
            up_proj: linear(self.n_embed, self.d_ff, self.bias),
            down_proj: linear(self.d_ff, self.n_embed, self.down_bias()),
            act: self.activation.init(device),
        })
    }
}

/// The gated feed-forward block (Shazeer 2020): `down(act(gate(x)) *
/// up(x))`. With `silu`, the `SwiGLU` of Llama, Qwen and their kin; with
/// tanh-approximated GELU, Gemma's `GeGLU`.
///
/// The fields carry the names `transformers` gives them.
///
/// Built by [`TransformerGatedMlpConfig`].
/// Implements [`TransformerGatedMlpMeta`].
#[derive(Module, Debug)]
pub struct TransformerGatedMlp {
    /// `n_embed -> d_ff`, the gate.
    pub gate_proj: Linear,

    /// `n_embed -> d_ff`.
    pub up_proj: Linear,

    /// `d_ff -> n_embed`.
    pub down_proj: Linear,

    /// The gate's activation.
    pub act: BunsenActivation,
}

impl TransformerGatedMlpMeta for TransformerGatedMlp {
    fn n_embed(&self) -> usize {
        self.gate_proj.weight.dims()[0]
    }

    fn d_ff(&self) -> usize {
        self.gate_proj.weight.dims()[1]
    }
}

impl TransformerGatedMlp {
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
        let gate = self.act.forward(self.gate_proj.forward(x.clone()));
        let y = self.down_proj.forward(gate * self.up_proj.forward(x));
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "n_embed"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("n_embed", self.n_embed())]
        );
        y
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

    /// [`TransformerGatedMlp`]: the gated feed-forward (`SwiGLU`, `GeGLU`).
    Gated(TransformerGatedMlpConfig),
}

impl From<TransformerMlpConfig> for TransformerFeedForwardConfig {
    fn from(config: TransformerMlpConfig) -> Self {
        Self::Mlp(config)
    }
}

impl From<TransformerGatedMlpConfig> for TransformerFeedForwardConfig {
    fn from(config: TransformerGatedMlpConfig) -> Self {
        Self::Gated(config)
    }
}

impl TransformerFeedForwardMeta for TransformerFeedForwardConfig {
    fn n_embed(&self) -> usize {
        match self {
            Self::Mlp(c) => TransformerMlpMeta::n_embed(c),
            Self::Gated(c) => TransformerGatedMlpMeta::n_embed(c),
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
        })
    }
}

/// A transformer layer's position-wise feed-forward, whichever its config
/// chose: the generic sublayer a layer holds, as
/// [`BunsenNormalization`] is for a norm. The variant
/// is not part of its tensors' paths for the loaders bunsen uses.
///
/// Built by [`TransformerFeedForwardConfig`].
/// Implements [`TransformerFeedForwardMeta`].
#[derive(Module, Debug)]
pub enum TransformerFeedForward {
    /// [`TransformerMlp`].
    Mlp(TransformerMlp),

    /// [`TransformerGatedMlp`].
    Gated(TransformerGatedMlp),
}

impl TransformerFeedForwardMeta for TransformerFeedForward {
    fn n_embed(&self) -> usize {
        match self {
            Self::Mlp(m) => TransformerMlpMeta::n_embed(m),
            Self::Gated(m) => TransformerGatedMlpMeta::n_embed(m),
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

    #[test]
    fn test_gated_mlp() {
        use crate::support::testing::cpu_device;
        let config = TransformerGatedMlpConfig::new(6, 10).with_bias(true);
        let mlp: TransformerGatedMlp = config.init(&cpu_device());
        assert_eq!(mlp.n_embed(), config.n_embed());
        assert_eq!(mlp.d_ff(), config.d_ff());
        assert!(mlp.up_proj.bias.is_some());
        // silu(0) = 0, so a zero input gives the down projection's bias.
        let y = mlp.forward(Tensor::zeros([2, 3, 6], &cpu_device()));
        assert_eq!(y.dims(), [2, 3, 6]);
    }

    /// The gated block applies its configured activation to the gate:
    /// `down(act(gate(x)) * up(x))`, `silu` by default.
    #[test]
    fn test_gated_mlp_activation() {
        use crate::support::testing::{
            assert_tensors_close,
            cpu_device,
            seeded_tensor,
        };
        let device = cpu_device();
        let x = seeded_tensor::<3>(7, [2, 3, 6], Distribution::Default, &device);
        for (config, act) in [
            (
                TransformerGatedMlpConfig::new(6, 10),
                burn::tensor::activation::silu::<3> as fn(Tensor<3>) -> Tensor<3>,
            ),
            (
                TransformerGatedMlpConfig::new(6, 10)
                    .with_activation(BunsenActivationConfig::GeluApproximate),
                burn::tensor::activation::gelu_approximate::<3>,
            ),
        ] {
            let mlp: TransformerGatedMlp = config.init(&device);
            let expected = mlp
                .down_proj
                .forward(act(mlp.gate_proj.forward(x.clone())) * mlp.up_proj.forward(x.clone()));
            assert_tensors_close(&mlp.forward(x.clone()), &expected, Default::default());
        }
    }

    /// The down projection's bias follows `bias` unless `down_bias` says
    /// otherwise.
    #[test]
    fn test_gated_mlp_down_bias() {
        use crate::support::testing::cpu_device;
        let device = cpu_device();
        let config = TransformerGatedMlpConfig::new(6, 10).with_down_bias(Some(true));
        assert!(!config.bias);
        assert!(config.down_bias());
        let mlp: TransformerGatedMlp = config.init(&device);
        assert!(mlp.gate_proj.bias.is_none());
        assert!(mlp.up_proj.bias.is_none());
        assert!(mlp.down_proj.bias.is_some());
        // With no gate bias, a zero input is the down projection's bias.
        let y = mlp.forward(Tensor::zeros([1, 2, 6], &device));
        let bias = mlp.down_proj.bias.as_ref().unwrap().val();
        crate::support::testing::assert_tensors_close(
            &y,
            &bias.unsqueeze_dims::<3>(&[0, 0]).repeat_dim(1, 2),
            burn::tensor::Tolerance::default(),
        );

        let config = TransformerGatedMlpConfig::new(6, 10)
            .with_bias(true)
            .with_down_bias(Some(false));
        let mlp: TransformerGatedMlp = config.init(&device);
        assert!(mlp.gate_proj.bias.is_some());
        assert!(mlp.down_proj.bias.is_none());
    }
}
