//! [`BunsenGatedMlp`]: `down(act(gate(x)) * up(x))`.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::Tensor,
    tensor::{
        Device,
        FloatDType,
    },
};

use crate::{
    blocks::activation::{
        BunsenActivation,
        BunsenActivationConfig,
    },
    burner::module::{
        InputDTypes,
        ModuleInit,
        ResultDType,
    },
    errors::BunsenResult,
};

/// Common meta for [`BunsenGatedMlp`] and [`BunsenGatedMlpConfig`].
pub trait BunsenGatedMlpMeta {
    /// The input and output width.
    fn n_embed(&self) -> usize;

    /// The hidden width.
    fn d_ff(&self) -> usize;
}

/// Config for [`BunsenGatedMlp`].
///
/// Implements [`BunsenGatedMlpMeta`].
#[derive(Config, Debug)]
pub struct BunsenGatedMlpConfig {
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

impl BunsenGatedMlpConfig {
    /// Whether the down projection carries a bias: `down_bias`, else `bias`.
    pub fn down_bias(&self) -> bool {
        self.down_bias.unwrap_or(self.bias)
    }
}

impl BunsenGatedMlpMeta for BunsenGatedMlpConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }

    fn d_ff(&self) -> usize {
        self.d_ff
    }
}

impl ModuleInit<BunsenGatedMlp> for BunsenGatedMlpConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<BunsenGatedMlp> {
        let linear = |i, o, bias| LinearConfig::new(i, o).with_bias(bias).init(device);
        Ok(BunsenGatedMlp {
            gate_proj: linear(self.n_embed, self.d_ff, self.bias),
            up_proj: linear(self.n_embed, self.d_ff, self.bias),
            down_proj: linear(self.d_ff, self.n_embed, self.down_bias()),
            act: self.activation.try_init(device)?,
        })
    }
}

/// The gated MLP (Shazeer 2020): `down(act(gate(x)) * up(x))` over the last
/// axis of a tensor of any rank. With `silu`, the `SwiGLU` of Llama, Qwen and
/// their kin; with tanh-approximated GELU, Gemma's `GeGLU`. The general form
/// of burn's `SwiGlu`.
///
/// The fields carry the names `transformers` gives them.
///
/// Built by [`BunsenGatedMlpConfig`].
/// Implements [`BunsenGatedMlpMeta`].
#[derive(Module, Debug)]
pub struct BunsenGatedMlp {
    /// `n_embed -> d_ff`, the gate.
    pub gate_proj: Linear,

    /// `n_embed -> d_ff`.
    pub up_proj: Linear,

    /// `d_ff -> n_embed`.
    pub down_proj: Linear,

    /// The gate's activation.
    pub act: BunsenActivation,
}

impl BunsenGatedMlpMeta for BunsenGatedMlp {
    fn n_embed(&self) -> usize {
        self.gate_proj.weight.dims()[0]
    }

    fn d_ff(&self) -> usize {
        self.gate_proj.weight.dims()[1]
    }
}

impl BunsenGatedMlp {
    /// The block over `x`'s last axis.
    ///
    /// # Arguments
    /// - `x`: a `[..., n_embed]` input.
    ///
    /// # Returns
    /// A `[..., n_embed]` result, its leading axes `x`'s.
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D> {
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            [..., "n_embed"],
            &x.dims(),
            &[("n_embed", self.n_embed())]
        );
        let gate = self.act.forward(self.gate_proj.forward(x.clone()));
        let y = self.down_proj.forward(gate * self.up_proj.forward(x));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            [..., "n_embed"],
            &y.dims(),
            &[("n_embed", self.n_embed())]
        );
        y
    }
}

impl InputDTypes for BunsenGatedMlp {
    /// Its projections' dtype.
    fn preferred_dtype(&self) -> FloatDType {
        FloatDType::from(self.up_proj.weight.val().dtype())
    }
}

impl ResultDType for BunsenGatedMlp {
    /// Its own: the result is in its projections' dtype.
    fn result_dtype(
        &self,
        _input: FloatDType,
    ) -> FloatDType {
        self.preferred_dtype()
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;

    #[test]
    fn test_gated_mlp() {
        use crate::support::testing::cpu_device;
        let config = BunsenGatedMlpConfig::new(6, 10).with_bias(true);
        let mlp: BunsenGatedMlp = config.init(&cpu_device());
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
                BunsenGatedMlpConfig::new(6, 10),
                burn::tensor::activation::silu::<3> as fn(Tensor<3>) -> Tensor<3>,
            ),
            (
                BunsenGatedMlpConfig::new(6, 10)
                    .with_activation(BunsenActivationConfig::GeluApproximate),
                burn::tensor::activation::gelu_approximate::<3>,
            ),
        ] {
            let mlp: BunsenGatedMlp = config.init(&device);
            let expected = mlp
                .down_proj
                .forward(act(mlp.gate_proj.forward(x.clone())) * mlp.up_proj.forward(x.clone()));
            assert_tensors_close(
                &mlp.forward(x.clone()),
                &expected,
                burn::tensor::Tolerance::default(),
            );
        }
    }

    /// The down projection's bias follows `bias` unless `down_bias` says
    /// otherwise.
    #[test]
    fn test_gated_mlp_down_bias() {
        use crate::support::testing::cpu_device;
        let device = cpu_device();
        let config = BunsenGatedMlpConfig::new(6, 10).with_down_bias(Some(true));
        assert!(!config.bias);
        assert!(config.down_bias());
        let mlp: BunsenGatedMlp = config.init(&device);
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

        let config = BunsenGatedMlpConfig::new(6, 10)
            .with_bias(true)
            .with_down_bias(Some(false));
        let mlp: BunsenGatedMlp = config.init(&device);
        assert!(mlp.gate_proj.bias.is_some());
        assert!(mlp.down_proj.bias.is_none());
    }

    /// The forward runs over the last axis of any rank, and the dtypes are its
    /// projections'.
    #[test]
    fn test_rank_and_dtypes() {
        use crate::support::testing::cpu_device;
        let mlp: BunsenGatedMlp = BunsenGatedMlpConfig::new(4, 6).init(&cpu_device());
        assert_eq!(
            mlp.forward(Tensor::<2>::zeros([5, 4], &cpu_device()))
                .dims(),
            [5, 4]
        );
        assert_eq!(
            mlp.forward(Tensor::<4>::zeros([2, 3, 5, 4], &cpu_device()))
                .dims(),
            [2, 3, 5, 4]
        );
        assert_eq!(mlp.preferred_dtype(), FloatDType::F32);
        assert_eq!(mlp.result_dtype(FloatDType::BF16), FloatDType::F32);
    }
}
