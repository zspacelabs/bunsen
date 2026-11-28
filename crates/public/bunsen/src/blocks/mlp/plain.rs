//! [`BunsenMlp`]: `linear2(act(linear1(x)))`.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout,
        DropoutConfig,
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
    blocks::{
        activation::{
            BunsenActivation,
            BunsenActivationConfig,
        },
        norm::{
            BunsenNormalization,
            BunsenNormalizationConfig,
        },
    },
    burner::module::{
        InputDTypes,
        ModuleInit,
        ResultDType,
    },
    errors::BunsenResult,
};

/// Common meta for [`BunsenMlp`] and [`BunsenMlpConfig`].
pub trait BunsenMlpMeta {
    /// The input width.
    fn d_input(&self) -> usize;

    /// The hidden width, `linear1`'s output.
    fn d_hidden(&self) -> usize;

    /// The output width.
    fn d_output(&self) -> usize;

    /// The dropout rate.
    fn dropout(&self) -> f64;
}

/// Config for [`BunsenMlp`].
#[derive(Config, Debug)]
pub struct BunsenMlpConfig {
    /// The input width.
    pub d_input: usize,

    /// The hidden width.
    pub d_hidden: usize,

    /// The output width.
    pub d_output: usize,

    /// The activation between the projections.
    #[config(default = "BunsenActivationConfig::Relu")]
    pub activation: BunsenActivationConfig,

    /// Whether the projections carry a bias term.
    #[config(default = true)]
    pub bias: bool,

    /// Whether `linear2` carries a bias; `None` follows `bias`. Swin V2's
    /// continuous position bias biases only `linear1`.
    #[config(default = "None")]
    pub down_bias: Option<bool>,

    /// The dropout rate, after the activation and after `linear2` (as
    /// `timm`'s `Mlp` drops); 0 is none.
    #[config(default = 0.0)]
    pub dropout: f64,

    /// A norm on the hidden width, after the activation's dropout and before
    /// `linear2`, sized to `d_hidden` whatever feature count it names
    /// (`BEiT3`'s sub-LN, timm's `Mlp` with a `norm_layer`); `None` builds
    /// none.
    #[config(default = "None")]
    pub hidden_norm: Option<BunsenNormalizationConfig>,
}

impl BunsenMlpConfig {
    /// Whether `linear2` carries a bias: `down_bias`, else `bias`.
    pub fn down_bias(&self) -> bool {
        self.down_bias.unwrap_or(self.bias)
    }
}

impl BunsenMlpMeta for BunsenMlpConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn d_hidden(&self) -> usize {
        self.d_hidden
    }

    fn d_output(&self) -> usize {
        self.d_output
    }

    fn dropout(&self) -> f64 {
        self.dropout
    }
}

impl BunsenMlpConfig {
    /// A transformer's plain feed-forward: `d_model` wide in and out, its
    /// hidden width `d_model * factor`, and no bias (`with_bias(true)` for a
    /// feed-forward with one).
    pub fn expansion(
        d_model: usize,
        factor: usize,
    ) -> Self {
        Self::new(d_model, d_model * factor, d_model).with_bias(false)
    }
}

impl ModuleInit<BunsenMlp> for BunsenMlpConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<BunsenMlp> {
        Ok(BunsenMlp {
            linear1: LinearConfig::new(self.d_input, self.d_hidden)
                .with_bias(self.bias)
                .init(device),
            act: self.activation.try_init(device)?,
            linear2: LinearConfig::new(self.d_hidden, self.d_output)
                .with_bias(self.down_bias())
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            hidden_norm: self
                .hidden_norm
                .clone()
                .map(|n| n.with_num_features(self.d_hidden).try_init(device))
                .transpose()?,
        })
    }
}

/// A position-wise MLP: `linear2(act(linear1(x)))` over the last axis of a
/// tensor of any rank, `d_input` to `d_hidden` to `d_output`; with a dropout
/// after the activation and after `linear2` (none at rate 0, or in
/// inference).
///
/// Built by [`BunsenMlpConfig`].
#[derive(Module, Debug)]
pub struct BunsenMlp {
    /// `d_input -> d_hidden`.
    pub linear1: Linear,

    /// The activation.
    pub act: BunsenActivation,

    /// `d_hidden -> d_output`.
    pub linear2: Linear,

    /// The dropout.
    pub dropout: Dropout,

    /// The norm on the hidden width before `linear2`, if any
    /// ([`BunsenMlpConfig::hidden_norm`]).
    pub hidden_norm: Option<BunsenNormalization>,
}

impl BunsenMlpMeta for BunsenMlp {
    fn d_input(&self) -> usize {
        self.linear1.weight.dims()[0]
    }

    fn d_hidden(&self) -> usize {
        self.linear1.weight.dims()[1]
    }

    fn d_output(&self) -> usize {
        self.linear2.weight.dims()[1]
    }

    fn dropout(&self) -> f64 {
        self.dropout.prob
    }
}

impl BunsenMlp {
    /// The MLP over `x`'s last axis.
    ///
    /// # Arguments
    /// - `x`: a `[..., d_input]` input.
    ///
    /// # Returns
    /// A `[..., d_output]` result, its leading axes `x`'s.
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D> {
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            [..., "in"],
            &x.dims(),
            &[("in", self.d_input())]
        );
        let x = self
            .dropout
            .forward(self.act.forward(self.linear1.forward(x)));
        let x = match &self.hidden_norm {
            Some(norm) => norm.forward(x),
            None => x,
        };
        let x = self.linear2.forward(x);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            [..., "out"],
            &x.dims(),
            &[("out", self.d_output())]
        );
        self.dropout.forward(x)
    }
}

impl InputDTypes for BunsenMlp {
    /// Its projections' dtype.
    fn preferred_dtype(&self) -> FloatDType {
        FloatDType::from(self.linear1.weight.val().dtype())
    }
}

impl ResultDType for BunsenMlp {
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
    use crate::support::testing::{
        cpu_device,
        seeded_tensor,
    };

    /// A config non-default in every field and the model built from it answer
    /// the meta alike.
    #[test]
    fn test_meta() {
        let config = BunsenMlpConfig::new(3, 7, 5)
            .with_activation(BunsenActivationConfig::Gelu)
            .with_bias(false)
            .with_dropout(0.25);
        let mlp: BunsenMlp = config.init(&cpu_device());
        for meta in [&config as &dyn BunsenMlpMeta, &mlp] {
            assert_eq!(
                (
                    meta.d_input(),
                    meta.d_hidden(),
                    meta.d_output(),
                    meta.dropout()
                ),
                (3, 7, 5, 0.25)
            );
        }
        assert!(mlp.linear1.bias.is_none() && mlp.linear2.bias.is_none());
        let up_only: BunsenMlp = BunsenMlpConfig::new(2, 4, 3)
            .with_down_bias(Some(false))
            .init(&cpu_device());
        assert!(up_only.linear1.bias.is_some() && up_only.linear2.bias.is_none());
    }

    /// The forward is the two projections around the activation, over the
    /// last axis of any rank.
    #[test]
    fn test_forward() {
        let device = cpu_device();
        let mlp: BunsenMlp = BunsenMlpConfig::new(4, 6, 3)
            .with_activation(BunsenActivationConfig::Gelu)
            .init(&device);
        let x = seeded_tensor::<4>(1, [2, 3, 5, 4], Distribution::Default, &device);
        let y = mlp.forward(x.clone());
        assert_eq!(y.dims(), [2, 3, 5, 3]);
        let expected = mlp.linear2.forward(burn::tensor::activation::gelu(
            mlp.linear1.forward(x.clone()),
        ));
        y.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), burn::tensor::Tolerance::default());
        let rows: Tensor<2> = x.reshape([30, 4]);
        assert_eq!(mlp.forward(rows).dims(), [30, 3]);
    }

    /// It computes in its projections' dtype, and its result is in it.
    #[test]
    fn test_dtypes() {
        let mlp: BunsenMlp = BunsenMlpConfig::new(2, 2, 2).init(&cpu_device());
        assert_eq!(mlp.preferred_dtype(), FloatDType::F32);
        assert_eq!(mlp.result_dtype(FloatDType::F16), FloatDType::F32);
    }
}
