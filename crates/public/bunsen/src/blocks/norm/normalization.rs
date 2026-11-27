//! The norm seam's enum and its config.

use burn::{
    nn::{
        BatchNorm,
        BatchNormConfig,
        GroupNorm,
        GroupNormConfig,
        Identity,
        LayerNormConfig,
        RmsNorm,
        RmsNormConfig,
    },
    prelude::*,
    tensor::FloatDType,
};

use super::{
    BunsenLayerNormConfig,
    EvoNormS0,
    EvoNormS0Config,
    LayerNorm,
};
use crate::{
    burner::module::{
        Forward1Module,
        InputDTypes,
        ModuleInit,
        ResultDType,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::norm::RmsNormOptions,
};

/// A [`BunsenNormalization`]'s config: its kind, and that kind's burn config.
#[derive(Config, Debug)]
pub enum BunsenNormalizationConfig {
    /// No normalization: the input as it is.
    Identity,

    /// burn's [`BatchNorm`].
    Batch(BatchNormConfig),

    /// burn's [`GroupNorm`]: its features are its `num_channels`, split into
    /// `num_groups` groups.
    Group(GroupNormConfig),

    /// bunsen's layer norm ([`BunsenNormalization::Layer`]); its serde form
    /// is burn's [`LayerNormConfig`]'s.
    Layer(BunsenLayerNormConfig),

    /// burn's [`RmsNorm`], its forward bunsen's ([`BunsenNormalization::Rms`]).
    Rms(RmsNormConfig),

    /// `EvoNorm-S0` ([`EvoNormS0`]): a norm with its own activation.
    EvoS0(EvoNormS0Config),
}

impl From<EvoNormS0Config> for BunsenNormalizationConfig {
    fn from(config: EvoNormS0Config) -> Self {
        Self::EvoS0(config)
    }
}

impl From<BatchNormConfig> for BunsenNormalizationConfig {
    fn from(config: BatchNormConfig) -> Self {
        Self::Batch(config)
    }
}

impl From<GroupNormConfig> for BunsenNormalizationConfig {
    fn from(config: GroupNormConfig) -> Self {
        Self::Group(config)
    }
}

impl From<BunsenLayerNormConfig> for BunsenNormalizationConfig {
    fn from(config: BunsenLayerNormConfig) -> Self {
        Self::Layer(config)
    }
}

impl From<LayerNormConfig> for BunsenNormalizationConfig {
    fn from(config: LayerNormConfig) -> Self {
        Self::Layer(config.into())
    }
}

impl From<RmsNormConfig> for BunsenNormalizationConfig {
    fn from(config: RmsNormConfig) -> Self {
        Self::Rms(config)
    }
}

/// The norm, on `device`.
///
/// # Errors
///
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a group norm
/// whose channels don't split into its groups.
impl ModuleInit<BunsenNormalization> for BunsenNormalizationConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<BunsenNormalization> {
        Ok(match self {
            Self::Identity => BunsenNormalization::Identity(Identity::new()),
            Self::Batch(config) => BunsenNormalization::Batch(config.init(device)),
            Self::Group(config) => {
                if config.num_groups == 0 || config.num_channels % config.num_groups != 0 {
                    return Err(BunsenError::illegal(format!(
                        "a group norm of {} channels in {} groups: the groups split the channels",
                        config.num_channels, config.num_groups
                    )));
                }
                BunsenNormalization::Group(config.init(device))
            }
            Self::Layer(config) => BunsenNormalization::Layer(config.try_init(device)?),
            Self::Rms(config) => BunsenNormalization::Rms(config.init(device)),
            Self::EvoS0(config) => BunsenNormalization::EvoS0(config.try_init(device)?),
        })
    }
}

impl BunsenNormalizationConfig {
    /// This config for `num_features` features; [`Identity`](Self::Identity)
    /// has none.
    pub fn with_num_features(
        self,
        num_features: usize,
    ) -> Self {
        match self {
            Self::Identity => self,
            Self::Batch(config) => BatchNormConfig {
                num_features,
                ..config
            }
            .into(),
            Self::Group(config) => GroupNormConfig {
                num_channels: num_features,
                ..config
            }
            .into(),
            Self::Layer(config) => BunsenLayerNormConfig {
                d_model: num_features,
                ..config
            }
            .into(),
            Self::Rms(config) => RmsNormConfig {
                d_model: num_features,
                ..config
            }
            .into(),
            Self::EvoS0(config) => EvoNormS0Config {
                num_features,
                ..config
            }
            .into(),
        }
    }

    /// The features it normalizes; 0 for [`Identity`](Self::Identity).
    pub fn num_features(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Batch(config) => config.num_features,
            Self::Group(config) => config.num_channels,
            Self::Layer(config) => config.d_model,
            Self::Rms(config) => config.d_model,
            Self::EvoS0(config) => config.num_features,
        }
    }
}

/// A normalization layer: burn's, behind bunsen's seam (the
/// [module docs](super)).
#[derive(Module, Debug)]
pub enum BunsenNormalization {
    /// No normalization.
    Identity(Identity),

    /// burn's [`BatchNorm`].
    Batch(BatchNorm),

    /// burn's [`GroupNorm`], over a `[batch, channels, ...]` input.
    Group(GroupNorm),

    /// A layer norm: burn's `LayerNorm`'s parameters, under bunsen's forward
    /// ([`LayerNorm::forward`]).
    Layer(LayerNorm),

    /// burn's [`RmsNorm`]'s parameters, under bunsen's forward
    /// ([`forward`](Self::forward)).
    Rms(RmsNorm),

    /// `EvoNorm-S0`, over a `[batch, channels, ...]` input; its activation
    /// its own.
    EvoS0(EvoNormS0),
}

impl From<RmsNorm> for BunsenNormalization {
    fn from(layer: RmsNorm) -> Self {
        Self::Rms(layer)
    }
}

impl From<BatchNorm> for BunsenNormalization {
    fn from(layer: BatchNorm) -> Self {
        Self::Batch(layer)
    }
}

impl From<GroupNorm> for BunsenNormalization {
    fn from(layer: GroupNorm) -> Self {
        Self::Group(layer)
    }
}

impl BunsenNormalization {
    /// `input`, normalized; in `input`'s dtype.
    ///
    /// [`Rms`](Self::Rms) and [`Layer`](Self::Layer) normalize over the last
    /// axis, an f16 or bf16 input in f32, their parameters cast to match.
    /// The other kinds are burn's forwards.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        match self {
            Self::Identity(norm) => norm.forward(input),
            Self::Batch(norm) => norm.forward(input),
            Self::Group(norm) => norm.forward(input),
            Self::Layer(norm) => norm.forward(input),
            Self::Rms(norm) => {
                let dtype = input.dtype();
                RmsNormOptions::default()
                    .with_eps(norm.epsilon as f32)
                    .with_widen_input(Some(FloatDType::F32))
                    .with_widen_sum(None)
                    .norm(input)
                    * norm.gamma.val().cast(dtype).unsqueeze()
            }
            Self::EvoS0(norm) => norm.forward(input),
        }
    }

    /// The features it normalizes; 0 for [`Identity`](Self::Identity).
    pub fn num_features(&self) -> usize {
        match self {
            Self::Identity(_) => 0,
            Self::Batch(norm) => norm.gamma.shape()[0],
            Self::Group(norm) => norm.num_channels,
            Self::Layer(norm) => norm.gamma.shape()[0],
            Self::Rms(norm) => norm.gamma.shape()[0],
            Self::EvoS0(norm) => norm.num_features(),
        }
    }

    /// The epsilon it adds to the variance; `None` for
    /// [`Identity`](Self::Identity).
    pub fn epsilon(&self) -> Option<f64> {
        match self {
            Self::Identity(_) => None,
            Self::Batch(norm) => Some(norm.epsilon),
            Self::Group(norm) => Some(norm.epsilon),
            Self::Layer(norm) => Some(norm.epsilon),
            Self::Rms(norm) => Some(norm.epsilon),
            Self::EvoS0(norm) => Some(norm.epsilon),
        }
    }

    /// The float dtype of its parameters; `None` for
    /// [`Identity`](Self::Identity) and a group norm without an affine.
    pub fn param_dtype(&self) -> Option<FloatDType> {
        let gamma = match self {
            Self::Identity(_) => return None,
            Self::Batch(norm) => norm.gamma.val(),
            Self::Group(norm) => norm.gamma.as_ref()?.val(),
            Self::Layer(norm) => norm.gamma.val(),
            Self::Rms(norm) => norm.gamma.val(),
            Self::EvoS0(norm) => norm.weight.val(),
        };
        Some(FloatDType::from(gamma.dtype()))
    }
}

impl<const D: usize> Forward1Module<Tensor<D>> for BunsenNormalization {
    type Output = Tensor<D>;

    /// [`BunsenNormalization::forward`].
    fn forward(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        BunsenNormalization::forward(self, input)
    }
}

impl InputDTypes for BunsenNormalization {
    /// Its parameters' dtype, or f32 for [`Identity`](Self::Identity).
    fn preferred_dtype(&self) -> FloatDType {
        self.param_dtype().unwrap_or(FloatDType::F32)
    }

    /// Every float dtype: it computes a half input in f32, and returns the
    /// input's dtype.
    fn accepts_dtype(
        &self,
        _dtype: FloatDType,
    ) -> bool {
        true
    }
}

impl ResultDType for BunsenNormalization {
    /// The input's.
    fn result_dtype(
        &self,
        input: FloatDType,
    ) -> FloatDType {
        input
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::norm::NormalizationConfig,
        tensor::DType,
    };

    use super::*;
    use crate::{
        burner::module::DTypeMapper,
        errors::testing::ErrorMatcher,
        support::testing::{
            cpu_device,
            matching::TensorMatcher,
        },
    };

    fn input(device: &Device) -> Tensor<2> {
        Tensor::from_data(
            [[0.5f32, -1.25, 2.0, 3.5], [-0.75, 4.0, 1.5, -2.25]],
            (device, DType::F32),
        )
    }

    /// In f32, the RMS forward is burn's.
    #[test]
    fn test_rms_f32_is_burns() {
        let device = cpu_device();
        let config = RmsNormConfig::new(4);
        let BunsenNormalization::Rms(norm) =
            BunsenNormalizationConfig::from(config.clone()).init(&device)
        else {
            unreachable!()
        };
        let ours = BunsenNormalization::Rms(norm.clone()).forward(input(&device));
        let burns = norm.forward(input(&device));
        ours.into_data()
            .assert_approx_eq::<f32>(&burns.into_data(), burn::tensor::Tolerance::absolute(1e-6));
    }

    /// A bf16 input is normalized in f32 and narrowed once: within bf16's
    /// rounding of the f32 result, and the output is bf16.
    #[test]
    fn test_rms_half_input_computes_in_f32() {
        let device = cpu_device();
        let norm = BunsenNormalizationConfig::from(RmsNormConfig::new(4)).init(&device);
        let reference = norm.forward(input(&device));
        let half = norm.forward(input(&device).cast(DType::BF16));
        TensorMatcher::close_to(&reference, burn::tensor::Tolerance::absolute(2e-2))
            .dtype(DType::BF16)
            .assert(&half);
    }

    /// A gain kept in f32 meets a bf16 input in the input's dtype: no mixed
    /// op (which burn-flex would refuse), and the output is bf16.
    #[test]
    fn test_rms_f32_gain_with_half_input() {
        let device = cpu_device();
        let norm = BunsenNormalizationConfig::from(RmsNormConfig::new(4)).init(&device);
        assert_eq!(norm.param_dtype(), Some(FloatDType::F32));
        let out = norm.forward(input(&device).cast(DType::BF16));
        assert_eq!(out.dtype(), DType::BF16);

        let half = norm.map(&mut DTypeMapper::new(DType::BF16));
        assert_eq!(half.preferred_dtype(), FloatDType::BF16);
        assert_eq!(half.forward(input(&device)).dtype(), DType::F32);
        assert_eq!(half.result_dtype(FloatDType::F32), FloatDType::F32);
    }

    /// The kinds and their features, as burn's wrapper has them.
    #[test]
    fn test_kinds_and_features() {
        let device = cpu_device();
        let identity = BunsenNormalizationConfig::Identity;
        assert_eq!(identity.clone().with_num_features(8).num_features(), 0);
        let x = input(&device);
        identity
            .init(&device)
            .forward(x.clone())
            .into_data()
            .assert_eq(&x.clone().into_data(), true);
        for config in [
            BunsenNormalizationConfig::from(LayerNormConfig::new(0)),
            BunsenNormalizationConfig::from(RmsNormConfig::new(0)),
            BunsenNormalizationConfig::from(BatchNormConfig::new(0)),
            BunsenNormalizationConfig::from(GroupNormConfig::new(2, 0)),
        ] {
            let config = config.with_num_features(4);
            assert_eq!(config.num_features(), 4);
            assert_eq!(config.init(&device).num_features(), 4);
        }
        let layer = BunsenNormalizationConfig::from(LayerNormConfig::new(4)).init(&device);
        assert_eq!(
            layer.forward(x.clone().cast(DType::BF16)).dtype(),
            DType::BF16
        );
    }

    /// The layer norm is burn's in f32, and computes a bf16 input in f32 with
    /// an f32 gain, which burn's own refuses on flex.
    #[test]
    fn test_layer_matches_burn_and_widens() {
        let device = cpu_device();
        let config = LayerNormConfig::new(4);
        let ours = BunsenNormalizationConfig::from(config.clone()).init(&device);
        let burns = config.init(&device);
        let x = input(&device);
        ours.forward(x.clone()).into_data().assert_approx_eq::<f32>(
            &burns.forward(x.clone()).into_data(),
            burn::tensor::Tolerance::absolute(1e-6),
        );
        let half = ours.forward(x.clone().cast(DType::BF16));
        TensorMatcher::close_to(&burns.forward(x), burn::tensor::Tolerance::absolute(2e-2))
            .dtype(DType::BF16)
            .assert(&half);
    }

    /// A config's serde form is burn's `NormalizationConfig`'s, so saved
    /// configs read either way.
    #[test]
    fn test_serde_form_is_burns() {
        for (ours, burns) in [
            (
                BunsenNormalizationConfig::Identity,
                NormalizationConfig::Identity,
            ),
            (
                RmsNormConfig::new(16).with_epsilon(1e-6).into(),
                RmsNormConfig::new(16).with_epsilon(1e-6).into(),
            ),
            (
                LayerNormConfig::new(8).into(),
                LayerNormConfig::new(8).into(),
            ),
            (
                BatchNormConfig::new(3).into(),
                BatchNormConfig::new(3).into(),
            ),
            (
                GroupNormConfig::new(32, 64).into(),
                GroupNormConfig::new(32, 64).into(),
            ),
        ] {
            let ours = serde_json::to_value(&ours).unwrap();
            assert_eq!(ours, serde_json::to_value(&burns).unwrap());
            let _: BunsenNormalizationConfig = serde_json::from_value(ours).unwrap();
        }
    }

    /// A group norm is burn's; channels its groups don't split are an
    /// error from `try_init`, not a panic.
    #[test]
    fn test_group_is_burns_and_checks_its_groups() {
        let device = cpu_device();
        let config = GroupNormConfig::new(2, 0);
        let ours = BunsenNormalizationConfig::from(config.clone())
            .with_num_features(4)
            .init(&device);
        assert_eq!(ours.num_features(), 4);
        assert_eq!(ours.epsilon(), Some(1e-5));
        assert_eq!(ours.param_dtype(), Some(FloatDType::F32));
        let burns = GroupNormConfig::new(2, 4).init(&device);
        let x: Tensor<3> = input(&device).reshape([1, 4, 2]);
        ours.forward(x.clone()).into_data().assert_approx_eq::<f32>(
            &burns.forward(x).into_data(),
            burn::tensor::Tolerance::absolute(1e-6),
        );

        let odd: BunsenResult<BunsenNormalization> = BunsenNormalizationConfig::from(config)
            .with_num_features(5)
            .try_init(&device);
        ErrorMatcher::illegal()
            .message_contains("5 channels in 2 groups")
            .assert_err(&odd);
    }
}
