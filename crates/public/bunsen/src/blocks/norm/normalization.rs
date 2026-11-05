//! The norm seam's enum and its config.

use burn::{
    nn::{
        BatchNorm,
        BatchNormConfig,
        Identity,
        LayerNormConfig,
        RmsNorm,
        RmsNormConfig,
    },
    prelude::*,
    tensor::FloatDType,
};

use super::LayerNorm;
use crate::burner::{
    module::{
        InputDTypes,
        ResultDType,
    },
    tensor::widen_half,
};

/// A [`BunsenNormalization`]'s config: its kind, and that kind's burn config.
#[derive(Config, Debug)]
pub enum BunsenNormalizationConfig {
    /// No normalization: the input as it is.
    Identity,

    /// burn's [`BatchNorm`].
    Batch(BatchNormConfig),

    /// A layer norm, by burn's [`LayerNormConfig`]
    /// ([`BunsenNormalization::Layer`]).
    Layer(LayerNormConfig),

    /// burn's [`RmsNorm`], its forward bunsen's ([`BunsenNormalization::Rms`]).
    Rms(RmsNormConfig),
}

impl From<BatchNormConfig> for BunsenNormalizationConfig {
    fn from(config: BatchNormConfig) -> Self {
        Self::Batch(config)
    }
}

impl From<LayerNormConfig> for BunsenNormalizationConfig {
    fn from(config: LayerNormConfig) -> Self {
        Self::Layer(config)
    }
}

impl From<RmsNormConfig> for BunsenNormalizationConfig {
    fn from(config: RmsNormConfig) -> Self {
        Self::Rms(config)
    }
}

impl BunsenNormalizationConfig {
    /// The norm, on `device`.
    pub fn init(
        &self,
        device: &Device,
    ) -> BunsenNormalization {
        match self {
            Self::Identity => BunsenNormalization::Identity(Identity::new()),
            Self::Batch(config) => BunsenNormalization::Batch(config.init(device)),
            Self::Layer(config) => BunsenNormalization::Layer(LayerNorm::init(config, device)),
            Self::Rms(config) => BunsenNormalization::Rms(config.init(device)),
        }
    }

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
            Self::Layer(config) => LayerNormConfig {
                d_model: num_features,
                ..config
            }
            .into(),
            Self::Rms(config) => RmsNormConfig {
                d_model: num_features,
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
            Self::Layer(config) => config.d_model,
            Self::Rms(config) => config.d_model,
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

    /// A layer norm: burn's `LayerNorm`'s parameters, under bunsen's forward
    /// ([`LayerNorm::forward`]).
    Layer(LayerNorm),

    /// burn's [`RmsNorm`]'s parameters, under bunsen's forward
    /// ([`forward`](Self::forward)).
    Rms(RmsNorm),
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
            Self::Layer(norm) => norm.forward(input),
            Self::Rms(norm) => rms_forward(norm, input),
        }
    }

    /// The features it normalizes; 0 for [`Identity`](Self::Identity).
    pub fn num_features(&self) -> usize {
        match self {
            Self::Identity(_) => 0,
            Self::Batch(norm) => norm.gamma.shape()[0],
            Self::Layer(norm) => norm.gamma.shape()[0],
            Self::Rms(norm) => norm.gamma.shape()[0],
        }
    }

    /// The epsilon it adds to the variance; `None` for
    /// [`Identity`](Self::Identity).
    pub fn epsilon(&self) -> Option<f64> {
        match self {
            Self::Identity(_) => None,
            Self::Batch(norm) => Some(norm.epsilon),
            Self::Layer(norm) => Some(norm.epsilon),
            Self::Rms(norm) => Some(norm.epsilon),
        }
    }

    /// The float dtype of its parameters; `None` for
    /// [`Identity`](Self::Identity).
    pub fn param_dtype(&self) -> Option<FloatDType> {
        let gamma = match self {
            Self::Identity(_) => return None,
            Self::Batch(norm) => norm.gamma.val(),
            Self::Layer(norm) => norm.gamma.val(),
            Self::Rms(norm) => norm.gamma.val(),
        };
        Some(FloatDType::from(gamma.dtype()))
    }
}

/// RMS normalization over the last axis, as `transformers`' Qwen3 and Llama
/// norms compute it: the input widened to f32 when half, normalized there,
/// narrowed back, then scaled by the gain in the input's dtype.
fn rms_forward<const D: usize>(
    norm: &RmsNorm,
    input: Tensor<D>,
) -> Tensor<D> {
    let dtype = input.dtype();
    let x = input.cast(widen_half(dtype));
    let rms = (x.clone().square().mean_dim(D - 1) + norm.epsilon).sqrt();
    let normed = (x / rms).cast(dtype);
    normed * norm.gamma.val().cast(dtype).unsqueeze()
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
        support::testing::cpu_device,
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
        assert_eq!(half.dtype(), DType::BF16);
        half.cast(DType::F32).into_data().assert_approx_eq::<f32>(
            &reference.into_data(),
            burn::tensor::Tolerance::absolute(2e-2),
        );
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
        assert_eq!(half.dtype(), DType::BF16);
        half.cast(DType::F32).into_data().assert_approx_eq::<f32>(
            &burns.forward(x).into_data(),
            burn::tensor::Tolerance::absolute(2e-2),
        );
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
        ] {
            let ours = serde_json::to_value(&ours).unwrap();
            assert_eq!(ours, serde_json::to_value(&burns).unwrap());
            let _: BunsenNormalizationConfig = serde_json::from_value(ours).unwrap();
        }
    }
}
