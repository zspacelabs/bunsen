//! # The norm seam
//!
//! [`NormConfig`] and [`Norm`] are bunsen's normalization layers: the kinds
//! bunsen's blocks use ([`Identity`](Norm::Identity), [`Batch`](Norm::Batch),
//! [`Layer`](Norm::Layer), [`Rms`](Norm::Rms)), each burn's layer, behind one
//! seam that bunsen owns. A block that takes a norm takes a [`NormConfig`],
//! so a norm's dtype handling, and later kinds, are bunsen's to change
//! without waiting on burn's releases.
//!
//! - **Half inputs compute in f32.** [`Rms`](Norm::Rms) normalizes an f16 or
//!   bf16 input in f32 and narrows it before the gain, as `transformers` does;
//!   burn's `RmsNorm` divides in the input's dtype. [`Layer`](Norm::Layer)
//!   normalizes a half input in f32 and narrows the result; burn's `LayerNorm`
//!   widens f16 alone, as bf16 has f32's range.
//! - **The parameters meet the input in one dtype.** Each casts its parameters
//!   to the dtype it computes in, so a gain kept in f32 never meets a half
//!   input in one op (burn-flex refuses that; on the cubecl backends it is
//!   wrong: [`burner::dtypes`](crate::burner::dtypes)).
//! - **The output is the input's dtype** ([`ResultDType`]).
//!
//! The variants and their configs keep burn's `NormalizationConfig` names,
//! so a config's serde form and a native record's keys are burn's. bunsen's
//! loaders skip enum variants, so a loaded tensor's path names no variant
//! either way.

use burn::{
    module::Param,
    nn::{
        BatchNorm,
        BatchNormConfig,
        Identity,
        Initializer,
        LayerNormConfig,
        RmsNorm,
        RmsNormConfig,
    },
    prelude::*,
    tensor::{
        DType,
        FloatDType,
    },
};

use crate::burner::module::{
    InputDTypes,
    ResultDType,
};

/// A [`Norm`]'s config: its kind, and that kind's burn config.
#[derive(Config, Debug)]
pub enum NormConfig {
    /// No normalization: the input as it is.
    Identity,

    /// burn's [`BatchNorm`].
    Batch(BatchNormConfig),

    /// A layer norm, by burn's [`LayerNormConfig`] ([`Norm::Layer`]).
    Layer(LayerNormConfig),

    /// burn's [`RmsNorm`], its forward bunsen's ([`Norm::Rms`]).
    Rms(RmsNormConfig),
}

impl From<BatchNormConfig> for NormConfig {
    fn from(config: BatchNormConfig) -> Self {
        Self::Batch(config)
    }
}

impl From<LayerNormConfig> for NormConfig {
    fn from(config: LayerNormConfig) -> Self {
        Self::Layer(config)
    }
}

impl From<RmsNormConfig> for NormConfig {
    fn from(config: RmsNormConfig) -> Self {
        Self::Rms(config)
    }
}

impl NormConfig {
    /// The norm, on `device`.
    pub fn init(
        &self,
        device: &Device,
    ) -> Norm {
        match self {
            Self::Identity => Norm::Identity(Identity::new()),
            Self::Batch(config) => Norm::Batch(config.init(device)),
            Self::Layer(config) => Norm::Layer(LayerNorm::init(config, device)),
            Self::Rms(config) => Norm::Rms(config.init(device)),
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
/// [module docs](self)).
#[derive(Module, Debug)]
pub enum Norm {
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

impl From<RmsNorm> for Norm {
    fn from(layer: RmsNorm) -> Self {
        Self::Rms(layer)
    }
}

impl From<BatchNorm> for Norm {
    fn from(layer: BatchNorm) -> Self {
        Self::Batch(layer)
    }
}

impl Norm {
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

/// A layer norm, [`Norm::Layer`]'s: burn's `LayerNorm`'s parameters (`gamma`,
/// and `beta` with a bias) and epsilon, under bunsen's forward.
///
/// It is named `LayerNorm`, as burn's is, because burn-store's adapters
/// recognize a layer norm by its type's name (`Struct:LayerNorm`): the
/// `PyTorch` adapter maps a checkpoint's `weight` and `bias` onto `gamma` and
/// `beta`, and the half-precision adapter can leave it out.
#[derive(Module, Debug)]
pub struct LayerNorm {
    /// The gain, `[d_model]`.
    pub gamma: Param<Tensor<1>>,

    /// The bias, `[d_model]`, when the config asks for one.
    pub beta: Option<Param<Tensor<1>>>,

    /// Added to the variance.
    pub epsilon: f64,
}

impl LayerNorm {
    /// Ones for the gain and zeros for the bias, as burn's `LayerNorm`.
    pub fn init(
        config: &LayerNormConfig,
        device: &Device,
    ) -> Self {
        Self {
            gamma: Initializer::Ones.init([config.d_model], device),
            beta: config
                .bias
                .then(|| Initializer::Zeros.init([config.d_model], device)),
            epsilon: config.epsilon,
        }
    }

    /// `input` normalized over its last axis: in f32 when it is half, its
    /// parameters cast to match, narrowed back to `input`'s dtype.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        let dtype = input.dtype();
        let compute = compute_dtype(dtype);
        burn::tensor::module::layer_norm(
            input.cast(compute),
            self.gamma.val().cast(compute),
            self.beta.as_ref().map(|b| b.val().cast(compute)),
            self.epsilon,
        )
        .cast(dtype)
    }
}

/// The dtype a half input is normalized in: f32; any other, its own.
fn compute_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
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
    let x = input.cast(compute_dtype(dtype));
    let rms = (x.clone().square().mean_dim(D - 1) + norm.epsilon).sqrt();
    let normed = (x / rms).cast(dtype);
    normed * norm.gamma.val().cast(dtype).unsqueeze()
}

impl InputDTypes for Norm {
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

impl ResultDType for Norm {
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
    use burn::nn::norm::NormalizationConfig;

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
        let Norm::Rms(norm) = NormConfig::from(config.clone()).init(&device) else {
            unreachable!()
        };
        let ours = Norm::Rms(norm.clone()).forward(input(&device));
        let burns = norm.forward(input(&device));
        ours.into_data()
            .assert_approx_eq::<f32>(&burns.into_data(), burn::tensor::Tolerance::absolute(1e-6));
    }

    /// A bf16 input is normalized in f32 and narrowed once: within bf16's
    /// rounding of the f32 result, and the output is bf16.
    #[test]
    fn test_rms_half_input_computes_in_f32() {
        let device = cpu_device();
        let norm = NormConfig::from(RmsNormConfig::new(4)).init(&device);
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
        let norm = NormConfig::from(RmsNormConfig::new(4)).init(&device);
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
        let identity = NormConfig::Identity;
        assert_eq!(identity.clone().with_num_features(8).num_features(), 0);
        let x = input(&device);
        identity
            .init(&device)
            .forward(x.clone())
            .into_data()
            .assert_eq(&x.clone().into_data(), true);
        for config in [
            NormConfig::from(LayerNormConfig::new(0)),
            NormConfig::from(RmsNormConfig::new(0)),
            NormConfig::from(BatchNormConfig::new(0)),
        ] {
            let config = config.with_num_features(4);
            assert_eq!(config.num_features(), 4);
            assert_eq!(config.init(&device).num_features(), 4);
        }
        let layer = NormConfig::from(LayerNormConfig::new(4)).init(&device);
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
        let ours = NormConfig::from(config.clone()).init(&device);
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
            (NormConfig::Identity, NormalizationConfig::Identity),
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
            let _: NormConfig = serde_json::from_value(ours).unwrap();
        }
    }
}
