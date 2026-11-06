//! [`GatedNorm`], a norm gated by a second input.

use burn::{
    config::Config,
    module::Module,
    nn::RmsNormConfig,
    prelude::Tensor,
    tensor::{
        Device,
        FloatDType,
    },
};

use super::{
    BunsenNormalization,
    BunsenNormalizationConfig,
};
use crate::{
    blocks::activation::{
        BunsenActivation,
        BunsenActivationConfig,
    },
    burner::{
        module::{
            Forward2Module,
            Gate,
            InputDTypes,
            ModuleInit,
            ResultDType,
        },
        tensor::Widening,
    },
    errors::BunsenResult,
};

/// Common meta for [`GatedNorm`] and [`GatedNormConfig`].
pub trait GatedNormMeta {
    /// The features it normalizes, the input's last axis; 0 for an
    /// [`Identity`](BunsenNormalization::Identity) norm.
    fn d_model(&self) -> usize;
}

/// Config for [`GatedNorm`].
#[derive(Config, Debug)]
pub struct GatedNormConfig {
    /// The norm.
    pub norm: BunsenNormalizationConfig,

    /// The gate's activation.
    #[config(default = "BunsenActivationConfig::Silu")]
    pub activation: BunsenActivationConfig,

    /// The widening the gate's product is taken in.
    #[config(default = "Widening::F32")]
    pub widening: Widening,
}

impl GatedNormConfig {
    /// A root-mean-square norm of `d_model` features, with `epsilon`, gated
    /// by `silu`, the product in f32 for a half input: the gated `DeltaNet`'s
    /// output norm, `w · x̂ · silu(z)` (`transformers`' `Qwen3_5RMSNormGated`,
    /// FLA's `FusedRMSNormGated`).
    pub fn rms(
        d_model: usize,
        epsilon: f64,
    ) -> Self {
        Self::new(RmsNormConfig::new(d_model).with_epsilon(epsilon).into())
    }

    /// The config, normalizing `d_model` features.
    pub fn with_num_features(
        self,
        d_model: usize,
    ) -> Self {
        Self {
            norm: self.norm.with_num_features(d_model),
            ..self
        }
    }
}

impl GatedNormMeta for GatedNormConfig {
    fn d_model(&self) -> usize {
        self.norm.num_features()
    }
}

impl ModuleInit<GatedNorm> for GatedNormConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<GatedNorm> {
        Ok(GatedNorm {
            norm: self.norm.init(device),
            activation: self.activation.init(device),
            widening: self.widening,
        })
    }
}

/// A norm whose output is gated by a second input: `norm(x) ⊙ act(gate)`,
/// the product taken in a [`Widening`] of the norm's output dtype and
/// narrowed back ([`Gate`]).
///
/// [`GatedNormConfig::rms`] is the gated `DeltaNet`'s output norm,
/// `w · x̂ · silu(z)`: the [`Rms`](BunsenNormalization::Rms) seam
/// normalizes a half input in f32 and narrows it before its gain, as
/// `transformers` does, and the gate's product is taken in f32.
///
/// Its record paths are its fields': a checkpoint's norm gain is
/// `norm.weight`.
///
/// Built by [`GatedNormConfig`].
#[derive(Module, Debug)]
pub struct GatedNorm {
    /// The norm.
    pub norm: BunsenNormalization,

    /// The gate's activation.
    pub activation: BunsenActivation,

    /// The widening the gate's product is taken in.
    #[module(skip)]
    pub widening: Widening,
}

impl GatedNormMeta for GatedNorm {
    fn d_model(&self) -> usize {
        self.norm.num_features()
    }
}

impl GatedNorm {
    /// The block as a [`Gate`] view of its norm and activation.
    pub fn gate(&self) -> Gate<'_, BunsenNormalization, BunsenActivation> {
        Gate::new(&self.norm, &self.activation).with_widening(self.widening)
    }

    /// `norm(x) ⊙ act(gate)`, over the last axis; in `x`'s dtype.
    ///
    /// # Arguments
    /// - `x`: a `[..., D]` tensor.
    /// - `gate`: the gate's logits, `x`'s shape.
    ///
    /// # Returns
    /// - a `[..., D]` tensor, in `x`'s dtype.
    pub fn forward<const R: usize>(
        &self,
        x: Tensor<R>,
        gate: Tensor<R>,
    ) -> Tensor<R> {
        self.gate().forward(x, gate)
    }
}

impl<const R: usize> Forward2Module<Tensor<R>, Tensor<R>> for GatedNorm {
    type Output = Tensor<R>;

    /// [`GatedNorm::forward`].
    fn forward(
        &self,
        x: Tensor<R>,
        gate: Tensor<R>,
    ) -> Tensor<R> {
        GatedNorm::forward(self, x, gate)
    }
}

impl InputDTypes for GatedNorm {
    /// Its norm's.
    fn preferred_dtype(&self) -> FloatDType {
        self.gate().preferred_dtype()
    }

    /// Its norm's.
    fn accepts_dtype(
        &self,
        dtype: FloatDType,
    ) -> bool {
        self.gate().accepts_dtype(dtype)
    }
}

impl ResultDType for GatedNorm {
    /// Its norm's.
    fn result_dtype(
        &self,
        input: FloatDType,
    ) -> FloatDType {
        self.gate().result_dtype(input)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        tensor::{
            DType,
            Distribution,
            Tolerance,
            activation::{
                sigmoid,
                silu,
            },
        },
    };

    use super::*;
    use crate::{
        burner::module::DTypeMapper,
        ops::norm::RmsNormOptions,
        support::testing::{
            assert_tensors_close,
            cpu_device,
            seeded_tensor,
        },
    };

    fn rms_gain(norm: &GatedNorm) -> Tensor<1> {
        match &norm.norm {
            BunsenNormalization::Rms(rms) => rms.gamma.val(),
            other => panic!("not an RMS norm: {other:?}"),
        }
    }

    /// `w · x / sqrt(mean(x²) + eps) · silu(z)`, per row, as written out.
    #[test]
    fn test_rms() {
        let device = cpu_device();
        let cfg = GatedNormConfig::rms(6, 1e-5);
        assert_eq!(cfg.d_model(), 6);
        assert_eq!(cfg.widening, Widening::F32);
        let mut norm: GatedNorm = cfg.init(&device);
        assert_eq!(norm.d_model(), 6);
        if let BunsenNormalization::Rms(rms) = &mut norm.norm {
            rms.gamma = Param::from_tensor(seeded_tensor::<1>(
                1,
                [6],
                Distribution::Uniform(0.5, 1.5),
                &device,
            ));
        }

        let x = seeded_tensor::<4>(2, [2, 3, 4, 6], Distribution::Default, &device);
        let z = seeded_tensor::<4>(3, [2, 3, 4, 6], Distribution::Default, &device);
        let rms = x.clone().square().mean_dim(3).add_scalar(1e-5).sqrt();
        let expected = x.clone() / rms * rms_gain(&norm).unsqueeze() * silu(z.clone());
        let y = norm.forward(x.clone(), z.clone());
        assert_eq!(y.dims(), [2, 3, 4, 6]);
        assert_tensors_close(&y, &expected, Tolerance::default());
        let by_trait = Forward2Module::forward(&norm, x, z);
        by_trait.into_data().assert_eq(&y.into_data(), true);
    }

    /// The default gain is ones (the norm, gated), and a zero gate closes
    /// the output.
    #[test]
    fn test_default_gain_and_closed_gate() {
        let device = cpu_device();
        let norm: GatedNorm = GatedNormConfig::rms(4, 1e-6).init(&device);
        let x = seeded_tensor::<2>(4, [3, 4], Distribution::Default, &device);
        let z = seeded_tensor::<2>(5, [3, 4], Distribution::Default, &device);
        let plain = RmsNormOptions::default().with_eps(1e-6).norm(x.clone());
        assert_tensors_close(
            &norm.forward(x.clone(), z.clone()),
            &(plain * silu(z)),
            Tolerance::default(),
        );
        let closed = norm.forward(x, Tensor::zeros([3, 4], &device));
        assert_eq!(closed.abs().max().into_scalar::<f32>(), 0.0);
    }

    /// A half input normalizes in f32 and narrows, takes the gain in its
    /// dtype, then meets the gate in f32, narrowed once, as `transformers`
    /// orders the casts; a bf16 gain under an f32 input returns f32.
    #[test]
    fn test_half_input() {
        let device = cpu_device();
        let norm: GatedNorm = GatedNormConfig::rms(4, 1e-6).init(&device);
        let x = seeded_tensor::<2>(6, [3, 4], Distribution::Default, &device);
        let z = seeded_tensor::<2>(7, [3, 4], Distribution::Default, &device);
        let half = x.clone().cast(DType::BF16);
        let x_hat = RmsNormOptions::default()
            .with_eps(1e-6)
            .with_widen_input(Some(FloatDType::F32))
            .with_widen_sum(None)
            .norm(half.clone())
            * rms_gain(&norm).cast(DType::BF16).unsqueeze();
        let expected = (x_hat.cast(DType::F32) * silu(z.clone())).cast(DType::BF16);
        let got = norm.forward(half, z.clone());
        assert_eq!(got.dtype(), DType::BF16);
        got.into_data().assert_eq(&expected.into_data(), true);

        let half_gain = norm.map(&mut DTypeMapper::new(DType::BF16));
        assert_eq!(half_gain.preferred_dtype(), FloatDType::BF16);
        assert!(half_gain.accepts_dtype(FloatDType::F32));
        assert_eq!(half_gain.forward(x, z).dtype(), DType::F32);
        assert_eq!(half_gain.result_dtype(FloatDType::F32), FloatDType::F32);
    }

    /// Another norm, activation and widening: a layer norm gated by a
    /// sigmoid, all in the input's dtype.
    #[test]
    fn test_other_parts() {
        let device = cpu_device();
        let cfg = GatedNormConfig::new(BunsenNormalizationConfig::Layer(
            burn::nn::LayerNormConfig::new(0),
        ))
        .with_activation(BunsenActivationConfig::Sigmoid)
        .with_widening(Widening::NONE)
        .with_num_features(5);
        assert_eq!(cfg.d_model(), 5);
        let norm: GatedNorm = cfg.init(&device);
        let x = seeded_tensor::<2>(8, [2, 5], Distribution::Default, &device);
        let z = seeded_tensor::<2>(9, [2, 5], Distribution::Default, &device);
        let expected = norm.norm.forward(x.clone()) * sigmoid(z.clone());
        norm.forward(x, z)
            .into_data()
            .assert_eq(&expected.into_data(), true);
    }

    #[test]
    #[should_panic(expected = "a gate has its gated output's shape")]
    fn test_gate_shape() {
        let device = cpu_device();
        let norm: GatedNorm = GatedNormConfig::rms(4, 1e-6).init(&device);
        let _ = norm.forward(
            Tensor::<2>::zeros([2, 4], &device),
            Tensor::zeros([3, 4], &device),
        );
    }
}
