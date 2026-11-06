//! [`GatedRmsNorm`], a root-mean-square norm gated by a second input.

use burn::{
    config::Config,
    module::Param,
    nn::Initializer,
    prelude::*,
    tensor::{
        FloatDType,
        activation::silu,
    },
};

use crate::{
    burner::{
        module::{
            InputDTypes,
            ResultDType,
        },
        tensor::widen_half,
    },
    ops::norm::RmsNormOptions,
};

/// Common meta for [`GatedRmsNorm`] and [`GatedRmsNormConfig`].
pub trait GatedRmsNormMeta {
    /// The features it normalizes, the input's last axis.
    fn d_model(&self) -> usize;
}

/// Config for [`GatedRmsNorm`].
#[derive(Config, Debug)]
pub struct GatedRmsNormConfig {
    /// The features it normalizes, the input's last axis.
    pub d_model: usize,

    /// Added to the mean of squares.
    #[config(default = 1e-6)]
    pub epsilon: f64,
}

impl GatedRmsNormMeta for GatedRmsNormConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }
}

impl GatedRmsNormConfig {
    /// Initializes the module: a gain of ones.
    pub fn init(
        &self,
        device: &Device,
    ) -> GatedRmsNorm {
        GatedRmsNorm {
            weight: Initializer::Ones.init([self.d_model], device),
            epsilon: self.epsilon,
        }
    }
}

/// A root-mean-square norm whose output is gated by a second input,
/// `w · x̂ · silu(z)`: the gated `DeltaNet`'s output norm (`transformers`'
/// `Qwen3_5RMSNormGated`, FLA's `FusedRMSNormGated`).
///
/// `x̂` is `x / sqrt(mean(x²) + eps)` over the last axis, a half input in
/// f32; it is narrowed back to `x`'s dtype and scaled by the gain, as stored
/// (not `1 + w`), in that dtype; the gate `silu(z)` is applied in f32, and
/// the result narrowed to `x`'s dtype, as `transformers` orders the casts.
///
/// Its gain is `weight`, the checkpoints' name: burn-store's adapters rename
/// a `weight` to `gamma` only on the norm types they know.
///
/// Built by [`GatedRmsNormConfig`].
#[derive(Module, Debug)]
pub struct GatedRmsNorm {
    /// The gain, `[d_model]`.
    pub weight: Param<Tensor<1>>,

    /// Added to the mean of squares.
    pub epsilon: f64,
}

impl GatedRmsNormMeta for GatedRmsNorm {
    fn d_model(&self) -> usize {
        self.weight.shape()[0]
    }
}

impl GatedRmsNorm {
    /// `w · x̂ · silu(gate)`, over the last axis; in `x`'s dtype.
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
        let shape = x.dims();
        assert_eq!(
            gate.dims(),
            shape,
            "a gated norm's gate has its input's shape"
        );
        assert_eq!(
            shape[R - 1],
            self.d_model(),
            "a gated norm normalizes {} features",
            self.d_model()
        );
        let dtype = x.dtype();
        let x_hat = RmsNormOptions::default()
            .with_eps(self.epsilon as f32)
            .with_widen_input(Some(FloatDType::F32))
            .with_widen_sum(None)
            .norm(x);
        let y = x_hat * self.weight.val().cast(dtype).unsqueeze();
        let compute = widen_half(dtype);
        (y.cast(compute) * silu(gate.cast(compute))).cast(dtype)
    }
}

impl InputDTypes for GatedRmsNorm {
    /// Its gain's dtype.
    fn preferred_dtype(&self) -> FloatDType {
        FloatDType::from(self.weight.val().dtype())
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

impl ResultDType for GatedRmsNorm {
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
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    /// `w · x / sqrt(mean(x²) + eps) · silu(z)`, per row, as written out.
    #[test]
    fn test_gated_rms_norm() {
        let device = cpu_device();
        let cfg = GatedRmsNormConfig::new(6).with_epsilon(1e-5);
        assert_eq!(cfg.d_model(), 6);
        let mut norm = cfg.init(&device);
        assert_eq!(norm.d_model(), 6);
        norm.weight = Param::from_tensor(seeded_tensor::<1>(
            1,
            [6],
            Distribution::Uniform(0.5, 1.5),
            &device,
        ));

        let x = seeded_tensor::<4>(2, [2, 3, 4, 6], Distribution::Default, &device);
        let z = seeded_tensor::<4>(3, [2, 3, 4, 6], Distribution::Default, &device);
        let rms = x.clone().square().mean_dim(3).add_scalar(1e-5).sqrt();
        let expected = x.clone() / rms
            * norm.weight.val().unsqueeze()
            * (z.clone() * burn::tensor::activation::sigmoid(z.clone()));
        let y = norm.forward(x, z);
        assert_eq!(y.dims(), [2, 3, 4, 6]);
        assert_tensors_close(&y, &expected, Tolerance::default());
    }

    /// The default gain is ones (the norm, gated), and a zero gate closes
    /// the output.
    #[test]
    fn test_default_gain_and_closed_gate() {
        let device = cpu_device();
        let norm = GatedRmsNormConfig::new(4).init(&device);
        assert_eq!(norm.epsilon, 1e-6);
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

    /// A half input computes in f32 and returns its own dtype, an f32 gain
    /// and gate meeting it in one dtype; a bf16 gain under an f32 input
    /// returns f32.
    #[test]
    fn test_half_input() {
        use burn::tensor::DType;

        use crate::burner::module::DTypeMapper;

        let device = cpu_device();
        let norm = GatedRmsNormConfig::new(4).init(&device);
        let x = seeded_tensor::<2>(6, [3, 4], Distribution::Default, &device);
        let z = seeded_tensor::<2>(7, [3, 4], Distribution::Default, &device);
        let reference = norm.forward(x.clone(), z.clone());
        let half = norm.forward(x.clone().cast(DType::BF16), z.clone());
        assert_eq!(half.dtype(), DType::BF16);
        half.cast(DType::F32)
            .into_data()
            .assert_approx_eq::<f32>(&reference.into_data(), Tolerance::absolute(2e-2));

        let half_gain = norm.map(&mut DTypeMapper::new(DType::BF16));
        assert_eq!(half_gain.preferred_dtype(), FloatDType::BF16);
        assert_eq!(half_gain.forward(x, z).dtype(), DType::F32);
        assert_eq!(half_gain.result_dtype(FloatDType::F32), FloatDType::F32);
    }

    #[test]
    #[should_panic(expected = "gate has its input's shape")]
    fn test_gate_shape() {
        let device = cpu_device();
        let norm = GatedRmsNormConfig::new(4).init(&device);
        let _ = norm.forward(
            Tensor::<2>::zeros([2, 4], &device),
            Tensor::zeros([3, 4], &device),
        );
    }
}
