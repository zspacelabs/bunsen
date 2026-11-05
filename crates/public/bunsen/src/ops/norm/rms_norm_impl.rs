use burn::{
    Tensor,
    tensor::FloatDType,
};

use crate::burner::tensor::widen_to;

/// Options for root-mean-square norm: `x / sqrt(mean(x²) + eps)` over the
/// last axis, with no trainable parameters, the result in the input's dtype.
///
/// Two dtypes are the caller's: the one the input is divided in
/// ([`widen_input`](Self::widen_input)), and the one its mean of squares is
/// taken in ([`widen_sum`](Self::widen_sum)). Each widens to at least the
/// dtype it names, and never narrows
/// ([`widen_to`](crate::burner::tensor::widen_to)):
///
/// | `widen_input` | `widen_sum` | |
/// |---|---|---|
/// | `None` | `Some(F32)` | the mean in f32, the division in the input's dtype (the default; `PyTorch`'s `F.rms_norm` on a half input) |
/// | `Some(F32)` | `None` | a half input widened to f32, normalized there, narrowed once (`transformers`' Llama and Qwen3 norms) |
/// | `None` | `None` | everything in the input's dtype |
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct RmsNormOptions {
    /// Epsilon value for numerical stability.
    pub eps: f32,

    /// The input is divided widened to at least this dtype, and the result
    /// narrowed back; `None`, divided in its own dtype.
    #[serde(default)]
    pub widen_input: Option<FloatDType>,

    /// The mean of squares is taken widened to at least this dtype; `None`,
    /// in the dtype the input is divided in.
    #[serde(default = "default_widen_sum")]
    pub widen_sum: Option<FloatDType>,
}

/// [`RmsNormOptions::widen_sum`]'s default: the mean in f32.
fn default_widen_sum() -> Option<FloatDType> {
    Some(FloatDType::F32)
}

impl Default for RmsNormOptions {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            widen_input: None,
            widen_sum: default_widen_sum(),
        }
    }
}

impl RmsNormOptions {
    /// Sets epsilon value.
    pub fn with_eps(
        mut self,
        eps: f32,
    ) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the dtype the input is divided in
    /// ([`widen_input`](Self::widen_input)).
    pub fn with_widen_input(
        mut self,
        widen_input: Option<FloatDType>,
    ) -> Self {
        self.widen_input = widen_input;
        self
    }

    /// Sets the dtype the mean of squares is taken in
    /// ([`widen_sum`](Self::widen_sum)).
    pub fn with_widen_sum(
        mut self,
        widen_sum: Option<FloatDType>,
    ) -> Self {
        self.widen_sum = widen_sum;
        self
    }

    /// `x`, root-mean-square normalized over its last axis; in `x`'s dtype.
    pub fn norm<const R: usize>(
        &self,
        x: Tensor<R>,
    ) -> Tensor<R> {
        let dtype = x.dtype();
        let x = match self.widen_input {
            Some(d) => x.cast(widen_to(dtype, d)),
            None => x,
        };
        let z = match self.widen_sum {
            Some(d) => x.clone().cast(widen_to(x.dtype(), d)),
            None => x.clone(),
        };
        let rms = z
            .square()
            .mean_dim(R - 1)
            .add_scalar(self.eps)
            .sqrt()
            .cast(x.dtype());
        (x / rms).cast(dtype)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        tensor::Distribution,
    };
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        performance_device,
    };

    #[test]
    #[serial]
    fn test_rms_norm() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let x: Tensor<3> = Tensor::random([2, 3, 4], Distribution::Default, &device);
        let options = RmsNormOptions::default();

        let y = options.norm(x.clone());

        let x_rms = x
            .clone()
            .square()
            .mean_dim(-1)
            .add_scalar(options.eps)
            .sqrt();
        let expected = x / x_rms;

        y.to_data().assert_eq(&expected.to_data(), true);
    }

    /// Each pair of widenings, on a bf16 input, against the same computation
    /// written out: the default (the mean in f32, the division in bf16),
    /// `transformers`' (all in f32, narrowed once), and all in bf16. In f32
    /// all of them agree.
    #[test]
    fn test_widenings() {
        use burn::tensor::DType;

        let device = crate::support::testing::cpu_device();
        let x: Tensor<2> = Tensor::from_data(
            [[0.5f32, -1.25, 2.0, 3.5], [-0.75, 4.0, 1.5, -2.25]],
            &device,
        );
        let eps = 1e-5;
        let rms = |t: Tensor<2>| t.square().mean_dim(1).add_scalar(eps).sqrt();
        let half = x.clone().cast(DType::BF16);
        let cases = [
            (
                RmsNormOptions::default(),
                half.clone() / rms(half.clone().cast(DType::F32)).cast(DType::BF16),
            ),
            (
                RmsNormOptions::default()
                    .with_widen_input(Some(FloatDType::F32))
                    .with_widen_sum(None),
                (half.clone().cast(DType::F32) / rms(half.clone().cast(DType::F32)))
                    .cast(DType::BF16),
            ),
            (
                RmsNormOptions::default().with_widen_sum(None),
                half.clone() / rms(half.clone()),
            ),
        ];
        for (options, expected) in cases {
            let got = options.norm(half.clone());
            assert_eq!(got.dtype(), DType::BF16, "{options:?}");
            got.into_data().assert_eq(&expected.into_data(), true);
            options.norm(x.clone()).into_data().assert_approx_eq::<f32>(
                &RmsNormOptions::default().norm(x.clone()).into_data(),
                burn::tensor::Tolerance::absolute(1e-6),
            );
        }
    }

    /// Options saved before the widenings existed (`{"eps": ...}`) read as
    /// the default.
    #[test]
    fn test_options_serde_defaults() {
        let options: RmsNormOptions = serde_json::from_str(r#"{"eps":1e-6}"#).unwrap();
        assert_eq!(options, RmsNormOptions::default().with_eps(1e-6));
    }
}
