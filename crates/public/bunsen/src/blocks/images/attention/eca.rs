//! [`Eca`]: efficient channel attention.

use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    prelude::Tensor,
    tensor::{
        Device,
        activation::sigmoid,
    },
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The view shared by [`EcaConfig`] and [`Eca`].
pub trait EcaMeta {
    /// The conv's kernel across the channels.
    fn kernel_size(&self) -> usize;
}

/// The config of an [`Eca`].
#[derive(Config, Debug)]
pub struct EcaConfig {
    /// The conv's kernel across the channels: odd.
    pub kernel_size: usize,
}

impl EcaConfig {
    /// timm's `EcaModule` of `channels`: the kernel is
    /// `t = int(|log2(channels) + 1| / 2)`, made odd (`t + 1` when even), at
    /// least 3 (the paper's `gamma` 2 and `beta` 1).
    pub fn timm(channels: usize) -> Self {
        // `math.log(channels, 2)`: `ln(channels) / ln(2)`, as Python computes
        // it, so a power of two truncates as timm's does.
        let t = (((channels as f64).ln() / 2f64.ln() + 1.0).abs() / 2.0) as usize;
        let odd = if t.is_multiple_of(2) { t + 1 } else { t };
        Self::new(odd.max(3))
    }
}

impl EcaMeta for EcaConfig {
    fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

impl ModuleInit<Eca> for EcaConfig {
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for an even
    /// kernel.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Eca> {
        if self.kernel_size.is_multiple_of(2) {
            return Err(BunsenError::illegal(format!(
                "EcaConfig: kernel_size {} is even; the conv keeps the channels with an odd one",
                self.kernel_size
            )));
        }
        let pad = (self.kernel_size - 1) / 2;
        Ok(Eca {
            conv: Conv1dConfig::new(1, 1, self.kernel_size)
                .with_padding(PaddingConfig1d::Explicit(pad, pad))
                .with_bias(false)
                .init(device),
        })
    }
}

/// Efficient channel attention: `x * sigmoid(conv(mean(x)))`, the mean over
/// the spatial axes, `conv` one `1d` conv (no bias) across the channels.
#[derive(Module, Debug)]
pub struct Eca {
    /// The conv across the channels, `1 -> 1`.
    pub conv: Conv1d,
}

impl EcaMeta for Eca {
    fn kernel_size(&self) -> usize {
        self.conv.weight.dims()[2]
    }
}

impl Eca {
    /// The input, its channels gated.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, channels, height, width]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        let [batch, channels, _, _] = input.dims();
        let means: Tensor<3> = input
            .clone()
            .mean_dim(2)
            .mean_dim(3)
            .reshape([batch, 1, channels]);
        let gate: Tensor<4> = sigmoid(self.conv.forward(means)).reshape([batch, channels, 1, 1]);
        input * gate
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::support::testing::cpu_device;

    /// timm's kernels (computed by `timm.layers.EcaModule`).
    #[test]
    fn test_timm_kernels() {
        for (channels, k) in [
            (8, 3),
            (64, 3),
            (128, 5),
            (256, 5),
            (1024, 5),
            (2048, 7),
            (4096, 7),
        ] {
            assert_eq!(EcaConfig::timm(channels).kernel_size, k, "{channels}");
        }
    }

    /// A zero kernel gates by one half; an even kernel is an error.
    #[test]
    fn test_forward_and_kernel_checks() {
        let device = cpu_device();
        let mut eca: Eca = EcaConfig::timm(64).init(&device);
        assert_eq!(eca.kernel_size(), 3);
        eca.conv.weight = eca.conv.weight.map(|w| w.zeros_like());
        let x = Tensor::<4>::random([2, 64, 3, 2], burn::tensor::Distribution::Default, &device);
        eca.forward(x.clone())
            .into_data()
            .assert_approx_eq::<f32>(&(x * 0.5).into_data(), Tolerance::absolute(1e-6));
        let even: BunsenResult<Eca> = EcaConfig::new(4).try_init(&device);
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Illegal)
            .message_contains("is even")
            .assert_err(&even);
    }
}
