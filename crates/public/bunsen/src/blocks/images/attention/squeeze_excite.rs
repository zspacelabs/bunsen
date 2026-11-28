//! [`SqueezeExcite`]: squeeze-and-excitation channel attention.

use burn::{
    config::Config,
    module::Module,
    nn::conv::{
        Conv2d,
        Conv2dConfig,
    },
    prelude::Tensor,
    tensor::{
        Device,
        activation::sigmoid,
    },
};

use crate::{
    blocks::activation::{
        BunsenActivation,
        BunsenActivationConfig,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    support::math::make_divisible,
};

/// The view shared by [`SqueezeExciteConfig`] and [`SqueezeExcite`].
pub trait SqueezeExciteMeta {
    /// The channels it gates.
    fn channels(&self) -> usize;

    /// The channels of its reduction.
    fn rd_channels(&self) -> usize;
}

/// The config of a [`SqueezeExcite`].
#[derive(Config, Debug)]
pub struct SqueezeExciteConfig {
    /// The channels it gates.
    pub channels: usize,

    /// The channels of its reduction.
    pub rd_channels: usize,

    /// The activation between the two convs.
    #[config(default = "BunsenActivationConfig::Relu")]
    pub act: BunsenActivationConfig,
}

impl SqueezeExciteConfig {
    /// timm's `SEModule` of `channels`: a reduction of `channels * rd_ratio`
    /// rounded to 8s ([`make_divisible`], no round limit), and a `ReLU`.
    /// timm's `'se'` attention is `rd_ratio` 1/16; `ResNet-RS`'s is 1/4.
    pub fn timm(
        channels: usize,
        rd_ratio: f64,
    ) -> Self {
        Self::new(
            channels,
            make_divisible(channels as f64 * rd_ratio, 8, None, 0.0),
        )
    }
}

impl SqueezeExciteMeta for SqueezeExciteConfig {
    fn channels(&self) -> usize {
        self.channels
    }

    fn rd_channels(&self) -> usize {
        self.rd_channels
    }
}

impl ModuleInit<SqueezeExcite> for SqueezeExciteConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SqueezeExcite> {
        Ok(SqueezeExcite {
            fc1: Conv2dConfig::new([self.channels, self.rd_channels], [1, 1]).init(device),
            act: self.act.try_init(device)?,
            fc2: Conv2dConfig::new([self.rd_channels, self.channels], [1, 1]).init(device),
        })
    }
}

/// Squeeze-and-excitation: `x * sigmoid(fc2(act(fc1(mean(x)))))`, the mean
/// over the spatial axes, `fc1` and `fc2` `1x1` convs with biases.
#[derive(Module, Debug)]
pub struct SqueezeExcite {
    /// The reduction, `channels -> rd_channels`.
    pub fc1: Conv2d,

    /// The activation.
    pub act: BunsenActivation,

    /// The expansion, `rd_channels -> channels`.
    pub fc2: Conv2d,
}

impl SqueezeExciteMeta for SqueezeExcite {
    fn channels(&self) -> usize {
        self.fc1.weight.dims()[1]
    }

    fn rd_channels(&self) -> usize {
        self.fc1.weight.dims()[0]
    }
}

impl SqueezeExcite {
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
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "channels", "height", "width"],
            &input.dims(),
            &[("channels", self.channels())],
        );
        let squeezed = input.clone().mean_dim(2).mean_dim(3);
        let gate = self
            .fc2
            .forward(self.act.forward(self.fc1.forward(squeezed)));
        input * sigmoid(gate)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::support::testing::cpu_device;

    /// timm's reductions: 1/16 of the channels in 8s, at least 8.
    #[test]
    fn test_timm_sizes() {
        assert_eq!(SqueezeExciteConfig::timm(256, 1.0 / 16.0).rd_channels, 16);
        assert_eq!(SqueezeExciteConfig::timm(64, 1.0 / 16.0).rd_channels, 8);
        assert_eq!(SqueezeExciteConfig::timm(2048, 0.25).rd_channels, 512);
    }

    /// The gate is the sigmoid of the convs over the spatial means: with zero
    /// weights and biases it is one half everywhere.
    #[test]
    fn test_forward_gates_by_the_means() {
        let device = cpu_device();
        let mut se: SqueezeExcite = SqueezeExciteConfig::timm(16, 1.0 / 16.0).init(&device);
        assert_eq!((se.channels(), se.rd_channels()), (16, 8));
        se.fc2.weight = se.fc2.weight.map(|w| w.zeros_like());
        se.fc2.bias = se.fc2.bias.map(|b| b.map(|b| b.zeros_like()));
        let x = Tensor::<4>::random([2, 16, 3, 5], burn::tensor::Distribution::Default, &device);
        se.forward(x.clone())
            .into_data()
            .assert_approx_eq::<f32>(&(x * 0.5).into_data(), Tolerance::absolute(1e-6));
    }
}
