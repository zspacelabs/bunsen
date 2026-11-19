//! [`SnakeActivation`]: Snake and `SnakeBeta`, periodic activations per channel.

use burn::{
    module::Param,
    prelude::*,
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`SnakeActivation`] and [`SnakeActivationConfig`].
pub trait SnakeActivationMeta {
    /// The channels: one frequency (and magnitude) each, on axis 1.
    fn channels(&self) -> usize;

    /// Whether the magnitude is a parameter of its own, `β` (`SnakeBeta`), or
    /// the frequency `α` (Snake).
    fn separate_magnitude(&self) -> bool;

    /// Whether the parameters are stored as logs, the forward reading their
    /// `exp`.
    fn log_scale(&self) -> bool;

    /// Added to the magnitude before it divides.
    fn epsilon(&self) -> f64;
}

/// Config for [`SnakeActivation`].
#[derive(Config, Debug)]
pub struct SnakeActivationConfig {
    /// The channels: one frequency (and magnitude) each, on axis 1.
    pub channels: usize,

    /// Whether the magnitude is a parameter of its own, `β` (`SnakeBeta`), or
    /// the frequency `α` (Snake, the default).
    #[config(default = false)]
    pub separate_magnitude: bool,

    /// Whether the parameters are stored as logs, the forward reading their
    /// `exp` (`BigVGAN`'s `alpha_logscale`). Off by default.
    #[config(default = false)]
    pub log_scale: bool,

    /// Added to the magnitude before it divides.
    #[config(default = 1e-9)]
    pub epsilon: f64,
}

impl SnakeActivationMeta for SnakeActivationConfig {
    fn channels(&self) -> usize {
        self.channels
    }

    fn separate_magnitude(&self) -> bool {
        self.separate_magnitude
    }

    fn log_scale(&self) -> bool {
        self.log_scale
    }

    fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl SnakeActivationConfig {
    /// `SnakeBeta` over `channels`, its parameters stored as logs: the
    /// activation of `BigVGAN`'s, DAC-style vocoders' and Qwen3-TTS's codec.
    pub fn snake_beta_log_scale(channels: usize) -> Self {
        Self::new(channels)
            .with_separate_magnitude(true)
            .with_log_scale(true)
    }
}

/// The activation: every parameter at its identity scale (1, or a log of 0).
impl ModuleInit<SnakeActivation> for SnakeActivationConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SnakeActivation> {
        let start = || {
            let value = if self.log_scale { 0.0 } else { 1.0 };
            Param::from_tensor(Tensor::full([self.channels], value, device))
        };
        Ok(SnakeActivation {
            alpha: start(),
            beta: self.separate_magnitude.then(start),
            log_scale: self.log_scale,
            epsilon: self.epsilon,
        })
    }
}

/// A periodic activation, per channel (Liu Ziyin et al., 2020, "Neural
/// Networks Fail to Learn Periodic Functions and How to Fix It"):
///
/// - Snake: `x + 1/(α + ε) · sin²(α x)`;
/// - `SnakeBeta`: `x + 1/(β + ε) · sin²(α x)`, the magnitude apart from the
///   frequency.
///
/// `α` and `β` are per channel, on axis 1 (`[B, C, ...]`); stored as logs
/// ([`log_scale`](SnakeActivationMeta::log_scale)), the forward takes their
/// `exp` each call. The parameters are named `alpha` and `beta`, as `BigVGAN`'s
/// checkpoints name them.
///
/// Built by [`SnakeActivationConfig`].
#[derive(Module, Debug)]
pub struct SnakeActivation {
    /// The frequency `α`, `[C]`; its log under `log_scale`.
    pub alpha: Param<Tensor<1>>,

    /// The magnitude `β`, `[C]`, when it is apart from `α`; its log under
    /// `log_scale`.
    pub beta: Option<Param<Tensor<1>>>,

    /// Whether the parameters are logs.
    pub log_scale: bool,

    /// Added to the magnitude before it divides.
    pub epsilon: f64,
}

impl SnakeActivationMeta for SnakeActivation {
    fn channels(&self) -> usize {
        self.alpha.dims()[0]
    }

    fn separate_magnitude(&self) -> bool {
        self.beta.is_some()
    }

    fn log_scale(&self) -> bool {
        self.log_scale
    }

    fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl SnakeActivation {
    /// A parameter `[C]` as a factor over axis 1 of a rank-`D` input: its
    /// `exp` under `log_scale`.
    fn factor<const D: usize>(
        &self,
        param: &Param<Tensor<1>>,
    ) -> Tensor<D> {
        let value = param.val();
        let value = if self.log_scale { value.exp() } else { value };
        let mut shape = [1; D];
        shape[1] = self.channels();
        value.reshape(shape)
    }

    /// `input`, activated: `[B, C, ...]` to the same; in `input`'s dtype.
    ///
    /// # Panics
    /// When `input` is of rank below 2, or its axis 1 is not `C`.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        assert!(D >= 2, "a Snake activation reads channels on axis 1");
        assert_eq!(
            input.dims()[1],
            self.channels(),
            "a Snake activation over {} channels",
            self.channels()
        );
        let alpha = self.factor::<D>(&self.alpha);
        let magnitude = match &self.beta {
            Some(beta) => self.factor::<D>(beta),
            None => alpha.clone(),
        };
        let wave = (input.clone() * alpha).sin().powi_scalar(2);
        input + wave * (magnitude + self.epsilon).recip()
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        TensorData,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    fn x(device: &Device) -> Tensor<3> {
        Tensor::from_floats([[[-1.5, 0.25, 2.0], [0.5, -0.75, 3.0]]], device)
    }

    /// The config and the module built from it answer the meta alike; a new
    /// activation is the identity scale.
    #[test]
    fn test_meta_and_init() {
        let device = cpu_device();
        for config in [
            SnakeActivationConfig::new(3),
            SnakeActivationConfig::snake_beta_log_scale(3),
        ] {
            let snake = config.init(&device);
            assert_eq!(snake.channels(), config.channels());
            assert_eq!(snake.separate_magnitude(), config.separate_magnitude());
            assert_eq!(snake.log_scale(), config.log_scale());
            assert_eq!(snake.epsilon(), config.epsilon());
            let start = if config.log_scale { 0.0 } else { 1.0 };
            snake
                .alpha
                .val()
                .into_data()
                .assert_eq(&TensorData::from([start; 3]), false);
        }
        assert!(SnakeActivationConfig::new(3).init(&device).beta.is_none());
    }

    /// Snake and log-scale `SnakeBeta` against values computed in float64.
    #[test]
    fn test_forward() {
        let device = cpu_device();
        let tolerance = Tolerance::rel_abs(1e-5, 1e-6);

        let mut snake = SnakeActivationConfig::new(2).init(&device);
        snake.alpha = Param::from_tensor(Tensor::from_floats([0.8, 1.7], &device));
        snake
            .forward(x(&device))
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[
                    [-0.41412893, 0.29933688, 3.24893423],
                    [0.83201309, -0.21174896, 3.50419578],
                ]]),
                tolerance,
            );

        let mut beta = SnakeActivationConfig::snake_beta_log_scale(2).init(&device);
        beta.alpha = Param::from_tensor(Tensor::from_floats([0.3, -0.2], &device));
        beta.beta = Some(Param::from_tensor(Tensor::from_floats(
            [-0.5, 0.4],
            &device,
        )));
        beta.forward(x(&device))
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[
                    [-0.16838032, 0.43074019, 2.30150467],
                    [0.60619590, -0.52746430, 3.26857547],
                ]]),
                tolerance,
            );
    }

    /// Axis 1 must be the channels.
    #[test]
    #[should_panic(expected = "over 3 channels")]
    fn test_channels_checked() {
        let device = cpu_device();
        SnakeActivationConfig::new(3)
            .init(&device)
            .forward(x(&device));
    }

    /// The Qwen3-TTS toy codec's last `SnakeBeta` (`decoder.decoder.3`, by
    /// `tools/gen_qwen3_tts_toy_fixtures.py`), on its own input.
    #[cfg(feature = "store")]
    #[test]
    fn test_qwen3_tts_toy_snake_beta() {
        use crate::support::testing::{
            ToyFixture,
            assert_io_close,
        };

        let device = cpu_device();
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_tts/toy.safetensors");
        let fixture = ToyFixture::load_io(path, &device);
        let alpha = fixture.io::<1>("decoder.decoder.3.alpha");
        let mut snake = SnakeActivationConfig::snake_beta_log_scale(alpha.dims()[0]).init(&device);
        snake.alpha = Param::from_tensor(alpha);
        snake.beta = Some(Param::from_tensor(
            fixture.io::<1>("decoder.decoder.3.beta"),
        ));
        assert_io_close(
            "decoder.decoder.3",
            snake.forward(fixture.io::<3>("io.codec.decoder_2")),
            fixture.io::<3>("io.codec.decoder_3"),
        );
    }
}
