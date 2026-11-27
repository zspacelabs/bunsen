//! # Noise as a value
//!
//! [`NoiseOptions`] is a [`Distribution`] plus an optional [`ClampOptions`],
//! held as one value object, with [`NoiseOptions::noise`] and
//! [`NoiseOptions::noise_like`] to sample it. `DropBlockOptions` carries one
//! to refill dropped regions.

use burn::{
    module::{
        Content,
        ModuleDisplay,
        ModuleDisplayDefault,
    },
    prelude::{
        Shape,
        Tensor,
    },
    tensor::{
        Device,
        Distribution,
    },
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    burner::distribution::DistributionDisplayAdapter,
    ops::clamp::ClampOptions,
};

/// Noise Configuration.
///
/// Carries a [`Distribution`] and an optional [`ClampOptions`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseOptions {
    /// The noise distribution.
    pub distribution: Distribution,

    /// The noise clip range.
    pub clamp: Option<ClampOptions>,
}

impl ModuleDisplay for NoiseOptions {}
impl ModuleDisplayDefault for NoiseOptions {
    fn content(
        &self,
        content: Content,
    ) -> Option<Content> {
        Some(
            content
                .add(
                    "distribution",
                    &DistributionDisplayAdapter::new(self.distribution),
                )
                .add("clamp", &self.clamp),
        )
    }
}

impl Default for NoiseOptions {
    fn default() -> Self {
        Self {
            distribution: Distribution::Default,
            clamp: None,
        }
    }
}

impl NoiseOptions {
    /// Extends the config with the given [`Distribution`].
    pub fn with_distribution(
        self,
        distribution: Distribution,
    ) -> Self {
        Self {
            distribution,
            ..self
        }
    }

    /// Extends the config with the given [`ClampOptions`].
    pub fn with_clamp<C>(
        self,
        clamp: C,
    ) -> Self
    where
        C: Into<Option<ClampOptions>>,
    {
        Self {
            clamp: clamp.into(),
            ..self
        }
    }

    /// Generates noise.
    ///
    /// Noise is drawn from the distribution; and optionally clamped.
    ///
    /// # Arguments
    ///
    /// - `shape` - the shape of the noise tensor to generate.
    /// - `device` - the device to build the tensor on.
    ///
    /// # Returns
    ///
    /// A new tensor with the given shape and device, filled with noise.
    pub fn noise<S, const D: usize>(
        &self,
        shape: S,
        device: &Device,
    ) -> Tensor<D>
    where
        S: Into<Shape>,
    {
        let noise = Tensor::random(shape.into(), self.distribution, device);
        match &self.clamp {
            None => noise,
            Some(clamp_cfg) => clamp_cfg.clamp(noise),
        }
    }

    /// Generates noise like a reference tensor.
    ///
    /// # Arguments
    ///
    /// - `tensor`: A reference tensor to match the shape and device.
    ///
    /// # Returns
    ///
    /// A new tensor with the same shape and device as the reference.
    pub fn noise_like<const D: usize>(
        &self,
        tensor: &Tensor<D>,
    ) -> Tensor<D> {
        self.noise(tensor.shape(), &tensor.device())
    }
}

#[cfg(test)]
mod tests {
    use burn::module::DisplaySettings;

    use super::*;
    use crate::support::testing::{
        assert_scalar_close,
        cpu_device,
    };

    #[test]
    fn test_noise_config_display() {
        let config = NoiseOptions::default().with_clamp(ClampOptions::new(0.5, 1.0));
        let settings = DisplaySettings::default();

        assert_eq!(
            config.format(settings),
            indoc::indoc! {r#"
                NoiseOptions {
                  distribution: Distribution::Default
                  clamp: ClampOptions {
                      min: 0.5
                      max: 1
                    }
                }"#
            }
        )
    }

    #[test]
    fn test_noise_default() {
        let cfg = NoiseOptions::default();
        assert_eq!(
            cfg,
            NoiseOptions {
                distribution: Distribution::Default,
                clamp: None
            }
        );

        let cfg = NoiseOptions::default()
            .with_distribution(Distribution::Bernoulli(0.3))
            .with_clamp(ClampOptions::default());
        assert_eq!(
            cfg,
            NoiseOptions {
                distribution: Distribution::Bernoulli(0.3),
                clamp: Some(ClampOptions::default())
            }
        );

        let cfg = NoiseOptions::default().with_clamp(Some(ClampOptions::default()));
        assert_eq!(
            cfg,
            NoiseOptions {
                distribution: Distribution::Default,
                clamp: Some(ClampOptions::default())
            }
        );

        let cfg = NoiseOptions::default()
            .with_clamp(ClampOptions::default())
            .with_clamp(None);
        assert_eq!(
            cfg,
            NoiseOptions {
                distribution: Distribution::Default,
                clamp: None,
            }
        );
    }

    #[test]
    fn test_noise_like_default_clamp() {
        let device = cpu_device();

        let reference: Tensor<2> = Tensor::ones([20, 20], &device);
        let numel = reference.shape().num_elements() as f64;

        let noise = NoiseOptions::default()
            .with_clamp(ClampOptions::default().with_min(0.5))
            .noise_like(&reference);

        assert_eq!(noise.shape(), reference.shape());
        assert_eq!(noise.device(), reference.device());

        // * Half of values should be exactly 0.5
        // * All values should be in [0.5, 1.0)

        // count 0.5
        let count_05 = noise
            .clone()
            .equal_elem(0.5)
            .int()
            .sum()
            .into_scalar::<f64>();
        assert_scalar_close(count_05 / numel, 0.5, 0.15);

        let count_ge_1 = noise
            .clone()
            .greater_equal_elem(1.0)
            .int()
            .sum()
            .into_scalar::<i64>();
        assert_eq!(count_ge_1, 0);
    }

    #[test]
    fn test_noise_like_bernoulli() {
        let device = cpu_device();

        let reference: Tensor<2> = Tensor::ones([20, 20], &device);

        let p = 0.1;

        let noise = NoiseOptions::default()
            .with_distribution(Distribution::Bernoulli(p))
            .noise_like(&reference);

        assert_eq!(noise.shape(), reference.shape());
        assert_eq!(noise.device(), reference.device());

        let ratio =
            noise.clone().sum().into_scalar::<f64>() / (noise.shape().num_elements() as f64);
        assert_scalar_close(ratio, p, 0.05);
    }
}
