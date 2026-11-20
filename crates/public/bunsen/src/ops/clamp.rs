//! # Clamping as a value
//!
//! [`ClampOptions`] is an optional minimum and an optional maximum, held as one
//! value object, with [`ClampOptions::clamp`] to apply it. It is for clamping
//! as a *setting* (a config field, a noise generator's range) rather than a
//! one-shot call, where burn's `Tensor::clamp` is enough.

use burn::{
    module::{
        Content,
        ModuleDisplay,
        ModuleDisplayDefault,
    },
    prelude::Tensor,
};
use serde::{
    Deserialize,
    Serialize,
};

/// Clamping operation: an optional min and an optional max.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClampOptions {
    /// The minimum value.
    min: Option<f64>,

    /// The maximum value.
    max: Option<f64>,
}

impl ClampOptions {}

impl ModuleDisplay for ClampOptions {}
impl ModuleDisplayDefault for ClampOptions {
    fn content(
        &self,
        content: Content,
    ) -> Option<Content> {
        Some(content.add("min", &self.min).add("max", &self.max))
    }
}

impl ClampOptions {
    /// Creates a new `ClampOptions`.
    pub fn new<A, B>(
        min: A,
        max: B,
    ) -> Self
    where
        A: Into<Option<f64>>,
        B: Into<Option<f64>>,
    {
        Self {
            min: min.into(),
            max: max.into(),
        }
    }

    /// Returns the minimum value.
    pub fn min(&self) -> Option<f64> {
        self.min
    }

    /// Returns the maximum value.
    pub fn max(&self) -> Option<f64> {
        self.max
    }

    /// Extends the clamp with a minimum value.
    pub fn with_min(
        self,
        min: f64,
    ) -> Self {
        Self {
            min: Some(min),
            ..self
        }
    }

    /// Extends the clamp with a maximum value.
    pub fn with_max(
        self,
        max: f64,
    ) -> Self {
        Self {
            max: Some(max),
            ..self
        }
    }

    /// Applies the clamp.
    pub fn clamp<const D: usize>(
        &self,
        tensor: Tensor<D>,
    ) -> Tensor<D> {
        match (self.min, self.max) {
            (Some(min), Some(max)) => tensor.clamp(min, max),
            (Some(min), None) => tensor.clamp_min(min),
            (None, Some(max)) => tensor.clamp_max(max),
            (None, None) => tensor,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::{
            DisplaySettings,
            ModuleDisplay,
        },
        tensor::TensorData,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_clamp_config_display() {
        let config = ClampOptions::default().with_min(0.5);
        let settings = DisplaySettings::default();

        assert_eq!(
            config.format(settings),
            indoc::indoc! {
                r#"
                ClampOptions {
                  min: 0.5
                  max: None
                }"#
            }
        )
    }

    #[test]
    fn test_config() {
        let device = cpu_device();

        let cfg = ClampOptions::default();
        assert_eq!(
            cfg,
            ClampOptions {
                min: None,
                max: None,
            }
        );
        let tensor = Tensor::<1>::from_data([-1.0, 0.0, 1.0], &device);
        let tensor = cfg.clamp(tensor);
        tensor
            .to_data()
            .assert_eq(&TensorData::from([-1.0, 0.0, 1.0]), false);

        let cfg = ClampOptions::default().with_min(-0.5).with_max(0.5);
        assert_eq!(
            cfg,
            ClampOptions {
                min: Some(-0.5),
                max: Some(0.5),
            }
        );
        let tensor = Tensor::<1>::from_data([-1.0, 0.0, 1.0], &device);
        let tensor = cfg.clamp(tensor);
        tensor
            .to_data()
            .assert_eq(&TensorData::from([-0.5, 0.0, 0.5]), false);
    }
}
