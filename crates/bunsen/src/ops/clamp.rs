//! # Tensor Clamping Support

use burn::{
    module::{
        Content,
        ModuleDisplay,
        ModuleDisplayDefault,
    },
    prelude::{
        Backend,
        Tensor,
    },
};
use serde::{
    Deserialize,
    Serialize,
};

/// Claming operation.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClampOp {
    /// The minimum value.
    min: Option<f64>,

    /// The maximum value.
    max: Option<f64>,
}

impl ClampOp {}

impl ModuleDisplay for ClampOp {}
impl ModuleDisplayDefault for ClampOp {
    fn content(
        &self,
        content: Content,
    ) -> Option<Content> {
        Some(content.add("min", &self.min).add("max", &self.max))
    }
}

impl ClampOp {
    /// Create a new `ClampConfig`..
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

    /// Get the minimum value.
    pub fn min(&self) -> Option<f64> {
        self.min
    }

    /// Get the maximum value.
    pub fn max(&self) -> Option<f64> {
        self.max
    }

    /// Extend the clamp with a minimum value.
    pub fn with_min(
        self,
        min: f64,
    ) -> Self {
        Self {
            min: Some(min),
            ..self
        }
    }

    /// Extend the clamp with a maximum value.
    pub fn with_max(
        self,
        max: f64,
    ) -> Self {
        Self {
            max: Some(max),
            ..self
        }
    }

    /// Apply the clamp.
    pub fn clamp<B: Backend, const D: usize>(
        &self,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D> {
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
    use crate::support::testing::SetupTestBackend;

    #[test]
    fn test_clamp_config_display() {
        let config = ClampOp::default().with_min(0.5);
        let settings = DisplaySettings::default();

        assert_eq!(
            config.format(settings),
            indoc::indoc! {
                r#"
                ClampOp {
                  min: 0.5
                  max: None
                }"#
            }
        )
    }

    #[test]
    fn test_config() {
        type B = SetupTestBackend;
        let device = Default::default();

        let cfg = ClampOp::default();
        assert_eq!(
            cfg,
            ClampOp {
                min: None,
                max: None,
            }
        );
        let tensor = Tensor::<B, 1>::from_data([-1.0, 0.0, 1.0], &device);
        let tensor = cfg.clamp(tensor);
        tensor
            .to_data()
            .assert_eq(&TensorData::from([-1.0, 0.0, 1.0]), false);

        let cfg = ClampOp::default().with_min(-0.5).with_max(0.5);
        assert_eq!(
            cfg,
            ClampOp {
                min: Some(-0.5),
                max: Some(0.5),
            }
        );
        let tensor = Tensor::<B, 1>::from_data([-1.0, 0.0, 1.0], &device);
        let tensor = cfg.clamp(tensor);
        tensor
            .to_data()
            .assert_eq(&TensorData::from([-0.5, 0.0, 0.5]), false);
    }
}
