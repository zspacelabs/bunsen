//! # timm's `config.json`
//!
//! What a timm hub repo (`timm/<architecture>.<tag>`) says of its model, in
//! its `config.json`: the architecture's registered name, the class count,
//! and the `pretrained_cfg` its weights were published with: the eval
//! transform, among others. A kit maps the architecture onto its own config
//! (`resnet::interop::timm`); the transform is every image kit's.
//!
//! - **Mirror:** [`TimmModelConfigMirror`], with its
//!   [`TimmPretrainedCfgMirror`]; upstream's field names.
//! - **Conversion:** [`TimmPretrainedCfgMirror::eval_transform_config`].
//! - **Reading:** [`timm_model_config_from_path`] and
//!   [`timm_model_config_from_str`].

use std::path::Path;

use serde::Deserialize;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        images::support::transform::{
            ImageCropMode,
            ImageEvalTransformConfig,
            ImageInterpolation,
        },
        support::interop::json_mirror_from_path,
    },
};

/// A timm hub repo's `config.json`, as timm's `push_to_hf_hub` writes it.
///
/// Fields keep upstream's names. Fields the kits do not read are ignored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TimmModelConfigMirror {
    /// The registered architecture: `resnet50`, `vit_base_patch16_224`.
    pub architecture: String,

    /// The head's class count; 0 for a headless model.
    pub num_classes: usize,

    /// The pooled features' width.
    pub num_features: usize,

    /// The head's pooling, for the models that state it (`token`, `avg`).
    #[serde(default)]
    pub global_pool: Option<String>,

    /// The weights' published settings.
    pub pretrained_cfg: TimmPretrainedCfgMirror,
}

/// The `pretrained_cfg` of a timm `config.json`: the settings a weight set
/// was published with.
///
/// Fields keep upstream's names. Fields the kits do not read are ignored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TimmPretrainedCfgMirror {
    /// The weight set's tag: `a1_in1k`.
    #[serde(default)]
    pub tag: Option<String>,

    /// The eval input's `[channels, height, width]`.
    pub input_size: [usize; 3],

    /// A larger input the weights test better at, when there is one.
    #[serde(default)]
    pub test_input_size: Option<[usize; 3]>,

    /// Whether the model takes only `input_size`.
    #[serde(default)]
    pub fixed_input_size: bool,

    /// The resize filter's name: `bicubic`.
    #[serde(default = "default_interpolation")]
    pub interpolation: String,

    /// The fraction of the scale size the crop keeps.
    #[serde(default = "default_crop_pct")]
    pub crop_pct: f64,

    /// The crop at `test_input_size`, when there is one.
    #[serde(default)]
    pub test_crop_pct: Option<f64>,

    /// How the image is fitted: `center`, `squash` or `border`.
    #[serde(default = "default_crop_mode")]
    pub crop_mode: String,

    /// The normalisation mean, per channel.
    pub mean: Vec<f32>,

    /// The normalisation standard deviation, per channel.
    pub std: Vec<f32>,

    /// The head's class count the weights were trained for.
    #[serde(default)]
    pub num_classes: Option<usize>,

    /// The weights' license, when stated: `apache-2.0`.
    #[serde(default)]
    pub license: Option<String>,

    /// Where the weights come from.
    #[serde(default)]
    pub origin_url: Option<String>,
}

/// timm's `resolve_data_config` fallback filter.
fn default_interpolation() -> String {
    "bicubic".to_string()
}

/// timm's `DEFAULT_CROP_PCT`.
fn default_crop_pct() -> f64 {
    0.875
}

/// timm's `DEFAULT_CROP_MODE`.
fn default_crop_mode() -> String {
    "center".to_string()
}

impl TimmPretrainedCfgMirror {
    /// The eval transform the weights were published with, at their
    /// `input_size` and `crop_pct` (timm's default; the test size is not
    /// used).
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// filter or crop mode that is not timm's eval set (`random`, …);
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for settings that do not validate
    /// ([`ImageEvalTransformConfig::try_validate`]).
    pub fn eval_transform_config(&self) -> BunsenResult<ImageEvalTransformConfig> {
        let interpolation =
            ImageInterpolation::from_timm_name(&self.interpolation).ok_or_else(|| {
                BunsenError::unsupported(format!(
                    "timm's {:?} interpolation in an eval transform",
                    self.interpolation
                ))
            })?;
        let crop_mode = match self.crop_mode.as_str() {
            "center" => ImageCropMode::Center,
            "squash" => ImageCropMode::Squash,
            "border" => ImageCropMode::Border,
            other => {
                return Err(BunsenError::unsupported(format!(
                    "timm's {other:?} crop mode"
                )));
            }
        };
        let [channels, height, width] = self.input_size;
        let transform = ImageEvalTransformConfig::new()
            .with_channels(channels)
            .with_size([height, width])
            .with_interpolation(interpolation)
            .with_crop_pct(self.crop_pct)
            .with_crop_mode(crop_mode)
            .with_mean(self.mean.clone())
            .with_std(self.std.clone());
        transform
            .try_validate()
            .map_err(|e| BunsenError::invalid_resource(format!("timm's pretrained_cfg: {e}")))?;
        Ok(transform)
    }
}

/// A timm `config.json`'s text as its mirror.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// text that is not a timm `config.json`.
pub fn timm_model_config_from_str(text: &str) -> BunsenResult<TimmModelConfigMirror> {
    serde_json::from_str(text)
        .map_err(|e| BunsenError::invalid_resource(format!("a timm config.json: {e}")))
}

/// A timm `config.json` file as its mirror.
///
/// # Errors
/// The `io::Error` of a file that cannot be read; an
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// one that is not a timm `config.json`.
pub fn timm_model_config_from_path(path: &Path) -> BunsenResult<TimmModelConfigMirror> {
    json_mirror_from_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    /// `timm/resnet50.a1_in1k`'s `config.json`, as the hub serves it.
    const RESNET50_A1: &str = r#"{
      "architecture": "resnet50",
      "num_classes": 1000,
      "num_features": 2048,
      "pretrained_cfg": {
        "tag": "a1_in1k",
        "custom_load": false,
        "input_size": [3, 224, 224],
        "test_input_size": [3, 288, 288],
        "fixed_input_size": false,
        "interpolation": "bicubic",
        "crop_pct": 0.95,
        "test_crop_pct": 1.0,
        "crop_mode": "center",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 1000,
        "pool_size": [7, 7],
        "first_conv": "conv1",
        "classifier": "fc",
        "origin_url": "https://github.com/huggingface/pytorch-image-models",
        "paper_ids": "arXiv:2110.00476"
      }
    }"#;

    #[test]
    fn test_a_timm_config_reads_and_converts() {
        let config = timm_model_config_from_str(RESNET50_A1).unwrap();
        assert_eq!(config.architecture, "resnet50");
        assert_eq!(config.num_classes, 1000);
        assert_eq!(config.pretrained_cfg.tag.as_deref(), Some("a1_in1k"));
        assert_eq!(config.pretrained_cfg.test_input_size, Some([3, 288, 288]));
        assert_eq!(
            config.pretrained_cfg.eval_transform_config().unwrap(),
            ImageEvalTransformConfig::new()
                .with_crop_pct(0.95)
                .with_interpolation(ImageInterpolation::Bicubic)
        );
    }

    #[test]
    fn test_eval_transform_config_refuses_a_training_filter() {
        let mut config = timm_model_config_from_str(RESNET50_A1).unwrap();
        config.pretrained_cfg.interpolation = "random".into();
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("\"random\" interpolation")
            .assert_err(&config.pretrained_cfg.eval_transform_config());
    }

    #[test]
    fn test_a_non_timm_file_is_an_invalid_resource() {
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("timm config.json")
            .assert_err(&timm_model_config_from_str("{\"model_type\": \"bert\"}"));
    }
}
