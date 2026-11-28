//! The conv backbone before a vision encoder's patch embedding.

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::images::models::conv_net::{
        ConvNetMeta,
        ConvNetModel,
        ConvNetStructureConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
    },
};

/// A vision encoder's conv backbone: the config of a [`VisionBackbone`]
/// (timm's `HybridEmbed`'s `backbone`).
///
/// The images go through the conv net first; the patch embedding cuts its
/// last map, not the images.
///
/// ## Discussion
///
/// The hybrid `ViT`'s backbone is a `ResNetV2`.
#[derive(Config, Debug)]
pub struct VisionBackboneConfig {
    /// The conv net, headless: its last map
    /// ([`feature_map`](ConvNetModel::feature_map)) is the patch embedding's
    /// input.
    pub net: ConvNetStructureConfig,

    /// The `[height, width]` of the images it takes.
    pub image_size: [usize; 2],
}

impl VisionBackboneConfig {
    /// The `[height, width]` of the conv net's last map over the images.
    ///
    /// # Errors
    ///
    /// As [`ConvNetStructureConfig::try_feature_size`].
    pub fn try_feature_size(&self) -> BunsenResult<[usize; 2]> {
        self.net.try_feature_size(self.image_size)
    }

    /// Checks the backbone: a valid, headless conv net.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a head of
    /// classes; as [`ConvNetStructureConfig::try_validate`].
    pub fn try_validate(&self) -> BunsenResult<()> {
        self.net.try_validate().context("the backbone")?;
        if self.net.num_classes() != 0 {
            return Err(ConstraintError::custom(
                "VisionBackboneConfig",
                "net.num_classes",
                format!(
                    "the backbone has a head of {} classes; its map is the patch embedding's \
                     input, so it has none",
                    self.net.num_classes()
                ),
            )
            .into());
        }
        Ok(())
    }
}

impl ModuleInit<VisionBackbone> for VisionBackboneConfig {
    /// Builds the conv net.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](VisionBackboneConfig::try_validate), and as the
    /// net's `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<VisionBackbone> {
        self.try_validate()?;
        Ok(VisionBackbone {
            net: self.net.try_init(device).context("the backbone")?,
            image_size: self.image_size,
        })
    }
}

/// A vision encoder's conv backbone; built by [`VisionBackboneConfig`].
#[derive(Module, Debug)]
pub struct VisionBackbone {
    /// The conv net, headless.
    pub net: ConvNetModel,

    /// The `[height, width]` of the images it takes.
    #[module(skip)]
    pub image_size: [usize; 2],
}

impl VisionBackbone {
    /// The conv net's last map.
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`, at the backbone's
    ///   image size.
    ///
    /// # Returns
    ///
    /// `[batch, d_features, map_height, map_width]`.
    pub fn forward(
        &self,
        images: Tensor<4>,
    ) -> Tensor<4> {
        self.net.feature_map(images)
    }
}
