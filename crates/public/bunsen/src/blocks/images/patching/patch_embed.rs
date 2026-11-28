//! # Patch embedding layers.
use burn::{
    config::Config,
    module::Module,
    nn::conv::{
        Conv2d,
        Conv2dConfig,
    },
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::norm::{
        BunsenNormalization,
        BunsenNormalizationConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// Common introspection interface for `PatchEmbed` module.
pub trait PatchEmbedMeta {
    /// Input resolution (height, width).
    fn input_resolution(&self) -> [usize; 2];

    /// Input height.
    fn input_height(&self) -> usize {
        self.input_resolution()[0]
    }

    /// Input width.
    fn input_width(&self) -> usize {
        self.input_resolution()[1]
    }

    /// Input feature dimension size.
    fn d_input(&self) -> usize;

    /// The size of each patch.
    fn patch_size(&self) -> usize;

    /// Image resolution, measured in patches.
    fn patches_resolution(&self) -> [usize; 2] {
        [self.patches_height(), self.patches_width()]
    }

    /// Height of the image, measured in patches.
    fn patches_height(&self) -> usize {
        self.input_height() / self.patch_size()
    }

    /// Width of the image, measured in patches.
    fn patches_width(&self) -> usize {
        self.input_width() / self.patch_size()
    }

    /// Total number of patches.
    fn num_patches(&self) -> usize {
        self.patches_height() * self.patches_width()
    }

    /// Output feature dimension size.
    fn d_output(&self) -> usize;

    /// Whether each patch token is normalized.
    fn enable_patch_norm(&self) -> bool;
}

/// Configuration for [`PatchEmbed`].
///
/// ## Discussion
///
/// - `norm`: a Swin's patch norm is a `LayerNorm`; a `ViT`'s patch embedding
///   has none.
/// - `bias`: CLIP's `ViT` is pre-norm, and its projection has no bias.
/// - `allow_remainder`: a `ViT` built by timm allows a remainder; a Swin
///   refuses one.
#[derive(Config, Debug)]
pub struct PatchEmbedConfig {
    /// Input resolution (height, width).
    input_resolution: [usize; 2],

    /// Patch size.
    patch_size: usize,

    /// Input feature dimension size.
    d_input: usize,

    /// Output feature dimension size.
    d_output: usize,

    /// The norm of each patch token, sized to `d_output` at init; `None` for
    /// none.
    #[config(default = "None")]
    norm: Option<BunsenNormalizationConfig>,

    /// Whether the projection has a bias. timm's `PatchEmbed` has one,
    /// except under a pre-norm vision transformer, which normalises the
    /// tokens before the first block.
    #[config(default = true)]
    bias: bool,

    /// Whether a side may be other than a whole number of patches, the conv
    /// dropping the pixels past the last whole patch (timm's `PatchEmbed`);
    /// `false`, such a geometry is refused.
    #[config(default = false)]
    allow_remainder: bool,
}

impl PatchEmbedMeta for PatchEmbedConfig {
    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn d_input(&self) -> usize {
        self.d_input
    }

    fn d_output(&self) -> usize {
        self.d_output
    }

    fn enable_patch_norm(&self) -> bool {
        self.norm.is_some()
    }
}

impl PatchEmbedConfig {
    /// Checks the geometry: a nonzero patch size that divides both sides of
    /// the input, or fits in both under `allow_remainder`.
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when it doesn't.
    pub fn try_validate(&self) -> BunsenResult<()> {
        let [h, w] = self.input_resolution;
        let p = self.patch_size;
        if p == 0 {
            return Err(ConstraintError::zero_or_empty("PatchEmbedConfig", "patch_size").into());
        }
        if self.allow_remainder && (h < p || w < p) {
            return Err(ConstraintError::custom(
                "PatchEmbedConfig",
                "input_resolution",
                format!("{h}x{w} images don't hold a {p}x{p} patch"),
            )
            .into());
        }
        if !self.allow_remainder && !(h.is_multiple_of(p) && w.is_multiple_of(p)) {
            return Err(ConstraintError::new(
                "PatchEmbedConfig",
                "input_resolution",
                Rule::NotMultiple {
                    value: if h.is_multiple_of(p) { w } else { h },
                    of: format!("patch_size ({p})"),
                },
            )
            .into());
        }
        Ok(())
    }
}

impl ModuleInit<PatchEmbed> for PatchEmbedConfig {
    /// The patch embedding.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](PatchEmbedConfig::try_validate), and as the
    /// norm's `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<PatchEmbed> {
        self.try_validate()?;
        let stride = [self.patch_size, self.patch_size];

        let module = PatchEmbed {
            input_resolution: self.input_resolution,
            patch_size: self.patch_size,
            projection: Conv2dConfig::new([self.d_input, self.d_output], stride)
                .with_stride(stride)
                .with_bias(self.bias)
                .init(device),
            norm: self
                .norm
                .clone()
                .map(|norm| norm.with_num_features(self.d_output).try_init(device))
                .transpose()?,
        };

        Ok(module)
    }
}

/// A patch embedding: a strided `patch_size` conv that cuts an image into
/// square patches and projects each to a token, then flattens the grid to a
/// row-major sequence, and optionally normalises each token: timm's
/// `PatchEmbed`.
///
/// Built by [`PatchEmbedConfig`].
///
/// ## Discussion
///
/// This is a `ViT`'s stem, and a Swin's, with the norm.
#[derive(Module, Debug)]
pub struct PatchEmbed {
    /// Input resolution (height, width).
    pub input_resolution: [usize; 2],

    /// Size of each patch.
    pub patch_size: usize,

    /// Convolutional layer for patch projection.
    pub projection: Conv2d,

    /// Patch normalization layer, if enabled.
    pub norm: Option<BunsenNormalization>,
}

impl PatchEmbedMeta for PatchEmbed {
    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn d_input(&self) -> usize {
        self.projection.weight.dims()[1]
    }

    fn d_output(&self) -> usize {
        self.projection.weight.dims()[0]
    }

    fn enable_patch_norm(&self) -> bool {
        self.norm.is_some()
    }
}

impl PatchEmbed {
    /// Applies the `PatchEmbed` module to an input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - `[B, C, H, W]` input tensor.
    ///
    /// # Returns
    ///
    /// * `[B, H/patch_size * W/patch_size, d_output]` output tensor.
    #[must_use]
    pub fn forward(
        &self,
        x: Tensor<4>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_input", "height", "width"],
            &x.dims(),
            &[
                ("d_input", self.d_input()),
                ("height", self.input_height()),
                ("width", self.input_width()),
            ]
        );

        #[cfg(any(debug_assertions, test))]
        let batch = x.dims()[0];

        let x = self.projection.forward(x);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_output", "patches_height", "patches_width"],
            &x.dims(),
            &[
                ("batch", batch),
                ("d_output", self.d_output()),
                ("patches_height", self.patches_height()),
                ("patches_width", self.patches_width()),
            ],
        );

        let x = x.flatten(2, 3);
        let x = x.swap_dims(1, 2);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "num_patches", "d_output"],
            &x.dims(),
            &[
                ("batch", batch),
                ("num_patches", self.num_patches()),
                ("d_output", self.d_output()),
            ],
        );

        let x = match self.norm {
            None => x,
            Some(ref norm) => norm.forward(x),
        };
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "num_patches", "d_output"],
            &x.dims(),
            &[
                ("batch", batch),
                ("num_patches", self.num_patches()),
                ("d_output", self.d_output()),
            ],
        );

        x
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::{
        blocks::norm::BunsenLayerNormConfig,
        errors::{
            WithOkOrPanic,
            testing::ErrorMatcher,
        },
        support::testing::cpu_device,
    };

    /// A `LayerNorm`, sized at init.
    fn layer_norm() -> BunsenNormalizationConfig {
        BunsenNormalizationConfig::Layer(BunsenLayerNormConfig::new(0))
    }

    #[test]
    fn test_patch_embed_meta() {
        let config = PatchEmbedConfig {
            input_resolution: [224, 224],
            patch_size: 16,
            d_input: 3,
            d_output: 768,
            norm: Some(layer_norm()),
            bias: true,
            allow_remainder: false,
        };

        assert_eq!(config.input_resolution(), [224, 224]);
        assert_eq!(config.patch_size(), 16);
        assert_eq!(config.d_input(), 3);
        assert_eq!(config.d_output(), 768);
        assert!(config.enable_patch_norm());
        assert_eq!(config.patches_resolution(), [14, 14]);
        assert_eq!(config.patches_height(), 14);
        assert_eq!(config.patches_width(), 14);
        assert_eq!(config.num_patches(), 196);
        assert_eq!(config.d_output(), 768);
        assert!(config.enable_patch_norm());

        let device = cpu_device();
        let patch_embed: PatchEmbed = config.try_init(&device).ok_or_panic();

        assert_eq!(patch_embed.input_resolution(), [224, 224]);
        assert_eq!(patch_embed.patch_size(), 16);
        assert_eq!(patch_embed.d_input(), 3);
        assert_eq!(patch_embed.d_output(), 768);
        assert!(patch_embed.enable_patch_norm());
        assert_eq!(patch_embed.patches_resolution(), [14, 14]);
        assert_eq!(patch_embed.patches_height(), 14);
        assert_eq!(patch_embed.patches_width(), 14);
        assert_eq!(patch_embed.num_patches(), 196);
        assert_eq!(patch_embed.d_output(), 768);
        assert!(patch_embed.enable_patch_norm());
    }

    #[test]
    fn test_patch_embed_invalid_resolution() {
        let device = cpu_device();
        for (config, message) in [
            (
                PatchEmbedConfig::new([224, 223], 16, 3, 768),
                "input_resolution: 223 is not a multiple of patch_size (16)",
            ),
            (
                PatchEmbedConfig::new([224, 224], 0, 3, 768),
                "PatchEmbedConfig.patch_size: must not be zero",
            ),
            (
                PatchEmbedConfig::new([224, 12], 16, 3, 768).with_allow_remainder(true),
                "PatchEmbedConfig.input_resolution: 224x12 images don't hold a 16x16 patch",
            ),
        ] {
            ErrorMatcher::illegal()
                .message_contains(message)
                .assert_err(&ModuleInit::<PatchEmbed>::try_init(&config, &device));
        }
    }

    /// Under `allow_remainder`, the pixels past the last whole patch are
    /// dropped: 38 pixels in 16-pixel patches are 2 patches a side.
    #[test]
    fn test_patch_embed_remainder() {
        let device = cpu_device();
        let config = PatchEmbedConfig::new([38, 38], 16, 3, 8).with_allow_remainder(true);
        assert_eq!(config.patches_resolution(), [2, 2]);
        let embed: PatchEmbed = config.init(&device);
        let output = embed.forward(Tensor::<4>::zeros([1, 3, 38, 38], &device));
        assert_eq!(output.dims(), [1, 4, 8]);
    }

    /// The norm is the seam's, sized to the tokens.
    #[test]
    fn test_patch_embed_norm_is_sized() {
        let device = cpu_device();
        let embed: PatchEmbed = PatchEmbedConfig::new([32, 32], 16, 3, 8)
            .with_norm(Some(layer_norm()))
            .init(&device);
        assert_eq!(embed.norm.as_ref().unwrap().num_features(), 8);
        assert!(embed.enable_patch_norm());
    }

    #[test]
    fn test_patch_embed_forward() {
        let config = PatchEmbedConfig {
            input_resolution: [224, 224],
            patch_size: 16,
            d_input: 3,
            d_output: 768,
            norm: Some(layer_norm()),
            bias: true,
            allow_remainder: false,
        };
        let device = cpu_device();
        let patch_embed = config.try_init(&device).ok_or_panic();

        let input = Tensor::<4>::from_data(
            TensorData::new(vec![1.0; 3 * 224 * 224], [1, 3, 224, 224]),
            &device,
        );

        let output = patch_embed.forward(input);
        assert_eq!(output.dims(), [1, 196, 768]);
    }

    /// A pre-norm `ViT`'s patch embedding has no conv bias.
    #[test]
    fn test_patch_embed_without_bias() {
        let device = cpu_device();
        let with: PatchEmbed = PatchEmbedConfig::new([32, 32], 16, 3, 8).init(&device);
        assert!(with.projection.bias.is_some());
        let without: PatchEmbed = PatchEmbedConfig::new([32, 32], 16, 3, 8)
            .with_bias(false)
            .init(&device);
        assert!(without.projection.bias.is_none());
    }

    #[test]
    fn test_patch_embed_without_norm() {
        let config = PatchEmbedConfig {
            input_resolution: [224, 224],
            patch_size: 16,
            d_input: 3,
            d_output: 768,
            norm: None,
            bias: true,
            allow_remainder: false,
        };
        let device = cpu_device();
        let patch_embed = config.try_init(&device).ok_or_panic();

        let input = Tensor::<4>::from_data(
            TensorData::new(vec![1.0; 3 * 224 * 224], [1, 3, 224, 224]),
            &device,
        );

        let output = patch_embed.forward(input);
        assert_eq!(output.dims(), [1, 196, 768]);
    }
}
