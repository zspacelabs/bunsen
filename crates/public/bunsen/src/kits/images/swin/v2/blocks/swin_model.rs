//! # Top-Level Swin Transformer v2 model components.

use alloc::{
    string::ToString,
    vec::Vec,
};

use burn::{
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        Dropout,
        DropoutConfig,
        Initializer,
        LayerNorm,
        LayerNormConfig,
        Linear,
        LinearConfig,
        pool::{
            AdaptiveAvgPool1d,
            AdaptiveAvgPool1dConfig,
        },
    },
    prelude::{
        Backend,
        Tensor,
    },
};

use crate::{
    blocks::images::{
        drop::rate_table::DropPathRateDepthTable,
        patching::patch_embed::{
            PatchEmbed,
            PatchEmbedConfig,
            PatchEmbedMeta,
        },
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::swin::v2::blocks::{
        PatchMerging,
        PatchMergingConfig,
        StochasticDepthTransformerBlockSequence,
        StochasticDepthTransformerBlockSequenceConfig,
        StochasticDepthTransformerBlockSequenceMeta,
    },
};

/// Configuration for a single layer in the Swin Transformer V2 model.
///
/// One entry per stage in a [`SwinTransformerV2ContractConfig`]'s
/// `layer_configs` vector. Records the per-stage block `depth` and attention
/// head count; stage-level resolution, channels and drop-path rates are derived
/// when the parent config lowers to a [`SwinTransformerV2StructureConfig`].
#[derive(Config, Debug, PartialEq, Eq)]
pub struct LayerConfig {
    /// The depth of the layer, i.e., the number of transformer blocks in this
    /// layer.
    pub depth: usize,

    /// The number of attention heads in the transformer blocks of this layer.
    pub num_heads: usize,
}

/// Meta trait for [`SwinTransformerV2`].
///
/// Implemented by [`SwinTransformerV2StructureConfig`], [`SwinTransformerV2`],
/// and the policy, [`SwinTransformerV2ContractConfig`].
pub trait SwinTransformerV2Meta {
    /// The input image resolution as [height, width].
    fn input_resolution(&self) -> [usize; 2];

    /// The input height of the image.
    fn input_height(&self) -> usize {
        self.input_resolution()[0]
    }

    /// The input width of the image.
    fn input_width(&self) -> usize {
        self.input_resolution()[1]
    }

    /// The number of input channels.
    fn d_input(&self) -> usize;

    /// The patch size of the input image.
    fn patch_size(&self) -> usize;

    /// Number of classes.
    fn num_classes(&self) -> usize;

    /// The size of the embedding dimension.
    fn d_embed(&self) -> usize;

    /// The window size for window attention.
    fn window_size(&self) -> usize;

    /// Depth of each layer.
    fn layer_configs(&self) -> Vec<LayerConfig>;

    /// Ratio of hidden dimension to input dimension in MLP.
    fn mlp_ratio(&self) -> f64;

    /// Whether to enable QKV bias.
    fn enable_qkv_bias(&self) -> bool;

    /// Dropout rate on the patch embeddings, and in each block's MLP and
    /// attention projection.
    fn drop_rate(&self) -> f64;

    /// Dropout rate for attention.
    fn attn_drop_rate(&self) -> f64;

    /// Drop path rate for stochastic depth.
    fn drop_path_rate(&self) -> f64;

    /// Enable APE (Absolute Positional Encoding).
    fn enable_ape(&self) -> bool;

    /// Enable patch normalization?
    fn enable_patch_norm(&self) -> bool;
}

/// Configuration for the [`SwinTransformerV2`] model: the policy of Swin's
/// Stacked Config.
///
/// Top-level config describing the patch embedding, per-stage
/// [`LayerConfig`]s, window size and training options. It implements
/// [`ToStructureConfig`], and its lowering is fallible:
/// [`try_to_structure`](ToStructureConfig::try_to_structure) checks that the
/// stages fit the input and the window, and returns the
/// [`SwinTransformerV2StructureConfig`] or a [`BunsenError::Invalid`] that
/// says what does not fit. `.init(device)`, from the trait's blanket
/// [`ModuleInit`] impl, lowers and builds in one step; then `forward` an image
/// tensor to get classification logits.
#[derive(Config, Debug)]
pub struct SwinTransformerV2ContractConfig {
    /// The input image resolution as [height, width].
    pub input_resolution: [usize; 2],

    /// The patch size of the input image.
    pub patch_size: usize,

    /// The number of input channels.
    pub d_input: usize,

    /// Number of classes.
    pub num_classes: usize,

    /// The size of the embedding dimension.
    pub d_embed: usize,

    /// Depth of each layer.
    pub layer_configs: Vec<LayerConfig>,

    /// The window size for window attention.
    #[config(default = 7)]
    pub window_size: usize,

    /// Ratio of hidden dimension to input dimension in MLP.
    #[config(default = 4.0)]
    pub mlp_ratio: f64,

    /// Whether to enable QKV bias.
    #[config(default = true)]
    pub enable_qkv_bias: bool,

    /// Dropout rate on the patch embeddings, and in each block's MLP and
    /// attention projection.
    #[config(default = 0.0)]
    pub drop_rate: f64,

    /// Dropout rate for attention.
    #[config(default = 0.0)]
    pub attn_drop_rate: f64,

    /// Drop path rate for stochastic depth.
    #[config(default = 0.1)]
    pub drop_path_rate: f64,

    /// Enable APE (Absolute Positional Encoding).
    #[config(default = true)]
    pub enable_ape: bool,

    /// Enable patch normalization?
    #[config(default = true)]
    pub enable_patch_norm: bool,
}

impl SwinTransformerV2Meta for SwinTransformerV2ContractConfig {
    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn d_input(&self) -> usize {
        self.d_input
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn d_embed(&self) -> usize {
        self.d_embed
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn layer_configs(&self) -> Vec<LayerConfig> {
        self.layer_configs.clone()
    }

    fn mlp_ratio(&self) -> f64 {
        self.mlp_ratio
    }

    fn enable_qkv_bias(&self) -> bool {
        self.enable_qkv_bias
    }

    fn drop_rate(&self) -> f64 {
        self.drop_rate
    }

    fn attn_drop_rate(&self) -> f64 {
        self.attn_drop_rate
    }

    fn drop_path_rate(&self) -> f64 {
        self.drop_path_rate
    }

    fn enable_ape(&self) -> bool {
        self.enable_ape
    }

    fn enable_patch_norm(&self) -> bool {
        self.enable_patch_norm
    }
}

impl ToStructureConfig for SwinTransformerV2ContractConfig {
    type Structure = SwinTransformerV2StructureConfig;

    /// Checks that the stages fit the input and the window, and resolves each
    /// stage's resolution, width, and drop-path rates.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] when `patch_size` is 0, when there are no
    /// stages, when the last stage's patch grid is empty or not a multiple of
    /// `window_size`, or when `input_resolution` is not that grid scaled back
    /// up by the merges and the patch size.
    fn try_to_structure(&self) -> BunsenResult<SwinTransformerV2StructureConfig> {
        if self.patch_size == 0 {
            return Err(BunsenError::Invalid(
                "patch_size must be non-zero".to_string(),
            ));
        }

        let patch_config = PatchEmbedConfig::new(
            self.input_resolution,
            self.patch_size,
            self.d_input,
            self.d_embed,
        )
        .with_enable_patch_norm(self.enable_patch_norm);

        if self.layer_configs.is_empty() {
            return Err(BunsenError::Invalid(
                "At least one layer configuration is required".to_string(),
            ));
        }

        let mut layer_resolutions: Vec<[usize; 2]> = Vec::with_capacity(self.layer_configs.len());
        let mut layer_dims: Vec<usize> = Vec::with_capacity(self.layer_configs.len());

        for layer_i in 0..self.layer_configs.len() {
            let layer_p = 2_usize.pow(layer_i as u32); // Power of 2 for each layer

            layer_resolutions.push([
                patch_config.patches_height() / layer_p,
                patch_config.patches_width() / layer_p,
            ]);
            layer_dims.push(patch_config.d_output() * layer_p);
        }

        let output_resolution = *layer_resolutions.last().unwrap();
        let [last_h, last_w] = output_resolution;
        if last_h == 0 || last_w == 0 {
            return Err(BunsenError::Invalid(format!(
                "Output resolution must be non-zero: {output_resolution:?}"
            )));
        }
        if !last_h.is_multiple_of(self.window_size) || !last_w.is_multiple_of(self.window_size) {
            return Err(BunsenError::Invalid(format!(
                "Output resolution must be divisible by window size: {:?} / {:?}",
                output_resolution, self.window_size
            )));
        }
        let expansion_scale = 2_usize.pow((self.layer_configs.len() - 1) as u32) * self.patch_size;
        let expected_resolution = [last_h * expansion_scale, last_w * expansion_scale];
        if patch_config.input_resolution() != expected_resolution {
            return Err(BunsenError::Invalid(format!(
                "Input resolution must match [<c> * <window_size:{:?}> * 2^(<layers:{:?}>-1) * <patch_size:{:?}>, ...]: {:?} != {:?}",
                self.window_size,
                self.layer_configs.len(),
                self.patch_size,
                patch_config.input_resolution(),
                expected_resolution,
            )));
        }

        // Stochastic depth delay rule
        let dpr_layer_rates = DropPathRateDepthTable::dpr_layer_rates(
            self.drop_path_rate,
            &self
                .layer_configs
                .iter()
                .map(|c| c.depth)
                .collect::<Vec<usize>>(),
        );

        let block_configs: Vec<StochasticDepthTransformerBlockSequenceConfig> =
            (0..self.layer_configs.len())
                .map(|layer_i| {
                    let cfg = &self.layer_configs[layer_i];

                    let layer_resolution = layer_resolutions[layer_i];
                    let layer_dim = layer_dims[layer_i];

                    StochasticDepthTransformerBlockSequenceConfig::new(
                        // Double the embedding size for each layer
                        layer_dim,
                        layer_resolution,
                        cfg.depth,
                        cfg.num_heads,
                        self.window_size,
                    )
                    .with_mlp_ratio(self.mlp_ratio())
                    .with_enable_qkv_bias(self.enable_qkv_bias())
                    .with_drop_path_rates(Some(dpr_layer_rates[layer_i].clone()))
                    .with_drop_rate(self.drop_rate())
                    .with_attn_drop_rate(self.attn_drop_rate())
                })
                .collect();

        Ok(SwinTransformerV2StructureConfig {
            patch_config,
            enable_ape: self.enable_ape,
            drop_rate: self.drop_rate,
            block_configs,
            num_classes: self.num_classes,
            attn_drop_rate: self.attn_drop_rate,
            drop_path_rate: self.drop_path_rate,
        })
    }
}

/// The unrolled structure of a [`SwinTransformerV2`].
///
/// [`SwinTransformerV2ContractConfig`] lowers to it through
/// [`try_to_structure`](ToStructureConfig::try_to_structure), which is where
/// the policy is checked. It holds the patch embedding, one block sequence per
/// stage (each with its resolution, width, heads, and per-block drop-path rates
/// resolved), and what the head needs; the patch merges between stages follow
/// from the block sequences. Call `.init(device)` to build the
/// [`SwinTransformerV2`].
///
/// Implements [`SwinTransformerV2Meta`].
#[derive(Config, Debug)]
pub struct SwinTransformerV2StructureConfig {
    /// The patch embedding configuration for the model.
    pub patch_config: PatchEmbedConfig,

    /// Whether to add an absolute positional encoding (APE) to the patches.
    pub enable_ape: bool,

    /// Dropout rate on the patch embeddings. Each block sequence carries the
    /// rate its MLP and attention projection apply, which the policy sets to
    /// the same value.
    pub drop_rate: f64,

    /// The block configurations for each layer, including stochastic depth.
    ///
    /// Between layers, a patch merge halves the resolution and doubles the
    /// width, so each layer's input is the one before it, merged.
    pub block_configs: Vec<StochasticDepthTransformerBlockSequenceConfig>,

    /// Number of classes.
    pub num_classes: usize,

    /// Dropout rate for attention, as the policy set it. Each block sequence
    /// carries the rate it applies.
    pub attn_drop_rate: f64,

    /// Drop path rate for stochastic depth, as the policy set it. Each block
    /// sequence carries its per-block rates.
    pub drop_path_rate: f64,
}

impl SwinTransformerV2StructureConfig {
    /// The `[height, width]` patch grid of each layer.
    pub fn layer_resolutions(&self) -> Vec<[usize; 2]> {
        self.block_configs
            .iter()
            .map(|c| c.input_resolution())
            .collect()
    }

    /// The embedding width of each layer.
    pub fn layer_dims(&self) -> Vec<usize> {
        self.block_configs.iter().map(|c| c.d_input()).collect()
    }
}

impl SwinTransformerV2Meta for SwinTransformerV2StructureConfig {
    fn input_resolution(&self) -> [usize; 2] {
        self.patch_config.input_resolution()
    }

    fn d_input(&self) -> usize {
        self.patch_config.d_input()
    }

    fn patch_size(&self) -> usize {
        self.patch_config.patch_size()
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn d_embed(&self) -> usize {
        self.patch_config.d_output()
    }

    fn window_size(&self) -> usize {
        self.block_configs[0].window_size()
    }

    fn layer_configs(&self) -> Vec<LayerConfig> {
        self.block_configs
            .iter()
            .map(|b| LayerConfig {
                depth: b.depth(),
                num_heads: b.num_heads(),
            })
            .collect()
    }

    fn mlp_ratio(&self) -> f64 {
        self.block_configs[0].mlp_ratio()
    }

    fn enable_qkv_bias(&self) -> bool {
        self.block_configs[0].enable_qkv_bias()
    }

    fn drop_rate(&self) -> f64 {
        self.drop_rate
    }

    fn attn_drop_rate(&self) -> f64 {
        self.attn_drop_rate
    }

    fn drop_path_rate(&self) -> f64 {
        self.drop_path_rate
    }

    fn enable_ape(&self) -> bool {
        self.enable_ape
    }

    fn enable_patch_norm(&self) -> bool {
        self.patch_config.enable_patch_norm()
    }
}

impl<B: Backend> ModuleInit<B, SwinTransformerV2<B>> for SwinTransformerV2StructureConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<SwinTransformerV2<B>> {
        let Some(last_block) = self.block_configs.last() else {
            return Err(BunsenError::Invalid(
                "At least one layer configuration is required".to_string(),
            ));
        };
        let grid_output_features = last_block.d_input();

        let patch_embed: PatchEmbed<B> = self.patch_config.try_init(device)?;

        // ape: trunc_normal: ([1, num_patches, d_embed], std=0.02)
        // defaults: (mean=0.0, a=-2.0, b=2.0)
        let patch_ape: Option<Param<Tensor<B, 3>>> = if self.enable_ape {
            Some(
                Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                }
                .init(
                    [1_usize, patch_embed.num_patches(), patch_embed.d_output()],
                    device,
                ),
            )
        } else {
            None
        };

        let grid_transformer_block_sequences = self
            .block_configs
            .iter()
            .map(|config| config.try_init(device))
            .collect::<BunsenResult<Vec<StochasticDepthTransformerBlockSequence<B>>>>()?;

        let grid_merge_layers = self.block_configs[..self.block_configs.len() - 1]
            .iter()
            .map(|config| {
                PatchMergingConfig::new(config.input_resolution(), config.d_input())
                    .try_init(device)
            })
            .collect::<BunsenResult<Vec<PatchMerging<B>>>>()?;

        let module = SwinTransformerV2 {
            patch_embed,
            patch_ape,
            grid_input_dropout: DropoutConfig::new(self.drop_rate).init(),
            grid_transformer_block_sequences,
            grid_merge_layers,
            grid_output_norm: LayerNormConfig::new(grid_output_features).init(device),
            grid_output_features,
            head_avgpool: AdaptiveAvgPool1dConfig::new(1).init(),
            head: LinearConfig::new(grid_output_features, self.num_classes).init(device),
            drop_rate: self.drop_rate,
            attn_drop_rate: self.attn_drop_rate,
            drop_path_rate: self.drop_path_rate,
        };

        Ok(module)
    }
}

/// High-level SWIN Transformer V2 model.
///
/// Implements [`SwinTransformerV2Meta`].
///
/// Built by [`SwinTransformerV2ContractConfig`] (high-level) or
/// [`SwinTransformerV2StructureConfig`].
#[derive(Module, Debug)]
pub struct SwinTransformerV2<B: Backend> {
    /// The patch embedding layer that converts the input image into patches.
    pub patch_embed: PatchEmbed<B>,

    /// The absolute positional encoding (APE) for the patches, if enabled.
    pub patch_ape: Option<Param<Tensor<B, 3>>>,

    /// The input dropout layer applied to the patch embeddings.
    pub grid_input_dropout: Dropout,

    /// The sequences of transformer blocks for the grid.
    pub grid_transformer_block_sequences: Vec<StochasticDepthTransformerBlockSequence<B>>,

    /// The patch merging layers that reduce the spatial dimensions of the grid.
    pub grid_merge_layers: Vec<PatchMerging<B>>,

    /// The layer normalization applied to the output of the grid transformer
    /// blocks.
    pub grid_output_norm: LayerNorm<B>,

    /// The number of output features after the grid transformer blocks.
    pub grid_output_features: usize,

    /// The average pooling layer to aggregate the grid output.
    pub head_avgpool: AdaptiveAvgPool1d,

    /// The final classification head.
    pub head: Linear<B>,

    /// Dropout rate on the patch embeddings, and in each block's MLP and
    /// attention projection.
    pub drop_rate: f64,

    /// Dropout rate for attention.
    pub attn_drop_rate: f64,

    /// Drop path rate for stochastic depth.
    pub drop_path_rate: f64,
}

impl<B: Backend> SwinTransformerV2Meta for SwinTransformerV2<B> {
    fn input_resolution(&self) -> [usize; 2] {
        self.patch_embed.input_resolution()
    }

    fn d_input(&self) -> usize {
        self.patch_embed.d_input()
    }

    fn patch_size(&self) -> usize {
        self.patch_embed.patch_size()
    }

    fn num_classes(&self) -> usize {
        self.head.weight.dims()[1]
    }

    fn d_embed(&self) -> usize {
        self.patch_embed.d_output()
    }

    fn window_size(&self) -> usize {
        self.grid_transformer_block_sequences[0].window_size()
    }

    fn layer_configs(&self) -> Vec<LayerConfig> {
        self.grid_transformer_block_sequences
            .iter()
            .map(|b| LayerConfig {
                depth: b.depth(),
                num_heads: b.num_heads(),
            })
            .collect()
    }

    fn mlp_ratio(&self) -> f64 {
        self.grid_transformer_block_sequences[0].mlp_ratio()
    }

    fn enable_qkv_bias(&self) -> bool {
        self.grid_transformer_block_sequences[0].enable_qkv_bias()
    }

    fn drop_rate(&self) -> f64 {
        self.drop_rate
    }

    fn attn_drop_rate(&self) -> f64 {
        self.attn_drop_rate
    }

    fn drop_path_rate(&self) -> f64 {
        self.drop_path_rate
    }

    fn enable_ape(&self) -> bool {
        self.patch_ape.is_some()
    }

    fn enable_patch_norm(&self) -> bool {
        self.patch_embed.enable_patch_norm()
    }
}

impl<B: Backend> SwinTransformerV2<B> {
    /// Applies patch embedding and absolute positional encoding (APE) to the
    /// input image tensor.
    #[inline(always)]
    #[must_use]
    fn apply_patching(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let x = self.patch_embed.forward(input);

        match self.patch_ape {
            Some(ref ape) => x + ape.val(),
            None => x,
        }
    }

    /// Applies the layer stack to a patch tensor.
    #[inline(always)]
    #[must_use]
    fn apply_stack(
        &self,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let mut x = self.grid_input_dropout.forward(input);

        for layer_i in 0..self.grid_transformer_block_sequences.len() {
            if layer_i > 0 {
                x = self.grid_merge_layers[layer_i - 1].forward(x);
            }

            x = self.grid_transformer_block_sequences[layer_i].forward(x);
        }
        // B L C

        self.grid_output_norm.forward(x)
        // B L C
    }

    /// Aggregates the grid into a single vector per batch.
    #[inline(always)]
    #[must_use]
    fn aggregate_grid(
        &self,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        // input: B L C
        let x = input.swap_dims(1, 2);
        let x = self.head_avgpool.forward(x);
        // B C 1
        x.squeeze_dim::<2>(2)
        // B C
    }

    /// Applies the final classification head to the transformed output.
    #[inline(always)]
    #[must_use]
    fn apply_head(
        &self,
        input: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.head.forward(input)
    }

    /// Applies the model to the input image tensor and returns the
    /// classification logits.
    ///
    /// # Arguments
    ///
    /// * `input`: A tensor of `[batch, channels, height, width]`,
    ///
    /// # Returns
    ///
    /// A 2D tensor of `[batch, num_classes]` of the classification logits.
    ///
    /// # Panics
    ///
    /// On shape contract failure.
    #[must_use]
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let [batch] = unpack_shape_contract!(
            ["batch", "d_input", "height", "width"],
            &input.dims(),
            &["batch"],
            &[
                ("d_input", self.d_input()),
                ("height", self.input_height()),
                ("width", self.input_width()),
            ]
        );

        let x = self.apply_patching(input);
        assert_shape_contract_periodically!(
            ["batch", "num_patches", "d_embed"],
            &x.dims(),
            &[
                ("num_patches", self.patch_embed.num_patches()),
                ("d_embed", self.d_embed()),
            ]
        );

        let x = self.apply_stack(x);
        let x = self.aggregate_grid(x);
        assert_shape_contract_periodically!(
            ["batch", "grid_output_features"],
            &x.dims(),
            &[("grid_output_features", self.grid_output_features)]
        );

        let x = self.apply_head(x);
        assert_shape_contract_periodically!(
            ["batch", "num_classes"],
            &x.dims(),
            &[("batch", batch), ("num_classes", self.num_classes())]
        );

        x
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use burn::tensor::{
        Distribution,
        Tolerance,
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        errors::WithOkOrPanic,
        kits::images::swin::v2::blocks::ShiftedWindowTransformerBlockMeta,
        support::testing::{
            CpuBackend,
            DeviceMemoryGuard,
            PerformanceBackend,
            default_device,
        },
    };

    #[test]
    #[serial]
    fn test_swin_transformer_v2_meta() {
        type B = PerformanceBackend;
        let config = SwinTransformerV2ContractConfig {
            input_resolution: [224, 224],
            patch_size: 4,
            d_input: 3,
            num_classes: 1000,
            d_embed: 96,
            layer_configs: vec![
                LayerConfig {
                    depth: 2,
                    num_heads: 3,
                },
                LayerConfig {
                    depth: 2,
                    num_heads: 6,
                },
                LayerConfig {
                    depth: 18,
                    num_heads: 12,
                },
            ],
            window_size: 7,
            mlp_ratio: 4.0,
            enable_qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.1,
            enable_ape: true,
            enable_patch_norm: true,
        };

        assert_eq!(config.input_resolution(), [224, 224]);
        assert_eq!(config.input_height(), 224);
        assert_eq!(config.input_width(), 224);
        assert_eq!(config.patch_size(), 4);
        assert_eq!(config.d_input(), 3);
        assert_eq!(config.num_classes(), 1000);
        assert_eq!(config.d_embed(), 96);
        assert_eq!(config.window_size(), 7);
        assert_eq!(config.mlp_ratio(), 4.0);
        assert!(config.enable_qkv_bias());
        assert_eq!(config.drop_rate(), 0.0);
        assert_eq!(config.attn_drop_rate(), 0.0);
        assert_eq!(config.drop_path_rate(), 0.1);
        assert!(config.enable_ape());
        assert!(config.enable_patch_norm());
        assert_eq!(
            config.layer_configs(),
            vec![
                LayerConfig {
                    depth: 2,
                    num_heads: 3,
                },
                LayerConfig {
                    depth: 2,
                    num_heads: 6,
                },
                LayerConfig {
                    depth: 18,
                    num_heads: 12,
                },
            ]
        );

        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);
        let model: SwinTransformerV2<B> = config.try_init(&device).ok_or_panic();

        assert_eq!(model.input_resolution(), [224, 224]);
        assert_eq!(model.input_height(), 224);
        assert_eq!(model.input_width(), 224);
        assert_eq!(model.patch_size(), 4);
        assert_eq!(model.d_input(), 3);
        assert_eq!(model.num_classes(), 1000);
        assert_eq!(model.d_embed(), 96);
        assert_eq!(model.window_size(), 7);
        assert_eq!(model.mlp_ratio(), 4.0);
        assert!(model.enable_qkv_bias());
        assert_eq!(model.drop_rate(), 0.0);
        assert_eq!(model.attn_drop_rate(), 0.0);
        assert_eq!(model.drop_path_rate(), 0.1);
        assert!(model.enable_ape());
        assert!(model.enable_patch_norm());
        assert_eq!(
            model.layer_configs(),
            vec![
                LayerConfig {
                    depth: 2,
                    num_heads: 3,
                },
                LayerConfig {
                    depth: 2,
                    num_heads: 6,
                },
                LayerConfig {
                    depth: 18,
                    num_heads: 12,
                },
            ]
        );
    }

    #[test]
    #[serial]
    fn test_smoke_test_ape() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let b = 2;
        let d_input = 3;

        let layer_configs = vec![
            LayerConfig {
                depth: 2,
                num_heads: 3,
            },
            LayerConfig {
                depth: 2,
                num_heads: 6,
            },
        ];

        let patch_size = 4;
        let window_size = 3;

        let last_wh = 2;
        let last_ww = 2;
        let last_h = last_wh * window_size;
        let last_w = last_ww * window_size;

        let merge_steps = (layer_configs.len() - 1) as u32;
        let expansion_scale = 2_usize.pow(merge_steps);
        let h = last_h * expansion_scale * patch_size;
        let w = last_w * expansion_scale * patch_size;

        let num_classes = 12;

        let d_embed = (d_input * patch_size * patch_size) / 2;

        let self1 = SwinTransformerV2ContractConfig::new(
            [h, w],
            patch_size,
            d_input,
            num_classes,
            d_embed,
            layer_configs,
        )
        .with_window_size(window_size);
        let model: SwinTransformerV2<B> = self1.try_init(&device).ok_or_panic();

        let distribution = Distribution::Normal(0.0, 0.02);
        let input = Tensor::<B, 4>::random([b, d_input, h, w], distribution, &device);

        let output = model.forward(input.clone());
        assert_eq!(output.dims(), [b, num_classes]);

        let expected: Tensor<B, 2> = {
            let patched = model.apply_patching(input.clone());
            assert_eq!(
                patched.dims(),
                [b, h * w / (patch_size * patch_size), d_embed]
            );

            let stacked = model.apply_stack(patched);
            assert_eq!(
                stacked.dims(),
                [b, last_h * last_w, model.grid_output_features]
            );

            let aggregated = model.aggregate_grid(stacked);
            assert_eq!(aggregated.dims(), [b, model.grid_output_features]);

            let classed = model.apply_head(aggregated);
            assert_eq!(classed.dims(), [b, num_classes]);

            classed
        };

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    #[serial]
    fn test_smoke_test_no_ape() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let b = 2;
        let d_input = 3;

        let layer_configs = vec![
            LayerConfig {
                depth: 2,
                num_heads: 3,
            },
            LayerConfig {
                depth: 2,
                num_heads: 6,
            },
        ];

        let patch_size = 4;
        let window_size = 3;

        let last_wh = 2;
        let last_ww = 2;
        let last_h = last_wh * window_size;
        let last_w = last_ww * window_size;

        let merge_steps = (layer_configs.len() - 1) as u32;
        let expansion_scale = 2_usize.pow(merge_steps);
        let h = last_h * expansion_scale * patch_size;
        let w = last_w * expansion_scale * patch_size;

        let num_classes = 12;

        let d_embed = (d_input * patch_size * patch_size) / 2;

        let self1 = SwinTransformerV2ContractConfig::new(
            [h, w],
            patch_size,
            d_input,
            num_classes,
            d_embed,
            layer_configs,
        )
        .with_enable_ape(false)
        .with_window_size(window_size);
        let model: SwinTransformerV2<B> = self1.try_init(&device).ok_or_panic();

        let distribution = Distribution::Normal(0.0, 0.02);
        let input = Tensor::<B, 4>::random([b, d_input, h, w], distribution, &device);

        let output = model.forward(input.clone());
        assert_eq!(output.dims(), [b, num_classes]);

        let expected: Tensor<B, 2> = {
            let patched = model.apply_patching(input.clone());
            assert_eq!(
                patched.dims(),
                [b, h * w / (patch_size * patch_size), d_embed]
            );

            let stacked = model.apply_stack(patched);
            assert_eq!(
                stacked.dims(),
                [b, last_h * last_w, model.grid_output_features]
            );

            let aggregated = model.aggregate_grid(stacked);
            assert_eq!(aggregated.dims(), [b, model.grid_output_features]);

            let classed = model.apply_head(aggregated);
            assert_eq!(classed.dims(), [b, num_classes]);

            classed
        };

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
    }

    /// A two-layer policy over `[48, 48]` images: a `[12, 12]` patch grid,
    /// merged once to `[6, 6]`, two windows of 3 a side.
    fn tiny_policy() -> SwinTransformerV2ContractConfig {
        SwinTransformerV2ContractConfig::new(
            [48, 48],
            4,
            3,
            12,
            24,
            vec![LayerConfig::new(1, 3), LayerConfig::new(1, 6)],
        )
        .with_window_size(3)
    }

    /// Asserts that `a` and `b` answer every [`SwinTransformerV2Meta`] method
    /// alike.
    fn assert_meta_agrees(
        a: &impl SwinTransformerV2Meta,
        b: &impl SwinTransformerV2Meta,
    ) {
        assert_eq!(a.input_resolution(), b.input_resolution());
        assert_eq!(a.input_height(), b.input_height());
        assert_eq!(a.input_width(), b.input_width());
        assert_eq!(a.d_input(), b.d_input());
        assert_eq!(a.patch_size(), b.patch_size());
        assert_eq!(a.num_classes(), b.num_classes());
        assert_eq!(a.d_embed(), b.d_embed());
        assert_eq!(a.window_size(), b.window_size());
        assert_eq!(a.layer_configs(), b.layer_configs());
        assert_eq!(a.mlp_ratio(), b.mlp_ratio());
        assert_eq!(a.enable_qkv_bias(), b.enable_qkv_bias());
        assert_eq!(a.drop_rate(), b.drop_rate());
        assert_eq!(a.attn_drop_rate(), b.attn_drop_rate());
        assert_eq!(a.drop_path_rate(), b.drop_path_rate());
        assert_eq!(a.enable_ape(), b.enable_ape());
        assert_eq!(a.enable_patch_norm(), b.enable_patch_norm());
    }

    /// The policy builds the same `SwinTransformerV2` through its structure
    /// as through the blanket `init`.
    #[test]
    #[serial]
    fn test_policy_pathways_agree() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let policy = tiny_policy()
            .with_mlp_ratio(2.0)
            .with_enable_qkv_bias(false)
            .with_drop_rate(0.1)
            .with_attn_drop_rate(0.1)
            .with_drop_path_rate(0.2)
            .with_enable_ape(false)
            .with_enable_patch_norm(false);

        let structure = policy.to_structure();
        assert_eq!(structure.layer_resolutions(), vec![[12, 12], [6, 6]]);
        assert_eq!(structure.layer_dims(), vec![24, 48]);
        assert_meta_agrees(&policy, &structure);

        let lowered: SwinTransformerV2<B> = structure.init(&device);
        let direct: SwinTransformerV2<B> = policy.init(&device);

        assert_meta_agrees(&direct, &lowered);
        assert_meta_agrees(&direct, &structure);
    }

    /// A policy whose stages do not fit is an error from `try_to_structure`,
    /// and so from `try_init`, rather than a panic.
    #[test]
    fn test_try_to_structure_rejects_bad_policy() {
        assert!(tiny_policy().try_to_structure().is_ok());

        let is_invalid = |policy: SwinTransformerV2ContractConfig| {
            matches!(policy.try_to_structure(), Err(BunsenError::Invalid(_)))
        };

        // No stages.
        assert!(is_invalid(SwinTransformerV2ContractConfig {
            layer_configs: vec![],
            ..tiny_policy()
        }));
        // The last stage's `[6, 6]` grid is not a multiple of the window.
        assert!(is_invalid(tiny_policy().with_window_size(4)));
        // `[50, 50]` still patches to a `[12, 12]` grid, but is not that grid
        // scaled back up.
        assert!(is_invalid(SwinTransformerV2ContractConfig {
            input_resolution: [50, 50],
            ..tiny_policy()
        }));

        let device: burn::prelude::Device<CpuBackend> = Default::default();
        let bad: BunsenResult<SwinTransformerV2<CpuBackend>> =
            tiny_policy().with_window_size(4).try_init(&device);
        assert!(matches!(bad, Err(BunsenError::Invalid(_))));
    }

    /// The policy's `drop_rate` reaches every block's MLP and attention
    /// projection dropout, as upstream's `BasicLayer(drop=drop_rate)` does,
    /// and not only the input dropout.
    #[test]
    fn test_drop_rate_reaches_the_blocks() {
        let policy = tiny_policy().with_drop_rate(0.25);

        let structure = policy.to_structure();
        assert_eq!(structure.drop_rate, 0.25);
        for sequence in &structure.block_configs {
            assert_eq!(sequence.drop_rate(), 0.25);
            for block in sequence.block_configs() {
                assert_eq!(block.drop_rate(), 0.25);
            }
        }

        let device: burn::prelude::Device<CpuBackend> = Default::default();
        let model: SwinTransformerV2<CpuBackend> = policy.init(&device);
        assert_eq!(model.grid_input_dropout.prob, 0.25);
        for sequence in &model.grid_transformer_block_sequences {
            assert_eq!(sequence.drop_rate(), 0.25);
        }
    }

    /// A zero patch size is an error from `try_to_structure`, and so from
    /// `try_init`, rather than a divide-by-zero panic.
    #[test]
    fn test_try_to_structure_rejects_zero_patch_size() {
        let policy = SwinTransformerV2ContractConfig {
            patch_size: 0,
            ..tiny_policy()
        };
        assert!(matches!(
            policy.try_to_structure(),
            Err(BunsenError::Invalid(_))
        ));

        let device: burn::prelude::Device<CpuBackend> = Default::default();
        let bad: BunsenResult<SwinTransformerV2<CpuBackend>> = policy.try_init(&device);
        assert!(matches!(bad, Err(BunsenError::Invalid(_))));
    }
}
