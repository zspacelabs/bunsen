use alloc::{
    vec,
    vec::Vec,
};

use burn::config::Config;

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        mlp::BunsenMlpConfig,
        norm::{
            BunsenLayerNormConfig,
            BunsenNormalizationConfig,
        },
        transformers::{
            attention::window::{
                RelativePositionBiasConfig,
                WindowAttentionConfig,
            },
            decoders::causal::CausalDecoderBlockConfig,
            models::{
                staged_vision_encoder::StagedVisionEncoderStructureConfig,
                vision_encoder::TokenPoolKind,
            },
            sublayer::NormPlacement,
        },
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    kits::images::classification::swin::common::{
        SWIN_NORM_EPSILON,
        SwinStages,
        swin_block_drop_paths,
    },
};

/// A Swin Transformer (V1)'s policy: timm's `SwinTransformer` knobs,
/// lowering to the generic [`StagedVisionEncoderStructureConfig`].
///
/// Stage `i` is `embed_dim * 2^i` wide with `num_heads[i]` heads and
/// `depths[i]` blocks; every stage after the first starts with a patch
/// merge. The model runs at `image_size`, its windows fixed to each stage's
/// grid.
#[derive(Config, Debug)]
pub struct SwinV1ContractConfig {
    /// The side of the square images it takes (timm's `img_size`).
    pub image_size: usize,

    /// The first stage's token width.
    pub embed_dim: usize,

    /// The blocks of each stage.
    pub depths: Vec<usize>,

    /// The heads of each stage.
    pub num_heads: Vec<usize>,

    /// The window: its side, before a stage's grid shrinks it.
    pub window_size: usize,

    /// The classes; 0 for a headless model.
    pub num_classes: usize,

    /// Each stage's window side, in place of
    /// [`window_size`](Self::window_size) (timm's per-stage `window_size`,
    /// Swin-S3's); empty, `window_size` for every stage.
    #[config(default = "vec![]")]
    pub stage_window_sizes: Vec<usize>,

    /// The side of a square patch.
    #[config(default = 4)]
    pub patch_size: usize,

    /// The image channels (timm's `in_chans`).
    #[config(default = 3)]
    pub in_channels: usize,

    /// The MLP's hidden width over the token width, rounded down.
    #[config(default = 4.0)]
    pub mlp_ratio: f64,

    /// Whether the attention's fused `qkv` projection has a bias.
    #[config(default = true)]
    pub qkv_bias: bool,

    /// The head's pool (timm's `global_pool`).
    #[config(default = "TokenPoolKind::Avg")]
    pub global_pool: TokenPoolKind,

    /// Every norm (timm's `norm_layer`): a layer norm of
    /// [`SWIN_NORM_EPSILON`](super::super::SWIN_NORM_EPSILON), its width
    /// filled in where it sits.
    #[config(default = "BunsenLayerNormConfig::new(0).with_epsilon(SWIN_NORM_EPSILON).into()")]
    pub norm: BunsenNormalizationConfig,

    /// The dropout before the classifier (training only).
    #[config(default = 0.0)]
    pub drop_rate: f64,

    /// The dropout on the attention weights (training only).
    #[config(default = 0.0)]
    pub attn_drop_rate: f64,

    /// The stochastic depth's last rate, ramped linearly over all the blocks
    /// in order, as timm's `calculate_drop_path_rates` (training only).
    #[config(default = 0.0)]
    pub drop_path_rate: f64,
}

impl SwinV1ContractConfig {
    /// Stage `i`'s token width: `embed_dim * 2^i`.
    pub fn stage_width(
        &self,
        i: usize,
    ) -> usize {
        self.embed_dim << i
    }

    /// Stage `i`'s window side, before its grid shrinks it.
    pub fn stage_window_size(
        &self,
        i: usize,
    ) -> usize {
        self.stage_window_sizes
            .get(i)
            .copied()
            .unwrap_or(self.window_size)
    }
}

impl ToStructureConfig for SwinV1ContractConfig {
    type Structure = StagedVisionEncoderStructureConfig;

    /// The staged vision encoder of this Swin: a patch embedding with a
    /// layer norm; stage `i` a patch merge (after the first, normed before
    /// its reduction), then `depths[i]` pre-norm blocks of window attention
    /// (every second shifted; scaled dot-product attention; a learnt
    /// relative position bias table) and a `GELU` MLP; a final layer norm,
    /// the pool, the head.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for no stages, a
    /// head count per stage that doesn't match the depths or divide the
    /// stage's width, a window per stage that doesn't match them, or a zero
    /// window; as the patch embedding's geometry.
    fn try_to_structure(&self) -> BunsenResult<StagedVisionEncoderStructureConfig> {
        let n = self.depths.len();
        if !self.stage_window_sizes.is_empty() && self.stage_window_sizes.len() != n {
            return Err(ConstraintError::new(
                "SwinV1ContractConfig",
                "stage_window_sizes",
                Rule::Relation {
                    lhs: (
                        "stage_window_sizes.len()".into(),
                        self.stage_window_sizes.len().to_string(),
                    ),
                    op: "==",
                    rhs: ("depths.len()".into(), n.to_string()),
                },
            )
            .into());
        }
        if (0..n).any(|i| self.stage_window_size(i) == 0) {
            return Err(ConstraintError::zero_or_empty(
                "SwinV1ContractConfig",
                "stage_window_sizes",
            )
            .into());
        }
        let drop_paths = swin_block_drop_paths(&self.depths, self.drop_path_rate);
        SwinStages {
            config: "SwinV1ContractConfig",
            image_size: self.image_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            embed_dim: self.embed_dim,
            depths: &self.depths,
            num_heads: &self.num_heads,
            num_classes: self.num_classes,
            global_pool: self.global_pool,
            norm: &self.norm,
            drop_rate: self.drop_rate,
            merge_norm_first: true,
        }
        .try_to_structure(|site| {
            let width = site.width;
            Ok(CausalDecoderBlockConfig::new(
                WindowAttentionConfig::new(
                    width,
                    site.heads,
                    site.grid,
                    [self.stage_window_size(site.stage); 2],
                )
                .with_shifted(site.index % 2 == 1)
                .with_qkv_bias(self.qkv_bias)
                .with_key_bias(self.qkv_bias)
                .with_bias(RelativePositionBiasConfig::Table)
                .with_attn_drop(self.attn_drop_rate)
                .into(),
                BunsenMlpConfig::new(width, (width as f64 * self.mlp_ratio) as usize, width)
                    .with_act(BunsenActivationConfig::Gelu)
                    .with_bias(true)
                    .into(),
            )
            .with_norm(self.norm.clone())
            .with_norm_placement(NormPlacement::Pre)
            .with_drop_path(drop_paths[site.stage][site.index]))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::models::staged_vision_encoder::{
            StagedVisionEncoderMeta,
            StagedVisionEncoderModel,
        },
        burner::module::ModuleInit,
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    fn tiny() -> SwinV1ContractConfig {
        SwinV1ContractConfig::new(32, 16, vec![2, 2], vec![2, 4], 4, 10)
    }

    /// The stages halve the grid and double the width; the blocks are
    /// pre-norm, every second shifted; the merges norm first.
    #[test]
    fn test_lowering() {
        let s = tiny().to_structure();
        assert_eq!(s.stage_grids(), [[8, 8], [4, 4]]);
        assert_eq!(s.stage_widths(), [16, 32]);
        assert!(s.stages[1].downsample.as_ref().unwrap().norm_first);
        let model: StagedVisionEncoderModel = tiny().init(&cpu_device());
        let block = &model.stages[0].stack.layers[1];
        assert!(block.attn.pre.is_some() && block.attn.post.is_none());
        assert_eq!(block.attn.branch.as_window().unwrap().shift, [2, 2]);
        assert_eq!(model.num_classes(), 10);
    }

    /// A window per stage: the second stage's 2x2 windows over its 4x4 grid.
    #[test]
    fn test_stage_windows() {
        let model: StagedVisionEncoderModel = tiny()
            .with_stage_window_sizes(vec![4, 2])
            .init(&cpu_device());
        let window = |i: usize| {
            model.stages[i].stack.layers[1]
                .attn
                .branch
                .as_window()
                .unwrap()
        };
        assert_eq!([window(0).window, window(0).shift], [[4, 4], [2, 2]]);
        assert_eq!([window(1).window, window(1).shift], [[2, 2], [1, 1]]);
    }

    #[test]
    fn test_illegal_policies() {
        let illegal = |config: SwinV1ContractConfig, message: &str| {
            ErrorMatcher::illegal()
                .message_contains(message)
                .assert_err(&config.try_to_structure());
        };
        illegal(
            SwinV1ContractConfig::new(32, 16, vec![2, 2], vec![2], 4, 10),
            "num_heads.len() (1) must be == depths.len() (2)",
        );
        illegal(
            tiny().with_stage_window_sizes(vec![4]),
            "stage_window_sizes.len() (1) must be == depths.len() (2)",
        );
        illegal(
            tiny().with_stage_window_sizes(vec![4, 0]),
            "SwinV1ContractConfig.stage_window_sizes: must not be zero",
        );
        illegal(
            SwinV1ContractConfig {
                image_size: 30,
                ..tiny()
            },
            "is not a multiple of patch_size",
        );
    }
}
