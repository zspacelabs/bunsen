use alloc::{
    format,
    vec,
    vec::Vec,
};

use burn::config::Config;

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        mlp::BunsenMlpConfig,
        transformers::{
            attention::window::{
                RelativePositionBiasConfig,
                WindowAttentionConfig,
            },
            decoders::causal::CausalDecoderBlockConfig,
            models::{
                staged_vision_encoder::StagedVisionEncoderStructureConfig,
                vision_encoder::TokenPoolConfig,
            },
            sublayer::NormPlacement,
        },
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::classification::swin::common::{
        SwinStages,
        swin_block_drop_paths,
        swin_norm,
    },
};

/// A Swin Transformer V2's policy: timm's `SwinTransformerV2` knobs, lowering
/// to the generic
/// [`StagedVisionEncoderStructureConfig`].
///
/// Stage `i` is `embed_dim * 2^i` wide with `num_heads[i]` heads and
/// `depths[i]` blocks; every stage after the first starts with a patch
/// merge. The model runs at `image_size`, its windows fixed to each stage's
/// grid.
#[derive(Config, Debug)]
pub struct SwinV2ContractConfig {
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

    /// The side of a square patch.
    #[config(default = "4")]
    pub patch_size: usize,

    /// The image channels (timm's `in_chans`).
    #[config(default = "3")]
    pub in_channels: usize,

    /// The window each stage's weights were trained at, if not this one (0:
    /// this one): timm's `pretrained_window_sizes`, the continuous position
    /// bias's scale. Empty for none.
    #[config(default = "vec![]")]
    pub pretrained_window_sizes: Vec<usize>,

    /// The MLP's hidden width over the token width, rounded down.
    #[config(default = "4.0")]
    pub mlp_ratio: f64,

    /// Whether the attention's query and value have biases (timm's
    /// `q_bias`, `v_bias`).
    #[config(default = "true")]
    pub qkv_bias: bool,

    /// The head's pool (timm's `global_pool`).
    #[config(default = "TokenPoolConfig::Avg")]
    pub global_pool: TokenPoolConfig,

    /// The dropout before the classifier (training only).
    #[config(default = "0.0")]
    pub drop_rate: f64,

    /// The dropout on the attention weights (training only).
    #[config(default = "0.0")]
    pub attn_drop_rate: f64,

    /// The stochastic depth's last rate, ramped linearly over all the blocks
    /// in order, as timm's `calculate_drop_path_rates` (training only).
    #[config(default = "0.0")]
    pub drop_path_rate: f64,
}

impl SwinV2ContractConfig {
    /// Stage `i`'s token width: `embed_dim * 2^i`.
    pub fn stage_width(
        &self,
        i: usize,
    ) -> usize {
        self.embed_dim << i
    }
}

impl ToStructureConfig for SwinV2ContractConfig {
    type Structure = StagedVisionEncoderStructureConfig;

    /// The staged vision encoder of this Swin V2: a patch embedding with a
    /// layer norm; stage `i` a patch merge (after the first), then `depths[i]`
    /// res-post-norm blocks of window attention (every second shifted;
    /// cosine attention; the continuous position bias, scaled by the stage's
    /// pretraining window if it has one; `q_bias` and `v_bias`, no key bias)
    /// and a `GELU` MLP; a final layer norm, the pool, the head.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for no stages, a
    /// head count per stage that doesn't match the depths or divide the
    /// stage's width, or a pretraining window per stage that doesn't match
    /// them; as the patch embedding's geometry.
    fn try_to_structure(&self) -> BunsenResult<StagedVisionEncoderStructureConfig> {
        let n = self.depths.len();
        if !self.pretrained_window_sizes.is_empty() && self.pretrained_window_sizes.len() != n {
            return Err(BunsenError::illegal(format!(
                "SwinV2ContractConfig: {} pretraining windows for {n} stages",
                self.pretrained_window_sizes.len()
            )));
        }
        let drop_paths = swin_block_drop_paths(&self.depths, self.drop_path_rate);
        SwinStages {
            config: "SwinV2ContractConfig",
            image_size: self.image_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            embed_dim: self.embed_dim,
            depths: &self.depths,
            num_heads: &self.num_heads,
            num_classes: self.num_classes,
            global_pool: self.global_pool,
            drop_rate: self.drop_rate,
            merge_norm_first: false,
        }
        .try_to_structure(|site| {
            let pretrained = match self.pretrained_window_sizes.get(site.stage) {
                Some(&p) if p > 0 => Some([p, p]),
                _ => None,
            };
            let width = site.width;
            Ok(CausalDecoderBlockConfig::new(
                WindowAttentionConfig::new(width, site.heads, site.grid, [self.window_size; 2])
                    .with_shifted(site.index % 2 == 1)
                    .with_qkv_bias(self.qkv_bias)
                    .with_key_bias(false)
                    .with_cosine(true)
                    .with_bias(RelativePositionBiasConfig::cpb(pretrained))
                    .with_attn_drop(self.attn_drop_rate)
                    .into(),
                BunsenMlpConfig::new(width, (width as f64 * self.mlp_ratio) as usize, width)
                    .with_activation(BunsenActivationConfig::Gelu)
                    .with_bias(true)
                    .into(),
            )
            .with_norm(swin_norm())
            .with_norm_placement(NormPlacement::ResPost)
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

    fn tiny() -> SwinV2ContractConfig {
        SwinV2ContractConfig::new(32, 16, vec![2, 2], vec![2, 4], 4, 10)
    }

    /// The stages halve the grid and double the width; the second's blocks
    /// shift.
    #[test]
    fn test_lowering() {
        let s = tiny().to_structure();
        assert_eq!(s.stage_grids(), [[8, 8], [4, 4]]);
        assert_eq!(s.stage_widths(), [16, 32]);
        assert!(s.stages[0].downsample.is_none() && s.stages[1].downsample.is_some());
        let model: StagedVisionEncoderModel = tiny().init(&cpu_device());
        let w = |i: usize, j: usize| {
            model.stages[i].stack.layers[j]
                .attn
                .branch
                .as_window()
                .unwrap()
                .shift
        };
        assert_eq!([w(0, 0), w(0, 1)], [[0, 0], [2, 2]]);
        assert_eq!(model.num_classes(), 10);
    }

    #[test]
    fn test_illegal_policies() {
        let illegal = |config: SwinV2ContractConfig, message: &str| {
            ErrorMatcher::illegal()
                .message_contains(message)
                .assert_err(&config.try_to_structure());
        };
        illegal(
            SwinV2ContractConfig::new(32, 16, vec![2, 2], vec![2], 4, 10),
            "2 stages of depths, 1 of heads",
        );
        illegal(
            SwinV2ContractConfig {
                num_heads: vec![3, 4],
                ..tiny()
            },
            "stage 0's 3 heads",
        );
        illegal(
            tiny().with_pretrained_window_sizes(vec![3]),
            "1 pretraining windows",
        );
        illegal(
            SwinV2ContractConfig {
                image_size: 30,
                ..tiny()
            },
            "don't divide into",
        );
    }
}
