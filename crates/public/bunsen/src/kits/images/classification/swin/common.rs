use alloc::{
    format,
    vec,
    vec::Vec,
};

use crate::{
    blocks::{
        images::{
            drop::rate_table::linear_drop_path_rate_schedule,
            models::classifier_head::ClassifierHeadConfig,
            patching::{
                patch_embed::PatchEmbedConfig,
                patch_merge::PatchMergeConfig,
            },
        },
        norm::BunsenNormalizationConfig,
        transformers::{
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::DecoderStackStructureConfig,
            },
            embedding::RotaryEmbeddingConfig,
            models::{
                staged_vision_encoder::{
                    StagedVisionEncoderStructureConfig,
                    VisionStageConfig,
                },
                vision_encoder::{
                    TokenPoolConfig,
                    TokenPoolKind,
                },
            },
        },
    },
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// The epsilon of Swin's default norm: `nn.LayerNorm`'s default, which timm
/// keeps.
pub const SWIN_NORM_EPSILON: f64 = 1e-5;

/// Each block's stochastic depth, by stage: one linear ramp from 0 to `rate`
/// over all the blocks, split at the stages (timm's
/// `calculate_drop_path_rates`).
pub(crate) fn swin_block_drop_paths(
    depths: &[usize],
    rate: f64,
) -> Vec<Vec<f64>> {
    let total: usize = depths.iter().sum();
    let ramp = match rate {
        0.0 => vec![0.0; total],
        rate => linear_drop_path_rate_schedule(rate, total),
    };
    let mut start = 0;
    depths
        .iter()
        .map(|&depth| {
            start += depth;
            ramp[start - depth..start].to_vec()
        })
        .collect()
}

/// What a Swin's stages share, V1's and V2's: the knobs their lowerings
/// read alike.
pub(crate) struct SwinStages<'a> {
    /// The policy's name, for errors.
    pub config: &'static str,
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub depths: &'a [usize],
    pub num_heads: &'a [usize],
    pub num_classes: usize,
    pub global_pool: TokenPoolKind,
    /// Every norm, its width filled in where it sits.
    pub norm: &'a BunsenNormalizationConfig,
    pub drop_rate: f64,
    /// Whether a merge norms before its reduction (V1), not after (V2).
    pub merge_norm_first: bool,
}

/// Where a block sits: what its config is built from.
pub(crate) struct SwinBlockSite {
    /// Its stage.
    pub stage: usize,
    /// Its place in the stage.
    pub index: usize,
    /// The stage's token width.
    pub width: usize,
    /// The stage's heads.
    pub heads: usize,
    /// The stage's grid.
    pub grid: [usize; 2],
}

impl SwinStages<'_> {
    /// The staged vision encoder: a patch embedding with a layer norm; stage
    /// `i` `embed_dim * 2^i` wide, a patch merge opening it after the first,
    /// then `depths[i]` blocks of `block`; a final layer norm, the pool, the
    /// head.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for no stages, a
    /// head count per stage that doesn't match the depths or divide the
    /// stage's width; as the patch embedding's geometry; as `block`.
    pub fn try_to_structure(
        &self,
        block: impl Fn(&SwinBlockSite) -> BunsenResult<CausalDecoderBlockConfig>,
    ) -> BunsenResult<StagedVisionEncoderStructureConfig> {
        let (config, n) = (self.config, self.depths.len());
        if n == 0 {
            return Err(ConstraintError::zero_or_empty(config, "depths").into());
        }
        if self.num_heads.len() != n {
            return Err(ConstraintError::new(
                config,
                "num_heads",
                Rule::Relation {
                    lhs: ("num_heads.len()".into(), self.num_heads.len().to_string()),
                    op: "==",
                    rhs: ("depths.len()".into(), n.to_string()),
                },
            )
            .into());
        }
        let patch_embed = PatchEmbedConfig::new(
            [self.image_size; 2],
            self.patch_size,
            self.in_channels,
            self.embed_dim,
        )
        .with_norm(Some(self.norm.clone()));
        patch_embed.try_validate()?;
        let width = |i: usize| self.embed_dim << i;
        let mut grid = [self.image_size / self.patch_size; 2];
        let mut stages = Vec::with_capacity(n);
        for stage in 0..n {
            let heads = self.num_heads[stage];
            if heads == 0 {
                return Err(
                    ConstraintError::zero_or_empty(config, format!("num_heads[{stage}]")).into(),
                );
            }
            if !width(stage).is_multiple_of(heads) {
                return Err(ConstraintError::new(
                    config,
                    format!("num_heads[{stage}]"),
                    Rule::NotMultiple {
                        value: width(stage),
                        of: format!("stage {stage}'s heads ({heads})"),
                    },
                )
                .into());
            }
            let downsample = (stage > 0).then(|| {
                PatchMergeConfig::new(grid, width(stage - 1), width(stage), self.norm.clone())
                    .with_norm_first(self.merge_norm_first)
            });
            if let Some(merge) = &downsample {
                grid = merge.output_grid();
            }
            let layers = (0..self.depths[stage])
                .map(|index| {
                    block(&SwinBlockSite {
                        stage,
                        index,
                        width: width(stage),
                        heads,
                        grid,
                    })
                })
                .collect::<BunsenResult<Vec<_>>>()?;
            stages.push(
                VisionStageConfig::new(DecoderStackStructureConfig::new(
                    layers,
                    BunsenNormalizationConfig::Identity,
                    RotaryEmbeddingConfig::unrotated(width(stage) / heads),
                ))
                .with_downsample(downsample),
            );
        }
        Ok(StagedVisionEncoderStructureConfig::new(
            patch_embed,
            stages,
            self.norm.clone(),
            ClassifierHeadConfig::new(width(n - 1), self.num_classes)
                .with_drop_rate(self.drop_rate),
        )
        .with_pool(TokenPoolConfig::try_from_kind(self.global_pool)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The drop path ramps over all the blocks, split at the stages.
    #[test]
    fn test_block_drop_paths() {
        let rates = swin_block_drop_paths(&[2, 2, 6, 2], 0.22);
        assert_eq!(rates.iter().map(Vec::len).collect::<Vec<_>>(), [2, 2, 6, 2]);
        for (k, rate) in rates.iter().flatten().enumerate() {
            assert!((rate - 0.02 * k as f64).abs() < 1e-12, "{rates:?}");
        }
        assert!(
            swin_block_drop_paths(&[2, 2], 0.0)
                .iter()
                .flatten()
                .all(|&r| r == 0.0)
        );
    }
}
