//! The staged vision encoder: its structure and the model.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::{
        images::{
            models::classifier_head::{
                ClassifierHead,
                ClassifierHeadConfig,
            },
            patching::{
                patch_embed::{
                    PatchEmbed,
                    PatchEmbedConfig,
                    PatchEmbedMeta,
                },
                patch_merge::{
                    PatchMerge,
                    PatchMergeConfig,
                },
            },
        },
        norm::{
            BunsenNormalization,
            BunsenNormalizationConfig,
        },
        transformers::{
            decoders::stack::{
                DecoderStack,
                DecoderStackMeta,
                DecoderStackStructureConfig,
            },
            models::vision_encoder::{
                TokenPool,
                TokenPoolConfig,
            },
        },
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
};

/// The view shared by [`StagedVisionEncoderStructureConfig`] and
/// [`StagedVisionEncoderModel`].
pub trait StagedVisionEncoderMeta {
    /// The image channels the patch embedding takes.
    fn in_channels(&self) -> usize;

    /// The `[height, width]` of the images it takes.
    fn image_size(&self) -> [usize; 2];

    /// Each stage's `[height, width]` grid, in tokens.
    fn stage_grids(&self) -> Vec<[usize; 2]>;

    /// Each stage's token width.
    fn stage_widths(&self) -> Vec<usize>;

    /// The classes of the head; 0 for a headless model.
    fn num_classes(&self) -> usize;

    /// The width of the pooled features: the last stage's.
    fn d_features(&self) -> usize {
        self.stage_widths().last().copied().unwrap_or(0)
    }
}

/// One stage's structure: an optional patch merge, then the blocks.
#[derive(Config, Debug)]
pub struct VisionStageConfig {
    /// The merge that starts the stage (none on the first).
    pub downsample: Option<PatchMergeConfig>,

    /// The blocks, over the stage's grid; no final norm of their own.
    pub stack: DecoderStackStructureConfig,
}

/// [`StagedVisionEncoderModel`]'s structure: the patch embedding, the
/// stages, the final norm, the pool and the head.
///
/// A family's policy (`SwinV2ContractConfig`) lowers to it. `try_init`
/// checks that the parts agree: each stage's merge over the grid before it,
/// one width through a stage, the head the last stage's width.
///
/// Implements [`StagedVisionEncoderMeta`].
#[derive(Config, Debug)]
pub struct StagedVisionEncoderStructureConfig {
    /// The patch embedding.
    pub patch_embed: PatchEmbedConfig,

    /// The stages, in order.
    pub stages: Vec<VisionStageConfig>,

    /// The norm after the last stage, sized at init.
    pub norm: BunsenNormalizationConfig,

    /// The pool over the last grid.
    #[config(default = "TokenPoolConfig::Avg")]
    pub pool: TokenPoolConfig,

    /// The head, over the pooled features.
    pub head: ClassifierHeadConfig,
}

impl StagedVisionEncoderMeta for StagedVisionEncoderStructureConfig {
    fn in_channels(&self) -> usize {
        self.patch_embed.d_input()
    }

    fn image_size(&self) -> [usize; 2] {
        self.patch_embed.input_resolution()
    }

    fn stage_grids(&self) -> Vec<[usize; 2]> {
        let mut grid = self.patch_embed.patches_resolution();
        self.stages
            .iter()
            .map(|stage| {
                if let Some(merge) = &stage.downsample {
                    grid = merge.output_grid();
                }
                grid
            })
            .collect()
    }

    fn stage_widths(&self) -> Vec<usize> {
        self.stages.iter().map(|s| s.stack.d_model()).collect()
    }

    fn num_classes(&self) -> usize {
        self.head.num_classes
    }
}

impl StagedVisionEncoderStructureConfig {
    /// Checks that the parts agree.
    ///
    /// See the [errors convention](crate::errors#convention-try_x-and-x).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for no stages; a
    /// merge over another grid than the one before it, or from another
    /// width; a stage whose blocks are not its merge's width (or the patch
    /// embedding's); a window block over another grid than its stage's; a
    /// stack with attention-residual routes; a head not the last stage's
    /// width, or distilled; a pool that reads prefix tokens; as the patch
    /// embedding's geometry.
    pub fn try_validate(&self) -> BunsenResult<()> {
        self.patch_embed.try_validate()?;
        if self.stages.is_empty() {
            return Err(ConstraintError::zero_or_empty(
                "StagedVisionEncoderStructureConfig",
                "stages",
            )
            .into());
        }
        let mut grid = self.patch_embed.patches_resolution();
        let mut width = self.patch_embed.d_output();
        for (i, stage) in self.stages.iter().enumerate() {
            if let Some(merge) = &stage.downsample {
                if merge.grid != grid || merge.d_input != width {
                    return Err(ConstraintError::new(
                        "StagedVisionEncoderStructureConfig",
                        "stages",
                        Rule::Chain {
                            index: i,
                            out: format!("a {grid:?} grid {width} wide"),
                            r#in: format!(
                                "a {:?} grid {} wide, to its merge",
                                merge.grid, merge.d_input
                            ),
                        },
                    )
                    .into());
                }
                grid = merge.output_grid();
                width = merge.d_output;
            }
            if stage.stack.d_model() != width {
                return Err(ConstraintError::new(
                    "StagedVisionEncoderStructureConfig",
                    "stages",
                    Rule::Chain {
                        index: i,
                        out: format!("{width} wide"),
                        r#in: format!("{} wide, to its blocks", stage.stack.d_model()),
                    },
                )
                .into());
            }
            if stage.stack.routes.is_some() {
                return Err(ConstraintError::custom(
                    "StagedVisionEncoderStructureConfig",
                    format!("stages[{i}].stack.routes"),
                    "a vision encoder's blocks have no attention-residual routes",
                )
                .into());
            }
            for (j, layer) in stage.stack.layers.iter().enumerate() {
                if let crate::blocks::transformers::mixers::TokenMixerConfig::Window(w) =
                    &layer.attn
                    && w.grid != grid
                {
                    return Err(ConstraintError::new(
                        "StagedVisionEncoderStructureConfig",
                        format!("stages[{i}].stack.layers[{j}].attn.grid"),
                        Rule::Relation {
                            lhs: ("the block's grid".into(), format!("{:?}", w.grid)),
                            op: "==",
                            rhs: ("the stage's grid".into(), format!("{grid:?}")),
                        },
                    )
                    .into());
                }
            }
        }
        if self.head.d_features != width || self.head.distilled {
            return Err(ConstraintError::custom(
                "StagedVisionEncoderStructureConfig",
                "head",
                format!(
                    "the head reads {} features{}; the last stage is {width} wide, and a staged \
                 encoder's head is not distilled",
                    self.head.d_features,
                    if self.head.distilled {
                        ", distilled"
                    } else {
                        ""
                    }
                ),
            )
            .into());
        }
        self.pool
            .try_check(0)
            .context("the pool, over a grid with no prefix")?;
        Ok(())
    }
}

impl ModuleInit<StagedVisionEncoderModel> for StagedVisionEncoderStructureConfig {
    /// Builds the model.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](StagedVisionEncoderStructureConfig::try_validate),
    /// and as each part's `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<StagedVisionEncoderModel> {
        self.try_validate()?;
        let width = self.stage_widths().last().copied().unwrap_or(0);
        Ok(StagedVisionEncoderModel {
            patch_embed: self.patch_embed.try_init(device)?,
            stages: self
                .stages
                .iter()
                .enumerate()
                .map(|(i, stage)| {
                    Ok(VisionStage {
                        downsample: stage
                            .downsample
                            .as_ref()
                            .map(|m| m.try_init(device))
                            .transpose()?,
                        stack: stage.stack.try_init(device)?,
                    })
                    .with_context(|| format!("stage {i}"))
                })
                .collect::<BunsenResult<_>>()?,
            norm: self
                .norm
                .clone()
                .with_num_features(width)
                .try_init(device)?,
            pool: self.pool.try_init(0, device)?,
            head: self.head.try_init(device).context("the head")?,
        })
    }
}

/// One stage of a [`StagedVisionEncoderModel`]: an optional patch merge,
/// then the blocks over the stage's grid.
#[derive(Module, Debug)]
pub struct VisionStage {
    /// The merge that starts the stage.
    pub downsample: Option<PatchMerge>,

    /// The blocks.
    pub stack: DecoderStack,
}

impl VisionStage {
    /// The stage over `x`, `[batch, tokens, d_in]`: the merge, then the
    /// blocks.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let x = match &self.downsample {
            Some(merge) => merge.forward(x),
            None => x,
        };
        self.stack.encoder_forward(x, None)
    }
}

/// The generic staged vision encoder: a [`PatchEmbed`], [`VisionStage`]s,
/// a final norm, a [`TokenPool`] and a [`ClassifierHead`]: timm's Swin
/// Transformer (V1 and V2) shape.
///
/// [`forward`](Self::forward) maps `[batch, in_channels, height, width]`
/// images to `[batch, num_classes]` logits;
/// [`feature_tokens`](Self::feature_tokens),
/// [`pooled_features`](Self::pooled_features) and
/// [`stage_maps`](Self::stage_maps) are the steps on the way. The model runs
/// at its own image size: each stage's windows are fixed to its grid.
///
/// Implements [`StagedVisionEncoderMeta`].
#[derive(Module, Debug)]
pub struct StagedVisionEncoderModel {
    /// The patch embedding.
    pub patch_embed: PatchEmbed,

    /// The stages, in order.
    pub stages: Vec<VisionStage>,

    /// The norm after the last stage.
    pub norm: BunsenNormalization,

    /// The pool over the last grid.
    pub pool: TokenPool,

    /// The head, over the pooled features.
    pub head: ClassifierHead,
}

impl StagedVisionEncoderMeta for StagedVisionEncoderModel {
    fn in_channels(&self) -> usize {
        self.patch_embed.d_input()
    }

    fn image_size(&self) -> [usize; 2] {
        self.patch_embed.input_resolution()
    }

    fn stage_grids(&self) -> Vec<[usize; 2]> {
        let mut grid = self.patch_embed.patches_resolution();
        self.stages
            .iter()
            .map(|stage| {
                if let Some(merge) = &stage.downsample {
                    grid = merge.output_grid();
                }
                grid
            })
            .collect()
    }

    fn stage_widths(&self) -> Vec<usize> {
        self.stages.iter().map(|s| s.stack.d_model()).collect()
    }

    fn num_classes(&self) -> usize {
        self.head.num_classes()
    }
}

/// Each stage's transformer's canonical names under `stages.{i}.stack.`;
/// every other path is its own.
impl CanonicalKeys for StagedVisionEncoderModel {
    fn canonical_rekey(&self) -> Rekey {
        self.stages
            .iter()
            .enumerate()
            .fold(Rekey::keep_rest(), |rekey, (i, stage)| {
                rekey.mount_canonical(format!("stages.{i}.stack."), &stage.stack)
            })
    }
}

impl StagedVisionEncoderModel {
    /// The logits: the head over the pooled features; for a headless model,
    /// the pooled features.
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, num_classes]`, or `[batch, d_features]` for a headless model.
    pub fn forward(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2> {
        #[cfg(any(debug_assertions, test))]
        let [batch] = self.unpack_images(&images);
        let logits = self
            .head
            .forward(self.pool.forward(self.feature_tokens(images)));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_output"],
            &logits.dims(),
            &[("batch", batch), ("d_output", self.head.d_output())]
        );
        logits
    }

    /// The batch of `images`, checked against the model's input:
    /// `[batch, in_channels, height, width]`.
    #[cfg(any(debug_assertions, test))]
    fn unpack_images(
        &self,
        images: &Tensor<4>,
    ) -> [usize; 1] {
        let [height, width] = self.image_size();
        crate::contracts::unpack_shape_contract!(
            ["batch", "in_channels", "height", "width"],
            &images.dims(),
            &["batch"],
            &[
                ("in_channels", self.in_channels()),
                ("height", height),
                ("width", width)
            ]
        )
    }

    /// The last stage's tokens, normed (timm's `forward_features`).
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, grid_h * grid_w, d_features]`, the last grid row by row.
    pub fn feature_tokens(
        &self,
        images: Tensor<4>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        let [batch] = self.unpack_images(&images);
        let x = self.patch_embed.forward(images);
        let x = self.stages.iter().fold(x, |x, stage| stage.forward(x));
        let x = self.norm.forward(x);
        #[cfg(any(debug_assertions, test))]
        {
            let [grid_h, grid_w] = *self.stage_grids().last().expect("a stage");
            crate::contracts::assert_shape_contract_periodically!(
                ["batch", "grid_h" * "grid_w", "d_features"],
                &x.dims(),
                &[
                    ("batch", batch),
                    ("grid_h", grid_h),
                    ("grid_w", grid_w),
                    ("d_features", self.d_features())
                ]
            );
        }
        x
    }

    /// The pooled features, the classifier's input.
    ///
    /// # Returns
    ///
    /// `[batch, d_features]`.
    pub fn pooled_features(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2> {
        #[cfg(any(debug_assertions, test))]
        let [batch] = self.unpack_images(&images);
        let pooled = self
            .head
            .pre_logits(self.pool.forward(self.feature_tokens(images)));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_features"],
            &pooled.dims(),
            &[("batch", batch), ("d_features", self.d_features())]
        );
        pooled
    }

    /// Each stage's output as a map, timm's `forward_intermediates` (no final
    /// norm).
    ///
    /// # Returns
    ///
    /// One `[batch, width_i, grid_h_i, grid_w_i]` map per stage.
    pub fn stage_maps(
        &self,
        images: Tensor<4>,
    ) -> Vec<Tensor<4>> {
        let grids = self.stage_grids();
        let mut x = self.patch_embed.forward(images);
        let mut maps = Vec::with_capacity(self.stages.len());
        for (stage, [h, w]) in self.stages.iter().zip(grids) {
            x = stage.forward(x);
            // The stage's tokens are its grid, row by row.
            let [b, d] = crate::contracts::unpack_shape_contract!(
                ["batch", "grid_h" * "grid_w", "width"],
                &x.dims(),
                &["batch", "width"],
                &[("grid_h", h), ("grid_w", w)]
            );
            maps.push(x.clone().reshape([b, h, w, d]).permute([0, 3, 1, 2]));
        }
        maps
    }

    /// The model with its head re-initialized for `num_classes` classes (0:
    /// headless).
    pub fn with_classes(
        self,
        num_classes: usize,
    ) -> Self {
        let device = self.patch_embed.projection.weight.device();
        Self {
            head: self.head.with_classes(num_classes, &device),
            ..self
        }
    }
}
