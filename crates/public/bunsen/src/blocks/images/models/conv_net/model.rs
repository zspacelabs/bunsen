//! The conv net: its resolved structure and the model.

use alloc::{
    format,
    vec,
    vec::Vec,
};

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::images::models::conv_net::{
        ConvChainMeta,
        ConvHead,
        ConvHeadConfig,
        ConvStage,
        ConvStageConfig,
        ConvStageMeta,
        ConvStem,
        ConvStemConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
        WithOkOrPanic,
    },
    ops::drop::DropBlockOptions,
    support::validators::{
        expect_probability,
        try_probability,
    },
};

/// The view shared by [`ConvNetStructureConfig`] and [`ConvNetModel`]: what a
/// caller reads back without walking the stages.
pub trait ConvNetMeta {
    /// The image channels the stem takes.
    fn in_channels(&self) -> usize;

    /// The number of stages.
    fn num_stages(&self) -> usize;

    /// The channels of the last stage's map: the pooled features' width.
    fn d_features(&self) -> usize;

    /// The classes of the head.
    fn num_classes(&self) -> usize;
}

/// [`ConvNetModel`]'s structure: the stem, every stage's resolved units, and
/// the head.
///
/// The structure of the conv net's [Stacked
/// config](crate::burner::module::ModuleInit#stacked-config):
/// [`ConvNetContractConfig`](super::ConvNetContractConfig) and the families'
/// policies lower to it. It is the tree, unit by unit, for building a net no
/// policy names: edit it, then `init` it. `try_init` checks that it chains.
///
/// Implements [`ConvNetMeta`].
#[derive(Config, Debug)]
pub struct ConvNetStructureConfig {
    /// The stem.
    pub stem: ConvStemConfig,

    /// The stages, in order.
    pub stages: Vec<ConvStageConfig>,

    /// The classifier head.
    pub head: ConvHeadConfig,
}

impl ConvNetMeta for ConvNetStructureConfig {
    fn in_channels(&self) -> usize {
        self.stem.in_channels()
    }

    fn num_stages(&self) -> usize {
        self.stages.len()
    }

    fn d_features(&self) -> usize {
        self.head.d_features
    }

    fn num_classes(&self) -> usize {
        self.head.num_classes
    }
}

impl ConvNetStructureConfig {
    /// Checks the tree: a valid stem, stages that are not empty, each valid,
    /// the stem chaining into the first stage, each stage into the next, and
    /// the last into the head.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause; a stage's own error under a frame naming
    /// it.
    pub fn try_validate(&self) -> BunsenResult<()> {
        const OWNER: &str = "ConvNetStructureConfig";
        self.stem.try_validate().context("the stem")?;
        if self.stages.is_empty() {
            return Err(ConstraintError::zero_or_empty(OWNER, "stages").into());
        }
        for (idx, stage) in self.stages.iter().enumerate() {
            stage
                .try_validate()
                .with_context(|| format!("conv net stage {idx} (stages[{idx}])"))?;
        }
        // The stem is the chain's source; a break names the stage.
        ConstraintError::check_chain(
            OWNER,
            "stages",
            core::iter::once((0, self.stem.in_channels(), self.stem.out_channels())).chain(
                self.stages
                    .iter()
                    .enumerate()
                    .map(|(idx, s)| (idx, s.in_channels(), s.out_channels())),
            ),
        )?;
        let channels = self.stages[self.stages.len() - 1].out_channels();
        if self.head.d_features != channels {
            return Err(ConstraintError::new(
                OWNER,
                "head",
                Rule::Relation {
                    lhs: (
                        "head.d_features".into(),
                        format!("{}", self.head.d_features),
                    ),
                    op: "==",
                    rhs: (
                        "the last stage's out_channels".into(),
                        format!("{channels}"),
                    ),
                },
            )
            .into());
        }
        Ok(())
    }

    /// Applies timm's stochastic-depth schedule to every unit.
    ///
    /// Unit `i` of the net's `n` gets a drop-path probability of
    /// `drop_path_rate * i / (n - 1)`, counting every unit of every stage:
    /// 0 at the first, `drop_path_rate` at the last. This is the rule of
    /// `make_blocks` in timm's `resnet.py`, and the one
    /// [`ConvNetModel::with_stochastic_path_depth`] applies to a built model.
    ///
    /// # Panics
    ///
    /// If `drop_path_rate` is not a probability.
    pub fn with_stochastic_depth_drop_path_rate(
        self,
        drop_path_rate: f64,
    ) -> Self {
        let drop_path_rate = expect_probability(drop_path_rate);
        let n = self.stages.iter().map(|s| s.len()).sum::<usize>();
        let mut i = 0;
        let stages = self
            .stages
            .into_iter()
            .map(|stage| {
                stage.map_units(&mut |_, unit| {
                    let rate = stochastic_depth_rate(drop_path_rate, i, n);
                    i += 1;
                    unit.with_drop_path_prob(rate)
                })
            })
            .collect();
        Self { stages, ..self }
    }

    /// Applies timm's `DropBlock` schedule to every unit of the last two
    /// stages.
    ///
    /// Every unit of the second-to-last stage gets a [`DropBlockOptions`]
    /// with `drop_prob`, a block size of 5 and a gamma scale of 0.25; every
    /// unit of the last stage one with a block size of 3 and a gamma scale
    /// of 1.0. The other stages get none, and a `drop_prob` of 0 clears every
    /// unit. On a 4-stage net these are stages 3 and 4, the rule of
    /// `drop_blocks` in timm's `resnet.py`, and the one
    /// [`ConvNetModel::try_with_stochastic_drop_block`] applies to a built
    /// model.
    ///
    /// timm picks stages 3 and 4 by position, so a net without them gets no
    /// `DropBlock` there. This takes the last two stages instead, the same
    /// stages on a 4-stage net, and rejects a nonzero `drop_prob` on a net
    /// with fewer than 2 stages rather than ignore it.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if `drop_prob` is
    /// not a probability, or if it is above 0 and the net has fewer than 2
    /// stages.
    pub fn try_with_standard_drop_block_prob(
        self,
        drop_prob: f64,
    ) -> BunsenResult<Self> {
        let options = standard_drop_block_options(drop_prob, self.stages.len())?;
        Ok(self.with_drop_block_options(options))
    }

    /// The panicking twin of
    /// [`try_with_standard_drop_block_prob`](Self::try_with_standard_drop_block_prob).
    ///
    /// # Panics
    ///
    /// As its twin's errors.
    pub fn with_standard_drop_block_prob(
        self,
        drop_prob: f64,
    ) -> Self {
        self.try_with_standard_drop_block_prob(drop_prob)
            .ok_or_panic()
    }

    /// Sets each stage's drop block: `options` has one entry per stage.
    ///
    /// # Panics
    ///
    /// If `options` does not have one entry per stage.
    pub fn with_drop_block_options(
        self,
        options: Vec<Option<DropBlockOptions>>,
    ) -> Self {
        assert_eq!(options.len(), self.stages.len(), "one option per stage");
        let stages = self
            .stages
            .into_iter()
            .zip(options)
            .map(|(stage, options)| stage.with_drop_block(options))
            .collect();
        Self { stages, ..self }
    }
}

/// The stochastic-depth linear decay rule of timm's `make_blocks`
/// (`resnet.py`): unit `i` of a net of `n` gets `drop_path_rate * i / (n -
/// 1)`. A net of one unit gets 0, where timm divides by zero.
fn stochastic_depth_rate(
    drop_path_rate: f64,
    i: usize,
    n: usize,
) -> f64 {
    if n < 2 {
        return 0.0;
    }
    drop_path_rate * (i as f64) / ((n - 1) as f64)
}

/// The `DropBlock` schedule of timm's `drop_blocks` (`resnet.py`), one entry
/// per stage of a net of `num_stages`: the second-to-last stage gets a block
/// size of 5 and a gamma scale of 0.25, the last a block size of 3 and a
/// gamma scale of 1.0, and the others none. A `drop_prob` of 0 gives none.
///
/// # Errors
///
/// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if `drop_prob` is
/// not a probability, or if it is above 0 and `num_stages` is below 2.
fn standard_drop_block_options(
    drop_prob: f64,
    num_stages: usize,
) -> BunsenResult<Vec<Option<DropBlockOptions>>> {
    let drop_prob = try_probability(drop_prob).context("drop_prob")?;
    let mut options = vec![None; num_stages];
    if drop_prob > 0.0 {
        if num_stages < 2 {
            return Err(BunsenError::illegal(format!(
                "the standard DropBlock schedule needs at least 2 stages, for its last two; got {num_stages}"
            )));
        }
        options[num_stages - 2] = DropBlockOptions::default()
            .with_drop_prob(drop_prob)
            .with_block_size(5)
            .with_gamma_scale(0.25)
            .into();
        options[num_stages - 1] = DropBlockOptions::default()
            .with_drop_prob(drop_prob)
            .with_block_size(3)
            .with_gamma_scale(1.0)
            .into();
    }
    Ok(options)
}

impl ModuleInit<ConvNetModel> for ConvNetStructureConfig {
    /// Builds the model.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](ConvNetStructureConfig::try_validate).
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConvNetModel> {
        self.try_validate()?;
        Ok(ConvNetModel {
            stem: self.stem.try_init(device)?,
            stages: self
                .stages
                .iter()
                .map(|s| s.try_init(device))
                .collect::<BunsenResult<_>>()?,
            head: self.head.try_init(device)?,
        })
    }
}

/// The generic conv net: a [`ConvStem`], [`ConvStage`]s of
/// [`ResidualUnit`](super::ResidualUnit)s, and a [`ConvHead`].
///
/// Every conv-net family is this type, built from its policy:
/// `ResNetContractConfig::init` returns one. [`forward`](Self::forward) maps
/// `[batch, in_channels, height, width]` images to `[batch, num_classes]`
/// logits; [`feature_map`](Self::feature_map),
/// [`pooled_features`](Self::pooled_features) and
/// [`stage_maps`](Self::stage_maps) are the steps on the way.
///
/// Implements [`ConvNetMeta`].
#[derive(Module, Debug)]
pub struct ConvNetModel {
    /// The stem.
    pub stem: ConvStem,

    /// The stages, in order.
    pub stages: Vec<ConvStage>,

    /// The classifier head.
    pub head: ConvHead,
}

impl ConvNetMeta for ConvNetModel {
    fn in_channels(&self) -> usize {
        self.stem.in_channels()
    }

    fn num_stages(&self) -> usize {
        self.stages.len()
    }

    fn d_features(&self) -> usize {
        self.head.d_features()
    }

    fn num_classes(&self) -> usize {
        self.head.num_classes()
    }
}

impl ConvNetModel {
    /// The logits: the head over the last stage's map.
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, num_classes]`.
    pub fn forward(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2> {
        self.head.forward(self.feature_map(images))
    }

    /// The last stage's map: the stem, then every stage.
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, d_features, height / stride, width / stride]`, each side
    /// rounded up; the stride is 32 on a `ResNet`.
    pub fn feature_map(
        &self,
        images: Tensor<4>,
    ) -> Tensor<4> {
        let x = self.stem.forward(images);
        self.stages.iter().fold(x, |x, stage| stage.forward(x))
    }

    /// The pooled features, the head's input.
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, pooled]`: `d_features` wide, or twice it for a
    /// [`CatAvgMax`](super::GlobalPool2dConfig::CatAvgMax) pool.
    pub fn pooled_features(
        &self,
        images: Tensor<4>,
    ) -> Tensor<2> {
        self.head.pool(self.feature_map(images))
    }

    /// The feature maps by stage, timm's `feature_info`: the stem's convs
    /// (before its pool), then each stage's output.
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `num_stages + 1` maps, `[batch, channels, h, w]` each.
    pub fn stage_maps(
        &self,
        images: Tensor<4>,
    ) -> Vec<Tensor<4>> {
        let stem = self.stem.forward_convs(images);
        let mut maps = vec![stem.clone()];
        let mut x = self.stem.forward_pool(stem);
        for stage in &self.stages {
            x = stage.forward(x);
            maps.push(x.clone());
        }
        maps
    }

    /// The model with its head re-initialized for `num_classes` classes.
    pub fn with_classes(
        self,
        num_classes: usize,
    ) -> Self {
        Self {
            head: self.head.with_classes(num_classes),
            ..self
        }
    }

    /// Applies timm's stochastic-depth schedule to every unit: the rule
    /// [`ConvNetStructureConfig::with_stochastic_depth_drop_path_rate`]
    /// applies to a structure.
    ///
    /// # Panics
    ///
    /// If `drop_path_rate` is not a probability.
    pub fn with_stochastic_path_depth(
        self,
        drop_path_rate: f64,
    ) -> Self {
        let drop_path_rate = expect_probability(drop_path_rate);
        let n = self.stages.iter().map(|s| s.len()).sum::<usize>();
        let mut i = 0;
        self.map_stages(|stages| {
            stages
                .into_iter()
                .map(|stage| {
                    stage.map_units(&mut |_, unit| {
                        let rate = stochastic_depth_rate(drop_path_rate, i, n);
                        i += 1;
                        unit.with_drop_path_prob(rate)
                    })
                })
                .collect()
        })
    }

    /// Applies timm's `DropBlock` schedule to every unit of the last two
    /// stages: the rule
    /// [`ConvNetStructureConfig::try_with_standard_drop_block_prob`] applies
    /// to a structure, which also says why a net with fewer than 2 stages is
    /// an error.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if `drop_prob` is
    /// not a probability, or if it is above 0 and the model has fewer than 2
    /// stages.
    pub fn try_with_stochastic_drop_block(
        self,
        drop_prob: f64,
    ) -> BunsenResult<Self> {
        let options = standard_drop_block_options(drop_prob, self.stages.len())?;
        Ok(self.with_drop_block_options(options))
    }

    /// The panicking twin of
    /// [`try_with_stochastic_drop_block`](Self::try_with_stochastic_drop_block).
    ///
    /// # Panics
    ///
    /// As its twin's errors.
    pub fn with_stochastic_drop_block(
        self,
        drop_prob: f64,
    ) -> Self {
        self.try_with_stochastic_drop_block(drop_prob).ok_or_panic()
    }

    /// Sets each stage's drop block: `options` has one entry per stage.
    ///
    /// # Panics
    ///
    /// If `options` does not have one entry per stage.
    pub fn with_drop_block_options(
        self,
        options: Vec<Option<DropBlockOptions>>,
    ) -> Self {
        assert_eq!(options.len(), self.stages.len(), "one option per stage");
        self.map_stages(|stages| {
            stages
                .into_iter()
                .zip(options)
                .map(|(stage, options)| stage.with_drop_block(options))
                .collect()
        })
    }

    /// Applies `f` to the stages.
    pub fn map_stages<F>(
        self,
        f: F,
    ) -> Self
    where
        F: FnOnce(Vec<ConvStage>) -> Vec<ConvStage>,
    {
        Self {
            stages: f(self.stages),
            ..self
        }
    }

    /// Freezes the stages (not the stem or the head): no gradients for their
    /// parameters.
    pub fn freeze_stages(self) -> Self {
        self.map_stages(|stages| stages.into_iter().map(|stage| stage.no_grad()).collect())
    }
}
