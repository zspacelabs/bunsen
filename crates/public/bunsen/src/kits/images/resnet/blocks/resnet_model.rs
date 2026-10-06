//! # `ResNet` Core Model

use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    module::Module,
    nn::{
        BatchNormConfig,
        Linear,
        LinearConfig,
        PaddingConfig2d,
        activation::ActivationConfig,
        conv::Conv2dConfig,
        norm::NormalizationConfig,
        pool::{
            AdaptiveAvgPool2d,
            AdaptiveAvgPool2dConfig,
            MaxPool2d,
            MaxPool2dConfig,
        },
    },
    prelude::{
        Backend,
        Config,
        Tensor,
    },
};

use crate::{
    blocks::conv::{
        ConvBlock2d,
        ConvBlock2dConfig,
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        ResultContext,
        WithOkOrPanic,
    },
    kits::images::resnet::{
        RESNET18_BLOCKS,
        blocks::{
            BottleneckPolicyConfig,
            LayerBlock,
            LayerBlockContractConfig,
            LayerBlockMeta,
            LayerBlockStructureConfig,
            ResidualBlock,
            ResidualBlockStructureConfig,
        },
    },
    ops::{
        conv::CONV_INTO_RELU_INITIALIZER,
        drop::DropBlockOptions,
    },
    support::validators::{
        expect_probability,
        try_probability,
    },
};

/// High-level [`ResNet`] model configuration: the policy of `ResNet`'s
/// Stacked Config.
///
/// The user-facing entry point for building a [`ResNet`]: stage depths, class
/// count, stem width, output stride, and bottleneck policy. It implements
/// [`ToStructureConfig`], lowering to the unrolled [`ResNetStructureConfig`],
/// and gets [`ModuleInit`] from that trait's blanket impl: call
/// `.init(device)` to lower and build the [`ResNet`] in one step, then drive
/// it with [`ResNet::forward`].
#[derive(Config, Debug)]
pub struct ResNetContractConfig {
    /// Layer block depths.
    /// Must have the same length as `channels`.
    pub layers: Vec<usize>,

    /// Number of classification classes.
    pub num_classes: usize,

    /// Number of channels in stem convolutions.
    /// TODO: Replace with a ``ResNetStem`` module.
    #[config(default = "64")]
    pub stem_width: usize,

    /// Output stride.
    #[config(default = "32")]
    pub output_stride: usize,

    /// When enabled, select [`BottleneckBlock`](`super::BottleneckBlock`);
    /// Otherwise, select [`BasicBlock`](`super::BasicBlock`).
    #[config(default = "None")]
    pub bottleneck_policy: Option<BottleneckPolicyConfig>,

    /// Normalization config.
    ///
    /// The feature size of this config will be replaced
    /// with the appropriate feature size for the input layer.
    #[config(default = "NormalizationConfig::Batch(BatchNormConfig::new(0))")]
    pub normalization: NormalizationConfig,

    /// Activation config.
    #[config(default = "ActivationConfig::Relu")]
    pub activation: ActivationConfig,
}

impl ResNetContractConfig {
    /// Enables default bottleneck policy.
    pub fn with_bottleneck(
        self,
        enable: bool,
    ) -> Self {
        let policy = if enable {
            Some(Default::default())
        } else {
            None
        };
        self.with_bottleneck_policy(policy)
    }

    /// Builds the [`LayerBlockContractConfig`] stack.
    #[allow(unused)]
    pub fn to_layer_contracts(&self) -> Vec<LayerBlockContractConfig> {
        let mut net_stride = 4;
        let mut dilation = 1;
        let mut prev_dilation = 1;
        let mut layers: Vec<LayerBlockContractConfig> = Default::default();
        let mut in_planes = self.stem_width;
        for (stage_idx, &num_blocks) in self.layers.iter().enumerate() {
            let downsample_input = {
                let mut stride = if stage_idx == 0 { 1 } else { 2 };
                if net_stride >= self.output_stride {
                    dilation *= stride;
                    stride = 1;
                } else {
                    net_stride *= stride;
                }
                stride != 1
            };

            let first_dilation = prev_dilation;

            let out_planes = if stage_idx == 0 {
                match &self.bottleneck_policy {
                    Some(policy) => in_planes * policy.pinch_factor,
                    None => in_planes,
                }
            } else {
                2 * in_planes
            };

            layers.push(
                LayerBlockContractConfig::new(num_blocks, in_planes, out_planes)
                    .with_downsample_input(downsample_input)
                    .with_first_dilation(Some(first_dilation))
                    .with_dilation(dilation)
                    .with_bottleneck_policy(self.bottleneck_policy.clone())
                    .with_normalization(self.normalization.clone())
                    .with_activation(self.activation.clone()),
            );

            in_planes = out_planes;
            prev_dilation = dilation;
        }

        layers
    }

    /// Creates a ResNet-18 model.
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(RESNET18_BLOCKS.to_vec(), num_classes) // .with_bottleneck(true)
    }
}

impl ToStructureConfig for ResNetContractConfig {
    type Structure = ResNetStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<ResNetStructureConfig> {
        Ok(ResNetStructureConfig::new(
            ConvBlock2dConfig::new(
                Conv2dConfig::new([3, self.stem_width], [7, 7])
                    .with_stride([2, 2])
                    .with_padding({
                        let d = 3;
                        PaddingConfig2d::Explicit(d, d, d, d)
                    })
                    .with_bias(false)
                    .with_initializer(CONV_INTO_RELU_INITIALIZER.clone()),
            )
            .with_norm(Some(BatchNormConfig::new(self.stem_width).into()))
            .with_act(Some(self.activation.clone())),
            self.to_layer_contracts()
                .iter()
                .map(|c| c.try_to_structure())
                .collect::<BunsenResult<Vec<_>>>()?,
            self.num_classes,
        ))
    }
}

/// [`ResNet`] Meta API.
///
/// The narrow view shared by [`ResNetStructureConfig`] and [`ResNet`]: what a
/// caller or a test reads back without walking the stages.
pub trait ResNetMeta {
    /// The number of [`LayerBlock`] stages.
    fn num_stages(&self) -> usize;

    /// The feature planes the classifier head reads: the last stage's output
    /// planes.
    ///
    /// # Panics
    ///
    /// On a [`ResNetStructureConfig`] with no stages, or whose last stage has
    /// no blocks.
    fn head_planes(&self) -> usize;

    /// The number of classification classes.
    fn num_classes(&self) -> usize;
}

/// [`ResNet`] Structure Config.
///
/// This config defines the structure of a converted [`ResNet`] model.
/// It is not a semantic configuration: `try_init` rejects a structure with no
/// stages, or with a stage that is empty or whose blocks' planes do not chain,
/// but does not check that one stage's planes chain into the next.
///
/// Holds the explicit stem, per-stage [`LayerBlockStructureConfig`]s, and head.
/// [`ResNetContractConfig`] lowers to it. Call `.init(device)` to build the
/// [`ResNet`] module, then drive it with [`ResNet::forward`].
///
/// Implements [`ResNetMeta`].
#[derive(Config, Debug)]
pub struct ResNetStructureConfig {
    /// The input Conv/Norm block configuration.
    pub input_cb: ConvBlock2dConfig,

    /// The inner layers configuration.
    pub layers: Vec<LayerBlockStructureConfig>,

    /// The number of classes.
    pub num_classes: usize,
}

impl ResNetMeta for ResNetStructureConfig {
    fn num_stages(&self) -> usize {
        self.layers.len()
    }

    fn head_planes(&self) -> usize {
        self.layers.last().unwrap().out_planes()
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

impl ResNetStructureConfig {
    /// Applies timm's `DropBlock` schedule to every block of the last two
    /// stages.
    ///
    /// Every block of the second-to-last stage gets a [`DropBlockOptions`]
    /// with `drop_prob`, a block size of 5 and a gamma scale of 0.25. Every
    /// block of the last stage gets one with a block size of 3 and a gamma
    /// scale of 1.0. The other stages get none, and a `drop_prob` of 0
    /// clears every block. On the standard 4-stage net these are stages 3
    /// and 4, the rule of `drop_blocks` and `make_blocks` in timm's
    /// `resnet.py`, and the one [`ResNet::try_with_stochastic_drop_block`]
    /// applies to a built model.
    ///
    /// timm picks stages 3 and 4 by position, so a net without them gets no
    /// `DropBlock` there. This method takes the last two stages instead,
    /// which are the same stages on a 4-stage net, and rejects a nonzero
    /// `drop_prob` on a net with fewer than 2 stages rather than ignore it.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if `drop_prob` is
    /// not a probability, or if it is above 0 and the structure has fewer
    /// than 2 stages.
    pub fn try_with_standard_drop_block_prob(
        self,
        drop_prob: f64,
    ) -> BunsenResult<Self> {
        let options = standard_drop_block_options(drop_prob, self.layers.len())?;
        Ok(self.with_drop_block_options(options))
    }

    /// Applies timm's `DropBlock` schedule to every block of the last two
    /// stages; the panicking twin of
    /// [`try_with_standard_drop_block_prob`](Self::try_with_standard_drop_block_prob).
    ///
    /// # Panics
    ///
    /// If `drop_prob` is not a probability, or if it is above 0 and the
    /// structure has fewer than 2 stages.
    pub fn with_standard_drop_block_prob(
        self,
        drop_prob: f64,
    ) -> Self {
        self.try_with_standard_drop_block_prob(drop_prob)
            .ok_or_panic()
    }

    /// Applies timm's stochastic-depth schedule to every block.
    ///
    /// Block `i` of the net's `n` gets a drop-path probability of
    /// `drop_path_rate * i / (n - 1)`, counting every block of every stage:
    /// 0 at the first block, `drop_path_rate` at the last. This is the rule
    /// of `make_blocks` in timm's `resnet.py`, and the one
    /// [`ResNet::with_stochastic_path_depth`] applies to a built model.
    ///
    /// # Panics
    ///
    /// If `drop_path_rate` is not a probability.
    pub fn with_stochastic_depth_drop_path_rate(
        self,
        drop_path_rate: f64,
    ) -> Self {
        let drop_path_rate = expect_probability(drop_path_rate);

        let net_num_blocks = self.layers.iter().map(|b| b.len()).sum::<usize>();
        let mut net_block_idx = 0;
        let mut update_drop_path = |_idx: usize, block: ResidualBlockStructureConfig| {
            let block_dpr = stochastic_depth_rate(drop_path_rate, net_block_idx, net_num_blocks);
            net_block_idx += 1;
            if block_dpr > 0.0 {
                block.with_drop_path_prob(block_dpr)
            } else {
                block
            }
        };

        Self {
            layers: self
                .layers
                .into_iter()
                .map(|b| b.map_blocks(&mut update_drop_path))
                .collect(),
            ..self
        }
    }

    /// Updates the config with the given drop block options.
    ///
    /// # Arguments
    ///
    /// - `options`: a vector of options, one for each layer.
    pub fn with_drop_block_options(
        self,
        options: Vec<Option<DropBlockOptions>>,
    ) -> Self {
        assert_eq!(options.len(), self.layers.len());
        Self {
            layers: self
                .layers
                .into_iter()
                .zip(options)
                .map(|(b, o)| b.with_drop_block(o))
                .collect(),
            ..self
        }
    }
}

/// The stochastic-depth linear decay rule of timm's `make_blocks`
/// (`resnet.py`): block `i` (`net_block_idx`) of a net of `n`
/// (`net_num_blocks`) gets a drop-path probability of
/// `drop_path_rate * i / (n - 1)`. A net of one block gets 0, where timm
/// divides by zero.
fn stochastic_depth_rate(
    drop_path_rate: f64,
    net_block_idx: usize,
    net_num_blocks: usize,
) -> f64 {
    if net_num_blocks < 2 {
        return 0.0;
    }
    drop_path_rate * (net_block_idx as f64) / ((net_num_blocks - 1) as f64)
}

/// The `DropBlock` schedule of timm's `drop_blocks` (`resnet.py`), one entry
/// per stage of a net of `num_stages`: the second-to-last stage gets a block
/// size of 5 and a gamma scale of 0.25, the last a block size of 3 and a
/// gamma scale of 1.0, and the others none. A `drop_prob` of 0 gives none
/// anywhere.
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

impl<B: Backend> ModuleInit<B, ResNet<B>> for ResNetStructureConfig {
    /// Builds the [`ResNet`] module.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, when there are no stages, or when a stage
    /// fails [`LayerBlockStructureConfig::try_validate`] (it has no blocks, or
    /// its blocks' planes do not chain); a frame names the stage.
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<ResNet<B>> {
        if self.layers.is_empty() {
            return Err(ConstraintError::zero_or_empty("ResNetStructureConfig", "layers").into());
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            layer
                .try_validate()
                .with_context(|| format!("ResNet stage {idx} (layers[{idx}])"))?;
        }

        let head_planes = self.head_planes();

        let module = ResNet {
            input_cb: self.input_cb.init(device),
            input_pool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding({
                    let d = 1;
                    PaddingConfig2d::Explicit(d, d, d, d)
                })
                .init(),

            layers: self
                .layers
                .iter()
                .map(|c| c.try_init(device))
                .collect::<BunsenResult<Vec<_>>>()?,

            output_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            output_fc: LinearConfig::new(head_planes, self.num_classes).init(device),
        };

        Ok(module)
    }
}

/// `ResNet` model.
///
/// The full image classification network: stem conv/norm/act + max-pool, a
/// sequence of [`LayerBlock`] stages, then adaptive-average-pool and a linear
/// classifier head. Configure via [`ResNetContractConfig`], call
/// `.init(device)` to build, then [`ResNet::forward`] to map a `[batch, 3, h,
/// w]` image batch to `[batch, num_classes]` logits.
///
/// Implements [`ResNetMeta`].
///
/// Built by [`ResNetContractConfig`] (high-level) or [`ResNetStructureConfig`].
#[derive(Module, Debug)]
pub struct ResNet<B: Backend> {
    /// Input conv/norm.
    pub input_cb: ConvBlock2d<B>,
    /// Input pool.
    pub input_pool: MaxPool2d,

    /// Layers.
    pub layers: Vec<LayerBlock<B>>,

    /// Head pooling.
    pub output_pool: AdaptiveAvgPool2d,
    /// Head classifier.
    pub output_fc: Linear<B>,
}

impl<B: Backend> ResNetMeta for ResNet<B> {
    fn num_stages(&self) -> usize {
        self.layers.len()
    }

    fn head_planes(&self) -> usize {
        self.output_fc.weight.dims()[0]
    }

    fn num_classes(&self) -> usize {
        self.output_fc.weight.dims()[1]
    }
}

impl<B: Backend> ResNet<B> {
    /// Debug Printout.
    pub fn debug_print(&self) {
        for (idx, layer) in self.layers.iter().enumerate() {
            println!(
                "# Stage[{idx:?}]/{}:: {} :> {}",
                layer.len(),
                layer.in_planes(),
                layer.out_planes()
            );
            layer.debug_print();
            println!();
        }
    }

    /// Forward pass.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        // Prep block
        let x = self.input_cb.forward(input);
        let x = self.input_pool.forward(x);

        // Residual blocks
        let mut x = x;
        for layer in self.layers.iter() {
            x = layer.forward(x);
        }

        // Head
        let x = self.output_pool.forward(x);
        // Reshape [B, C, 1, 1] -> [B, C]
        let x = x.flatten(1, 3);
        self.output_fc.forward(x)
    }

    /// Loads weights from a `PyTorch` weights path.
    ///
    /// # Errors
    ///
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) if the file is
    /// missing or unreadable (see [`sys_at`](crate::errors::sys_at)); other
    /// I/O failures by their kind.
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
    /// with the [`PytorchStoreError`](burn_store::PytorchStoreError) as its
    /// cause, if the checkpoint does not load into the model.
    #[cfg(feature = "store_pytorch")]
    pub fn load_pytorch_weights(
        mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> BunsenResult<Self> {
        use burn_store::{
            ModuleSnapshot,
            PytorchStore,
            PytorchStoreError,
            pytorch::PytorchError,
        };

        use crate::errors::sys_at;

        let path = path.into();
        let mut store = PytorchStore::from_file(path.clone())
            .skip_enum_variants(true)
            .with_key_remapping(r"bn(\d+)\.weight", "bn$1.gamma")
            .with_key_remapping(r"bn(\d+)\.bias", "bn$1.beta")
            .with_key_remapping(r"^conv1\.", "input_cb.conv.")
            .with_key_remapping(r"^bn1\.", "input_cb.norm.")
            .with_key_remapping(r"bn(\d+)\.", "cb$1.norm.")
            .with_key_remapping(r"conv(\d+)\.", "cb$1.conv.")
            .with_key_remapping(r"downsample\.0\.", "downsample.conv.")
            .with_key_remapping(r"downsample\.1\.", "downsample.norm.")
            .with_key_remapping(r"fc\.", "output_fc.")
            .with_key_remapping(r"layer(\d+)\.", "layers.$1.blocks.");

        self.load_from(&mut store).map_err(|e| match e {
            PytorchStoreError::Io(io) | PytorchStoreError::Reader(PytorchError::Io(io)) => {
                sys_at("read", &path)(io)
            }
            e => BunsenError::invalid_resource(format!(
                "cannot load PyTorch weights from {}",
                path.display()
            ))
            .with_cause(e),
        })?;

        Ok(self)
    }

    /// Re-initializes the last layer with the specified number of output
    /// classes.
    pub fn with_classes(
        mut self,
        num_classes: usize,
    ) -> Self {
        let [d_input, _d_output] = self.output_fc.weight.dims();
        self.output_fc =
            LinearConfig::new(d_input, num_classes).init(&self.output_fc.weight.device());
        self
    }

    /// Applies timm's stochastic-depth schedule to every block.
    ///
    /// Block `i` of the net's `n` gets a drop-path probability of
    /// `drop_path_rate * i / (n - 1)`, counting every block of every stage:
    /// 0 at the first block, `drop_path_rate` at the last. This is the rule
    /// of `make_blocks` in timm's `resnet.py`, and the one
    /// [`ResNetStructureConfig::with_stochastic_depth_drop_path_rate`]
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

        let net_num_blocks = self.layers.iter().map(|b| b.len()).sum::<usize>();
        let mut net_block_idx = 0;
        let mut update_drop_path = |_idx: usize, block: ResidualBlock<B>| {
            let block_dpr = stochastic_depth_rate(drop_path_rate, net_block_idx, net_num_blocks);
            net_block_idx += 1;
            if block_dpr > 0.0 {
                block.with_drop_path_prob(block_dpr)
            } else {
                block
            }
        };

        Self {
            layers: self
                .layers
                .into_iter()
                .map(|b| b.map_blocks(&mut update_drop_path))
                .collect(),
            ..self
        }
    }

    /// Updates the config with the given drop block options.
    ///
    /// # Arguments
    ///
    /// - `options`: a vector of options, one for each layer.
    pub fn with_drop_block_options(
        self,
        options: Vec<Option<DropBlockOptions>>,
    ) -> Self {
        assert_eq!(options.len(), self.layers.len());
        Self {
            layers: self
                .layers
                .into_iter()
                .zip(options)
                .map(|(b, o)| b.with_drop_block(o))
                .collect(),
            ..self
        }
    }

    /// Applies timm's `DropBlock` schedule to every block of the last two
    /// stages.
    ///
    /// Every block of the second-to-last stage gets a [`DropBlockOptions`]
    /// with `drop_prob`, a block size of 5 and a gamma scale of 0.25. Every
    /// block of the last stage gets one with a block size of 3 and a gamma
    /// scale of 1.0. The other stages get none, and a `drop_prob` of 0
    /// clears every block. On the standard 4-stage net these are stages 3
    /// and 4, the rule of `drop_blocks` and `make_blocks` in timm's
    /// `resnet.py`, and the one
    /// [`ResNetStructureConfig::try_with_standard_drop_block_prob`] applies
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
        let options = standard_drop_block_options(drop_prob, self.layers.len())?;
        Ok(self.with_drop_block_options(options))
    }

    /// Applies timm's `DropBlock` schedule to every block of the last two
    /// stages; the panicking twin of
    /// [`try_with_stochastic_drop_block`](Self::try_with_stochastic_drop_block).
    ///
    /// # Panics
    ///
    /// If `drop_prob` is not a probability, or if it is above 0 and the model
    /// has fewer than 2 stages.
    pub fn with_stochastic_drop_block(
        self,
        drop_prob: f64,
    ) -> Self {
        self.try_with_stochastic_drop_block(drop_prob).ok_or_panic()
    }

    /// Applies a mapping over layers.
    pub fn map_layers<F>(
        self,
        f: F,
    ) -> Self
    where
        F: Fn(Vec<LayerBlock<B>>) -> Vec<LayerBlock<B>>,
    {
        Self {
            layers: f(self.layers),
            ..self
        }
    }

    /// Freezes the layers.
    pub fn freeze_layers(self) -> Self {
        self.map_layers(|layers| layers.into_iter().map(|layer| layer.no_grad()).collect())
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::images::resnet::{
            RESNET34_BLOCKS,
            RESNET50_BLOCKS,
        },
        support::testing::{
            CpuBackend,
            DeviceMemoryGuard,
            PerformanceBackend,
            cpu_device, performance_device,
        },
    };

    /// Fetches a checkpoint through the kit's factory and reads it into
    /// its prefab's model.
    #[cfg(all(feature = "store_pytorch", feature = "cache", feature = "fetch"))]
    fn test_load_pytorch<B: Backend>(spec: &str) -> BunsenResult<()> {
        use crate::{
            data::pretrained::{
                PretrainedCache,
                PretrainedCacheOptions,
            },
            kits::images::resnet::pretrained::default_resnet_factory,
        };

        let device = default_device();
        let cache = PretrainedCache::new(PretrainedCacheOptions::default())?;
        let loaded = default_resnet_factory()?.load::<B>(spec, &cache, &device)?;
        let _model: &ResNet<B> = &loaded.handle;
        Ok(())
    }

    #[test]
    #[serial]
    #[cfg(all(feature = "store_pytorch", feature = "cache", feature = "fetch"))]
    fn test_load_pytorch_prefab() -> BunsenResult<()> {
        test_load_pytorch::<PerformanceBackend>("torchvision/resnet18")
    }

    #[test]
    #[serial]
    #[cfg(all(feature = "store_pytorch", feature = "cache", feature = "fetch"))]
    fn test_load_pytorch_prefab_cuda() -> BunsenResult<()> {
        test_load_pytorch::<PerformanceBackend>("torchvision/resnet34")
    }

    #[test]
    fn test_to_layers_34_basic() {
        let cfg = ResNetContractConfig::new(RESNET34_BLOCKS.to_vec(), 1000);

        let layers = cfg.to_layer_contracts();

        println!("{:#?}", layers);

        // assert!(false);
    }

    #[test]
    #[serial]
    fn test_to_layers_50_bottleneck() {
        type B = PerformanceBackend;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let cfg = ResNetContractConfig::new(RESNET50_BLOCKS.to_vec(), 1000).with_bottleneck(true);
        let layers = cfg.to_layer_contracts();

        let first_stage = layers[0].clone();
        println!("block[0] cfg:\n{:#?}", first_stage);
        println!();

        let blocks = first_stage
            .to_block_contracts()
            .into_iter()
            .map(|b| b.to_structure())
            .collect::<Vec<_>>();
        println!("blocks ...");
        println!("{:#?}", blocks);
        println!();

        let model: ResNet<B> = cfg.to_structure().init(&device);

        model.debug_print();

        // assert!(false);
    }

    /// Asserts that `a` and `b` answer every [`ResNetMeta`] method alike.
    fn assert_meta_agrees(
        a: &impl ResNetMeta,
        b: &impl ResNetMeta,
    ) {
        assert_eq!(a.num_stages(), b.num_stages());
        assert_eq!(a.head_planes(), b.head_planes());
        assert_eq!(a.num_classes(), b.num_classes());
    }

    /// A policy builds the same `ResNet` through its structure as through the
    /// blanket `init`.
    #[test]
    #[serial]
    fn test_policy_pathways_agree() {
        type B = PerformanceBackend;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let policy = ResNetContractConfig::new(vec![1, 1], 7)
            .with_stem_width(8)
            .with_bottleneck(true);

        let structure = policy.to_structure();
        assert_eq!(structure.num_stages(), 2);
        // A pinch factor of 4 on an 8-plane stem, doubled by the second stage.
        assert_eq!(structure.head_planes(), 64);
        assert_eq!(structure.num_classes(), 7);

        let lowered: ResNet<B> = structure.init(&device);
        let direct: ResNet<B> = policy.init(&device);

        assert_meta_agrees(&direct, &lowered);
        assert_meta_agrees(&direct, &structure);
    }

    /// A structure with no stages is an error from `try_init`, not a panic:
    /// the head has no stage to take its width from.
    #[test]
    fn test_try_init_rejects_no_stages() {
        let device = cpu_device();

        let structure = ResNetStructureConfig {
            layers: vec![],
            ..ResNetContractConfig::new(vec![1], 7)
                .with_stem_width(8)
                .to_structure()
        };
        let no_stages = ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .has_cause::<ConstraintError>()
            .message_contains("ResNetStructureConfig.layers");
        let bad: BunsenResult<ResNet<CpuBackend>> = structure.try_init(&device);
        no_stages.assert_err(&bad);

        // The policy reaches the same check through the blanket `try_init`.
        let bad: BunsenResult<ResNet<CpuBackend>> = ResNetContractConfig::new(vec![], 7)
            .with_stem_width(8)
            .try_init(&device);
        no_stages.assert_err(&bad);
    }

    /// A stage with no blocks is an error from `try_init`, not a panic.
    #[test]
    fn test_try_init_rejects_an_empty_stage() {
        let device = cpu_device();

        let bad: BunsenResult<ResNet<CpuBackend>> = ResNetContractConfig::new(vec![1, 0], 7)
            .with_stem_width(8)
            .try_init(&device);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .has_cause::<ConstraintError>()
            .frame_contains("ResNet stage 1")
            .assert_err(&bad);
    }

    /// timm's stochastic-depth schedule (`make_blocks` in `resnet.py`) for a
    /// `[2, 2, 2, 2]` net at a rate of 0.1: block `i` of the net's `n` gets
    /// `0.1 * i / (n - 1)`, counting every block, each stage's first
    /// included. The values are python3's.
    const TIMM_DROP_PATH_2222_AT_0_1: [f64; 8] = [
        0.0,
        0.014285714285714287,
        0.028571428571428574,
        0.042857142857142864,
        0.05714285714285715,
        0.07142857142857142,
        0.08571428571428573,
        0.1,
    ];

    /// Each block's drop-path probability in a structure, in net order.
    fn structure_drop_path_probs(structure: &ResNetStructureConfig) -> Vec<f64> {
        structure
            .layers
            .iter()
            .flat_map(|layer| layer.blocks.iter())
            .map(|block| match block {
                ResidualBlockStructureConfig::Basic(config) => config.drop_path_prob,
                ResidualBlockStructureConfig::Bottleneck(config) => config.drop_path_prob,
            })
            .collect()
    }

    /// Each block's drop-path probability in a module, in net order; 0 for a
    /// block without a `DropPath`.
    fn module_drop_path_probs<B: Backend>(model: &ResNet<B>) -> Vec<f64> {
        model
            .layers
            .iter()
            .flat_map(|layer| layer.blocks.iter())
            .map(|block| {
                let drop_path = match block {
                    ResidualBlock::Basic(block) => &block.drop_path,
                    ResidualBlock::Bottleneck(block) => &block.drop_path,
                };
                drop_path.as_ref().map_or(0.0, |d| d.drop_prob)
            })
            .collect()
    }

    fn assert_rates_eq(
        actual: &[f64],
        expected: &[f64],
    ) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} != {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{actual:?} != {expected:?}");
        }
    }

    /// The structure's stochastic-depth schedule is timm's.
    #[test]
    fn test_structure_drop_path_schedule_matches_timm() {
        let structure = ResNetContractConfig::new(vec![2, 2, 2, 2], 10)
            .with_stem_width(8)
            .to_structure()
            .with_stochastic_depth_drop_path_rate(0.1);

        assert_rates_eq(
            &structure_drop_path_probs(&structure),
            &TIMM_DROP_PATH_2222_AT_0_1,
        );
    }

    /// The module's stochastic-depth schedule is timm's.
    #[test]
    fn test_module_drop_path_schedule_matches_timm() {
        let device = cpu_device();
        let model: ResNet<CpuBackend> = ResNetContractConfig::new(vec![2, 2, 2, 2], 10)
            .with_stem_width(8)
            .init(&device);
        let model = model.with_stochastic_path_depth(0.1);

        assert_rates_eq(&module_drop_path_probs(&model), &TIMM_DROP_PATH_2222_AT_0_1);
    }

    /// One block per stage: timm gives `[0.0, 0.1]` for two blocks, and both
    /// schedules agree with it.
    #[test]
    fn test_drop_path_schedules_agree_with_one_block_per_stage() {
        let structure = ResNetContractConfig::new(vec![1, 1], 10)
            .with_stem_width(8)
            .to_structure();

        let device = cpu_device();
        let model: ResNet<CpuBackend> = structure.init(&device);
        let model = model.with_stochastic_path_depth(0.1);
        assert_rates_eq(&module_drop_path_probs(&model), &[0.0, 0.1]);

        let structure = structure.with_stochastic_depth_drop_path_rate(0.1);
        assert_rates_eq(&structure_drop_path_probs(&structure), &[0.0, 0.1]);
    }

    /// timm's DropBlock schedule (`drop_blocks` and `make_blocks` in
    /// `resnet.py`) for a `[2, 2, 2, 2]` net at a rate of 0.1: none in the
    /// first two stages; every block of stage 3 gets a block size of 5 and a
    /// gamma scale of 0.25, and every block of stage 4 a block size of 3 and
    /// a gamma scale of 1.0.
    fn timm_drop_blocks_2222_at_0_1() -> Vec<Option<DropBlockOptions>> {
        let stage3 = DropBlockOptions::default()
            .with_drop_prob(0.1)
            .with_block_size(5)
            .with_gamma_scale(0.25);
        let stage4 = DropBlockOptions::default()
            .with_drop_prob(0.1)
            .with_block_size(3)
            .with_gamma_scale(1.0);
        vec![
            None,
            None,
            None,
            None,
            Some(stage3.clone()),
            Some(stage3),
            Some(stage4.clone()),
            Some(stage4),
        ]
    }

    /// Each block's DropBlock options in a structure, in net order.
    fn structure_drop_blocks(structure: &ResNetStructureConfig) -> Vec<Option<DropBlockOptions>> {
        structure
            .layers
            .iter()
            .flat_map(|layer| layer.blocks.iter())
            .map(|block| match block {
                ResidualBlockStructureConfig::Basic(config) => config.drop_block.clone(),
                ResidualBlockStructureConfig::Bottleneck(config) => config.drop_block.clone(),
            })
            .collect()
    }

    /// Each block's DropBlock options in a module, in net order.
    fn module_drop_blocks<B: Backend>(model: &ResNet<B>) -> Vec<Option<DropBlockOptions>> {
        model
            .layers
            .iter()
            .flat_map(|layer| layer.blocks.iter())
            .map(|block| {
                let drop_block = match block {
                    ResidualBlock::Basic(block) => &block.drop_block,
                    ResidualBlock::Bottleneck(block) => &block.drop_block,
                };
                drop_block.as_ref().map(|d| d.options.clone())
            })
            .collect()
    }

    /// The structure's DropBlock schedule is timm's, and a module built from
    /// the structure keeps it.
    #[test]
    fn test_structure_drop_block_schedule_matches_timm() {
        let structure = ResNetContractConfig::new(vec![2, 2, 2, 2], 10)
            .with_stem_width(8)
            .to_structure()
            .with_standard_drop_block_prob(0.1);
        assert_eq!(
            structure_drop_blocks(&structure),
            timm_drop_blocks_2222_at_0_1()
        );

        let device = cpu_device();
        let model: ResNet<CpuBackend> = structure.init(&device);
        assert_eq!(module_drop_blocks(&model), timm_drop_blocks_2222_at_0_1());
    }

    /// The module's DropBlock schedule is timm's.
    #[test]
    fn test_module_drop_block_schedule_matches_timm() {
        let device = cpu_device();
        let model: ResNet<CpuBackend> = ResNetContractConfig::new(vec![2, 2, 2, 2], 10)
            .with_stem_width(8)
            .init(&device);
        let model = model.with_stochastic_drop_block(0.1);

        assert_eq!(module_drop_blocks(&model), timm_drop_blocks_2222_at_0_1());
    }

    /// With fewer than 2 stages there is no second-to-last stage for the
    /// `DropBlock` schedule: a nonzero rate is an error from both `try_`
    /// forms. A rate of 0 places nothing, and is fine.
    #[test]
    fn test_try_drop_block_schedules_reject_one_stage() {
        let structure = ResNetContractConfig::new(vec![2], 10)
            .with_stem_width(8)
            .to_structure();
        let device = cpu_device();
        let model: ResNet<CpuBackend> = structure.init(&device);

        let one_stage =
            ErrorMatcher::kind(BunsenErrorKind::Illegal).message_contains("at least 2 stages");
        one_stage.assert_err(&structure.clone().try_with_standard_drop_block_prob(0.1));
        one_stage.assert_err(&model.clone().try_with_stochastic_drop_block(0.1));

        // A rate that is not a probability names the field.
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .frame_contains("drop_prob")
            .assert_err(&structure.clone().try_with_standard_drop_block_prob(1.5));

        let structure = structure.try_with_standard_drop_block_prob(0.0).unwrap();
        assert_eq!(structure_drop_blocks(&structure), vec![None, None]);
        let model = model.try_with_stochastic_drop_block(0.0).unwrap();
        assert_eq!(module_drop_blocks(&model), vec![None, None]);
    }

    /// The panicking twin panics with a message that names the problem.
    #[test]
    #[should_panic(expected = "at least 2 stages")]
    fn test_drop_block_schedule_panics_on_one_stage() {
        let _ = ResNetContractConfig::new(vec![2], 10)
            .with_stem_width(8)
            .to_structure()
            .with_standard_drop_block_prob(0.1);
    }
}
