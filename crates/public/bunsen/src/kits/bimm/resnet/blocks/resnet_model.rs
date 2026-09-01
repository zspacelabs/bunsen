//! # `ResNet` Core Model
//!
//! [`ResNet`] is the core `ResNet` module.
//!
//! [`ResNetContractConfig`] implements [`Config`], and provides
//! a high-level configuration interface.
//! It provides [`ResNetContractConfig::to_structure`] to convert
//! to a [`ResNetStructureConfig`].
//!
//! [`ResNetStructureConfig`] implements [`Config`], and provides
//! [`ResNetStructureConfig::init`] to initialize a [`ResNet`].
//!
//! [`ResNet`] implements [`Module`], and provides
//! [`ResNet::forward`].

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

#[allow(unused_imports)]
use crate::errors::BunsenError;
use crate::{
    blocks::conv::{
        ConvBlock2d,
        ConvBlock2dConfig,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    kits::bimm::resnet::{
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
    support::validators::expect_probability,
};

/// High-level [`ResNet`] model configuration.
///
/// The user-facing entry point for building a [`ResNet`]: stage depths, class
/// count, stem width, output stride, and bottleneck policy. Lowers to a
/// [`ResNetStructureConfig`] via [`ResNetContractConfig::to_structure`]; call
/// `.init(device)` to build the [`ResNet`] module, then drive it with
/// [`ResNet::forward`].
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

    /// Converts to a [`ResNetStructureConfig`].
    pub fn to_structure(&self) -> ResNetStructureConfig {
        ResNetStructureConfig::new(
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
                .into_iter()
                .map(|c| c.into())
                .collect::<Vec<_>>(),
            self.num_classes,
        )
    }

    /// Creates a ResNet-18 model.
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(RESNET18_BLOCKS.to_vec(), num_classes) // .with_bottleneck(true)
    }
}

impl<B: Backend> ModuleInit<B, ResNet<B>> for ResNetContractConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<ResNet<B>> {
        self.to_structure().try_init(device)
    }
}

impl From<ResNetContractConfig> for ResNetStructureConfig {
    #[allow(unused)]
    fn from(config: ResNetContractConfig) -> Self {
        config.to_structure()
    }
}

/// [`ResNet`] Structure Config.
///
/// This config defines the structure of a converted [`ResNet`] model.
/// It is not a semantic configuration and does not check the validity
/// of the internal sizes before or during construction.
///
/// Holds the explicit stem, per-stage [`LayerBlockStructureConfig`]s, and head.
/// Call `.init(device)` to build the [`ResNet`] module, then drive it with
/// [`ResNet::forward`].
#[derive(Config, Debug)]
pub struct ResNetStructureConfig {
    /// The input Conv/Norm block configuration.
    pub input_cb: ConvBlock2dConfig,

    /// The inner layers configuration.
    pub layers: Vec<LayerBlockStructureConfig>,

    /// The number of classes.
    pub num_classes: usize,
}

impl ResNetStructureConfig {
    /// Applies the given standard drop block probability scheme.
    pub fn with_standard_drop_block_prob(
        self,
        drop_prob: f64,
    ) -> Self {
        let drop_prob = expect_probability(drop_prob);
        let k = self.layers.len();
        let mut blocks = vec![None; k];
        if drop_prob > 0.0 {
            blocks[k - 2] = DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_block_size(5)
                .with_gamma_scale(0.25)
                .into();
            blocks[k - 1] = DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_block_size(3)
                .with_gamma_scale(1.0)
                .into();
        }
        self.with_drop_block_options(blocks)
    }

    /// Updates the config with stochastic depth.
    pub fn with_stochastic_depth_drop_path_rate(
        self,
        drop_path_rate: f64,
    ) -> Self {
        let drop_path_rate = expect_probability(drop_path_rate);

        let net_num_blocks = self.layers.iter().map(|b| b.len()).sum::<usize>() - self.layers.len();
        let mut net_block_idx = 0;
        let mut update_drop_path = |idx: usize, block: ResidualBlockStructureConfig| {
            // stochastic depth linear decay rule
            let block_dpr = drop_path_rate * (net_block_idx as f64) / ((net_num_blocks - 1) as f64);
            net_block_idx += 1;
            if idx != 0 && block_dpr > 0.0 {
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

impl<B: Backend> ModuleInit<B, ResNet<B>> for ResNetStructureConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<ResNet<B>> {
        let head_planes = self.layers.last().unwrap().out_planes();

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
                .map(|c| c.init(device))
                .collect::<Vec<_>>(),

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
    #[cfg(feature = "store")]
    pub fn load_pytorch_weights(
        mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> BunsenResult<Self> {
        use burn_store::{
            ModuleSnapshot,
            PytorchStore,
        };
        let mut store = PytorchStore::from_file(path)
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

        self.load_from(&mut store)
            .map_err(|e| BunsenError::External(e.to_string()))?;

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

    /// Updates the config with stochastic depth.
    pub fn with_stochastic_path_depth(
        self,
        drop_path_rate: f64,
    ) -> Self {
        let drop_path_rate = expect_probability(drop_path_rate);

        let net_num_blocks = self.layers.iter().map(|b| b.len()).sum::<usize>();
        let mut net_block_idx = 0;
        let mut update_drop_path = |_idx: usize, block: ResidualBlock<B>| {
            // stochastic depth linear decay rule
            let block_dpr = drop_path_rate * (net_block_idx as f64) / ((net_num_blocks - 1) as f64);
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

    /// Applies the given standard drop block probability scheme.
    pub fn with_stochastic_drop_block(
        self,
        drop_prob: f64,
    ) -> Self {
        let drop_prob = expect_probability(drop_prob);
        let k = self.layers.len();
        let mut blocks = vec![None; k];
        if drop_prob > 0.0 {
            blocks[k - 2] = DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_block_size(5)
                .with_gamma_scale(0.25)
                .into();
            blocks[k - 1] = DropBlockOptions::default()
                .with_drop_prob(drop_prob)
                .with_block_size(3)
                .with_gamma_scale(1.0)
                .into();
        }
        self.with_drop_block_options(blocks)
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
        data::cache::BunsenDiskCache,
        kits::bimm::resnet::{
            RESNET34_BLOCKS,
            RESNET50_BLOCKS,
        },
        support::testing::PerformanceBackend,
    };

    #[cfg(feature = "store")]
    fn test_load_pytorch<B: Backend>(
        prefab: &str,
        pretrained: &str,
    ) -> BunsenResult<()> {
        use crate::kits::bimm::resnet::PREFAB_RESNET_MAP;

        let device = Default::default();

        let prefab = PREFAB_RESNET_MAP.expect_lookup_prefab(&prefab);

        let resnet_config = prefab.to_config().to_structure();
        println!("{:#?}", resnet_config);
        let model: ResNet<B> = resnet_config.init(&device);

        let path = prefab
            .expect_lookup_pretrained_weights(pretrained)
            .fetch_weights(&mut BunsenDiskCache::default())
            .map_err(|e| BunsenError::External(e.to_string()))?;

        let _model: ResNet<B> = model.load_pytorch_weights(path.clone())?;

        Ok(())
    }

    #[test]
    #[serial]
    #[cfg(feature = "store")]
    fn test_load_pytorch_prefab() -> BunsenResult<()> {
        type B = PerformanceBackend;
        let prefab = "resnet18";
        let pretrained = "tv_in1k";
        test_load_pytorch::<B>(&prefab, &pretrained)
    }

    #[test]
    #[serial]
    #[cfg(feature = "store")]
    fn test_load_pytorch_prefab_cuda() -> BunsenResult<()> {
        type B = PerformanceBackend;
        let prefab = "resnet34";
        let pretrained = "tv_in1k";
        test_load_pytorch::<B>(&prefab, &pretrained)
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
        let device = Default::default();

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
}
