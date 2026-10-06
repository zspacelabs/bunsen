//! # Residual Block Wrapper

use burn::{
    nn::{
        BatchNormConfig,
        activation::ActivationConfig,
        norm::NormalizationConfig,
    },
    prelude::{
        Config,
        Module,
        Tensor,
    },
    tensor::Device,
};

use crate::{
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    errors::BunsenResult,
    kits::images::resnet::blocks::{
        BasicBlock,
        BasicBlockConfig,
        BasicBlockMeta,
        BottleneckBlock,
        BottleneckBlockConfig,
        BottleneckBlockMeta,
        BottleneckPolicyConfig,
    },
    ops::{
        conv::stride_div_output_resolution,
        drop::DropBlockOptions,
    },
    support::validators::expect_probability,
};

/// Abstract [`ResidualBlock`] Config.
///
/// High-level description of a single residual unit: channel sizes, dilation,
/// downsampling, and whether to select a [`BasicBlock`] or [`BottleneckBlock`].
/// It implements [`ToStructureConfig`], lowering to a
/// [`ResidualBlockStructureConfig`], and gets [`ModuleInit`] from that trait's
/// blanket impl: call `.init(device)` to build the [`ResidualBlock`] module,
/// then drive it with [`ResidualBlock::forward`].
#[derive(Config, Debug)]
pub struct ResidualBlockContractConfig {
    /// The number of input feature planes.
    pub in_planes: usize,

    /// The number of output feature planes.
    pub out_planes: usize,

    /// Dilation rate for conv layers.
    #[config(default = 1)]
    pub dilation: usize,

    /// If set, override the first dilation rate.
    #[config(default = "None")]
    pub first_dilation: Option<usize>,

    /// Downsample the input by 2x?
    #[config(default = "false")]
    pub downsample_input: bool,

    /// Select between [`BasicBlock`] and [`BottleneckBlock`].
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

impl ToStructureConfig for ResidualBlockContractConfig {
    type Structure = ResidualBlockStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<ResidualBlockStructureConfig> {
        let stride = if self.downsample_input { 2 } else { 1 };

        Ok(match &self.bottleneck_policy {
            None => BasicBlockConfig::new(self.in_planes, self.out_planes)
                .with_stride(stride)
                .with_dilation(self.dilation)
                .with_first_dilation(self.first_dilation)
                .with_normalization(self.normalization.clone())
                .with_activation(self.activation.clone())
                .into(),
            Some(policy) => BottleneckBlockConfig::new(self.in_planes, self.out_planes)
                .with_stride(stride)
                .with_dilation(self.dilation)
                .with_first_dilation(self.first_dilation)
                .with_normalization(self.normalization.clone())
                .with_activation(self.activation.clone())
                .with_policy(policy.clone())
                .into(),
        })
    }
}

/// [`ResidualBlock`] Meta API.
///
/// Defines a shared API for [`ResidualBlock`] and
/// [`ResidualBlockStructureConfig`].
pub trait ResidualBlockMeta {
    /// The number of input feature planes.
    fn in_planes(&self) -> usize;

    /// The number of outpu feature planes.
    fn out_planes(&self) -> usize;

    /// The stride of convolution.
    ///
    /// Affects downsample behavior.
    fn stride(&self) -> usize;

    /// Returns the output resolution for a given input resolution.
    ///
    /// The input must be a multiple of the stride.
    ///
    /// # Arguments
    ///
    /// - `input_resolution`: \ `[in_height=out_height*stride,
    ///   in_width=out_width*stride]`.
    ///
    /// # Returns
    ///
    /// `[out_height, out_width]`
    ///
    /// # Panics
    ///
    /// If the input resolution is not a multiple of the stride.
    fn output_resolution(
        &self,
        input_resolution: [usize; 2],
    ) -> [usize; 2] {
        stride_div_output_resolution(input_resolution, self.stride())
    }
}

/// [`ResidualBlock`] Config.
///
/// The concrete, resolved choice of inner block ([`BasicBlockConfig`] or
/// [`BottleneckBlockConfig`]) for one residual unit.
/// [`ResidualBlockContractConfig`] lowers to it, and each variant converts
/// from its inner config with `From`. Call `.init(device)` to build the
/// [`ResidualBlock`] module, then drive it with [`ResidualBlock::forward`].
///
/// Implements [`ResidualBlockMeta`].
#[derive(Config, Debug)]
pub enum ResidualBlockStructureConfig {
    /// A `ResNet` [`BasicBlock`].
    Basic(BasicBlockConfig),

    /// A `ResNet` [`BottleneckBlock`].
    Bottleneck(BottleneckBlockConfig),
}

impl ResidualBlockMeta for ResidualBlockStructureConfig {
    fn in_planes(&self) -> usize {
        match self {
            Self::Basic(config) => config.in_planes(),
            Self::Bottleneck(config) => config.in_planes(),
        }
    }

    fn out_planes(&self) -> usize {
        match self {
            Self::Basic(config) => config.out_planes(),
            Self::Bottleneck(config) => config.out_planes(),
        }
    }

    fn stride(&self) -> usize {
        match self {
            Self::Basic(config) => config.stride(),
            Self::Bottleneck(config) => config.stride(),
        }
    }

    fn output_resolution(
        &self,
        input_resolution: [usize; 2],
    ) -> [usize; 2] {
        match self {
            Self::Basic(config) => config.output_resolution(input_resolution),
            Self::Bottleneck(config) => config.output_resolution(input_resolution),
        }
    }
}

impl From<BasicBlockConfig> for ResidualBlockStructureConfig {
    fn from(config: BasicBlockConfig) -> Self {
        Self::Basic(config)
    }
}

impl From<BottleneckBlockConfig> for ResidualBlockStructureConfig {
    fn from(config: BottleneckBlockConfig) -> Self {
        Self::Bottleneck(config)
    }
}

impl ResidualBlockStructureConfig {
    /// Sets drop block options.
    pub fn with_drop_block(
        self,
        options: Option<DropBlockOptions>,
    ) -> Self {
        match self {
            Self::Basic(config) => config.with_drop_block(options).into(),
            Self::Bottleneck(config) => config.with_drop_block(options).into(),
        }
    }

    /// Sets the drop path probability.
    pub fn with_drop_path_prob(
        self,
        drop_path_prob: f64,
    ) -> Self {
        let drop_path_prob = expect_probability(drop_path_prob);
        match self {
            Self::Basic(config) => config.with_drop_path_prob(drop_path_prob).into(),
            Self::Bottleneck(config) => config.with_drop_path_prob(drop_path_prob).into(),
        }
    }
}

impl ModuleInit<ResidualBlock> for ResidualBlockStructureConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ResidualBlock> {
        Ok(match self {
            Self::Basic(config) => config.try_init(device)?.into(),
            Self::Bottleneck(config) => config.try_init(device)?.into(),
        })
    }
}

/// A `ResNet` [`BasicBlock`] or [`BottleneckBlock`] wrapper.
///
/// A uniform residual-unit module that dispatches to either a [`BasicBlock`] or
/// a [`BottleneckBlock`], letting a stage hold a homogeneous list of blocks.
/// Configure via [`ResidualBlockContractConfig`], call `.init(device)` to
/// build, then [`ResidualBlock::forward`] to apply.
///
/// A block built on its own converts with `From`: from a [`BasicBlock`] or
/// a [`BottleneckBlock`].
///
/// Implements [`ResidualBlockMeta`].
///
/// Built by [`ResidualBlockContractConfig`] (high-level) or
/// [`ResidualBlockStructureConfig`].
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ResidualBlock {
    /// A `ResNet` [`BasicBlock`].
    Basic(BasicBlock),

    /// A `ResNet` [`BottleneckBlock`].
    Bottleneck(BottleneckBlock),
}

impl From<BasicBlock> for ResidualBlock {
    fn from(block: BasicBlock) -> Self {
        Self::Basic(block)
    }
}

impl From<BottleneckBlock> for ResidualBlock {
    fn from(block: BottleneckBlock) -> Self {
        Self::Bottleneck(block)
    }
}

impl ResidualBlockMeta for ResidualBlock {
    fn in_planes(&self) -> usize {
        match self {
            Self::Basic(block) => block.in_planes(),
            Self::Bottleneck(block) => block.in_planes(),
        }
    }

    fn out_planes(&self) -> usize {
        match self {
            Self::Basic(block) => block.out_planes(),
            Self::Bottleneck(block) => block.out_planes(),
        }
    }

    fn stride(&self) -> usize {
        match self {
            Self::Basic(block) => block.stride(),
            Self::Bottleneck(block) => block.stride(),
        }
    }
}

impl ResidualBlock {
    /// Debug print.
    pub fn debug_print(&self) {
        match self {
            Self::Basic(block) => block.debug_print(),
            Self::Bottleneck(block) => block.debug_print(),
        }
    }

    /// Applies the wrapped block to the input.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_planes, in_height=out_height*stride,
    ///   in_width=out_width*stride]`.
    ///
    /// # Returns
    ///
    /// A `[batch, out_planes, out_height, out_width]` tensor;
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        match self {
            Self::Basic(block) => block.forward(input),
            Self::Bottleneck(block) => block.forward(input),
        }
    }

    /// Sets the drop path probability.
    pub fn with_drop_path_prob(
        self,
        drop_path_prob: f64,
    ) -> Self {
        let drop_path_prob = expect_probability(drop_path_prob);
        match self {
            Self::Basic(block) => block.with_drop_path_prob(drop_path_prob).into(),
            Self::Bottleneck(block) => block.with_drop_path_prob(drop_path_prob).into(),
        }
    }

    /// Sets drop block options.
    pub fn with_drop_block(
        self,
        options: Option<DropBlockOptions>,
    ) -> Self {
        match self {
            Self::Basic(config) => config.with_drop_block(options).into(),
            Self::Bottleneck(config) => config.with_drop_block(options).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::{
            DeviceMemoryGuard,
            performance_device,
        },
    };

    #[test]
    fn test_residual_block_config() {
        let in_planes = 16;
        let out_planes = 32;

        {
            let inner_cfg = BasicBlockConfig::new(in_planes, out_planes).with_stride(2);

            let cfg: ResidualBlockStructureConfig = inner_cfg.clone().into();
            assert!(matches!(cfg, ResidualBlockStructureConfig::Basic(_)));
            assert_eq!(cfg.in_planes(), in_planes);
            assert_eq!(cfg.out_planes(), out_planes);
            assert_eq!(cfg.stride(), 2);
            assert_eq!(cfg.output_resolution([20, 20]), [10, 10]);
        }

        {
            let inner_cfg = BottleneckBlockConfig::new(in_planes, out_planes).with_stride(2);

            let cfg: ResidualBlockStructureConfig = inner_cfg.clone().into();
            assert!(matches!(cfg, ResidualBlockStructureConfig::Bottleneck(_)));
            assert_eq!(cfg.in_planes(), in_planes);
            assert_eq!(cfg.out_planes(), out_planes);
            assert_eq!(cfg.stride(), 2);
            assert_eq!(cfg.output_resolution([20, 20]), [10, 10]);
        }
    }

    #[test]
    #[serial]
    fn test_residual_block_basic_block() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let batch_size = 2;
        let in_planes = 16;
        let planes = 32;
        let in_height = 8;
        let in_width = 8;
        let out_height = 4;
        let out_width = 4;

        let cfg: ResidualBlockStructureConfig = BasicBlockConfig::new(in_planes, planes)
            .with_stride(2)
            .into();

        let block: ResidualBlock = cfg.init(&device);
        assert!(matches!(block, ResidualBlock::Basic(_)));
        assert_eq!(block.in_planes(), in_planes);
        assert_eq!(block.out_planes(), planes);
        assert_eq!(block.stride(), 2);
        assert_eq!(block.output_resolution([20, 20]), [10, 10]);

        let input = Tensor::ones([batch_size, in_planes, in_height, in_width], &device);
        let output = block.forward(input);

        assert_shape_contract!(
            ["batch", "out_channels", "out_height", "out_width"],
            &output.dims(),
            &[
                ("batch", batch_size),
                ("out_channels", planes),
                ("out_height", out_height),
                ("out_width", out_width)
            ],
        );
    }

    #[test]
    #[serial]
    fn test_residual_block_bottleneck_block() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let batch_size = 2;
        let in_planes = 16;
        let planes = 32;
        let in_height = 8;
        let in_width = 8;
        let out_height = 4;
        let out_width = 4;

        let cfg: ResidualBlockStructureConfig = BottleneckBlockConfig::new(in_planes, planes)
            .with_stride(2)
            .into();

        let block: ResidualBlock = cfg.init(&device);
        assert!(matches!(block, ResidualBlock::Bottleneck(_)));
        assert_eq!(block.in_planes(), in_planes);
        assert_eq!(block.out_planes(), planes);
        assert_eq!(block.stride(), 2);
        assert_eq!(block.output_resolution([20, 20]), [10, 10]);

        let input = Tensor::ones([batch_size, in_planes, in_height, in_width], &device);
        let output = block.forward(input);

        assert_shape_contract!(
            ["batch", "out_planes", "out_height", "out_width"],
            &output.dims(),
            &[
                ("batch", batch_size),
                ("out_planes", planes),
                ("out_height", out_height),
                ("out_width", out_width)
            ],
        );
    }

    /// Asserts that `a` and `b` answer every [`ResidualBlockMeta`] method
    /// alike.
    fn assert_meta_agrees(
        a: &impl ResidualBlockMeta,
        b: &impl ResidualBlockMeta,
    ) {
        assert_eq!(a.in_planes(), b.in_planes());
        assert_eq!(a.out_planes(), b.out_planes());
        assert_eq!(a.stride(), b.stride());
        assert_eq!(a.output_resolution([8, 12]), b.output_resolution([8, 12]));
    }

    /// A policy builds the same `ResidualBlock` through its structure as
    /// through the blanket `init`.
    #[test]
    #[serial]
    fn test_policy_pathways_agree() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let policy = ResidualBlockContractConfig::new(8, 32)
            .with_downsample_input(true)
            .with_bottleneck_policy(Some(BottleneckPolicyConfig::default()));

        let structure = policy.to_structure();
        assert!(matches!(
            structure,
            ResidualBlockStructureConfig::Bottleneck(_)
        ));
        assert_eq!(structure.stride(), 2);

        let lowered: ResidualBlock = structure.init(&device);
        let direct: ResidualBlock = policy.init(&device);
        assert!(matches!(direct, ResidualBlock::Bottleneck(_)));

        assert_meta_agrees(&direct, &lowered);
        assert_meta_agrees(&direct, &structure);
    }
}
