//! `ResNet`'s policy, and its lowering onto the generic conv net.

use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    nn::BatchNormConfig,
    prelude::Config,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        images::models::conv_net::{
            ConvLayersConfig,
            ConvNetContractConfig,
            ConvNetStructureConfig,
            ConvStageContractConfig,
            ConvStemContractConfig,
            ConvUnitKind,
        },
        norm::BunsenNormalizationConfig,
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// ResNet-18 block depths.
pub const RESNET18_BLOCKS: [usize; 4] = [2, 2, 2, 2];
/// ResNet-34 block depths.
pub const RESNET34_BLOCKS: [usize; 4] = [3, 4, 6, 3];
/// ResNet-50 block depths.
pub const RESNET50_BLOCKS: [usize; 4] = [3, 4, 6, 3];
/// ResNet-101 block depths.
pub const RESNET101_BLOCKS: [usize; 4] = [3, 4, 23, 3];
/// ResNet-152 block depths.
pub const RESNET152_BLOCKS: [usize; 4] = [3, 8, 36, 3];

/// The default pinch factor of a bottleneck: its units put out 4x their
/// mixer's channels.
pub const BOTTLENECK_BLOCK_DEFAULT_PINCH_FACTOR: usize = 4;

/// A `ResNet`'s bottleneck policy: its units are bottlenecks, whose output
/// is `pinch_factor` times their mixer's channels.
#[derive(Config, Debug)]
pub struct BottleneckPolicyConfig {
    /// The output channels over the mixer's.
    #[config(default = "BOTTLENECK_BLOCK_DEFAULT_PINCH_FACTOR")]
    pub pinch_factor: usize,
}

impl Default for BottleneckPolicyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// `ResNet`'s policy: the stage depths and widths, the classes, the input's
/// channels, the stem's width, the output stride, the bottleneck policy, the
/// norm and the activation.
///
/// The user-facing way to name a `ResNet`, in its own terms; its knobs are
/// timm's `ResNet`'s, by their meaning. It lowers to the generic conv net's
/// policy ([`try_to_conv_net`](Self::try_to_conv_net)), and
/// through it, by [`ToStructureConfig`], to a [`ConvNetStructureConfig`];
/// `.init(device)` builds the
/// [`ConvNetModel`](crate::blocks::images::models::conv_net::ConvNetModel) in
/// one step, through the blanket
/// [`ModuleInit`](crate::burner::module::ModuleInit).
///
/// The catalog's rows state their geometry as this config, serialized.
#[derive(Config, Debug)]
pub struct ResNetContractConfig {
    /// The units per stage; a stage per entry.
    pub layers: Vec<usize>,

    /// The classes of the head.
    pub num_classes: usize,

    /// The input's channels: 3 for an RGB image (timm's `in_chans`).
    #[config(default = "3")]
    pub in_channels: usize,

    /// Each stage's planes, one per entry of `layers` (timm's `channels`): a
    /// basic unit's width, a bottleneck's mixer's.
    #[config(default = "vec![64, 128, 256, 512]")]
    pub channels: Vec<usize>,

    /// The `7x7` stem's width: 64 in every timm and torchvision `ResNet`.
    #[config(default = "64")]
    pub conv_stem_width: usize,

    /// The net's stride at its last stage: 32, or 16 or 8 for a dilated net.
    #[config(default = "32")]
    pub output_stride: usize,

    /// Bottleneck units when set; else basic units.
    #[config(default = "None")]
    pub bottleneck_policy: Option<BottleneckPolicyConfig>,

    /// The norm after each conv, the stem's included; its feature count is
    /// matched to each conv.
    #[config(default = "BunsenNormalizationConfig::Batch(BatchNormConfig::new(0))")]
    pub normalization: BunsenNormalizationConfig,

    /// The activation.
    #[config(default = "BunsenActivationConfig::Relu")]
    pub activation: BunsenActivationConfig,
}

impl ResNetContractConfig {
    /// Sets the default bottleneck policy, or none.
    pub fn with_bottleneck(
        self,
        enable: bool,
    ) -> Self {
        self.with_bottleneck_policy(enable.then(BottleneckPolicyConfig::default))
    }

    /// The generic conv net's policy of this `ResNet`: a `conv_stem_width`
    /// wide `7x7` stem (stride 2) and its `3x3` max pool; then a stage per
    /// entry of `layers`, the first at stride 1 and each later one at stride
    /// 2.
    ///
    /// Each stage's mixers are its `channels` wide. A basic unit puts out its
    /// mixer's width; a bottleneck `pinch_factor` times it, so a `resnet50`'s
    /// stages put out 256, 512, 1024 and 2048 channels.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when `channels`
    /// doesn't give one width per stage.
    pub fn try_to_conv_net(&self) -> BunsenResult<ConvNetContractConfig> {
        if self.channels.len() != self.layers.len() {
            return Err(BunsenError::illegal(format!(
                "ResNetContractConfig: {} channels for {} stages; one width per stage",
                self.channels.len(),
                self.layers.len()
            )));
        }
        let expansion = self
            .bottleneck_policy
            .as_ref()
            .map_or(1, |policy| policy.pinch_factor);
        let unit = match self.bottleneck_policy {
            Some(_) => ConvUnitKind::Bottleneck,
            None => ConvUnitKind::Basic,
        };
        let stages = self
            .layers
            .iter()
            .zip(&self.channels)
            .enumerate()
            .map(|(idx, (&depth, &mid))| {
                ConvStageContractConfig::new(depth, unit, mid * expansion, mid)
                    .with_stride(if idx == 0 { 1 } else { 2 })
            })
            .collect();
        Ok(ConvNetContractConfig::new(
            ConvStemContractConfig::new(vec![self.conv_stem_width]),
            stages,
            self.num_classes,
        )
        .with_in_channels(self.in_channels)
        .with_output_stride(self.output_stride)
        .with_layers(
            ConvLayersConfig::new()
                .with_norm(self.normalization.clone())
                .with_act(self.activation.clone()),
        ))
    }

    /// [`try_to_conv_net`](Self::try_to_conv_net), for a policy known to be
    /// sound.
    ///
    /// # Panics
    ///
    /// When it is not.
    pub fn to_conv_net(&self) -> ConvNetContractConfig {
        self.try_to_conv_net().unwrap_or_else(|e| panic!("{e}"))
    }
}

impl ToStructureConfig for ResNetContractConfig {
    type Structure = ConvNetStructureConfig;

    /// Lowers through [`try_to_conv_net`](Self::try_to_conv_net).
    ///
    /// # Errors
    ///
    /// As [`try_to_conv_net`](Self::try_to_conv_net), and as
    /// [`ConvNetContractConfig`]'s lowering.
    fn try_to_structure(&self) -> BunsenResult<ConvNetStructureConfig> {
        self.try_to_conv_net()?.try_to_structure()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::images::models::conv_net::{
            ConvNetMeta,
            ConvNetModel,
            ResidualUnitMeta,
        },
        burner::module::ModuleInit,
        data::pretrained::Prefabs,
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::cpu_device,
    };

    /// The prefabs' stage widths and depths are torchvision's: a basic
    /// `resnet34` puts out 64 to 512 channels, a bottleneck `resnet50` 256 to
    /// 2048 around mixers of 64 to 512.
    #[test]
    fn test_prefabs_lower_to_torchvisions_widths() {
        let widths = |name: &str| -> Vec<(usize, usize, usize)> {
            ResNetContractConfig::expect_prefab(name)
                .to_conv_net()
                .stages
                .iter()
                .map(|s| (s.depth, s.mid_width, s.width))
                .collect()
        };
        assert_eq!(
            widths("resnet34"),
            [(3, 64, 64), (4, 128, 128), (6, 256, 256), (3, 512, 512)]
        );
        assert_eq!(
            widths("resnet50"),
            [(3, 64, 256), (4, 128, 512), (6, 256, 1024), (3, 512, 2048)]
        );
        let structure = ResNetContractConfig::expect_prefab("resnet50").to_structure();
        assert_eq!(structure.num_stages(), 4);
        assert_eq!(structure.d_features(), 2048);
        assert_eq!(structure.num_classes(), 1000);
        let strides: Vec<usize> = structure.stages.iter().map(|s| s.stride()).collect();
        assert_eq!(strides, [1, 2, 2, 2]);
    }

    /// The policy builds the same model through its structure as through
    /// the blanket `init`; the norm and activation reach every conv.
    #[test]
    fn test_policy_pathways_agree() {
        let device = cpu_device();
        let policy = ResNetContractConfig::new(vec![1, 1], 7)
            .with_channels(vec![8, 16])
            .with_conv_stem_width(8)
            .with_bottleneck(true)
            .with_activation(BunsenActivationConfig::Gelu);
        let structure = policy.to_structure();
        assert_eq!(structure.d_features(), 64);
        assert!(matches!(
            structure.stem.convs[0].act,
            Some(BunsenActivationConfig::Gelu)
        ));
        assert!(matches!(
            structure.stages[1].units[0].act,
            Some(BunsenActivationConfig::Gelu)
        ));
        let direct: ConvNetModel = policy.init(&device);
        let lowered: ConvNetModel = structure.init(&device);
        assert_eq!(direct.d_features(), lowered.d_features());
        assert_eq!(direct.num_classes(), 7);
    }

    /// A policy with no stages, an empty one, or other than one width per
    /// stage is an error from `try_init`.
    #[test]
    fn test_try_init_rejects_bad_stages() {
        let device = cpu_device();
        let illegal = || ErrorMatcher::kind(BunsenErrorKind::Illegal);
        let none: BunsenResult<ConvNetModel> = ResNetContractConfig::new(vec![], 7)
            .with_channels(vec![])
            .try_init(&device);
        illegal()
            .message_contains("ConvNetStructureConfig.stages")
            .assert_err(&none);
        let empty: BunsenResult<ConvNetModel> = ResNetContractConfig::new(vec![1, 0], 7)
            .with_channels(vec![8, 16])
            .try_init(&device);
        illegal()
            .frame_contains("conv net stage 1")
            .assert_err(&empty);
        let unmatched: BunsenResult<ConvNetModel> =
            ResNetContractConfig::new(vec![1, 1], 7).try_init(&device);
        illegal()
            .message_contains("4 channels for 2 stages")
            .assert_err(&unmatched);
    }

    /// The input's channels reach the stem; the stem's width and the stage
    /// widths are their own knobs.
    #[test]
    fn test_in_channels_and_widths() {
        let device = cpu_device();
        let model: ConvNetModel = ResNetContractConfig::new(vec![1, 1], 5)
            .with_in_channels(1)
            .with_conv_stem_width(12)
            .with_channels(vec![8, 24])
            .init(&device);
        assert_eq!(model.in_channels(), 1);
        assert_eq!(model.d_features(), 24);
        let logits = model.forward(burn::Tensor::<4>::zeros([2, 1, 32, 32], &device));
        assert_eq!(logits.dims(), [2, 5]);
    }
}
