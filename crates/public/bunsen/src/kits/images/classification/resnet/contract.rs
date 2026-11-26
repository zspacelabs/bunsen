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
            ConvAaKind,
            ConvAttentionKind,
            ConvLayersConfig,
            ConvNetContractConfig,
            ConvNetStructureConfig,
            ConvShortcutKind,
            ConvStageContractConfig,
            ConvStemContractConfig,
            ConvStemPoolKind,
            ConvUnitKind,
            GlobalPool2dConfig,
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

/// A `ResNet`'s bottleneck policy: its units are bottlenecks, which put out
/// `pinch_factor` times their stage's planes, around a grouped mixer.
///
/// The mixer is `planes * base_width / 64` channels (rounded down) per group,
/// in `cardinality` groups, as timm's: a `resnet50` is 1 group of 64 per 64
/// planes, a `resnext50_32x4d` 32 groups of 4, a `wide_resnet50_2` 1 of 128.
#[derive(Config, Debug)]
pub struct BottleneckPolicyConfig {
    /// The output channels over the stage's planes (timm's expansion).
    #[config(default = "BOTTLENECK_BLOCK_DEFAULT_PINCH_FACTOR")]
    pub pinch_factor: usize,

    /// The mixer's groups (timm's `cardinality`).
    #[config(default = "1")]
    pub cardinality: usize,

    /// The mixer's channels per group, per 64 planes (timm's `base_width`).
    #[config(default = "64")]
    pub base_width: usize,
}

impl BottleneckPolicyConfig {
    /// The mixer's width of a stage of `planes` planes:
    /// `floor(planes * base_width / 64) * cardinality`.
    pub fn mixer_width(
        &self,
        planes: usize,
    ) -> usize {
        planes * self.base_width / 64 * self.cardinality
    }
}

impl Default for BottleneckPolicyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A `ResNet`'s stem: its convs, before the max pool (timm's `stem_type`).
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ResNetStemKind {
    /// One `7x7` conv, `conv_stem_width` wide (timm's `''`, torchvision's).
    Single,

    /// Three `3x3` convs, `stem_width`, `stem_width` and twice it wide
    /// (timm's `'deep'`: the C, D, E and S variants).
    Deep,

    /// Three `3x3` convs, `3 * (stem_width / 4)`, `stem_width` and twice it
    /// wide (timm's `'deep_tiered'`: the T variants).
    Tiered,
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

    /// The stem's kind.
    #[config(default = "ResNetStemKind::Single")]
    pub stem_type: ResNetStemKind,

    /// A deep or tiered stem's width (timm's `stem_width`): its output is
    /// twice it.
    #[config(default = "64")]
    pub stem_width: usize,

    /// A single-conv stem's width: 64 in every timm and torchvision `ResNet`.
    #[config(default = "64")]
    pub conv_stem_width: usize,

    /// Average-pool shortcuts (timm's `avg_down`, the D, E and T variants): a
    /// `2x2` average pool of the unit's stride, then a `1x1` conv; else a
    /// strided `1x1` conv.
    #[config(default = "false")]
    pub avg_down: bool,

    /// Each unit's channel attention, after its last norm (timm's
    /// `block_args.attn_layer`): `'se'` is
    /// [`ConvAttentionKind::SE`], `'eca'` [`ConvAttentionKind::Eca`].
    #[config(default = "None")]
    pub attention: Option<ConvAttentionKind>,

    /// Anti-aliased downsampling (timm's `aa_layer`): each unit that strides
    /// 2 moves its stride from its mixer onto this pool, after the mixer's
    /// activation, and the stem's max pool becomes a `2x2` average pool
    /// ([`Avg`](ConvAaKind::Avg), `nn.AvgPool2d`) or a stride-1 max pool and
    /// a blur of stride 2 ([`Blur`](ConvAaKind::Blur), `BlurPool2d`).
    #[config(default = "None")]
    pub anti_alias: Option<ConvAaKind>,

    /// The stem's max pool replaced by a `3x3` conv of stride 2, its norm and
    /// its activation (timm's `replace_stem_pool`, `ResNet-RS`'s).
    #[config(default = "false")]
    pub replace_stem_pool: bool,

    /// The head's global pool (timm's `global_pool`): `avg`, `max`, `avgmax`
    /// or `catavgmax`.
    #[config(default = "GlobalPool2dConfig::Avg")]
    pub global_pool: GlobalPool2dConfig,

    /// The dropout before the classifier (timm's `drop_rate`; training
    /// only).
    #[config(default = "0.0")]
    pub drop_rate: f64,

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

    /// The generic conv net's policy of this `ResNet`: its stem
    /// ([`ResNetStemKind`]; its first conv strides 2) and a `3x3` max pool
    /// (or an anti-aliased pool, [`anti_alias`](Self::anti_alias));
    /// then a stage per entry of `layers`, the first at stride 1 and each
    /// later one at stride 2.
    ///
    /// Each stage's units are its `channels` planes. A basic unit is that
    /// wide throughout; a bottleneck puts out `pinch_factor` times it, so a
    /// `resnet50`'s stages put out 256, 512, 1024 and 2048 channels, around
    /// its policy's grouped mixer
    /// ([`mixer_width`](BottleneckPolicyConfig::mixer_width)).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for no stages, or
    /// when `channels` doesn't give one width per stage;
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// replaced stem pool with anti-aliasing.
    pub fn try_to_conv_net(&self) -> BunsenResult<ConvNetContractConfig> {
        if self.layers.is_empty() {
            return Err(BunsenError::illegal(
                "ResNetContractConfig: a ResNet has at least one stage",
            ));
        }
        if self.channels.len() != self.layers.len() {
            return Err(BunsenError::illegal(format!(
                "ResNetContractConfig: {} channels for {} stages; one width per stage",
                self.channels.len(),
                self.layers.len()
            )));
        }
        let stages = self
            .layers
            .iter()
            .zip(&self.channels)
            .enumerate()
            .map(|(idx, (&depth, &planes))| {
                let stage = match &self.bottleneck_policy {
                    Some(policy) => ConvStageContractConfig::new(
                        depth,
                        ConvUnitKind::Bottleneck,
                        planes * policy.pinch_factor,
                        policy.mixer_width(planes),
                    )
                    .with_groups(policy.cardinality),
                    None => {
                        ConvStageContractConfig::new(depth, ConvUnitKind::Basic, planes, planes)
                    }
                };
                stage.with_stride(if idx == 0 { 1 } else { 2 })
            })
            .collect();
        let w = self.stem_width;
        let stem = match self.stem_type {
            ResNetStemKind::Single => ConvStemContractConfig::new(vec![self.conv_stem_width]),
            ResNetStemKind::Deep => {
                ConvStemContractConfig::new(vec![w, w, 2 * w]).with_kernel_size(3)
            }
            ResNetStemKind::Tiered => {
                ConvStemContractConfig::new(vec![3 * (w / 4), w, 2 * w]).with_kernel_size(3)
            }
        };
        let stem = stem.with_pool(match (self.replace_stem_pool, self.anti_alias) {
            (true, None) => ConvStemPoolKind::Conv,
            (true, Some(_)) => {
                return Err(BunsenError::unsupported(
                    "ResNetContractConfig: an anti-aliased conv stem pool (timm's \
                     replace_stem_pool with an aa_layer) is not built",
                ));
            }
            (false, None) => ConvStemPoolKind::Max,
            (false, Some(ConvAaKind::Avg)) => ConvStemPoolKind::Avg,
            (false, Some(ConvAaKind::Blur)) => ConvStemPoolKind::MaxBlur,
        });
        let shortcut = if self.avg_down {
            ConvShortcutKind::AvgConv
        } else {
            ConvShortcutKind::Conv
        };
        Ok(ConvNetContractConfig::new(stem, stages, self.num_classes)
            .with_global_pool(self.global_pool)
            .with_drop_rate(self.drop_rate)
            .with_in_channels(self.in_channels)
            .with_output_stride(self.output_stride)
            .with_layers(
                ConvLayersConfig::new()
                    .with_norm(self.normalization.clone())
                    .with_act(self.activation.clone())
                    .with_shortcut(shortcut)
                    .with_attn(self.attention)
                    .with_aa(self.anti_alias),
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
            ConvChainMeta,
            ConvNetMeta,
            ConvNetModel,
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

    /// The grouped and widened prefabs' mixers are timm's
    /// `floor(planes * base_width / 64) * cardinality`, in `cardinality`
    /// groups; their outputs are a `resnet50`'s.
    #[test]
    fn test_grouped_prefabs_lower_to_timms_widths() {
        let stages = |name: &str| -> Vec<(usize, usize, usize)> {
            ResNetContractConfig::expect_prefab(name)
                .to_conv_net()
                .stages
                .iter()
                .map(|s| (s.mid_width, s.width, s.groups))
                .collect()
        };
        assert_eq!(
            stages("resnext50_32x4d"),
            [
                (128, 256, 32),
                (256, 512, 32),
                (512, 1024, 32),
                (1024, 2048, 32)
            ]
        );
        assert_eq!(
            stages("resnext101_64x4d"),
            [
                (256, 256, 64),
                (512, 512, 64),
                (1024, 1024, 64),
                (2048, 2048, 64)
            ]
        );
        assert_eq!(
            stages("wide_resnet50_2"),
            [
                (128, 256, 1),
                (256, 512, 1),
                (512, 1024, 1),
                (1024, 2048, 1)
            ]
        );
        assert_eq!(stages("resnet50")[0], (64, 256, 1));
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
            .message_contains("a ResNet has at least one stage")
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
