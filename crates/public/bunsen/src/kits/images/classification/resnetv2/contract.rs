// `ResNetV2`'s policy: timm's `ResNetV2` knobs, and their lowering onto the
// generic conv net.

use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    config::Config,
    nn::GroupNormConfig,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        images::models::conv_net::{
            ConvFinalKind,
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
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
    support::math::make_divisible,
};

/// `BiT`'s weight standardization epsilon (timm's `StdConv2d(eps=1e-8)`).
pub const BIT_STD_CONV_EPSILON: f64 = 1e-8;

/// A `ResNetV2`'s stem, timm's `stem_type`.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ResNetV2StemKind {
    /// A `7x7` conv of stride 2, then a `3x3` max pool of stride 2 (timm's
    /// `''`).
    Plain,

    /// The `7x7` conv, then a zero pad of 1 and an unpadded `3x3` max pool
    /// (timm's `'fixed'`, `BiT`'s).
    Fixed,

    /// The `7x7` conv, then a `3x3` max pool of stride 2 over the map
    /// SAME-padded from its own size (timm's `'same'`, the hybrid `ViT`'s
    /// backbone).
    Same,

    /// Three `3x3` convs, `stem_chs / 2`, `stem_chs / 2`, `stem_chs` wide,
    /// the first of stride 2, then the max pool (timm's `'deep'`).
    Deep,
}

/// A `ResNetV2`'s policy: timm's `ResNetV2` knobs, by their names, lowering
/// to the generic [`ConvNetContractConfig`].
///
/// A `ResNetV2` is a pre-activation `ResNet`: a stem with no norm of its own,
/// four stages of pre-activation units (each unit's input normalized and
/// activated once, read by its path and its projection shortcut, the
/// identity shortcut the raw input), a final norm and activation, and a
/// pooled classifier. `BiT` standardizes every conv's weight
/// ([`std_conv`](Self::std_conv)) and group-normalizes.
///
/// Not [`preact`](Self::preact), it is the hybrid `ViT`'s backbone (timm's
/// `_resnetv2`): a normed stem, post-activation bottlenecks (a `ResNet`'s,
/// the projection shortcut normed), no final norm, and usually no head; with
/// no [`layers`](Self::layers), its stem alone.
#[derive(Config, Debug)]
pub struct ResNetV2ContractConfig {
    /// Each stage's units (timm's `layers`); none for a stem alone (the
    /// hybrid `ViT`'s `vit_tiny_r_s16_p8`).
    pub layers: Vec<usize>,

    /// The classes; 0 for a headless model.
    pub num_classes: usize,

    /// Each stage's output width before [`width_factor`](Self::width_factor)
    /// (timm's `channels`).
    #[config(default = "vec![256, 512, 1024, 2048]")]
    pub channels: Vec<usize>,

    /// The widths' multiplier (timm's `width_factor`, `BiT`'s `x1` to `x4`);
    /// every width is then rounded to 8s.
    #[config(default = 1)]
    pub width_factor: usize,

    /// The stem's width before [`width_factor`](Self::width_factor) (timm's
    /// `stem_chs`).
    #[config(default = 64)]
    pub stem_chs: usize,

    /// The stem (timm's `stem_type`).
    #[config(default = "ResNetV2StemKind::Plain")]
    pub stem_type: ResNetV2StemKind,

    /// Each conv's weight standardization epsilon (timm's `conv_layer`
    /// `StdConv2d`, `BiT`'s [`BIT_STD_CONV_EPSILON`]); none when `None`.
    #[config(default = "None")]
    pub std_conv: Option<f64>,

    /// Whether each conv SAME-pads its input from its own size (timm's
    /// `conv_layer` `StdConv2dSame` or `Conv2dSame`; the hybrid `ViT`'s).
    #[config(default = false)]
    pub same_pad: bool,

    /// Pre-activation units, a bare stem and a final norm (timm's `preact`,
    /// every published `ResNetV2`'s); `false`, the hybrid `ViT`'s backbone.
    #[config(default = true)]
    pub preact: bool,

    /// The norm (timm's `norm_layer`), its width filled in where it sits: a
    /// 32-group norm by default (`GroupNormAct`, `BiT`'s), a batch norm
    /// (`BatchNormAct2d`), or `EvoNorm-S0` (`EvoNorm2dS0`), which brings its
    /// own activation.
    #[config(default = "BunsenNormalizationConfig::Group(GroupNormConfig::new(32, 0))")]
    pub norm: BunsenNormalizationConfig,

    /// The activation after each norm (timm's `act_layer`); none after an
    /// `EvoNorm-S0`, which brings its own.
    #[config(default = "BunsenActivationConfig::Relu")]
    pub act: BunsenActivationConfig,

    /// Two-conv units (timm's `basic`, `PreActBasic`) rather than
    /// bottlenecks.
    #[config(default = false)]
    pub basic: bool,

    /// A bottleneck's width over its unit's (timm's `bottle_ratio`): 0.25,
    /// or 1 for basic units.
    #[config(default = 0.25)]
    pub bottle_ratio: f64,

    /// An average pool before each projection shortcut's conv (timm's
    /// `avg_down`).
    #[config(default = false)]
    pub avg_down: bool,

    /// The image channels (timm's `in_chans`).
    #[config(default = 3)]
    pub in_channels: usize,

    /// The net's stride at its last stage: 32, or 16 or 8 for a dilated net.
    #[config(default = 32)]
    pub output_stride: usize,

    /// The head's global pool.
    #[config(default = "GlobalPool2dConfig::Avg")]
    pub global_pool: GlobalPool2dConfig,

    /// The dropout before the classifier (timm's `drop_rate`; training
    /// only).
    #[config(default = 0.0)]
    pub drop_rate: f64,
}

impl ResNetV2ContractConfig {
    /// The stem's width: `make_divisible(stem_chs * width_factor)`.
    pub fn stem_width(&self) -> usize {
        make_divisible((self.stem_chs * self.width_factor) as f64, 8, None, 0.9)
    }

    /// Each stage's `(width, bottleneck width)`, as timm rounds them:
    /// `make_divisible(channels * width_factor)`, and
    /// `make_divisible(width * bottle_ratio)`.
    pub fn stage_widths(&self) -> Vec<(usize, usize)> {
        self.channels
            .iter()
            .map(|&c| {
                let width = make_divisible((c * self.width_factor) as f64, 8, None, 0.9);
                let mid = make_divisible(width as f64 * self.bottle_ratio, 8, None, 0.9);
                (width, mid)
            })
            .collect()
    }

    /// The generic conv net's policy of this `ResNetV2`.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for other than
    /// one width per stage, no stages and a pre-activation net (whose stem
    /// is bare), or basic units on a net that is not pre-activation (timm
    /// builds its bottlenecks whatever `basic` says).
    pub fn try_to_conv_net(&self) -> BunsenResult<ConvNetContractConfig> {
        if self.layers.len() != self.channels.len() {
            return Err(ConstraintError::new(
                "ResNetV2ContractConfig",
                "channels",
                Rule::Relation {
                    lhs: ("channels.len()".into(), self.channels.len().to_string()),
                    op: "==",
                    rhs: ("layers.len()".into(), self.layers.len().to_string()),
                },
            )
            .into());
        }
        if self.layers.is_empty() && self.preact {
            return Err(ConstraintError::custom(
                "ResNetV2ContractConfig",
                "layers",
                "a pre-activation net of no stages, its stem's conv bare",
            )
            .into());
        }
        let unit = match (self.preact, self.basic) {
            (true, true) => ConvUnitKind::PreBasic,
            (true, false) => ConvUnitKind::PreBottleneck,
            (false, false) => ConvUnitKind::Bottleneck,
            (false, true) => {
                return Err(ConstraintError::custom(
                    "ResNetV2ContractConfig",
                    "basic",
                    "basic units on a net that is not pre-activation; timm's are bottlenecks",
                )
                .into());
            }
        };
        let stages = self
            .layers
            .iter()
            .zip(self.stage_widths())
            .enumerate()
            .map(|(idx, (&depth, (width, mid)))| {
                ConvStageContractConfig::new(depth, unit, width, mid).with_stride(if idx == 0 {
                    1
                } else {
                    2
                })
            })
            .collect();
        let norm = self.norm.clone();
        let act = match norm {
            BunsenNormalizationConfig::EvoS0(_) => BunsenActivationConfig::Identity,
            _ => self.act.clone(),
        };
        let layers = ConvLayersConfig::new()
            .with_norm(norm)
            .with_act(act)
            .with_shortcut(match self.avg_down {
                true => ConvShortcutKind::AvgConv,
                false => ConvShortcutKind::Conv,
            })
            .with_weight_std(self.std_conv)
            .with_same_pad(self.same_pad)
            .with_pre_identity(false);
        let stem_width = self.stem_width();
        let stem = match self.stem_type {
            ResNetV2StemKind::Plain | ResNetV2StemKind::Fixed | ResNetV2StemKind::Same => {
                ConvStemContractConfig::new(vec![stem_width])
            }
            ResNetV2StemKind::Deep => {
                ConvStemContractConfig::new(vec![stem_width / 2, stem_width / 2, stem_width])
                    .with_kernel_size(3)
            }
        }
        .with_pool(match self.stem_type {
            ResNetV2StemKind::Fixed => ConvStemPoolKind::ZeroPadMax,
            ResNetV2StemKind::Same => ConvStemPoolKind::SameMax,
            ResNetV2StemKind::Plain | ResNetV2StemKind::Deep => ConvStemPoolKind::Max,
        })
        .with_last_norm_act(!self.preact);
        Ok(ConvNetContractConfig::new(stem, stages, self.num_classes)
            .with_in_channels(self.in_channels)
            .with_output_stride(self.output_stride)
            .with_layers(layers)
            .with_final_conv(self.preact.then_some(ConvFinalKind::NormAct))
            .with_global_pool(self.global_pool)
            .with_drop_rate(self.drop_rate))
    }
}

impl ToStructureConfig for ResNetV2ContractConfig {
    type Structure = ConvNetStructureConfig;

    /// The conv net of this `ResNetV2` ([`try_to_conv_net`](Self::try_to_conv_net)),
    /// resolved.
    ///
    /// # Errors
    ///
    /// As [`try_to_conv_net`](Self::try_to_conv_net) and the conv net's lowering,
    /// under a frame naming the `ResNetV2`.
    fn try_to_structure(&self) -> BunsenResult<ConvNetStructureConfig> {
        self.try_to_conv_net()
            .and_then(|net| net.try_to_structure())
            .context("ResNetV2ContractConfig")
    }
}

#[cfg(test)]
mod tests {
    use burn::nn::BatchNormConfig;

    use super::*;
    use crate::{
        blocks::{
            conv::ConvBlock2dMeta,
            images::models::conv_net::{
                ConvChainMeta,
                ConvNetMeta,
                ConvNetModel,
                ConvStemPoolConfig,
            },
        },
        burner::module::ModuleInit,
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    /// timm's `resnetv2_50x1_bit`.
    fn bit50() -> ResNetV2ContractConfig {
        ResNetV2ContractConfig::new(vec![3, 4, 6, 3], 1000)
            .with_stem_type(ResNetV2StemKind::Fixed)
            .with_std_conv(Some(BIT_STD_CONV_EPSILON))
    }

    /// `BiT`'s widths and lowering: a bare `7x7` stem conv and the
    /// zero-padded pool, pre-activation bottlenecks whose identity shortcut
    /// is the raw input, every conv standardized, a final norm.
    #[test]
    fn test_bit_lowering() {
        assert_eq!(
            bit50().stage_widths(),
            [(256, 64), (512, 128), (1024, 256), (2048, 512)]
        );
        let x3 = bit50().with_width_factor(3);
        assert_eq!(x3.stem_width(), 192);
        assert_eq!(x3.stage_widths()[0], (768, 192));

        let structure = bit50().to_structure();
        assert_eq!(structure.stem.stride(), 4);
        let unit = &structure.stages[0].units[1];
        assert!(unit.pre.is_some() && unit.shortcut.is_none() && !unit.pre_identity);
        assert!(structure.stages[1].units[0].shortcut.is_some());
        assert!(structure.final_conv.is_some());
        assert_eq!(structure.d_features(), 2048);

        let device = cpu_device();
        let model: ConvNetModel = ResNetV2ContractConfig::new(vec![1, 1, 1, 1], 7)
            .with_channels(vec![16, 32, 64, 128])
            .with_stem_chs(16)
            .with_norm(BatchNormConfig::new(0).into())
            .init(&device);
        assert_eq!(model.num_classes(), 7);
    }

    /// The deep stem's widths, and basic units.
    #[test]
    fn test_deep_basic() {
        let config = ResNetV2ContractConfig::new(vec![2, 2, 2, 2], 1000)
            .with_channels(vec![64, 128, 256, 512])
            .with_basic(true)
            .with_bottle_ratio(1.0)
            .with_stem_type(ResNetV2StemKind::Deep)
            .with_avg_down(true);
        let structure = config.to_structure();
        let widths: Vec<_> = structure
            .stem
            .convs
            .iter()
            .map(|c| c.out_channels())
            .collect();
        assert_eq!(widths, [32, 32, 64]);
        assert_eq!(structure.stages[0].units[0].convs.len(), 2);
    }

    /// The hybrid `ViT`'s backbone (timm's `_resnetv2((3, 4, 9))`): a normed
    /// stem and its SAME-padded pool, post-activation bottlenecks, every conv
    /// standardized and SAME-padded, no final norm, no head; with no layers,
    /// its stem alone.
    #[test]
    fn test_hybrid_backbone_lowering() {
        let r50 = ResNetV2ContractConfig::new(vec![3, 4, 9], 0)
            .with_channels(vec![256, 512, 1024])
            .with_stem_type(ResNetV2StemKind::Same)
            .with_std_conv(Some(BIT_STD_CONV_EPSILON))
            .with_same_pad(true)
            .with_preact(false);
        let structure = r50.to_structure();
        let stem = &structure.stem.convs[0];
        assert!(stem.norm.is_some() && stem.act.is_some() && stem.same_pad);
        assert!(matches!(
            structure.stem.pool,
            Some(ConvStemPoolConfig::SameMax(_))
        ));
        let unit = &structure.stages[1].units[0];
        assert!(unit.pre.is_none() && unit.act.is_some());
        let shortcut = &unit.shortcut.as_ref().unwrap().block;
        assert!(shortcut.norm.is_some() && shortcut.act.is_none() && shortcut.same_pad);
        assert!(structure.final_conv.is_none());
        assert_eq!(structure.d_features(), 1024);
        assert_eq!(structure.try_feature_size([384, 384]).unwrap(), [24, 24]);

        let stem_only = ResNetV2ContractConfig {
            layers: vec![],
            channels: vec![],
            ..r50
        };
        let structure = stem_only.to_structure();
        assert!(structure.stages.is_empty());
        assert_eq!(structure.d_features(), 64);
        assert_eq!(structure.try_feature_size([224, 224]).unwrap(), [56, 56]);
    }

    #[test]
    fn test_illegal_policies() {
        let illegal = || ErrorMatcher::illegal();
        illegal()
            .message_contains("channels.len() (4) must be == layers.len() (3)")
            .assert_err(&ResNetV2ContractConfig::new(vec![3, 4, 6], 1000).try_to_structure());
        illegal()
            .message_contains("a pre-activation net of no stages")
            .assert_err(
                &ResNetV2ContractConfig::new(vec![], 0)
                    .with_channels(vec![])
                    .try_to_structure(),
            );
        illegal()
            .message_contains("basic units on a net that is not pre-activation")
            .assert_err(
                &ResNetV2ContractConfig::new(vec![2, 2, 2, 2], 0)
                    .with_basic(true)
                    .with_preact(false)
                    .try_to_structure(),
            );
    }
}
