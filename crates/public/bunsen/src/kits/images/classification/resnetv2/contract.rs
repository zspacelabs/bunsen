// `ResNetV2`'s policy: timm's `ResNetV2` knobs, and their lowering onto the
// generic conv net.

use alloc::{
    format,
    vec,
    vec::Vec,
};

use burn::{
    config::Config,
    nn::{
        BatchNormConfig,
        GroupNormConfig,
    },
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
        norm::{
            BunsenNormalizationConfig,
            EvoNormS0Config,
        },
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
    },
    support::math::make_divisible,
};

/// `BiT`'s weight standardization epsilon (timm's `StdConv2d(eps=1e-8)`).
pub const BIT_STD_CONV_EPSILON: f64 = 1e-8;

/// A `ResNetV2`'s stem, timm's `stem_type`.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ResNetV2Stem {
    /// A `7x7` conv of stride 2, then a `3x3` max pool of stride 2 (timm's
    /// `''`).
    Plain,

    /// The `7x7` conv, then a zero pad of 1 and an unpadded `3x3` max pool
    /// (timm's `'fixed'`, `BiT`'s).
    Fixed,

    /// Three `3x3` convs, `stem_chs / 2`, `stem_chs / 2`, `stem_chs` wide,
    /// the first of stride 2, then the max pool (timm's `'deep'`).
    Deep,
}

/// A `ResNetV2`'s norm, timm's `norm_layer`: each with its activation.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ResNetV2Norm {
    /// A 32-group norm and the activation (timm's `GroupNormAct`, the
    /// default, `BiT`'s).
    GroupNorm,

    /// A batch norm and the activation (timm's `BatchNormAct2d`).
    BatchNorm,

    /// `EvoNorm-S0` of 32 groups, its activation its own (timm's
    /// `EvoNorm2dS0`).
    EvoNormS0,
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
#[derive(Config, Debug)]
pub struct ResNetV2ContractConfig {
    /// Each stage's units (timm's `layers`).
    pub layers: Vec<usize>,

    /// The classes; 0 for a headless model.
    pub num_classes: usize,

    /// Each stage's output width before [`width_factor`](Self::width_factor)
    /// (timm's `channels`).
    #[config(default = "vec![256, 512, 1024, 2048]")]
    pub channels: Vec<usize>,

    /// The widths' multiplier (timm's `width_factor`, `BiT`'s `x1` to `x4`);
    /// every width is then rounded to 8s.
    #[config(default = "1")]
    pub width_factor: usize,

    /// The stem's width before [`width_factor`](Self::width_factor) (timm's
    /// `stem_chs`).
    #[config(default = "64")]
    pub stem_chs: usize,

    /// The stem (timm's `stem_type`).
    #[config(default = "ResNetV2Stem::Plain")]
    pub stem_type: ResNetV2Stem,

    /// Each conv's weight standardization epsilon (timm's `conv_layer`
    /// `StdConv2d`, `BiT`'s [`BIT_STD_CONV_EPSILON`]); none when `None`.
    #[config(default = "None")]
    pub std_conv: Option<f64>,

    /// The norm, with its activation (timm's `norm_layer`).
    #[config(default = "ResNetV2Norm::GroupNorm")]
    pub norm_layer: ResNetV2Norm,

    /// The activation after each norm (timm's `act_layer`); `EvoNorm-S0`
    /// brings its own.
    #[config(default = "BunsenActivationConfig::Relu")]
    pub act_layer: BunsenActivationConfig,

    /// Two-conv units (timm's `basic`, `PreActBasic`) rather than
    /// bottlenecks.
    #[config(default = "false")]
    pub basic: bool,

    /// A bottleneck's width over its unit's (timm's `bottle_ratio`): 0.25,
    /// or 1 for basic units.
    #[config(default = "0.25")]
    pub bottle_ratio: f64,

    /// An average pool before each projection shortcut's conv (timm's
    /// `avg_down`).
    #[config(default = "false")]
    pub avg_down: bool,

    /// The image channels (timm's `in_chans`).
    #[config(default = "3")]
    pub in_channels: usize,

    /// The net's stride at its last stage: 32, or 16 or 8 for a dilated net.
    #[config(default = "32")]
    pub output_stride: usize,

    /// The head's global pool.
    #[config(default = "GlobalPool2dConfig::Avg")]
    pub global_pool: GlobalPool2dConfig,

    /// The dropout before the classifier (timm's `drop_rate`; training
    /// only).
    #[config(default = "0.0")]
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
    /// one width per stage, or no stages.
    pub fn try_conv_net(&self) -> BunsenResult<ConvNetContractConfig> {
        if self.layers.is_empty() || self.layers.len() != self.channels.len() {
            return Err(BunsenError::illegal(format!(
                "ResNetV2ContractConfig: {} stages of layers and {} of channels",
                self.layers.len(),
                self.channels.len()
            )));
        }
        let unit = match self.basic {
            true => ConvUnitKind::PreBasic,
            false => ConvUnitKind::PreBottleneck,
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
        let (norm, act) = match self.norm_layer {
            ResNetV2Norm::GroupNorm => (
                BunsenNormalizationConfig::Group(GroupNormConfig::new(32, 0)),
                self.act_layer.clone(),
            ),
            ResNetV2Norm::BatchNorm => (
                BunsenNormalizationConfig::Batch(BatchNormConfig::new(0)),
                self.act_layer.clone(),
            ),
            ResNetV2Norm::EvoNormS0 => (
                BunsenNormalizationConfig::EvoS0(EvoNormS0Config::new(0)),
                BunsenActivationConfig::Identity,
            ),
        };
        let layers = ConvLayersConfig::new()
            .with_norm(norm)
            .with_act(act)
            .with_shortcut(match self.avg_down {
                true => ConvShortcutKind::AvgConv,
                false => ConvShortcutKind::Conv,
            })
            .with_weight_std(self.std_conv)
            .with_pre_identity(false);
        let stem_width = self.stem_width();
        let stem = match self.stem_type {
            ResNetV2Stem::Plain | ResNetV2Stem::Fixed => {
                ConvStemContractConfig::new(vec![stem_width])
            }
            ResNetV2Stem::Deep => {
                ConvStemContractConfig::new(vec![stem_width / 2, stem_width / 2, stem_width])
                    .with_kernel_size(3)
            }
        }
        .with_pool(match self.stem_type {
            ResNetV2Stem::Fixed => ConvStemPoolKind::ZeroPadMax,
            ResNetV2Stem::Plain | ResNetV2Stem::Deep => ConvStemPoolKind::Max,
        })
        .with_last_norm_act(false);
        Ok(ConvNetContractConfig::new(stem, stages, self.num_classes)
            .with_in_channels(self.in_channels)
            .with_output_stride(self.output_stride)
            .with_layers(layers)
            .with_final_conv(Some(ConvFinalKind::NormAct))
            .with_global_pool(self.global_pool)
            .with_drop_rate(self.drop_rate))
    }
}

impl ToStructureConfig for ResNetV2ContractConfig {
    type Structure = ConvNetStructureConfig;

    /// The conv net of this `ResNetV2` ([`try_conv_net`](Self::try_conv_net)),
    /// resolved.
    ///
    /// # Errors
    ///
    /// As [`try_conv_net`](Self::try_conv_net) and the conv net's lowering,
    /// under a frame naming the `ResNetV2`.
    fn try_to_structure(&self) -> BunsenResult<ConvNetStructureConfig> {
        self.try_conv_net()
            .and_then(|net| net.try_to_structure())
            .context("ResNetV2ContractConfig")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::{
            conv::ConvBlock2dMeta,
            images::models::conv_net::{
                ConvChainMeta,
                ConvNetMeta,
                ConvNetModel,
            },
        },
        burner::module::ModuleInit,
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::cpu_device,
    };

    /// timm's `resnetv2_50x1_bit`.
    fn bit50() -> ResNetV2ContractConfig {
        ResNetV2ContractConfig::new(vec![3, 4, 6, 3], 1000)
            .with_stem_type(ResNetV2Stem::Fixed)
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
            .with_norm_layer(ResNetV2Norm::BatchNorm)
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
            .with_stem_type(ResNetV2Stem::Deep)
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

    #[test]
    fn test_illegal_policies() {
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("3 stages of layers and 4 of channels")
            .assert_err(&ResNetV2ContractConfig::new(vec![3, 4, 6], 1000).try_to_structure());
    }
}
