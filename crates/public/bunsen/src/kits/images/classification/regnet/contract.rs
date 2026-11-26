// `RegNet`'s policy: timm's `RegNetCfg`, its stages, and its lowering onto
// the generic conv net.

use alloc::{
    format,
    vec,
    vec::Vec,
};

use burn::config::Config;
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        images::models::conv_net::{
            ConvAttentionKind,
            ConvAttentionPlace,
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
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
    },
    support::math::{
        make_divisible,
        round_ties_even,
    },
};

/// The width every generated `RegNet` width is a multiple of (timm's
/// `generate_regnet` `quant`).
pub const REGNET_WIDTH_QUANT: usize = 8;

/// One stage of a `RegNet`, as timm resolves it: its depth, its units' output
/// width, their bottleneck (mixer) width, and the mixer's groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegNetStage {
    /// The units.
    pub depth: usize,

    /// Each unit's output channels.
    pub width: usize,

    /// The bottleneck's channels, the grouped `3x3` mixer's.
    pub mid_width: usize,

    /// The mixer's groups.
    pub groups: usize,
}

/// A `RegNet` unit's shortcut where it changes shape, timm's `downsample`.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum RegNetDownsample {
    /// A `1x1` conv of the unit's stride and a norm (timm's `'conv1x1'`).
    Conv1x1,

    /// A `2x2` average pool of the stride, then a `1x1` conv and a norm
    /// (timm's `'avg'`).
    Avg,

    /// None: a unit that changes shape has no residual (timm's `None`,
    /// `RegNetZ`'s).
    None,
}

/// A `RegNet`'s policy: timm's `RegNetCfg`, by its names, lowering to the
/// generic [`ConvNetContractConfig`].
///
/// A `RegNet` is a `3x3` stem of stride 2 (no pool), four stages of
/// bottleneck units (a `1x1`, a grouped `3x3` mixer that strides, a
/// squeeze-excite after it when [`se_ratio`](Self::se_ratio) is set, a `1x1`;
/// a shortcut where a unit changes shape, as
/// [`downsample`](Self::downsample) says), and a pooled linear head;
/// `RegNetV` pre-activates its units ([`preact`](Self::preact)), `RegNetZ`
/// inverts its bottleneck, ends each unit linear and has no shortcut convs
/// ([`bottle_ratio`](Self::bottle_ratio),
/// [`linear_out`](Self::linear_out)). Its widths are not stated but generated from a linear rule
/// quantized to a geometric one ([`try_stages`](Self::try_stages)).
#[derive(Config, Debug)]
pub struct RegNetContractConfig {
    /// The units, over all four stages (timm's `depth`).
    pub depth: usize,

    /// The first unit's width (timm's `w0`), a multiple of
    /// [`REGNET_WIDTH_QUANT`].
    pub w0: usize,

    /// The widths' slope per unit (timm's `wa`).
    pub wa: f64,

    /// The ratio of one stage's width to the last's (timm's `wm`), over 1.
    pub wm: f64,

    /// The channels of each group of the mixer (timm's `group_size`).
    pub group_size: usize,

    /// The classes; 0 for a headless trunk.
    pub num_classes: usize,

    /// The bottleneck's width over the unit's (timm's `bottle_ratio`).
    #[config(default = "1.0")]
    pub bottle_ratio: f64,

    /// The squeeze-excite after the mixer, its reduction `round(in *
    /// se_ratio)` of the unit's input (timm's `se_ratio`); 0, none
    /// (`RegNetX`).
    #[config(default = "0.0")]
    pub se_ratio: f64,

    /// The floor under which a bottleneck width rounded to its groups
    /// rounds up instead (timm's `group_min_ratio`, the `_tv` archs' 0.9);
    /// 0 rounds to the nearest.
    #[config(default = "0.0")]
    pub group_min_ratio: f64,

    /// The stem's width (timm's `stem_width`).
    #[config(default = "32")]
    pub stem_width: usize,

    /// The shortcut of a unit that changes shape (timm's `downsample`).
    #[config(default = "RegNetDownsample::Conv1x1")]
    pub downsample: RegNetDownsample,

    /// No activation after each unit's add, and one after the last stage
    /// instead (timm's `linear_out`, `RegNetZ`'s).
    #[config(default = "false")]
    pub linear_out: bool,

    /// Pre-activation units (timm's `preact`, `RegNetV`'s `PreBottleneck`): a
    /// bare stem conv, each unit's input normalized and activated before its
    /// path and shortcut, each later conv after its own norm and activation,
    /// and an activation after the last stage.
    #[config(default = "false")]
    pub preact: bool,

    /// A `1x1` conv, norm and activation to this many features after the
    /// last stage (timm's `num_features`, `regnetz_040_h`'s 1536); 0, none.
    #[config(default = "0")]
    pub num_features: usize,

    /// The activation after each conv's norm and each unit's add (timm's
    /// `act_layer`).
    #[config(default = "BunsenActivationConfig::Relu")]
    pub act_layer: BunsenActivationConfig,

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

impl RegNetContractConfig {
    /// The per-unit widths of timm's `generate_regnet`, computed as it
    /// computes them, in float32: `w0 + i * wa` for each unit `i`, its
    /// exponent `round(log(w / w0) / log(wm))`, and the width `w0 *
    /// wm^exponent` rounded to [`REGNET_WIDTH_QUANT`]s, rounding ties to
    /// even as `torch.round` does.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a negative
    /// slope, a `wm` not over 1, a `w0` of 0 or not a multiple of
    /// [`REGNET_WIDTH_QUANT`], or no units.
    pub fn try_unit_widths(&self) -> BunsenResult<Vec<usize>> {
        let quant = REGNET_WIDTH_QUANT;
        if self.wa < 0.0
            || self.wm <= 1.0
            || self.w0 == 0
            || !self.w0.is_multiple_of(quant)
            || self.depth == 0
        {
            return Err(BunsenError::illegal(format!(
                "RegNetContractConfig: widths need wa >= 0, wm > 1, w0 a nonzero multiple of \
                 {quant} and a depth; wa {}, wm {}, w0 {}, depth {}",
                self.wa, self.wm, self.w0, self.depth
            )));
        }
        // torch casts the Python floats to the tensors' float32; `math.log`
        // is a double, cast the same way.
        let (w0, wa, wm) = (self.w0 as f32, self.wa as f32, self.wm as f32);
        let log_wm = self.wm.ln() as f32;
        Ok((0..self.depth)
            .map(|i| {
                let width = i as f32 * wa + w0;
                let exponent = round_ties_even(f64::from((width / w0).ln() / log_wm)) as f32;
                round_ties_even(f64::from(w0 * wm.powf(exponent) / quant as f32)) * quant
            })
            .collect())
    }

    /// The four stages, as timm's `RegNet._get_stage_args` resolves them:
    /// each distinct generated width a stage, its count the depth; the
    /// bottleneck widths `int(width * bottle_ratio)` rounded to the groups
    /// (`adjust_widths_groups_comp`: to the nearest, or by `make_divisible`
    /// under [`group_min_ratio`](Self::group_min_ratio)), each group at most
    /// the bottleneck; the widths back from them. A unit's mixer is then
    /// `round(width * bottle_ratio)` wide, in groups of the group size.
    ///
    /// # Errors
    ///
    /// As [`try_unit_widths`](Self::try_unit_widths);
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a group size
    /// of 0, a bottleneck ratio not over 0, or widths that make other than
    /// four stages.
    pub fn try_stages(&self) -> BunsenResult<Vec<RegNetStage>> {
        let widths = self.try_unit_widths()?;
        if self.group_size == 0 || self.bottle_ratio <= 0.0 {
            return Err(BunsenError::illegal(format!(
                "RegNetContractConfig: a group size of {} and a bottleneck ratio of {}",
                self.group_size, self.bottle_ratio
            )));
        }
        // `torch.unique(.., return_counts=True)`: the distinct widths in
        // order, each with its count.
        let mut sorted = widths;
        sorted.sort_unstable();
        let mut runs: Vec<(usize, usize)> = vec![];
        for width in sorted {
            match runs.last_mut() {
                Some((w, count)) if *w == width => *count += 1,
                _ => runs.push((width, 1)),
            }
        }
        if runs.len() != 4 {
            return Err(BunsenError::illegal(format!(
                "RegNetContractConfig: the widths make {} stages, not 4: {runs:?}",
                runs.len()
            )));
        }
        let b = self.bottle_ratio;
        Ok(runs
            .into_iter()
            .map(|(width, depth)| {
                let w_bot = (width as f64 * b) as usize;
                let group = self.group_size.min(w_bot);
                let w_bot = if self.group_min_ratio != 0.0 {
                    make_divisible(w_bot as f64, group, None, self.group_min_ratio)
                } else {
                    // timm's `quantize_float`: `int(round(f / q) * q)`.
                    round_ties_even(w_bot as f64 / group as f64) * group
                };
                let width = (w_bot as f64 / b) as usize;
                let mid_width = round_ties_even(width as f64 * b);
                RegNetStage {
                    depth,
                    width,
                    mid_width,
                    groups: mid_width / group,
                }
            })
            .collect())
    }

    /// The generic conv net's policy of this `RegNet`.
    ///
    /// # Errors
    ///
    /// As [`try_stages`](Self::try_stages).
    pub fn try_conv_net(&self) -> BunsenResult<ConvNetContractConfig> {
        let stages = self
            .try_stages()?
            .into_iter()
            .map(|stage| {
                ConvStageContractConfig::new(
                    stage.depth,
                    match self.preact {
                        true => ConvUnitKind::PreBottleneck,
                        false => ConvUnitKind::Bottleneck,
                    },
                    stage.width,
                    stage.mid_width,
                )
                .with_groups(stage.groups)
                .with_stride(2)
            })
            .collect();
        let layers = ConvLayersConfig::new()
            .with_act(self.act_layer.clone())
            .with_shortcut(match self.downsample {
                RegNetDownsample::Conv1x1 => ConvShortcutKind::Conv,
                RegNetDownsample::Avg => ConvShortcutKind::AvgConv,
                RegNetDownsample::None => ConvShortcutKind::None,
            })
            .with_out_act(!self.linear_out)
            .with_attn(
                (self.se_ratio != 0.0).then_some(ConvAttentionKind::SeFromInput {
                    se_ratio: self.se_ratio,
                }),
            )
            .with_attn_place(ConvAttentionPlace::Mixer);
        Ok(ConvNetContractConfig::new(
            ConvStemContractConfig::new(vec![self.stem_width])
                .with_kernel_size(3)
                .with_stride(2)
                .with_pool(ConvStemPoolKind::None)
                .with_last_norm_act(!self.preact),
            stages,
            self.num_classes,
        )
        .with_in_channels(self.in_channels)
        .with_output_stride(self.output_stride)
        .with_layers(layers)
        // timm's `final_conv`: a conv block to `num_features`, else an
        // activation where the units end without one.
        .with_final_conv(match self.num_features {
            0 => (self.linear_out || self.preact).then_some(ConvFinalKind::Act),
            width => Some(ConvFinalKind::Conv { width }),
        })
        .with_global_pool(self.global_pool)
        .with_drop_rate(self.drop_rate))
    }
}

impl ToStructureConfig for RegNetContractConfig {
    type Structure = ConvNetStructureConfig;

    /// The conv net of this `RegNet` ([`try_conv_net`](Self::try_conv_net)),
    /// resolved.
    ///
    /// # Errors
    ///
    /// As [`try_conv_net`](Self::try_conv_net) and the conv net's lowering,
    /// under a frame naming the `RegNet`.
    fn try_to_structure(&self) -> BunsenResult<ConvNetStructureConfig> {
        self.try_conv_net()
            .and_then(|net| net.try_to_structure())
            .context("RegNetContractConfig")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::images::{
            attention::ChannelAttentionConfig,
            models::conv_net::{
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

    /// timm's `regnetx_002`: `RegNetCfg(w0=24, wa=36.44, wm=2.49,
    /// group_size=8, depth=13)`.
    fn x002() -> RegNetContractConfig {
        RegNetContractConfig::new(13, 24, 36.44, 2.49, 8, 1000)
    }

    /// `regnetx_002`'s widths and stages, as timm builds them.
    #[test]
    fn test_x002_stages() {
        assert_eq!(
            x002().try_unit_widths().unwrap(),
            [
                24, 56, 152, 152, 152, 152, 368, 368, 368, 368, 368, 368, 368
            ]
        );
        let stage = |depth, width, groups| RegNetStage {
            depth,
            width,
            mid_width: width,
            groups,
        };
        assert_eq!(
            x002().try_stages().unwrap(),
            [
                stage(1, 24, 3),
                stage(1, 56, 7),
                stage(4, 152, 19),
                stage(7, 368, 46)
            ]
        );
    }

    /// The lowering: a `3x3` stem of stride 2 and no pool, bottleneck units
    /// of expansion 1 whose grouped mixer strides, a conv shortcut where a
    /// unit changes shape; under `se_ratio`, `RegNet`'s squeeze-excite after
    /// the mixer.
    #[test]
    fn test_lowering() {
        let structure = x002().to_structure();
        assert_eq!(structure.stem.convs.len(), 1);
        assert!(structure.stem.pool.is_none());
        assert_eq!(structure.stem.stride(), 2);
        let first = &structure.stages[0].units[0];
        assert_eq!((first.in_channels(), first.out_channels()), (32, 24));
        assert!(first.shortcut.is_some());
        assert!(first.attn.is_none());
        assert!(structure.stages[2].units[1].shortcut.is_none());
        assert_eq!(structure.d_features(), 368);

        let y = x002().with_se_ratio(0.25).to_structure();
        let unit = &y.stages[1].units[0];
        assert_eq!(unit.attn_place, ConvAttentionPlace::Mixer);
        let Some(ChannelAttentionConfig::Se(se)) = &unit.attn else {
            panic!("a squeeze-excite");
        };
        // Over the 56-wide mixer, reduced to round(24 * 0.25).
        assert_eq!((se.channels, se.rd_channels), (56, 6));

        let device = cpu_device();
        let model: ConvNetModel = RegNetContractConfig {
            num_classes: 5,
            ..x002()
        }
        .init(&device);
        assert_eq!(model.num_classes(), 5);
    }

    #[test]
    fn test_illegal_policies() {
        let illegal = |config: RegNetContractConfig, message: &str| {
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .message_contains(message)
                .assert_err(&config.try_to_structure());
        };
        illegal(RegNetContractConfig { wm: 1.0, ..x002() }, "wm > 1");
        illegal(
            RegNetContractConfig::new(13, 20, 36.44, 2.49, 8, 1000),
            "w0 a nonzero multiple of 8",
        );
        illegal(
            RegNetContractConfig::new(4, 24, 36.44, 2.49, 8, 1000),
            "stages, not 4",
        );
        illegal(
            RegNetContractConfig {
                group_size: 0,
                ..x002()
            },
            "a group size of 0",
        );
    }
}
