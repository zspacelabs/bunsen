//! The conv net's policy: a stem, stage specs, and the layer choices.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    nn::{
        BatchNormConfig,
        PaddingConfig2d,
        conv::Conv2dConfig,
        pool::{
            AvgPool2dConfig,
            MaxPool2dConfig,
        },
    },
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        conv::ConvBlock2dConfig,
        images::{
            anti_alias::{
                AntiAliasConfig,
                BlurPool2dConfig,
            },
            attention::{
                ChannelAttentionConfig,
                EcaConfig,
                SqueezeExciteConfig,
            },
            models::{
                classifier_head::ClassifierHeadConfig,
                conv_net::{
                    ConvAttentionPlace,
                    ConvChainMeta,
                    ConvFinalConfig,
                    ConvNetStructureConfig,
                    ConvShortcutConfig,
                    ConvStageConfig,
                    ConvStemConfig,
                    ConvStemPoolConfig,
                    GlobalPool2dConfig,
                    MaxBlurPoolConfig,
                    NormActConfig,
                    PreActConvConfig,
                    ResidualConvConfig,
                    ResidualUnitConfig,
                },
            },
        },
        norm::BunsenNormalizationConfig,
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
    },
    ops::conv::{
        CONV_INTO_RELU_INITIALIZER,
        build_square_conv2d_padding_config,
    },
};

/// The kind of a stage's residual units: the conv path a
/// [`ConvStageContractConfig`] lowers each unit to.
///
/// Every kind lowers to a [`ResidualUnitConfig`]; the kinds differ in the
/// path. The `ResNet` kinds are built; the families after it add kinds
/// (`CSPNet`'s dark unit, the pre-activation units of `ResNetV2`).
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ConvUnitKind {
    /// Two `kxk` convs, `in -> mid -> width`; the first strides and is the
    /// mixer (`ResNet`'s `BasicBlock`).
    Basic,

    /// A `1x1` reduce, a `kxk` mixer (grouped; it strides) and a `1x1`
    /// expand, `in -> mid -> mid -> width` (`ResNet`'s `Bottleneck`).
    Bottleneck,

    /// [`Bottleneck`](Self::Bottleneck)'s convs, pre-activated: the unit's
    /// input normalized and activated once, read by the path and the
    /// shortcut, then each later conv after its own norm and activation; no
    /// norm on a conv's output and no activation after the add, and a
    /// shortcut conv with no norm (`RegNetV`'s `PreBottleneck`).
    PreBottleneck,
}

/// A residual unit's shortcut, as a policy names it: what a
/// [`ConvShortcutConfig`] is built from.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ConvShortcutKind {
    /// A strided `1x1` conv and a norm (torchvision's `downsample`).
    Conv,

    /// An average pool, then a `1x1` conv and a norm (timm's `avg_down`, the
    /// `ResNet-D` variants): [`ConvShortcutConfig::timm_avg_down`].
    AvgConv,

    /// None: a unit that changes the channels or strides has no residual,
    /// its output its path's (`RegNetZ`'s `downsample=None`); the others add
    /// the identity.
    None,
}

/// A residual unit's channel attention, as a policy names it: sized from the
/// channels where it applies ([`ConvLayersConfig::attn_place`]) when lowered.
#[derive(Config, Debug, Copy, PartialEq)]
pub enum ConvAttentionKind {
    /// Squeeze-and-excitation, its reduction `rd_ratio` of the channels in 8s
    /// ([`SqueezeExciteConfig::timm`]): timm's `'se'` is 1/16, `ResNet-RS`'s
    /// 1/4.
    Se {
        /// The reduction's share of the channels.
        rd_ratio: f64,
    },

    /// Efficient channel attention, its kernel from the channels
    /// ([`EcaConfig::timm`]): timm's `'eca'`.
    Eca,

    /// Squeeze-and-excitation whose reduction is `round(in_channels *
    /// se_ratio)` of the *unit's input* channels (ties to even, as Python's
    /// `round`), its activation the layers': `RegNet`'s `SEModule`.
    SeFromInput {
        /// The reduction's share of the unit's input channels.
        se_ratio: f64,
    },
}

impl ConvAttentionKind {
    /// timm's `'se'`: a reduction of 1/16.
    pub const SE: Self = Self::Se {
        rd_ratio: 1.0 / 16.0,
    };

    /// The attention over `channels`, in a unit of `in_channels` input
    /// channels and `act` activations.
    pub fn to_config(
        self,
        channels: usize,
        in_channels: usize,
        act: &BunsenActivationConfig,
    ) -> ChannelAttentionConfig {
        match self {
            Self::Se { rd_ratio } => {
                ChannelAttentionConfig::Se(SqueezeExciteConfig::timm(channels, rd_ratio))
            }
            Self::Eca => ChannelAttentionConfig::Eca(EcaConfig::timm(channels)),
            Self::SeFromInput { se_ratio } => ChannelAttentionConfig::Se(
                SqueezeExciteConfig::new(
                    channels,
                    crate::support::math::round_ties_even(in_channels as f64 * se_ratio),
                )
                .with_act(act.clone()),
            ),
        }
    }
}

/// A unit's anti-aliased downsampling, as a policy names it (timm's
/// `aa_layer`): where a unit strides 2, or enters a dilated stage, its mixer
/// keeps stride 1 and this pool, after the mixer's activation, takes the
/// stride.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ConvAaKind {
    /// An average pool of the stride (timm's `nn.AvgPool2d`).
    Avg,

    /// A `3x3` binomial blur, then the stride (timm's `BlurPool2d`).
    Blur,
}

impl ConvAaKind {
    /// The pool at `stride`.
    pub fn to_config(
        self,
        stride: usize,
    ) -> AntiAliasConfig {
        match self {
            Self::Avg => AntiAliasConfig::avg(stride),
            Self::Blur => AntiAliasConfig::blur(stride),
        }
    }
}

/// The layer a policy puts between the last stage and the pool: what a
/// [`ConvFinalConfig`] is built from.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ConvFinalKind {
    /// The layers' activation alone (a pre-activation or linear-output
    /// net's).
    Act,

    /// A `1x1` conv to `width` channels, the layers' norm and activation
    /// (`RegNetZ`'s `num_features`).
    Conv {
        /// The features' width.
        width: usize,
    },
}

/// The stem's pool, as a policy names it.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ConvStemPoolKind {
    /// A `3x3` max pool, stride 2, padding 1 (`ResNet`'s).
    Max,

    /// A `2x2` average pool, stride 2 (an average-anti-aliased `ResNet`'s).
    Avg,

    /// A `3x3` max pool, stride 1, padding 1, then a `3x3` blur of stride 2
    /// (a blur-anti-aliased `ResNet`'s).
    MaxBlur,

    /// A `3x3` conv of stride 2 that keeps the channels, its norm and its
    /// activation (`ResNet-RS`'s, timm's `replace_stem_pool`).
    Conv,

    /// No pool.
    None,
}

/// The layer choices of a conv net, stated once for the model and replaced
/// per stage ([`ConvStageContractConfig::layers`]).
#[derive(Config, Debug)]
pub struct ConvLayersConfig {
    /// The norm after each conv; its feature count is matched to the conv.
    #[config(default = "BunsenNormalizationConfig::Batch(BatchNormConfig::new(0))")]
    pub norm: BunsenNormalizationConfig,

    /// The activation after each conv's norm, but a unit's last, and after
    /// each unit's add.
    #[config(default = "BunsenActivationConfig::Relu")]
    pub act: BunsenActivationConfig,

    /// The shortcut of a unit that changes the channels or strides.
    #[config(default = "ConvShortcutKind::Conv")]
    pub shortcut: ConvShortcutKind,

    /// Each unit's channel attention; none when `None`.
    #[config(default = "None")]
    pub attn: Option<ConvAttentionKind>,

    /// Where each unit's channel attention applies: over its path's output
    /// (`ResNet`'s), or after its mixer (`RegNet`'s).
    #[config(default = "ConvAttentionPlace::Out")]
    pub attn_place: ConvAttentionPlace,

    /// Each striding unit's anti-aliased downsampling; none when `None`.
    #[config(default = "None")]
    pub aa: Option<ConvAaKind>,

    /// Whether each unit activates after its add: `false`, a linear output
    /// (`RegNetZ`'s `linear_out`). A pre-activation unit has none whatever
    /// this says.
    #[config(default = "true")]
    pub out_act: bool,
}

impl Default for ConvLayersConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvLayersConfig {
    /// A `k x k` conv: no bias, padded to keep the size at stride 1.
    fn conv2d(
        channels: [usize; 2],
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Conv2dConfig {
        Conv2dConfig::new(channels, [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_dilation([dilation, dilation])
            .with_padding(build_square_conv2d_padding_config(
                kernel_size,
                stride,
                dilation,
            ))
            .with_groups(groups)
            .with_bias(false)
    }

    /// A conv block of these layers: a [`conv2d`](Self::conv2d), the norm,
    /// and the activation when `act`.
    fn conv(
        &self,
        channels: [usize; 2],
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        act: bool,
    ) -> ConvBlock2dConfig {
        ConvBlock2dConfig::new(Self::conv2d(
            channels,
            kernel_size,
            stride,
            dilation,
            groups,
        ))
        .with_norm(Some(self.norm.clone()))
        .with_act(act.then(|| self.act.clone()))
        .match_norm_features()
    }

    /// A pre-activation conv of these layers: the norm and the activation
    /// over its input when `norm_act`, then a [`conv2d`](Self::conv2d).
    fn pre_conv(
        &self,
        channels: [usize; 2],
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        norm_act: bool,
    ) -> PreActConvConfig {
        let conv = PreActConvConfig::new(Self::conv2d(
            channels,
            kernel_size,
            stride,
            dilation,
            groups,
        ));
        match norm_act {
            true => conv
                .with_norm(Some(self.norm.clone()))
                .with_act(Some(self.act.clone())),
            false => conv,
        }
    }
}

/// The stem, as a policy states it: a run of convs, then a pool.
#[derive(Config, Debug)]
pub struct ConvStemContractConfig {
    /// Each conv's output channels: `[64]` is `ResNet`'s single `7x7` stem,
    /// `[32, 32, 64]` its deep stem.
    pub widths: Vec<usize>,

    /// Every conv's kernel: 7 for a single conv, 3 for a deep stem.
    #[config(default = 7)]
    pub kernel_size: usize,

    /// The first conv's stride; the others' are 1.
    #[config(default = 2)]
    pub stride: usize,

    /// The pool after the convs.
    #[config(default = "ConvStemPoolKind::Max")]
    pub pool: ConvStemPoolKind,

    /// Whether each conv has its norm and activation: `false`, bare convs,
    /// for a pre-activation net, whose first unit normalizes its input
    /// (`RegNetV`'s stem).
    #[config(default = "true")]
    pub norm_act: bool,
}

impl ConvStemContractConfig {
    /// The stem's structure, on `in_channels` image channels, with `layers`'
    /// norm and activation after each conv.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for no widths.
    pub fn try_to_structure(
        &self,
        in_channels: usize,
        layers: &ConvLayersConfig,
    ) -> BunsenResult<ConvStemConfig> {
        if self.widths.is_empty() {
            return Err(ConstraintError::zero_or_empty("ConvStemContractConfig", "widths").into());
        }
        let mut channels = in_channels;
        let convs = self
            .widths
            .iter()
            .enumerate()
            .map(|(idx, &width)| {
                let stride = if idx == 0 { self.stride } else { 1 };
                let mut conv = layers.conv([channels, width], self.kernel_size, stride, 1, 1, true);
                if !self.norm_act {
                    conv = conv.with_norm(None).with_act(None);
                }
                conv.conv = conv
                    .conv
                    .with_initializer(CONV_INTO_RELU_INITIALIZER.clone());
                channels = width;
                conv
            })
            .collect();
        let max = |stride: usize| {
            MaxPool2dConfig::new([3, 3])
                .with_strides([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        };
        let pool = match self.pool {
            ConvStemPoolKind::Max => Some(ConvStemPoolConfig::Max(max(2))),
            ConvStemPoolKind::Avg => Some(ConvStemPoolConfig::Avg(
                AvgPool2dConfig::new([2, 2]).with_strides([2, 2]),
            )),
            ConvStemPoolKind::MaxBlur => Some(ConvStemPoolConfig::MaxBlur(MaxBlurPoolConfig::new(
                max(1),
                BlurPool2dConfig::new(),
            ))),
            ConvStemPoolKind::Conv => Some(ConvStemPoolConfig::Conv(layers.conv(
                [channels, channels],
                3,
                2,
                1,
                1,
                true,
            ))),
            ConvStemPoolKind::None => None,
        };
        Ok(ConvStemConfig { convs, pool })
    }
}

/// One stage, as a policy states it: `depth` units of a kind and widths,
/// and the stage's stride.
#[derive(Config, Debug)]
pub struct ConvStageContractConfig {
    /// The number of units.
    pub depth: usize,

    /// The units' kind.
    pub unit: ConvUnitKind,

    /// Each unit's output channels.
    pub width: usize,

    /// The mixer's channels: a bottleneck's inner width, a basic unit's
    /// first conv's output.
    pub mid_width: usize,

    /// The mixer's groups (a `ResNeXt`'s cardinality); a basic unit's
    /// second conv's too.
    #[config(default = 1)]
    pub groups: usize,

    /// The mixer's kernel (and a basic unit's second conv's).
    #[config(default = 3)]
    pub kernel_size: usize,

    /// The stage's stride, the first unit's: 2 halves each side. Past the
    /// net's output stride it becomes dilation.
    #[config(default = 2)]
    pub stride: usize,

    /// The layer choices of this stage, replacing the model's whole.
    #[config(default = "None")]
    pub layers: Option<ConvLayersConfig>,
}

impl ConvStageContractConfig {
    /// One unit of this stage: from `in_channels`, at `stride`, its mixer
    /// dilated by `first_dilation` and a basic unit's second conv by
    /// `dilation`.
    fn try_unit(
        &self,
        layers: &ConvLayersConfig,
        in_channels: usize,
        stride: usize,
        first_dilation: usize,
        dilation: usize,
    ) -> BunsenResult<ResidualUnitConfig> {
        let (mid, width, k, groups) = (self.mid_width, self.width, self.kernel_size, self.groups);
        // timm's `use_aa`: the pool takes the stride where a unit strides 2
        // or enters a dilated stage; the mixer keeps stride 1.
        let aa = layers
            .aa
            .filter(|_| stride == 2 || first_dilation != dilation)
            .map(|kind| kind.to_config(stride));
        let mixer_stride = if aa.is_some() { 1 } else { stride };
        let preact = self.unit == ConvUnitKind::PreBottleneck;
        let (convs, mixer): (Vec<ResidualConvConfig>, usize) = match self.unit {
            ConvUnitKind::Basic => (
                alloc::vec![
                    layers
                        .conv(
                            [in_channels, mid],
                            k,
                            mixer_stride,
                            first_dilation,
                            groups,
                            true
                        )
                        .into(),
                    layers
                        .conv([mid, width], k, 1, dilation, groups, false)
                        .into(),
                ],
                0,
            ),
            ConvUnitKind::Bottleneck => (
                alloc::vec![
                    layers.conv([in_channels, mid], 1, 1, 1, 1, true).into(),
                    layers
                        .conv([mid, mid], k, mixer_stride, first_dilation, groups, true)
                        .into(),
                    layers.conv([mid, width], 1, 1, 1, 1, false).into(),
                ],
                1,
            ),
            // The unit's `pre` normalizes the input; each later conv its own.
            ConvUnitKind::PreBottleneck => (
                alloc::vec![
                    layers
                        .pre_conv([in_channels, mid], 1, 1, 1, 1, false)
                        .into(),
                    layers
                        .pre_conv([mid, mid], k, mixer_stride, first_dilation, groups, true)
                        .into(),
                    layers.pre_conv([mid, width], 1, 1, 1, 1, true).into(),
                ],
                1,
            ),
        };
        // A pre-activation shortcut's conv has no norm: its input is normed.
        let shortcut_conv = |stride| {
            let conv = layers.conv([in_channels, width], 1, stride, 1, 1, false);
            match preact {
                true => conv.with_norm(None),
                false => conv,
            }
        };
        let reshapes = stride != 1 || in_channels != width;
        let shortcut = match (reshapes, layers.shortcut) {
            (false, _) | (true, ConvShortcutKind::None) => None,
            (true, ConvShortcutKind::Conv) => Some(ConvShortcutConfig::conv(shortcut_conv(stride))),
            (true, ConvShortcutKind::AvgConv) => Some(ConvShortcutConfig::timm_avg_down(
                shortcut_conv(1),
                stride,
                dilation,
            )),
        };
        let residual = !(reshapes && layers.shortcut == ConvShortcutKind::None);
        Ok(ResidualUnitConfig {
            convs,
            mixer,
            pre: preact.then(|| {
                NormActConfig::new(layers.norm.clone()).with_act(Some(layers.act.clone()))
            }),
            residual,
            aa,
            shortcut,
            attn: layers.attn.map(|kind| {
                let channels = match layers.attn_place {
                    ConvAttentionPlace::Out => width,
                    ConvAttentionPlace::Mixer => mid,
                };
                kind.to_config(channels, in_channels, &layers.act)
            }),
            attn_place: layers.attn_place,
            act: (layers.out_act && !preact).then(|| layers.act.clone()),
            drop_block: None,
            drop_path_prob: 0.0,
        })
    }
}

/// The generic conv net's policy: a stem, stage specs, the layer choices
/// and the head.
///
/// The policy of the conv net's [Stacked
/// config](crate::burner::module::ModuleInit#stacked-config): it lowers
/// through [`ToStructureConfig`] to a [`ConvNetStructureConfig`], every unit
/// resolved, and builds a
/// [`ConvNetModel`](super::ConvNetModel) through the blanket
/// [`ModuleInit`](crate::burner::module::ModuleInit). A family's policy
/// (`ResNetContractConfig`) lowers to this one, in its own terms.
///
/// The lowering is timm's: the stem's stride (its conv's and its pool's)
/// starts the net's; each stage's stride multiplies it until it reaches
/// `output_stride`, past which a stage's stride becomes dilation. A stage's
/// first unit takes the stride, the stage's input channels, and the dilation
/// before the stage (`resnet.py`'s `first_dilation`); the others take none,
/// the stage's width, and the stage's dilation.
#[derive(Config, Debug)]
pub struct ConvNetContractConfig {
    /// The stem.
    pub stem: ConvStemContractConfig,

    /// The stages, in order.
    pub stages: Vec<ConvStageContractConfig>,

    /// The classes of the head.
    pub num_classes: usize,

    /// The image channels.
    #[config(default = 3)]
    pub in_channels: usize,

    /// The net's stride at its last stage: 32, or 16 or 8 for a dilated net.
    #[config(default = 32)]
    pub output_stride: usize,

    /// The layer choices, for every stage that does not replace them.
    #[config(default = "ConvLayersConfig::new()")]
    pub layers: ConvLayersConfig,

    /// The layer between the last stage and the pool; none when `None`.
    #[config(default = "None")]
    pub final_conv: Option<ConvFinalKind>,

    /// The head's global pool.
    #[config(default = "GlobalPool2dConfig::Avg")]
    pub global_pool: GlobalPool2dConfig,

    /// The dropout before the classifier (timm's `drop_rate`; training
    /// only).
    #[config(default = "0.0")]
    pub drop_rate: f64,
}

impl ToStructureConfig for ConvNetContractConfig {
    type Structure = ConvNetStructureConfig;

    /// Resolves every unit.
    ///
    /// # Errors
    ///
    /// As [`ConvStemContractConfig::try_to_structure`], under a frame naming
    /// the stem; [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported)
    /// for a kind not built yet, under a frame naming its stage.
    fn try_to_structure(&self) -> BunsenResult<ConvNetStructureConfig> {
        let stem = self
            .stem
            .try_to_structure(self.in_channels, &self.layers)
            .context("the conv net's stem")?;
        let mut net_stride = stem.stride();
        let mut dilation = 1;
        let mut channels = stem.out_channels();
        let mut stages = Vec::with_capacity(self.stages.len());
        for (idx, spec) in self.stages.iter().enumerate() {
            let layers = spec.layers.as_ref().unwrap_or(&self.layers);
            let first_dilation = dilation;
            let mut stride = spec.stride;
            if net_stride >= self.output_stride {
                dilation *= stride;
                stride = 1;
            } else {
                net_stride *= stride;
            }
            let units = (0..spec.depth)
                .map(|u| {
                    if u == 0 {
                        spec.try_unit(layers, channels, stride, first_dilation, dilation)
                    } else {
                        spec.try_unit(layers, spec.width, 1, dilation, dilation)
                    }
                })
                .collect::<BunsenResult<Vec<_>>>()
                .with_context(|| format!("conv net stage {idx} (stages[{idx}])"))?;
            stages.push(ConvStageConfig { units });
            channels = spec.width;
        }
        let final_conv = self.final_conv.map(|kind| match kind {
            ConvFinalKind::Act => ConvFinalConfig::Act(self.layers.act.clone()),
            ConvFinalKind::Conv { width } => {
                ConvFinalConfig::Conv(self.layers.conv([channels, width], 1, 1, 1, 1, true))
            }
        });
        if let Some(ConvFinalKind::Conv { width }) = self.final_conv {
            channels = width;
        }
        Ok(ConvNetStructureConfig {
            stem,
            stages,
            final_conv,
            pool: self.global_pool,
            head: ClassifierHeadConfig::new(
                self.global_pool.pooled_features(channels),
                self.num_classes,
            )
            .with_drop_rate(self.drop_rate),
        })
    }
}
