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
            mixers::{
                Res2ConvConfig,
                SelectiveKernelConfig,
                SplitAttnConfig,
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
        BunsenError,
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
    ops::conv::{
        CONV_INTO_RELU_INITIALIZER,
        Conv2dVariantOptions,
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

    /// [`Basic`](Self::Basic)'s two convs, pre-activated as
    /// [`PreBottleneck`](Self::PreBottleneck) is (timm's `PreActBasic`,
    /// `ResNetV2`'s).
    PreBasic,

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

    /// `ResNeSt`'s `avd`: a `3x3` average pool of the stride, padded 1, the
    /// padding counted ([`AntiAliasConfig::avd`]); where a unit strides 2
    /// alone, not where it enters a dilated stage.
    Avd,
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
            Self::Avd => AntiAliasConfig::avd(stride),
        }
    }
}

/// A unit's spatial mixer, the conv that strides: what its
/// [`ResidualConvConfig`] is built as.
///
/// A seam: a kind a unit can't hold is
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) there.
#[derive(Config, Debug, Copy, PartialEq)]
pub enum ConvMixerKind {
    /// One conv block (`ResNet`'s `3x3`).
    Conv,

    /// `Res2Net`'s scale split of `scale` chunks of the mixer's width
    /// ([`Res2ConvConfig`]; a bottleneck's alone). Each branch is the layers'
    /// conv, norm and activation, `mid_width / scale` wide; in a unit that
    /// strides or reshapes, the branches are apart and the last chunk is
    /// pooled ([`Res2ConvConfig::timm_pool`]).
    Res2 {
        /// The chunks (timm's `scale`).
        scale: usize,
    },

    /// `ResNeSt`'s split attention of `radix` splits
    /// ([`SplitAttnConfig::timm`]; a bottleneck's alone), in the stage's
    /// groups, the layers' norm and activation, its attention width
    /// `rd_ratio` of the splits' input in `rd_divisor`s.
    SplitAttn {
        /// The splits (timm's `radix`).
        radix: usize,
        /// The attention width's share of the splits' input (timm's
        /// `rd_ratio`, 0.25).
        rd_ratio: f64,
        /// The attention width's divisor (timm's `rd_divisor`, 8).
        rd_divisor: usize,
    },

    /// `SKNet`'s selective kernel ([`SelectiveKernelConfig::timm`]; a basic
    /// unit's first conv, a bottleneck's `3x3`): a path per kernel, the
    /// layers' norm and activation, an attention `rd_ratio` of the channels in
    /// `rd_divisor`s.
    SelectiveKernel {
        /// The paths' kernels (timm's `kernel_size`, `[3, 5]`).
        kernel_size: [usize; 2],
        /// Whether each kernel is kept `3x3` and dilated to its reach instead
        /// (timm's `keep_3x3`).
        keep_3x3: bool,
        /// The attention width's share of the channels (timm's `rd_ratio`,
        /// 1/16).
        rd_ratio: f64,
        /// The attention width's divisor (timm's `rd_divisor`, 8).
        rd_divisor: usize,
        /// Whether the input splits across the paths (timm's `split_input`).
        split_input: bool,
    },
}

/// The layer a policy puts between the last stage and the pool: what a
/// [`ConvFinalConfig`] is built from.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ConvFinalKind {
    /// The layers' activation alone (a pre-activation or linear-output
    /// net's).
    Act,

    /// The layers' norm and activation (a pre-activation net's last norm,
    /// `ResNetV2`'s `norm`).
    NormAct,

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

    /// A zero pad of 1, then a `3x3` max pool of stride 2 and no padding of
    /// its own (`BiT`'s `fixed` stem).
    ZeroPadMax,

    /// A `3x3` max pool of stride 2 over the input SAME-padded from its own
    /// size with `-inf` (timm's `MaxPool2dSame`, the hybrid `ViT`'s `same`
    /// stem).
    SameMax,

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
    #[config(default = true)]
    pub out_act: bool,

    /// Each conv's weight standardization, by this `eps` (timm's
    /// `StdConv2d`, `BiT`'s 1e-8); none when `None`.
    #[config(default = "None")]
    pub weight_std: Option<f64>,

    /// Whether a pre-activation unit's identity shortcut reads the
    /// pre-activated input (`RegNetV`'s) or the raw one (`ResNetV2`'s).
    #[config(default = true)]
    pub pre_identity: bool,

    /// Whether each conv SAME-pads its input from its own size, the conv
    /// itself unpadded (timm's `Conv2dSame`, and `StdConv2dSame` with
    /// [`weight_std`](Self::weight_std); the hybrid `ViT`'s backbone).
    #[config(default = false)]
    pub same_pad: bool,

    /// Each unit's spatial mixer.
    #[config(default = "ConvMixerKind::Conv")]
    pub mixer: ConvMixerKind,

    /// Whether the anti-aliased downsampling ([`aa`](Self::aa)) goes before
    /// the mixer rather than after it (`ResNeSt`'s `avd_first`).
    #[config(default = false)]
    pub aa_first: bool,
}

impl Default for ConvLayersConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvLayersConfig {
    /// The conv variant [`weight_std`](Self::weight_std) and
    /// [`same_pad`](Self::same_pad) state, as the blocks hold it.
    pub fn variant(&self) -> Conv2dVariantOptions {
        Conv2dVariantOptions::new()
            .with_weight_std(self.weight_std)
            .with_same_pad(self.same_pad)
    }

    /// A `k x k` conv: no bias, padded to keep the size at stride 1; under
    /// [`same_pad`](Self::same_pad), unpadded (its input is padded).
    fn conv2d(
        &self,
        channels: [usize; 2],
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Conv2dConfig {
        Conv2dConfig::new(channels, [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_dilation([dilation, dilation])
            .with_padding(match self.same_pad {
                true => PaddingConfig2d::Valid,
                false => build_square_conv2d_padding_config(kernel_size, stride, dilation),
            })
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
        ConvBlock2dConfig::new(self.conv2d(channels, kernel_size, stride, dilation, groups))
            .with_norm(Some(self.norm.clone()))
            .with_act(act.then(|| self.act.clone()))
            .with_variant(self.variant())
            .match_norm_features()
    }

    /// A unit's spatial mixer of these layers' [`mixer`](Self::mixer), from
    /// `channels[0]` to `channels[1]`; `is_first` for a unit that strides or
    /// reshapes (timm's `Bottle2neck.is_first`).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a scale split
    /// that doesn't keep the channels or split them evenly.
    fn try_mixer(
        &self,
        channels: [usize; 2],
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        is_first: bool,
    ) -> BunsenResult<ResidualConvConfig> {
        match self.mixer {
            ConvMixerKind::Conv => Ok(self
                .conv(channels, kernel_size, stride, dilation, groups, true)
                .into()),
            ConvMixerKind::Res2 { scale } => {
                let [c_in, c_out] = channels;
                if scale == 0 || c_in != c_out || !c_in.is_multiple_of(scale) {
                    return Err(ConstraintError::custom(
                        "ConvLayersConfig",
                        "mixer",
                        format!(
                            "a scale split of {scale} chunks from {c_in} to {c_out} channels: it \
                             keeps the channels, in whole chunks"
                        ),
                    )
                    .into());
                }
                let width = c_in / scale;
                let convs = (0..scale.saturating_sub(1).max(1))
                    .map(|_| self.conv([width, width], kernel_size, stride, dilation, groups, true))
                    .collect();
                Ok(Res2ConvConfig::new(scale, convs)
                    .with_hierarchical(!is_first)
                    .with_pool((is_first && scale > 1).then(|| Res2ConvConfig::timm_pool(stride)))
                    .into())
            }
            ConvMixerKind::SelectiveKernel {
                kernel_size: kernels,
                keep_3x3,
                rd_ratio,
                rd_divisor,
                split_input,
            } => {
                if kernels.iter().any(|&k| k < 3 || k % 2 == 0)
                    || rd_divisor == 0
                    || rd_ratio <= 0.0
                {
                    return Err(ConstraintError::custom(
                        "ConvLayersConfig",
                        "mixer",
                        format!(
                            "a selective kernel of kernels {kernels:?}, its attention {rd_ratio} of \
                             the channels in {rd_divisor}s: its kernels are odd and at least 3, its \
                             attention positive"
                        ),
                    )
                    .into());
                }
                if split_input && !channels[0].is_multiple_of(kernels.len()) {
                    return Err(ConstraintError::new(
                        "ConvLayersConfig",
                        "mixer",
                        Rule::NotMultiple {
                            value: channels[0],
                            of: format!("the selective kernel's {} paths", kernels.len()),
                        },
                    )
                    .into());
                }
                Ok(SelectiveKernelConfig::timm(
                    channels,
                    kernels,
                    keep_3x3,
                    stride,
                    dilation,
                    groups,
                    rd_ratio,
                    rd_divisor,
                    split_input,
                    self.norm.clone(),
                    self.act.clone(),
                )
                .into())
            }
            ConvMixerKind::SplitAttn {
                radix,
                rd_ratio,
                rd_divisor,
            } => {
                if radix == 0 || rd_divisor == 0 || rd_ratio <= 0.0 {
                    return Err(ConstraintError::custom(
                        "ConvLayersConfig",
                        "mixer",
                        format!(
                            "a split attention of radix {radix}, its attention {rd_ratio} of its \
                             input in {rd_divisor}s: the radix, the ratio and the divisor are positive"
                        ),
                    )
                    .into());
                }
                Ok(SplitAttnConfig::timm(
                    channels,
                    kernel_size,
                    stride,
                    dilation,
                    groups,
                    radix,
                    rd_ratio,
                    rd_divisor,
                    self.norm.clone(),
                    self.act.clone(),
                )
                .into())
            }
        }
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
        let conv =
            PreActConvConfig::new(self.conv2d(channels, kernel_size, stride, dilation, groups))
                .with_variant(self.variant());
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

    /// Whether the last conv has its norm and activation: `false`, a bare
    /// last conv, for a pre-activation net, whose first unit normalizes its
    /// input (`RegNetV`'s and `ResNetV2`'s stems; a deep stem's earlier convs
    /// keep theirs).
    #[config(default = true)]
    pub last_norm_act: bool,
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
                if !self.last_norm_act && idx + 1 == self.widths.len() {
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
            ConvStemPoolKind::ZeroPadMax => Some(ConvStemPoolConfig::ZeroPadMax(
                MaxPool2dConfig::new([3, 3]).with_strides([2, 2]),
            )),
            ConvStemPoolKind::SameMax => Some(ConvStemPoolConfig::SameMax(
                MaxPool2dConfig::new([3, 3]).with_strides([2, 2]),
            )),
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
            .filter(|kind| stride == 2 || (*kind != ConvAaKind::Avd && first_dilation != dilation))
            .map(|kind| kind.to_config(stride));
        let mixer_stride = if aa.is_some() { 1 } else { stride };
        let preact = matches!(
            self.unit,
            ConvUnitKind::PreBasic | ConvUnitKind::PreBottleneck
        );
        let reshapes = stride != 1 || in_channels != width;
        if layers.mixer != ConvMixerKind::Conv {
            let selective = matches!(layers.mixer, ConvMixerKind::SelectiveKernel { .. });
            let holds = match self.unit {
                ConvUnitKind::Bottleneck => true,
                ConvUnitKind::Basic => selective,
                ConvUnitKind::PreBasic | ConvUnitKind::PreBottleneck => false,
            };
            if !holds {
                return Err(BunsenError::unsupported(format!(
                    "a {:?} mixer in a {:?} unit",
                    layers.mixer, self.unit
                )));
            }
            if aa.is_some() && !matches!(layers.mixer, ConvMixerKind::SplitAttn { .. }) {
                return Err(BunsenError::unsupported(format!(
                    "a {:?} mixer with anti-aliasing",
                    layers.mixer
                )));
            }
        }
        let (convs, mixer): (Vec<ResidualConvConfig>, usize) = match self.unit {
            ConvUnitKind::Basic => (
                alloc::vec![
                    layers.try_mixer(
                        [in_channels, mid],
                        k,
                        mixer_stride,
                        first_dilation,
                        groups,
                        reshapes
                    )?,
                    layers
                        .conv([mid, width], k, 1, dilation, groups, false)
                        .into(),
                ],
                0,
            ),
            ConvUnitKind::Bottleneck => (
                alloc::vec![
                    layers.conv([in_channels, mid], 1, 1, 1, 1, true).into(),
                    layers.try_mixer(
                        [mid, mid],
                        k,
                        mixer_stride,
                        first_dilation,
                        groups,
                        reshapes
                    )?,
                    layers.conv([mid, width], 1, 1, 1, 1, false).into(),
                ],
                1,
            ),
            // The unit's `pre` normalizes the input; each later conv its own.
            ConvUnitKind::PreBasic => (
                alloc::vec![
                    layers
                        .pre_conv(
                            [in_channels, mid],
                            k,
                            mixer_stride,
                            first_dilation,
                            groups,
                            false
                        )
                        .into(),
                    layers
                        .pre_conv([mid, width], k, 1, dilation, groups, true)
                        .into(),
                ],
                0,
            ),
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
            pre_identity: layers.pre_identity,
            aa,
            aa_first: layers.aa_first,
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
    #[config(default = 0.0)]
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
            ConvFinalKind::NormAct => ConvFinalConfig::NormAct(
                NormActConfig::new(self.layers.norm.clone())
                    .with_act(Some(self.layers.act.clone())),
            ),
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
