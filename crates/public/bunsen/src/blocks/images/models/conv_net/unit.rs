//! The residual unit: a conv path, a shortcut, and the add.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::Conv2d,
        pool::{
            AvgPool2d,
            AvgPool2dConfig,
        },
    },
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::{
        activation::{
            BunsenActivation,
            BunsenActivationConfig,
        },
        conv::{
            ConvBlock2d,
            ConvBlock2dConfig,
            ConvBlock2dMeta,
        },
        images::{
            anti_alias::{
                AntiAlias,
                AntiAliasConfig,
            },
            attention::{
                ChannelAttention,
                ChannelAttentionConfig,
                SqueezeExciteMeta,
            },
            drop::{
                drop_block::{
                    DropBlock2d,
                    DropBlock2dConfig,
                },
                drop_path::{
                    DropPath,
                    DropPathConfig,
                },
            },
        },
        norm::BunsenNormalization,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
    ops::drop::DropBlockOptions,
    support::validators::{
        expect_probability,
        try_probability,
    },
};

/// The view shared by every link of a conv net's chain and its config (the
/// stem, a stage, a residual unit, a unit's convs and shortcut): what goes
/// in, what comes out, and how far it strides.
pub trait ConvChainMeta {
    /// The input channels.
    fn in_channels(&self) -> usize;

    /// The output channels.
    fn out_channels(&self) -> usize;

    /// The stride: each side of the output is `ceil(in / stride)`.
    fn stride(&self) -> usize;
}

/// One conv of a [`ResidualUnit`]'s path: the config of a [`ResidualConv`].
///
/// A seam: each kind is a module of its own. Built today: a conv block
/// ([`ConvBlock2dConfig`]). The spatial mixers of the `ResNet` kin (`Res2Net`'s
/// scale split, `ResNeSt`'s split attention, `SKNet`'s selective kernel) are
/// further kinds.
#[derive(Config, Debug)]
pub enum ResidualConvConfig {
    /// A conv, norm and activation.
    Block(ConvBlock2dConfig),
}

impl From<ConvBlock2dConfig> for ResidualConvConfig {
    fn from(config: ConvBlock2dConfig) -> Self {
        Self::Block(config)
    }
}

impl ConvChainMeta for ResidualConvConfig {
    fn in_channels(&self) -> usize {
        match self {
            Self::Block(config) => config.in_channels(),
        }
    }

    fn out_channels(&self) -> usize {
        match self {
            Self::Block(config) => config.out_channels(),
        }
    }

    fn stride(&self) -> usize {
        match self {
            Self::Block(config) => config.stride()[0],
        }
    }
}

impl ModuleInit<ResidualConv> for ResidualConvConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ResidualConv> {
        Ok(match self {
            Self::Block(config) => ResidualConv::Block(config.try_init(device)?),
        })
    }
}

/// One conv of a [`ResidualUnit`]'s path; built by [`ResidualConvConfig`].
///
/// The variant is not part of a parameter's path: a block's conv weight is
/// `convs.{k}.conv.weight`.
#[derive(Module, Debug)]
pub enum ResidualConv {
    /// A conv, norm and activation.
    Block(ConvBlock2d),
}

impl ConvChainMeta for ResidualConv {
    fn in_channels(&self) -> usize {
        match self {
            Self::Block(block) => block.in_channels(),
        }
    }

    fn out_channels(&self) -> usize {
        match self {
            Self::Block(block) => block.out_channels(),
        }
    }

    fn stride(&self) -> usize {
        match self {
            Self::Block(block) => block.stride()[0],
        }
    }
}

impl ResidualConv {
    /// Applies the conv, with `f` run where a regularizer goes: for a block,
    /// after its norm and before its activation
    /// ([`ConvBlock2d::map_forward`]).
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, in_height, in_width]`.
    /// - `f`: an endofunction over the normalized output.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`.
    pub fn map_forward<F>(
        &self,
        input: Tensor<4>,
        f: F,
    ) -> Tensor<4>
    where
        F: FnOnce(Tensor<4>) -> Tensor<4>,
    {
        match self {
            Self::Block(block) => block.map_forward(input, f),
        }
    }

    /// Applies the conv.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, in_height, in_width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        self.map_forward(input, |x| x)
    }
}

/// A [`ResidualUnit`]'s shortcut, when its output differs from its input in
/// channels or size: the config of a [`ConvShortcut`]. An optional average
/// pool, then a conv block: a strided `1x1` conv and a norm with no pool
/// (torchvision's `downsample`, [`conv`](Self::conv)), or a pool before a
/// `1x1` conv at stride 1 (timm's `avg_down`, the `ResNet-D` variants,
/// [`timm_avg_down`](Self::timm_avg_down)).
#[derive(Config, Debug)]
pub struct ConvShortcutConfig {
    /// The pool before the conv, if any.
    pub pool: Option<AvgPool2dConfig>,

    /// The conv, its norm and its activation.
    pub block: ConvBlock2dConfig,
}

impl ConvShortcutConfig {
    /// A conv block alone: torchvision's `downsample`.
    pub fn conv(block: ConvBlock2dConfig) -> Self {
        Self { pool: None, block }
    }

    /// timm's `downsample_avg` before `block` (a conv at stride 1), for a
    /// unit of `stride` and `dilation`:
    ///
    /// - at a stride, a `2x2` pool of that stride, in ceil mode, the padding
    ///   not counted (`AvgPool2d(2, stride, ceil_mode=True,
    ///   count_include_pad=False)`);
    /// - at stride 1, dilated, a `2x2` pool of stride 1 over the input padded
    ///   by one zero row and column at the bottom and right, the zeros counted
    ///   (timm's `AvgPool2dSame`);
    /// - at stride 1, undilated, no pool.
    pub fn timm_avg_down(
        block: ConvBlock2dConfig,
        stride: usize,
        dilation: usize,
    ) -> Self {
        let pool = if stride != 1 {
            Some(
                AvgPool2dConfig::new([2, 2])
                    .with_strides([stride, stride])
                    .with_ceil_mode(true)
                    .with_count_include_pad(false),
            )
        } else if dilation != 1 {
            Some(
                AvgPool2dConfig::new([2, 2])
                    .with_strides([1, 1])
                    .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
                    .with_count_include_pad(true),
            )
        } else {
            None
        };
        Self { pool, block }
    }
}

impl ConvChainMeta for ConvShortcutConfig {
    fn in_channels(&self) -> usize {
        self.block.in_channels()
    }

    fn out_channels(&self) -> usize {
        self.block.out_channels()
    }

    /// The pool's stride, times the conv's.
    fn stride(&self) -> usize {
        self.pool.as_ref().map_or(1, |p| p.strides[0]) * self.block.stride()[0]
    }
}

impl ModuleInit<ConvShortcut> for ConvShortcutConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConvShortcut> {
        let ConvBlock2d { conv, norm, act } = self.block.try_init(device)?;
        Ok(ConvShortcut {
            pool: self.pool.as_ref().map(AvgPool2dConfig::init),
            conv,
            norm,
            act,
        })
    }
}

/// A [`ResidualUnit`]'s shortcut: an optional average pool, then a conv
/// block. Built by [`ConvShortcutConfig`].
///
/// Its conv and norm are a conv block's, under the same names:
/// `shortcut.conv.weight` and `shortcut.norm.*`, with or without the pool.
#[derive(Module, Debug)]
pub struct ConvShortcut {
    /// The pool before the conv, if any.
    pub pool: Option<AvgPool2d>,

    /// The conv.
    pub conv: Conv2d,

    /// The norm after the conv.
    pub norm: Option<BunsenNormalization>,

    /// The activation after the norm.
    pub act: Option<BunsenActivation>,
}

impl ConvChainMeta for ConvShortcut {
    fn in_channels(&self) -> usize {
        self.conv.weight.dims()[1] * self.conv.groups
    }

    fn out_channels(&self) -> usize {
        self.conv.weight.dims()[0]
    }

    fn stride(&self) -> usize {
        self.pool.as_ref().map_or(1, |p| p.stride[0]) * self.conv.stride[0]
    }
}

impl ConvShortcut {
    /// Applies the shortcut: the pool, the conv, the norm, the activation.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, in_height, in_width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        #[cfg(any(debug_assertions, test))]
        let [batch] = {
            let [_, in_channels, _, _] = self.conv.weight.dims();
            crate::contracts::unpack_shape_contract!(
                ["batch", "in_channels", _, _],
                &input.dims(),
                &["batch"],
                &[("in_channels", in_channels)]
            )
        };
        let x = match &self.pool {
            Some(pool) => pool.forward(input),
            None => input,
        };
        let x = self.conv.forward(x);
        let x = match &self.norm {
            Some(norm) => norm.forward(x),
            None => x,
        };
        let x = match &self.act {
            Some(act) => act.forward(x),
            None => x,
        };
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "out_channels", _, _],
            &x.dims(),
            &[
                ("batch", batch),
                ("out_channels", self.conv.weight.dims()[0])
            ]
        );
        x
    }
}

/// A [`ResidualUnit`]'s config: its conv path, which conv is its spatial
/// mixer, its shortcut, the activation after the add, and its regularizers.
///
/// Every residual unit kind lowers to this one shape
/// ([`ConvUnitKind`](super::ConvUnitKind)): a basic unit is two `3x3` convs
/// whose first is the mixer, a bottleneck a `1x1`, a `3x3` mixer and a `1x1`.
/// The last conv carries no activation: the unit's `act` follows the add.
///
/// Implements [`ConvChainMeta`].
#[derive(Config, Debug)]
pub struct ResidualUnitConfig {
    /// The residual path, in order; each conv's output is the next one's
    /// input.
    pub convs: Vec<ResidualConvConfig>,

    /// The index in `convs` of the spatial mixer, the conv that strides: the
    /// drop block follows its norm.
    pub mixer: usize,

    /// The anti-aliased downsampling after the mixer's activation, which then
    /// takes the stride from the mixer (timm's `aa`).
    #[config(default = "None")]
    pub aa: Option<AntiAliasConfig>,

    /// The shortcut, when the path changes the channels or the size; else the
    /// identity.
    #[config(default = "None")]
    pub shortcut: Option<ConvShortcutConfig>,

    /// The channel attention over the path's output, before the add (timm's
    /// `se` slot: squeeze-excite, ECA).
    #[config(default = "None")]
    pub attn: Option<ChannelAttentionConfig>,

    /// The activation after the add.
    #[config(default = "Some(BunsenActivationConfig::Relu)")]
    pub act: Option<BunsenActivationConfig>,

    /// The drop block after the mixer's norm (training only).
    #[config(default = "None")]
    pub drop_block: Option<DropBlockOptions>,

    /// The probability the whole path is dropped (stochastic depth, training
    /// only).
    #[config(default = "0.0")]
    pub drop_path_prob: f64,
}

impl ConvChainMeta for ResidualUnitConfig {
    fn in_channels(&self) -> usize {
        self.convs[0].in_channels()
    }

    fn out_channels(&self) -> usize {
        self.convs[self.convs.len() - 1].out_channels()
    }

    fn stride(&self) -> usize {
        self.convs.iter().map(|c| c.stride()).product::<usize>()
            * self.aa.as_ref().map_or(1, AntiAliasConfig::stride)
    }
}

impl ResidualUnitConfig {
    /// Checks the unit's shape: a path that is not empty and chains, a mixer
    /// in it, a shortcut (or none) that matches the path, and a drop-path
    /// probability.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, naming the field.
    pub fn try_validate(&self) -> BunsenResult<()> {
        const OWNER: &str = "ResidualUnitConfig";
        if self.convs.is_empty() {
            return Err(ConstraintError::zero_or_empty(OWNER, "convs").into());
        }
        ConstraintError::check_chain(
            OWNER,
            "convs",
            self.convs
                .iter()
                .enumerate()
                .map(|(idx, c)| (idx, c.in_channels(), c.out_channels())),
        )?;
        if self.mixer >= self.convs.len() {
            return Err(ConstraintError::new(
                OWNER,
                "mixer",
                Rule::Relation {
                    lhs: ("mixer".into(), format!("{}", self.mixer)),
                    op: "<",
                    rhs: ("convs.len()".into(), format!("{}", self.convs.len())),
                },
            )
            .into());
        }
        let (in_channels, out_channels, stride) =
            (self.in_channels(), self.out_channels(), self.stride());
        match &self.shortcut {
            Some(shortcut) => {
                for (what, path, short) in [
                    ("in_channels", in_channels, shortcut.in_channels()),
                    ("out_channels", out_channels, shortcut.out_channels()),
                    ("stride", stride, shortcut.stride()),
                ] {
                    if path != short {
                        return Err(ConstraintError::new(
                            OWNER,
                            "shortcut",
                            Rule::Relation {
                                lhs: (format!("shortcut {what}"), format!("{short}")),
                                op: "==",
                                rhs: (format!("path {what}"), format!("{path}")),
                            },
                        )
                        .into());
                    }
                }
            }
            None if in_channels != out_channels || stride != 1 => {
                return Err(BunsenError::from(ConstraintError::new(
                    OWNER,
                    "shortcut",
                    Rule::Missing,
                ))
                .context(format!(
                    "the path maps {in_channels} channels to {out_channels} at stride {stride}; \
                     the identity cannot"
                )));
            }
            None => {}
        }
        if let Some(ChannelAttentionConfig::Se(se)) = &self.attn
            && se.channels() != out_channels
        {
            return Err(ConstraintError::new(
                OWNER,
                "attn",
                Rule::Relation {
                    lhs: ("attn channels".into(), format!("{}", se.channels())),
                    op: "==",
                    rhs: ("path out_channels".into(), format!("{out_channels}")),
                },
            )
            .into());
        }
        try_probability(self.drop_path_prob).context("drop_path_prob")?;
        Ok(())
    }
}

impl ModuleInit<ResidualUnit> for ResidualUnitConfig {
    /// Builds the unit.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](ResidualUnitConfig::try_validate).
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ResidualUnit> {
        self.try_validate()?;
        Ok(ResidualUnit {
            convs: self
                .convs
                .iter()
                .map(|c| c.try_init(device))
                .collect::<BunsenResult<_>>()?,
            mixer: self.mixer,
            aa: self.aa.as_ref().map(|aa| aa.try_init(device)).transpose()?,
            shortcut: self
                .shortcut
                .as_ref()
                .map(|s| s.try_init(device))
                .transpose()?,
            attn: self.attn.as_ref().map(|a| a.try_init(device)).transpose()?,
            act: self.act.as_ref().map(|a| a.init(device)),
            drop_block: self
                .drop_block
                .as_ref()
                .map(|options| DropBlock2dConfig::from(options.clone()).init(device)),
            drop_path: drop_path_layer(self.drop_path_prob, device),
        })
    }
}

/// A [`DropPath`] for a nonzero probability; none for 0.
fn drop_path_layer(
    drop_path_prob: f64,
    device: &Device,
) -> Option<DropPath> {
    (drop_path_prob != 0.0).then(|| {
        DropPathConfig::new()
            .with_drop_prob(drop_path_prob)
            .init(device)
    })
}

/// A residual unit: a conv path added to a shortcut, then an activation.
///
/// `act(shortcut(x) + drop_path(attn(path(x))))`, where the path runs
/// `convs` in order, a drop block follows the mixer's norm and the
/// anti-aliasing pool its activation, and `attn` is the channel attention,
/// when there is one. The shortcut is the
/// identity when there is none. Built by [`ResidualUnitConfig`].
///
/// Implements [`ConvChainMeta`].
#[derive(Module, Debug)]
pub struct ResidualUnit {
    /// The residual path.
    pub convs: Vec<ResidualConv>,

    /// The index in `convs` of the spatial mixer.
    pub mixer: usize,

    /// The anti-aliased downsampling after the mixer.
    pub aa: Option<AntiAlias>,

    /// The shortcut; the identity when none.
    pub shortcut: Option<ConvShortcut>,

    /// The channel attention over the path's output.
    pub attn: Option<ChannelAttention>,

    /// The activation after the add.
    pub act: Option<BunsenActivation>,

    /// The drop block after the mixer's norm.
    pub drop_block: Option<DropBlock2d>,

    /// Stochastic depth over the path.
    pub drop_path: Option<DropPath>,
}

impl ConvChainMeta for ResidualUnit {
    fn in_channels(&self) -> usize {
        self.convs[0].in_channels()
    }

    fn out_channels(&self) -> usize {
        self.convs[self.convs.len() - 1].out_channels()
    }

    fn stride(&self) -> usize {
        self.convs.iter().map(|c| c.stride()).product::<usize>()
            * self.aa.as_ref().map_or(1, AntiAlias::stride)
    }
}

impl ResidualUnit {
    /// Applies the unit.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, in_height, in_width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`, each side
    /// `ceil(in / stride)`, as torch's.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        let [batch, in_height, in_width] = crate::contracts::unpack_shape_contract!(
            ["batch", "in_channels", "in_height", "in_width"],
            &input.dims(),
            &["batch", "in_height", "in_width"],
            &[("in_channels", self.in_channels())],
        );
        let stride = self.stride();
        let (out_height, out_width) = (in_height.div_ceil(stride), in_width.div_ceil(stride));
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (batch, out_height, out_width);

        let shortcut = match &self.shortcut {
            Some(shortcut) => shortcut.forward(input.clone()),
            None => input.clone(),
        };

        let mut x = input;
        for (idx, conv) in self.convs.iter().enumerate() {
            x = match &self.drop_block {
                Some(drop_block) if idx == self.mixer => {
                    conv.map_forward(x, |x| drop_block.forward(x))
                }
                _ => conv.forward(x),
            };
            if idx == self.mixer
                && let Some(aa) = &self.aa
            {
                x = aa.forward(x);
            }
        }
        if let Some(attn) = &self.attn {
            x = attn.forward(x);
        }
        if let Some(drop_path) = &self.drop_path {
            x = drop_path.forward(x);
        }
        let x = x + shortcut;
        let x = match &self.act {
            Some(act) => act.forward(x),
            None => x,
        };

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "out_channels", "out_height", "out_width"],
            &x.dims(),
            &[
                ("batch", batch),
                ("out_channels", self.out_channels()),
                ("out_height", out_height),
                ("out_width", out_width),
            ],
        );
        x
    }

    /// Sets the drop-path probability: a [`DropPath`] for a nonzero one, none
    /// for 0.
    ///
    /// # Panics
    ///
    /// If `drop_path_prob` is not a probability.
    pub fn with_drop_path_prob(
        self,
        drop_path_prob: f64,
    ) -> Self {
        let drop_path_prob = expect_probability(drop_path_prob);
        let device = self.devices()[0].clone();
        Self {
            drop_path: drop_path_layer(drop_path_prob, &device),
            ..self
        }
    }

    /// Sets the drop block after the mixer's norm; none clears it.
    pub fn with_drop_block(
        self,
        options: Option<DropBlockOptions>,
    ) -> Self {
        let device = self.devices()[0].clone();
        Self {
            drop_block: options.map(|options| DropBlock2dConfig::from(options).init(&device)),
            ..self
        }
    }
}
