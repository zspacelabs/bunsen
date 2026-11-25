//! The residual unit: a conv path, a shortcut, and the add.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    module::Module,
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
        images::drop::{
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

/// The view shared by a residual unit's parts and their configs: what goes
/// in, what comes out, and how far it strides.
pub trait ResidualUnitMeta {
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

impl ResidualUnitMeta for ResidualConvConfig {
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

impl ResidualUnitMeta for ResidualConv {
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
/// channels or size: the config of a [`ConvShortcut`].
///
/// A seam: each kind is a module of its own. Built today: a strided `1x1`
/// conv and a norm (torchvision's `downsample`). The average-pool-then-conv
/// shortcut of the `ResNet-D` variants is a further kind.
#[derive(Config, Debug)]
pub enum ConvShortcutConfig {
    /// A conv and a norm, with no activation.
    Conv(ConvBlock2dConfig),
}

impl ResidualUnitMeta for ConvShortcutConfig {
    fn in_channels(&self) -> usize {
        match self {
            Self::Conv(config) => config.in_channels(),
        }
    }

    fn out_channels(&self) -> usize {
        match self {
            Self::Conv(config) => config.out_channels(),
        }
    }

    fn stride(&self) -> usize {
        match self {
            Self::Conv(config) => config.stride()[0],
        }
    }
}

impl ModuleInit<ConvShortcut> for ConvShortcutConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConvShortcut> {
        Ok(match self {
            Self::Conv(config) => ConvShortcut::Conv(config.try_init(device)?),
        })
    }
}

/// A [`ResidualUnit`]'s shortcut; built by [`ConvShortcutConfig`].
///
/// The variant is not part of a parameter's path: a conv shortcut's weight is
/// `shortcut.conv.weight`.
#[derive(Module, Debug)]
pub enum ConvShortcut {
    /// A conv and a norm, with no activation.
    Conv(ConvBlock2d),
}

impl ResidualUnitMeta for ConvShortcut {
    fn in_channels(&self) -> usize {
        match self {
            Self::Conv(block) => block.in_channels(),
        }
    }

    fn out_channels(&self) -> usize {
        match self {
            Self::Conv(block) => block.out_channels(),
        }
    }

    fn stride(&self) -> usize {
        match self {
            Self::Conv(block) => block.stride()[0],
        }
    }
}

impl ConvShortcut {
    /// Applies the shortcut.
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
        match self {
            Self::Conv(block) => block.forward(input),
        }
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
/// Implements [`ResidualUnitMeta`].
#[derive(Config, Debug)]
pub struct ResidualUnitConfig {
    /// The residual path, in order; each conv's output is the next one's
    /// input.
    pub convs: Vec<ResidualConvConfig>,

    /// The index in `convs` of the spatial mixer, the conv that strides: the
    /// drop block follows its norm.
    pub mixer: usize,

    /// The shortcut, when the path changes the channels or the size; else the
    /// identity.
    #[config(default = "None")]
    pub shortcut: Option<ConvShortcutConfig>,

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

impl ResidualUnitMeta for ResidualUnitConfig {
    fn in_channels(&self) -> usize {
        self.convs[0].in_channels()
    }

    fn out_channels(&self) -> usize {
        self.convs[self.convs.len() - 1].out_channels()
    }

    fn stride(&self) -> usize {
        self.convs.iter().map(|c| c.stride()).product()
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
        for idx in 1..self.convs.len() {
            let (prev, curr) = (&self.convs[idx - 1], &self.convs[idx]);
            if prev.out_channels() != curr.in_channels() {
                return Err(ConstraintError::new(
                    OWNER,
                    "convs",
                    Rule::Chain {
                        index: idx,
                        out: format!("{} channels", prev.out_channels()),
                        r#in: format!("{} channels", curr.in_channels()),
                    },
                )
                .into());
            }
        }
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
            shortcut: self
                .shortcut
                .as_ref()
                .map(|s| s.try_init(device))
                .transpose()?,
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
/// `act(shortcut(x) + drop_path(path(x)))`, where the path runs `convs` in
/// order and a drop block follows the mixer's norm. The shortcut is the
/// identity when there is none. Built by [`ResidualUnitConfig`].
///
/// Implements [`ResidualUnitMeta`].
#[derive(Module, Debug)]
pub struct ResidualUnit {
    /// The residual path.
    pub convs: Vec<ResidualConv>,

    /// The index in `convs` of the spatial mixer.
    pub mixer: usize,

    /// The shortcut; the identity when none.
    pub shortcut: Option<ConvShortcut>,

    /// The activation after the add.
    pub act: Option<BunsenActivation>,

    /// The drop block after the mixer's norm.
    pub drop_block: Option<DropBlock2d>,

    /// Stochastic depth over the path.
    pub drop_path: Option<DropPath>,
}

impl ResidualUnitMeta for ResidualUnit {
    fn in_channels(&self) -> usize {
        self.convs[0].in_channels()
    }

    fn out_channels(&self) -> usize {
        self.convs[self.convs.len() - 1].out_channels()
    }

    fn stride(&self) -> usize {
        self.convs.iter().map(|c| c.stride()).product()
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
