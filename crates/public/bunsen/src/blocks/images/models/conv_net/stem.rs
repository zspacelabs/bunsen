//! The stem: the convs before the first stage, and a pool.

use alloc::vec::Vec;

use burn::{
    config::Config,
    module::Module,
    nn::pool::{
        AvgPool2d,
        AvgPool2dConfig,
        MaxPool2d,
        MaxPool2dConfig,
    },
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::{
        conv::{
            ConvBlock2d,
            ConvBlock2dConfig,
            ConvBlock2dMeta,
        },
        images::{
            anti_alias::{
                BlurPool2d,
                BlurPool2dConfig,
                BlurPool2dMeta,
            },
            models::conv_net::ConvChainMeta,
        },
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
    },
};

/// The stem's pool: the config of a [`ConvStemPool`].
///
/// A seam: each kind is a module of its own. Built: a max pool (`ResNet`'s
/// `3x3`, stride 2), and the anti-aliased nets' (timm's `aa_layer`): an
/// average pool, or a max pool of stride 1 and a blur of the stride; and
/// `ResNet-RS`'s strided conv block.
#[derive(Config, Debug)]
pub enum ConvStemPoolConfig {
    /// A max pool.
    Max(MaxPool2dConfig),

    /// A strided conv, its norm and its activation (timm's
    /// `replace_stem_pool`).
    Conv(ConvBlock2dConfig),

    /// An average pool.
    Avg(AvgPool2dConfig),

    /// A max pool, then a blur that strides.
    MaxBlur(MaxBlurPoolConfig),

    /// A zero pad of 1 on each side, then a max pool of no padding of its
    /// own (`BiT`'s `fixed` stem: `ConstantPad2d(1, 0)` and `MaxPool2d(3, 2,
    /// padding=0)`); the zeros, unlike a max pool's own padding, can win
    /// over a negative map.
    ZeroPadMax(MaxPool2dConfig),
}

impl ConvStemPoolConfig {
    /// The pool's stride.
    pub fn stride(&self) -> usize {
        match self {
            Self::Max(config) => config.strides[0],
            Self::Conv(config) => config.stride()[0],
            Self::Avg(config) => config.strides[0],
            Self::MaxBlur(config) => config.max.strides[0] * config.blur.stride(),
            Self::ZeroPadMax(config) => config.strides[0],
        }
    }
}

/// The pool.
///
/// # Errors
///
/// As [`BlurPool2dConfig`]'s and the conv block's `try_init`.
impl ModuleInit<ConvStemPool> for ConvStemPoolConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConvStemPool> {
        Ok(match self {
            Self::Max(config) => ConvStemPool::Max(config.init()),
            Self::Conv(config) => ConvStemPool::Conv(config.try_init(device)?),
            Self::Avg(config) => ConvStemPool::Avg(config.init()),
            Self::MaxBlur(config) => ConvStemPool::MaxBlur(MaxBlurPool {
                max: config.max.init(),
                blur: config.blur.try_init(device)?,
            }),
            Self::ZeroPadMax(config) => ConvStemPool::ZeroPadMax(config.init()),
        })
    }
}

/// The config of a [`MaxBlurPool`].
#[derive(Config, Debug)]
pub struct MaxBlurPoolConfig {
    /// The max pool, at stride 1 in timm's.
    pub max: MaxPool2dConfig,

    /// The blur, which strides.
    pub blur: BlurPool2dConfig,
}

/// A max pool, then a blur that strides: an anti-aliased `ResNet`'s stem
/// pool (timm's `MaxPool2d(3, 1, 1)` and `BlurPool2d`).
#[derive(Module, Debug)]
pub struct MaxBlurPool {
    /// The max pool.
    pub max: MaxPool2d,

    /// The blur.
    pub blur: BlurPool2d,
}

/// The stem's pool; built by [`ConvStemPoolConfig`]. Only the conv kind has
/// parameters: `stem.pool.conv` and `stem.pool.norm`.
// A burn module enum holds its kinds by value, as the other seams do.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum ConvStemPool {
    /// A max pool.
    Max(MaxPool2d),

    /// A strided conv, its norm and its activation.
    Conv(ConvBlock2d),

    /// An average pool.
    Avg(AvgPool2d),

    /// A max pool, then a blur that strides.
    MaxBlur(MaxBlurPool),

    /// A zero pad of 1, then a max pool.
    ZeroPadMax(MaxPool2d),
}

impl ConvStemPool {
    /// The pool's stride.
    pub fn stride(&self) -> usize {
        match self {
            Self::Max(pool) => pool.stride[0],
            Self::Conv(block) => block.stride()[0],
            Self::Avg(pool) => pool.stride[0],
            Self::MaxBlur(pool) => pool.max.stride[0] * pool.blur.stride(),
            Self::ZeroPadMax(pool) => pool.stride[0],
        }
    }

    /// Applies the pool: `[batch, channels, height, width]` to
    /// `[batch, channels, out_height, out_width]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        #[cfg(any(debug_assertions, test))]
        let [batch, channels] = crate::contracts::unpack_shape_contract!(
            ["batch", "channels", _, _],
            &input.dims(),
            &["batch", "channels"]
        );
        let x = match self {
            Self::Max(pool) => pool.forward(input),
            Self::Conv(block) => block.forward(input),
            Self::Avg(pool) => pool.forward(input),
            Self::MaxBlur(pool) => pool.blur.forward(pool.max.forward(input)),
            Self::ZeroPadMax(pool) => pool.forward(input.pad([(1, 1), (1, 1)], 0.0)),
        };
        // A pool keeps the channels.
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "channels", _, _],
            &x.dims(),
            &[("batch", batch), ("channels", channels)]
        );
        x
    }
}

/// [`ConvStem`]'s config: its convs, in order, and its pool.
///
/// A single `7x7` conv is `ResNet`'s stem; a run of `3x3`s its deep and
/// tiered stems; a conv whose kernel is its stride, with no norm, a
/// patchify stem.
///
/// Implements [`ConvChainMeta`].
#[derive(Config, Debug)]
pub struct ConvStemConfig {
    /// The convs, in order; each one's output is the next one's input.
    pub convs: Vec<ConvBlock2dConfig>,

    /// The pool after them, if any.
    #[config(default = "None")]
    pub pool: Option<ConvStemPoolConfig>,
}

impl ConvChainMeta for ConvStemConfig {
    fn in_channels(&self) -> usize {
        self.convs[0].in_channels()
    }

    fn out_channels(&self) -> usize {
        match &self.pool {
            Some(ConvStemPoolConfig::Conv(block)) => block.out_channels(),
            _ => self.convs[self.convs.len() - 1].out_channels(),
        }
    }

    fn stride(&self) -> usize {
        let convs: usize = self.convs.iter().map(|c| c.stride()[0]).product();
        convs * self.pool.as_ref().map_or(1, |p| p.stride())
    }
}

impl ConvStemConfig {
    /// Checks the stem's shape: convs that are not empty and chain, into a
    /// conv pool's input.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause.
    pub fn try_validate(&self) -> BunsenResult<()> {
        if self.convs.is_empty() {
            return Err(ConstraintError::zero_or_empty("ConvStemConfig", "convs").into());
        }
        let convs = || {
            self.convs
                .iter()
                .enumerate()
                .map(|(idx, c)| (idx, c.in_channels(), c.out_channels()))
        };
        ConstraintError::check_chain("ConvStemConfig", "convs", convs())?;
        if let Some(ConvStemPoolConfig::Conv(pool)) = &self.pool {
            // The last conv is the pool's source; a break names the pool's
            // place after the convs.
            ConstraintError::check_chain(
                "ConvStemConfig",
                "pool",
                convs().next_back().into_iter().chain(core::iter::once((
                    self.convs.len(),
                    pool.in_channels(),
                    pool.out_channels(),
                ))),
            )?;
        }
        Ok(())
    }
}

impl ModuleInit<ConvStem> for ConvStemConfig {
    /// Builds the stem.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](ConvStemConfig::try_validate).
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConvStem> {
        self.try_validate()?;
        Ok(ConvStem {
            convs: self
                .convs
                .iter()
                .map(|c| c.try_init(device))
                .collect::<BunsenResult<_>>()?,
            pool: self
                .pool
                .as_ref()
                .map(|pool| pool.try_init(device))
                .transpose()?,
        })
    }
}

/// A conv net's stem: convs, then a pool. Built by [`ConvStemConfig`].
///
/// Implements [`ConvChainMeta`].
#[derive(Module, Debug)]
pub struct ConvStem {
    /// The convs, in order.
    pub convs: Vec<ConvBlock2d>,

    /// The pool, if any.
    pub pool: Option<ConvStemPool>,
}

impl ConvChainMeta for ConvStem {
    fn in_channels(&self) -> usize {
        self.convs[0].in_channels()
    }

    fn out_channels(&self) -> usize {
        match &self.pool {
            Some(ConvStemPool::Conv(block)) => block.out_channels(),
            _ => self.convs[self.convs.len() - 1].out_channels(),
        }
    }

    fn stride(&self) -> usize {
        let convs: usize = self.convs.iter().map(|c| c.stride()[0]).product();
        convs * self.pool.as_ref().map_or(1, ConvStemPool::stride)
    }
}

impl ConvStem {
    /// The convs' output, before the pool: timm's first feature map (`act1`
    /// on a `ResNet`, at stride 2).
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, conv_height, conv_width]`.
    pub fn forward_convs(
        &self,
        images: Tensor<4>,
    ) -> Tensor<4> {
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "in_channels", "height", "width"],
            &images.dims(),
            &[("in_channels", self.in_channels())],
        );
        self.convs.iter().fold(images, |x, conv| conv.forward(x))
    }

    /// The pool, over [`forward_convs`](Self::forward_convs)' output.
    pub fn forward_pool(
        &self,
        x: Tensor<4>,
    ) -> Tensor<4> {
        match &self.pool {
            Some(pool) => pool.forward(x),
            None => x,
        }
    }

    /// The whole stem: convs, then the pool.
    ///
    /// # Arguments
    ///
    /// - `images`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`.
    pub fn forward(
        &self,
        images: Tensor<4>,
    ) -> Tensor<4> {
        self.forward_pool(self.forward_convs(images))
    }
}
