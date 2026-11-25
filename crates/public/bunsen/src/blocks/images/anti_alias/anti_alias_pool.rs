//! [`AntiAlias`]: the anti-aliased downsampling seam.

use burn::{
    config::Config,
    module::Module,
    nn::pool::{
        AvgPool2d,
        AvgPool2dConfig,
    },
    prelude::Tensor,
};

use super::{
    BlurPool2d,
    BlurPool2dConfig,
    BlurPool2dMeta,
};
use crate::errors::BunsenResult;

/// The config of an [`AntiAlias`]: its kind, and that kind's config.
#[derive(Config, Debug)]
pub enum AntiAliasConfig {
    /// A binomial blur, then the stride.
    Blur(BlurPool2dConfig),

    /// An average pool.
    Avg(AvgPool2dConfig),
}

impl AntiAliasConfig {
    /// timm's `aa_layer=BlurPool2d` at `stride`: a `3x3` blur.
    pub fn blur(stride: usize) -> Self {
        Self::Blur(BlurPool2dConfig::new().with_stride(stride))
    }

    /// timm's `aa_layer=nn.AvgPool2d` at `stride`: `AvgPool2d(stride)`, its
    /// kernel and stride both `stride`, no padding.
    pub fn avg(stride: usize) -> Self {
        Self::Avg(AvgPool2dConfig::new([stride, stride]).with_strides([stride, stride]))
    }

    /// The stride.
    pub fn stride(&self) -> usize {
        match self {
            Self::Blur(config) => config.stride(),
            Self::Avg(config) => config.strides[0],
        }
    }

    /// The pool.
    ///
    /// # Errors
    ///
    /// As [`BlurPool2dConfig::try_init`].
    pub fn try_init(&self) -> BunsenResult<AntiAlias> {
        Ok(match self {
            Self::Blur(config) => AntiAlias::Blur(config.try_init()?),
            Self::Avg(config) => AntiAlias::Avg(config.init()),
        })
    }
}

/// An anti-aliased downsampling pool; it has no parameters.
#[derive(Module, Debug)]
pub enum AntiAlias {
    /// A binomial blur, then the stride.
    Blur(BlurPool2d),

    /// An average pool.
    Avg(AvgPool2d),
}

impl AntiAlias {
    /// The stride.
    pub fn stride(&self) -> usize {
        match self {
            Self::Blur(pool) => pool.stride(),
            Self::Avg(pool) => pool.stride[0],
        }
    }

    /// The input, pooled: `[batch, channels, height, width]` to `[batch,
    /// channels, out_height, out_width]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        match self {
            Self::Blur(pool) => pool.forward(input),
            Self::Avg(pool) => pool.forward(input),
        }
    }
}
