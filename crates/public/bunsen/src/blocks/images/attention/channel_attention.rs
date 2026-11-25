//! [`ChannelAttention`]: the channel attention seam.

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::Device,
};

use super::{
    Eca,
    EcaConfig,
    SqueezeExcite,
    SqueezeExciteConfig,
};
use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// The config of a [`ChannelAttention`]: its kind, and that kind's config.
#[derive(Config, Debug)]
pub enum ChannelAttentionConfig {
    /// Squeeze-and-excitation.
    Se(SqueezeExciteConfig),

    /// Efficient channel attention.
    Eca(EcaConfig),
}

impl ModuleInit<ChannelAttention> for ChannelAttentionConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ChannelAttention> {
        Ok(match self {
            Self::Se(config) => ChannelAttention::Se(config.try_init(device)?),
            Self::Eca(config) => ChannelAttention::Eca(config.try_init(device)?),
        })
    }
}

/// A channel attention: a feature map's channels gated by the map itself.
///
/// The variant is not part of a parameter's path: a squeeze-excite's first
/// conv is `fc1`, an ECA's conv `conv`, under the slot's own name.
// A burn module enum holds its kinds by value, as the other seams do.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum ChannelAttention {
    /// Squeeze-and-excitation.
    Se(SqueezeExcite),

    /// Efficient channel attention.
    Eca(Eca),
}

impl ChannelAttention {
    /// The input, its channels gated.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, channels, height, width]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        match self {
            Self::Se(se) => se.forward(input),
            Self::Eca(eca) => eca.forward(input),
        }
    }
}
