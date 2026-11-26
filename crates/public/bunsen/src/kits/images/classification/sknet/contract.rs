// `SKNet`'s policy: a `ResNet`'s, its mixer the selective kernel.

use burn::config::Config;

use crate::{
    blocks::images::models::conv_net::{
        ConvMixerKind,
        ConvNetContractConfig,
        ConvNetStructureConfig,
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
    },
    kits::images::classification::resnet::ResNetContractConfig,
};

/// An `SKNet`'s policy: timm's `sknet.py` knobs, the `ResNet` around the
/// blocks and the blocks' `sk_kwargs` (`rd_ratio`, `rd_divisor`,
/// `split_input`), lowering to the generic [`ConvNetContractConfig`].
///
/// An `SKNet` (timm's `skresnet`, `skresnext`) is timm's `ResNet` whose
/// spatial mixer, a basic block's first conv or a bottleneck's `3x3`, is the
/// selective kernel
/// ([`SelectiveKernelConfig`](crate::blocks::images::mixers::SelectiveKernelConfig)):
/// two paths of two receptive fields, weighed per channel by an attention over
/// their sum.
#[derive(Config, Debug)]
pub struct SKNetContractConfig {
    /// The `ResNet` around the blocks: basic blocks (`skresnet18`, `34`), or
    /// bottlenecks of its policy's `base_width` and `cardinality`
    /// (`skresnext50_32x4d`).
    pub resnet: ResNetContractConfig,

    /// The mixer's paths' kernels (timm's `kernel_size`).
    #[config(default = "[3, 5]")]
    pub kernel_size: [usize; 2],

    /// Whether each path is kept `3x3`, dilated to its kernel's reach
    /// (timm's `keep_3x3`).
    #[config(default = true)]
    pub keep_3x3: bool,

    /// The attention width's share of the mixer's channels (timm's
    /// `rd_ratio`: `1/16` by default, `1/8` for the basic nets).
    #[config(default = 0.0625)]
    pub rd_ratio: f64,

    /// The attention width's divisor (timm's `rd_divisor`).
    #[config(default = 8)]
    pub rd_divisor: usize,

    /// Whether the mixer's input splits across its paths (timm's
    /// `split_input`).
    #[config(default = true)]
    pub split_input: bool,
}

impl SKNetContractConfig {
    /// The generic conv net's policy of this `SKNet`: the `ResNet`'s
    /// ([`try_to_conv_net`](ResNetContractConfig::try_to_conv_net)), its
    /// mixer the selective kernel.
    ///
    /// # Errors
    ///
    /// As the `ResNet`'s; [`Illegal`](crate::errors::BunsenErrorKind::Illegal)
    /// for a `ResNet` that anti-aliases (timm's blocks pass `aa_layer` to
    /// the paths, which the kit doesn't build).
    pub fn try_conv_net(&self) -> BunsenResult<ConvNetContractConfig> {
        if self.resnet.anti_alias.is_some() {
            return Err(BunsenError::illegal(
                "SKNetContractConfig: an anti-aliased SKNet",
            ));
        }
        let mut net = self.resnet.try_to_conv_net()?;
        net.layers.mixer = ConvMixerKind::SelectiveKernel {
            kernel_size: self.kernel_size,
            keep_3x3: self.keep_3x3,
            rd_ratio: self.rd_ratio,
            rd_divisor: self.rd_divisor,
            split_input: self.split_input,
        };
        Ok(net)
    }
}

impl ToStructureConfig for SKNetContractConfig {
    type Structure = ConvNetStructureConfig;

    /// The conv net of this `SKNet` ([`try_conv_net`](Self::try_conv_net)),
    /// resolved.
    ///
    /// # Errors
    ///
    /// As [`try_conv_net`](Self::try_conv_net) and the conv net's lowering,
    /// under a frame naming the `SKNet`.
    fn try_to_structure(&self) -> BunsenResult<ConvNetStructureConfig> {
        self.try_conv_net()
            .and_then(|net| net.try_to_structure())
            .context("SKNetContractConfig")
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{
        blocks::{
            conv::ConvBlock2dMeta,
            images::models::conv_net::{
                ConvAaKind,
                ResidualConvConfig,
            },
        },
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
    };

    /// timm's `skresnet18`: basic blocks, the selective kernel their first
    /// conv, the attention `1/8` of the channels in 16s.
    #[test]
    fn test_skresnet18_lowering() {
        let config = SKNetContractConfig::new(ResNetContractConfig::new(vec![2, 2, 2, 2], 1000))
            .with_rd_ratio(0.125)
            .with_rd_divisor(16);
        let structure = config.to_structure();
        let ResidualConvConfig::SelectiveKernel(sk) = &structure.stages[1].units[0].convs[0] else {
            panic!("a selective kernel");
        };
        assert_eq!(
            (sk.in_channels(), sk.out_channels(), sk.stride()),
            (64, 128, 2)
        );
        assert_eq!(sk.reduce.out_channels(), 16);
    }

    #[test]
    fn test_illegal_policies() {
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("an anti-aliased SKNet")
            .assert_err(
                &SKNetContractConfig::new(
                    ResNetContractConfig::new(vec![2, 2, 2, 2], 1000)
                        .with_anti_alias(Some(ConvAaKind::Blur)),
                )
                .try_to_structure(),
            );
    }
}
