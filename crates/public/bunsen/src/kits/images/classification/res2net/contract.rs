// `Res2Net`'s policy: a `ResNet`'s, its bottleneck's mixer the scale split.

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

/// A `Res2Net`'s policy: timm's `res2net.py` knobs, the `ResNet` around the
/// blocks and the `Bottle2neck`'s `scale`, lowering to the generic
/// [`ConvNetContractConfig`].
///
/// A `Res2Net` is timm's `ResNet` (its stem, depths, shortcuts and head)
/// whose bottlenecks' `3x3` is the scale split
/// ([`Res2ConvConfig`](crate::blocks::images::mixers::Res2ConvConfig)): the
/// first `1x1` widens to `scale` chunks of the mixer's width (timm's
/// `floor(planes * base_width / 64) * cardinality`, the `ResNet`'s
/// [`mixer_width`](crate::kits::images::classification::resnet::BottleneckPolicyConfig::mixer_width)),
/// and each chunk but the last gets its own `3x3`.
#[derive(Config, Debug)]
pub struct Res2NetContractConfig {
    /// The `ResNet` around the blocks: bottlenecks of its
    /// [`bottleneck_policy`](ResNetContractConfig::bottleneck_policy)'s
    /// `base_width` (26, 48, 14 or 4) and `cardinality`.
    pub resnet: ResNetContractConfig,

    /// The chunks (timm's `block_args.scale`).
    #[config(default = 4)]
    pub scale: usize,
}

impl Res2NetContractConfig {
    /// The generic conv net's policy of this `Res2Net`: the `ResNet`'s
    /// ([`try_to_conv_net`](ResNetContractConfig::try_to_conv_net)), each
    /// stage's mixer `scale` times as wide, the mixer the scale split.
    ///
    /// # Errors
    ///
    /// As the `ResNet`'s; [`Illegal`](crate::errors::BunsenErrorKind::Illegal)
    /// for basic blocks, or a scale of 0.
    pub fn try_conv_net(&self) -> BunsenResult<ConvNetContractConfig> {
        if self.resnet.bottleneck_policy.is_none() {
            return Err(BunsenError::illegal(
                "Res2NetContractConfig: a Res2Net's blocks are bottlenecks",
            ));
        }
        if self.scale == 0 {
            return Err(BunsenError::illegal("Res2NetContractConfig: a scale of 0"));
        }
        let mut net = self.resnet.try_to_conv_net()?;
        for stage in &mut net.stages {
            stage.mid_width *= self.scale;
        }
        net.layers.mixer = ConvMixerKind::Res2 { scale: self.scale };
        Ok(net)
    }
}

impl ToStructureConfig for Res2NetContractConfig {
    type Structure = ConvNetStructureConfig;

    /// The conv net of this `Res2Net` ([`try_conv_net`](Self::try_conv_net)),
    /// resolved.
    ///
    /// # Errors
    ///
    /// As [`try_conv_net`](Self::try_conv_net) and the conv net's lowering,
    /// under a frame naming the `Res2Net`.
    fn try_to_structure(&self) -> BunsenResult<ConvNetStructureConfig> {
        self.try_conv_net()
            .and_then(|net| net.try_to_structure())
            .context("Res2NetContractConfig")
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{
        blocks::images::models::conv_net::{
            ConvChainMeta,
            ResidualConvConfig,
        },
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::images::classification::resnet::BottleneckPolicyConfig,
    };

    /// timm's `res2net50_26w_4s`: 26-wide chunks per 64 planes, four of
    /// them, around `ResNet-50`'s stages.
    #[test]
    fn test_res2net50_lowering() {
        let config = Res2NetContractConfig::new(
            ResNetContractConfig::new(vec![3, 4, 6, 3], 1000)
                .with_bottleneck_policy(Some(BottleneckPolicyConfig::new().with_base_width(26))),
        );
        let structure = config.to_structure();
        let unit = &structure.stages[1].units[0];
        assert_eq!(unit.convs[0].out_channels(), 52 * 4);
        let ResidualConvConfig::Res2(res2) = &unit.convs[1] else {
            panic!("a scale split");
        };
        assert_eq!((res2.width(), res2.scale, res2.stride()), (52, 4, 2));
        assert_eq!(unit.out_channels(), 512);
    }

    #[test]
    fn test_illegal_policies() {
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("a Res2Net's blocks are bottlenecks")
            .assert_err(
                &Res2NetContractConfig::new(ResNetContractConfig::new(vec![2, 2, 2, 2], 1000))
                    .try_to_structure(),
            );
    }
}
