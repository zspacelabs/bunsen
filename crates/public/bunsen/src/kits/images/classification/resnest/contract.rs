// `ResNeSt`'s policy: a `ResNet`'s, its bottleneck's mixer the split
// attention.

use burn::config::Config;

use crate::{
    blocks::images::models::conv_net::{
        ConvAaKind,
        ConvMixerKind,
        ConvNetContractConfig,
        ConvNetStructureConfig,
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
    kits::images::classification::resnet::ResNetContractConfig,
};

/// A `ResNeSt`'s policy: timm's `resnest.py` knobs, the `ResNet` around the
/// blocks, the `ResNestBottleneck`'s `block_args` (`radix`, `avd`,
/// `avd_first`) and its `SplitAttn`'s (`rd_ratio`, `rd_divisor`), lowering to
/// the generic [`ConvNetContractConfig`].
///
/// A `ResNeSt` is timm's `ResNet` (a deep stem, average-pool shortcuts, its
/// depths and head) whose bottlenecks' `3x3` is the split attention
/// ([`SplitAttnConfig`](crate::blocks::images::mixers::SplitAttnConfig)) of
/// `radix` splits in the `ResNet`'s `cardinality` groups, and whose striding
/// units downsample by an average pool (`avd`) after the mixer, or before it.
#[derive(Config, Debug)]
pub struct ResNeStContractConfig {
    /// The `ResNet` around the blocks: bottlenecks of its
    /// [`bottleneck_policy`](ResNetContractConfig::bottleneck_policy)'s
    /// `base_width` and `cardinality`.
    pub resnet: ResNetContractConfig,

    /// The splits (timm's `radix`): 2, 4, or 1 for a sigmoid gate.
    #[config(default = 2)]
    pub radix: usize,

    /// Whether a striding unit downsamples by a `3x3` average pool, its mixer
    /// at stride 1 (timm's `avd`).
    #[config(default = true)]
    pub avd: bool,

    /// Whether that pool goes before the mixer rather than after it (timm's
    /// `avd_first`).
    #[config(default = false)]
    pub avd_first: bool,

    /// The split attention's width over its input (timm's `SplitAttn`
    /// `rd_ratio`).
    #[config(default = 0.25)]
    pub rd_ratio: f64,

    /// The split attention width's divisor (timm's `SplitAttn` `rd_divisor`).
    #[config(default = 8)]
    pub rd_divisor: usize,
}

impl ResNeStContractConfig {
    /// The generic conv net's policy of this `ResNeSt`: the `ResNet`'s
    /// ([`try_to_conv_net`](ResNetContractConfig::try_to_conv_net)), its
    /// mixer the split attention, its anti-aliasing `avd`'s pool.
    ///
    /// # Errors
    ///
    /// As the `ResNet`'s; [`Illegal`](crate::errors::BunsenErrorKind::Illegal)
    /// for basic blocks, a radix of 0, or a `ResNet` that anti-aliases or
    /// attends itself (timm's `ResNestBottleneck` refuses both).
    pub fn try_to_conv_net(&self) -> BunsenResult<ConvNetContractConfig> {
        if self.resnet.bottleneck_policy.is_none() {
            return Err(ConstraintError::new(
                "ResNeStContractConfig",
                "resnet.bottleneck_policy",
                Rule::Missing,
            )
            .into());
        }
        if self.radix == 0 {
            return Err(ConstraintError::zero_or_empty("ResNeStContractConfig", "radix").into());
        }
        if self.resnet.anti_alias.is_some() || self.resnet.attention.is_some() {
            return Err(ConstraintError::custom(
                "ResNeStContractConfig",
                "resnet",
                "a ResNeSt's blocks take no anti-aliasing or attention of the ResNet's",
            )
            .into());
        }
        let mut net = self.resnet.try_to_conv_net()?;
        net.layers.mixer = ConvMixerKind::SplitAttn {
            radix: self.radix,
            rd_ratio: self.rd_ratio,
            rd_divisor: self.rd_divisor,
        };
        net.layers.aa = self.avd.then_some(ConvAaKind::Avd);
        net.layers.aa_first = self.avd_first;
        Ok(net)
    }
}

impl ToStructureConfig for ResNeStContractConfig {
    type Structure = ConvNetStructureConfig;

    /// The conv net of this `ResNeSt` ([`try_to_conv_net`](Self::try_to_conv_net)),
    /// resolved.
    ///
    /// # Errors
    ///
    /// As [`try_to_conv_net`](Self::try_to_conv_net) and the conv net's lowering,
    /// under a frame naming the `ResNeSt`.
    fn try_to_structure(&self) -> BunsenResult<ConvNetStructureConfig> {
        self.try_to_conv_net()
            .and_then(|net| net.try_to_structure())
            .context("ResNeStContractConfig")
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
        errors::testing::ErrorMatcher,
        kits::images::classification::resnet::BottleneckPolicyConfig,
        support::testing::expect_variant,
    };

    /// timm's `resnest50d`: radix 2, `avd` after the mixer, around
    /// `ResNet-50`'s stages.
    #[test]
    fn test_resnest50d_lowering() {
        let config = ResNeStContractConfig::new(
            ResNetContractConfig::new(vec![3, 4, 6, 3], 1000).with_bottleneck(true),
        );
        let structure = config.to_structure();
        let unit = &structure.stages[1].units[0];
        expect_variant!(
            let ResidualConvConfig::SplitAttn(attn) = &unit.convs[1],
            "a split attention",
        );
        assert_eq!(
            (attn.radix, attn.stride(), attn.out_channels()),
            (2, 1, 128)
        );
        assert!(unit.aa.is_some() && !unit.aa_first);
        assert_eq!(unit.stride(), 2);
        assert!(structure.stages[0].units[0].aa.is_none());
    }

    #[test]
    fn test_illegal_policies() {
        let illegal = |config: ResNeStContractConfig, message: &str| {
            ErrorMatcher::illegal()
                .message_contains(message)
                .assert_err(&config.try_to_structure());
        };
        illegal(
            ResNeStContractConfig::new(ResNetContractConfig::new(vec![2, 2, 2, 2], 1000)),
            "resnet.bottleneck_policy: is required",
        );
        illegal(
            ResNeStContractConfig::new(
                ResNetContractConfig::new(vec![2, 2, 2, 2], 1000)
                    .with_bottleneck_policy(Some(BottleneckPolicyConfig::new()))
                    .with_anti_alias(Some(ConvAaKind::Blur)),
            ),
            "take no anti-aliasing or attention",
        );
    }
}
