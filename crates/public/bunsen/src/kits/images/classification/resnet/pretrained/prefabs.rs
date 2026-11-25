//! # `ResNet` prefabs

use alloc::{
    vec,
    vec::Vec,
};

use burn::nn::GroupNormConfig;

use crate::{
    blocks::images::models::conv_net::{
        ConvAaKind,
        ConvAttentionKind,
    },
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::images::classification::resnet::{
        BottleneckPolicyConfig,
        ResNetContractConfig,
        ResNetStemKind,
    },
};

impl ResNetContractConfig {
    /// The `resnet18` prefab: ResNet-18 [2, 2, 2, 2] basic blocks.
    pub fn resnet18() -> Self {
        ResNetContractConfig::new(vec![2, 2, 2, 2], 1000)
    }

    /// The `resnet26` prefab: ResNet-26 [2, 2, 2, 2] Bottleneck.
    pub fn resnet26() -> Self {
        ResNetContractConfig::new(vec![2, 2, 2, 2], 1000).with_bottleneck(true)
    }

    /// The `resnet34` prefab: ResNet-34 [3, 4, 6, 3] basic blocks.
    pub fn resnet34() -> Self {
        ResNetContractConfig::new(vec![3, 4, 6, 3], 1000)
    }

    /// The `resnet50` prefab: ResNet-50 [3, 4, 6, 3] Bottleneck.
    pub fn resnet50() -> Self {
        ResNetContractConfig::new(vec![3, 4, 6, 3], 1000).with_bottleneck(true)
    }

    /// The `resnet50_gn` prefab: ResNet-50 with timm's group norm (32
    /// groups, epsilon `1e-5`) for every norm, the stem's included.
    pub fn resnet50_gn() -> Self {
        Self::resnet50().with_normalization(GroupNormConfig::new(32, 0).into())
    }

    /// The `resnet101` prefab: ResNet-101 [3, 4, 23, 3] Bottleneck.
    pub fn resnet101() -> Self {
        ResNetContractConfig::new(vec![3, 4, 23, 3], 1000).with_bottleneck(true)
    }

    /// The `resnet152` prefab: ResNet-152 [3, 8, 36, 3] Bottleneck.
    pub fn resnet152() -> Self {
        ResNetContractConfig::new(vec![3, 8, 36, 3], 1000).with_bottleneck(true)
    }

    /// A `ResNet` of `layers` on a deep or tiered stem of `stem_width`, with
    /// average-pool shortcuts when `avg_down`.
    fn deep(
        layers: Vec<usize>,
        bottleneck: bool,
        stem_type: ResNetStemKind,
        stem_width: usize,
        avg_down: bool,
    ) -> Self {
        ResNetContractConfig::new(layers, 1000)
            .with_bottleneck(bottleneck)
            .with_stem_type(stem_type)
            .with_stem_width(stem_width)
            .with_avg_down(avg_down)
    }

    /// The `resnet10t` prefab: ResNet-10-T [1, 1, 1, 1] basic blocks; a tiered
    /// stem of width 32, average-pool shortcuts.
    pub fn resnet10t() -> Self {
        Self::deep(vec![1, 1, 1, 1], false, ResNetStemKind::Tiered, 32, true)
    }

    /// The `resnet14t` prefab: ResNet-14-T [1, 1, 1, 1] Bottleneck; a tiered
    /// stem of width 32, average-pool shortcuts.
    pub fn resnet14t() -> Self {
        Self::deep(vec![1, 1, 1, 1], true, ResNetStemKind::Tiered, 32, true)
    }

    /// The `resnet18d` prefab: ResNet-18-D [2, 2, 2, 2] basic blocks; a deep
    /// stem of width 32, average-pool shortcuts.
    pub fn resnet18d() -> Self {
        Self::deep(vec![2, 2, 2, 2], false, ResNetStemKind::Deep, 32, true)
    }

    /// The `resnet26d` prefab: ResNet-26-D [2, 2, 2, 2] Bottleneck; a deep stem
    /// of width 32, average-pool shortcuts.
    pub fn resnet26d() -> Self {
        Self::deep(vec![2, 2, 2, 2], true, ResNetStemKind::Deep, 32, true)
    }

    /// The `resnet26t` prefab: ResNet-26-T [2, 2, 2, 2] Bottleneck; a tiered
    /// stem of width 32, average-pool shortcuts.
    pub fn resnet26t() -> Self {
        Self::deep(vec![2, 2, 2, 2], true, ResNetStemKind::Tiered, 32, true)
    }

    /// The `resnet34d` prefab: ResNet-34-D [3, 4, 6, 3] basic blocks; a deep
    /// stem of width 32, average-pool shortcuts.
    pub fn resnet34d() -> Self {
        Self::deep(vec![3, 4, 6, 3], false, ResNetStemKind::Deep, 32, true)
    }

    /// The `resnet50c` prefab: ResNet-50-C [3, 4, 6, 3] Bottleneck; a deep stem
    /// of width 32.
    pub fn resnet50c() -> Self {
        Self::deep(vec![3, 4, 6, 3], true, ResNetStemKind::Deep, 32, false)
    }

    /// The `resnet50d` prefab: ResNet-50-D [3, 4, 6, 3] Bottleneck; a deep stem
    /// of width 32, average-pool shortcuts.
    pub fn resnet50d() -> Self {
        Self::deep(vec![3, 4, 6, 3], true, ResNetStemKind::Deep, 32, true)
    }

    /// The `resnet50s` prefab: ResNet-50-S [3, 4, 6, 3] Bottleneck; a deep stem
    /// of width 64.
    pub fn resnet50s() -> Self {
        Self::deep(vec![3, 4, 6, 3], true, ResNetStemKind::Deep, 64, false)
    }

    /// The `resnet50t` prefab: ResNet-50-T [3, 4, 6, 3] Bottleneck; a tiered
    /// stem of width 32, average-pool shortcuts.
    pub fn resnet50t() -> Self {
        Self::deep(vec![3, 4, 6, 3], true, ResNetStemKind::Tiered, 32, true)
    }

    /// The `resnet101c` prefab: ResNet-101-C [3, 4, 23, 3] Bottleneck; a deep
    /// stem of width 32.
    pub fn resnet101c() -> Self {
        Self::deep(vec![3, 4, 23, 3], true, ResNetStemKind::Deep, 32, false)
    }

    /// The `resnet101d` prefab: ResNet-101-D [3, 4, 23, 3] Bottleneck; a deep
    /// stem of width 32, average-pool shortcuts.
    pub fn resnet101d() -> Self {
        Self::deep(vec![3, 4, 23, 3], true, ResNetStemKind::Deep, 32, true)
    }

    /// The `resnet101s` prefab: ResNet-101-S [3, 4, 23, 3] Bottleneck; a deep
    /// stem of width 64.
    pub fn resnet101s() -> Self {
        Self::deep(vec![3, 4, 23, 3], true, ResNetStemKind::Deep, 64, false)
    }

    /// The `resnet152c` prefab: ResNet-152-C [3, 8, 36, 3] Bottleneck; a deep
    /// stem of width 32.
    pub fn resnet152c() -> Self {
        Self::deep(vec![3, 8, 36, 3], true, ResNetStemKind::Deep, 32, false)
    }

    /// The `resnet152d` prefab: ResNet-152-D [3, 8, 36, 3] Bottleneck; a deep
    /// stem of width 32, average-pool shortcuts.
    pub fn resnet152d() -> Self {
        Self::deep(vec![3, 8, 36, 3], true, ResNetStemKind::Deep, 32, true)
    }

    /// The `resnet152s` prefab: ResNet-152-S [3, 8, 36, 3] Bottleneck; a deep
    /// stem of width 64.
    pub fn resnet152s() -> Self {
        Self::deep(vec![3, 8, 36, 3], true, ResNetStemKind::Deep, 64, false)
    }

    /// The `resnet200d` prefab: ResNet-200-D [3, 24, 36, 3] Bottleneck; a deep
    /// stem of width 32, average-pool shortcuts.
    pub fn resnet200d() -> Self {
        Self::deep(vec![3, 24, 36, 3], true, ResNetStemKind::Deep, 32, true)
    }

    /// The `resnext50d_32x4d` prefab: ResNeXt-50 [3, 4, 6, 3], 32 groups of
    /// 4, on a deep stem of width 32, average-pool shortcuts.
    pub fn resnext50d_32x4d() -> Self {
        Self::resnext50_32x4d()
            .with_stem_type(ResNetStemKind::Deep)
            .with_stem_width(32)
            .with_avg_down(true)
    }

    /// The `ecaresnet26t` prefab: ECA-ResNet-26-T: `resnet26t` with ECA.
    pub fn ecaresnet26t() -> Self {
        Self::resnet26t().with_attention(Some(ConvAttentionKind::Eca))
    }

    /// The `ecaresnet50d` prefab: ECA-ResNet-50-D: `resnet50d` with ECA.
    pub fn ecaresnet50d() -> Self {
        Self::resnet50d().with_attention(Some(ConvAttentionKind::Eca))
    }

    /// The `ecaresnet50t` prefab: ECA-ResNet-50-T: `resnet50t` with ECA.
    pub fn ecaresnet50t() -> Self {
        Self::resnet50t().with_attention(Some(ConvAttentionKind::Eca))
    }

    /// The `ecaresnet200d` prefab: ECA-ResNet-200-D: `resnet200d` with ECA.
    pub fn ecaresnet200d() -> Self {
        Self::resnet200d().with_attention(Some(ConvAttentionKind::Eca))
    }

    /// The `ecaresnetlight` prefab: ECA-ResNet-Light [1, 1, 11, 3] Bottleneck,
    /// average-pool shortcuts, ECA.
    pub fn ecaresnetlight() -> Self {
        ResNetContractConfig::new(vec![1, 1, 11, 3], 1000)
            .with_bottleneck(true)
            .with_avg_down(true)
            .with_attention(Some(ConvAttentionKind::Eca))
    }

    /// The `ecaresnet101d` prefab: ECA-ResNet-101-D: `resnet101d` with ECA.
    pub fn ecaresnet101d() -> Self {
        Self::resnet101d().with_attention(Some(ConvAttentionKind::Eca))
    }

    /// The `ecaresnet269d` prefab: ECA-ResNet-269-D [3, 30, 48, 8] Bottleneck,
    /// deep stem of width 32, average-pool shortcuts, ECA.
    pub fn ecaresnet269d() -> Self {
        Self::deep(vec![3, 30, 48, 8], true, ResNetStemKind::Deep, 32, true)
            .with_attention(Some(ConvAttentionKind::Eca))
    }

    /// The `ecaresnext26t_32x4d` prefab: ECA-ResNeXt-26-T [2, 2, 2, 2], 32
    /// groups of 4, tiered stem of width 32, average-pool shortcuts, ECA.
    pub fn ecaresnext26t_32x4d() -> Self {
        Self::grouped(vec![2, 2, 2, 2], 32, 4)
            .with_stem_type(ResNetStemKind::Tiered)
            .with_stem_width(32)
            .with_avg_down(true)
            .with_attention(Some(ConvAttentionKind::Eca))
    }

    /// The `ecaresnext50t_32x4d` prefab: ECA-ResNeXt-50-T [3, 4, 6, 3], 32
    /// groups of 4, tiered stem of width 32, average-pool shortcuts, ECA.
    pub fn ecaresnext50t_32x4d() -> Self {
        Self::grouped(vec![3, 4, 6, 3], 32, 4)
            .with_stem_type(ResNetStemKind::Tiered)
            .with_stem_width(32)
            .with_avg_down(true)
            .with_attention(Some(ConvAttentionKind::Eca))
    }

    /// The `seresnet18` prefab: SE-ResNet-18: `resnet18` with squeeze-excite.
    pub fn seresnet18() -> Self {
        Self::resnet18().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnet34` prefab: SE-ResNet-34: `resnet34` with squeeze-excite.
    pub fn seresnet34() -> Self {
        Self::resnet34().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnet50` prefab: SE-ResNet-50: `resnet50` with squeeze-excite.
    pub fn seresnet50() -> Self {
        Self::resnet50().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnet50t` prefab: SE-ResNet-50-T: `resnet50t` with
    /// squeeze-excite.
    pub fn seresnet50t() -> Self {
        Self::resnet50t().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnet101` prefab: SE-ResNet-101: `resnet101` with
    /// squeeze-excite.
    pub fn seresnet101() -> Self {
        Self::resnet101().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnet152` prefab: SE-ResNet-152: `resnet152` with
    /// squeeze-excite.
    pub fn seresnet152() -> Self {
        Self::resnet152().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnet152d` prefab: SE-ResNet-152-D: `resnet152d` with
    /// squeeze-excite.
    pub fn seresnet152d() -> Self {
        Self::resnet152d().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnet200d` prefab: SE-ResNet-200-D: `resnet200d` with
    /// squeeze-excite.
    pub fn seresnet200d() -> Self {
        Self::resnet200d().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnet269d` prefab: SE-ResNet-269-D [3, 30, 48, 8] Bottleneck,
    /// deep stem of width 32, average-pool shortcuts, squeeze-excite.
    pub fn seresnet269d() -> Self {
        Self::deep(vec![3, 30, 48, 8], true, ResNetStemKind::Deep, 32, true)
            .with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnext26d_32x4d` prefab: SE-ResNeXt-26-D [2, 2, 2, 2], 32 groups
    /// of 4, deep stem of width 32, average-pool shortcuts, squeeze-excite.
    pub fn seresnext26d_32x4d() -> Self {
        Self::grouped(vec![2, 2, 2, 2], 32, 4)
            .with_stem_type(ResNetStemKind::Deep)
            .with_stem_width(32)
            .with_avg_down(true)
            .with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnext26t_32x4d` prefab: SE-ResNeXt-26-T [2, 2, 2, 2], 32 groups
    /// of 4, tiered stem of width 32, average-pool shortcuts, squeeze-excite.
    pub fn seresnext26t_32x4d() -> Self {
        Self::grouped(vec![2, 2, 2, 2], 32, 4)
            .with_stem_type(ResNetStemKind::Tiered)
            .with_stem_width(32)
            .with_avg_down(true)
            .with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnext50_32x4d` prefab: SE-ResNeXt-50: `resnext50_32x4d` with
    /// squeeze-excite.
    pub fn seresnext50_32x4d() -> Self {
        Self::resnext50_32x4d().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnext101_32x4d` prefab: SE-ResNeXt-101: `resnext101_32x4d` with
    /// squeeze-excite.
    pub fn seresnext101_32x4d() -> Self {
        Self::resnext101_32x4d().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnext101_32x8d` prefab: SE-ResNeXt-101: `resnext101_32x8d` with
    /// squeeze-excite.
    pub fn seresnext101_32x8d() -> Self {
        Self::resnext101_32x8d().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnext101d_32x8d` prefab: SE-ResNeXt-101-D: `resnext101_32x8d`
    /// on a deep stem of width 32, average-pool shortcuts, squeeze-excite.
    pub fn seresnext101d_32x8d() -> Self {
        Self::resnext101_32x8d()
            .with_stem_type(ResNetStemKind::Deep)
            .with_stem_width(32)
            .with_avg_down(true)
            .with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `seresnext101_64x4d` prefab: SE-ResNeXt-101: `resnext101_64x4d` with
    /// squeeze-excite.
    pub fn seresnext101_64x4d() -> Self {
        Self::resnext101_64x4d().with_attention(Some(ConvAttentionKind::SE))
    }

    /// The `resnetblur18` prefab: ResNet-18 with blur anti-aliasing.
    pub fn resnetblur18() -> Self {
        Self::resnet18().with_anti_alias(Some(ConvAaKind::Blur))
    }

    /// The `resnetblur50` prefab: ResNet-50 with blur anti-aliasing.
    pub fn resnetblur50() -> Self {
        Self::resnet50().with_anti_alias(Some(ConvAaKind::Blur))
    }

    /// The `resnetblur50d` prefab: ResNet-50-D with blur anti-aliasing.
    pub fn resnetblur50d() -> Self {
        Self::resnet50d().with_anti_alias(Some(ConvAaKind::Blur))
    }

    /// The `resnetblur101d` prefab: ResNet-101-D with blur anti-aliasing.
    pub fn resnetblur101d() -> Self {
        Self::resnet101d().with_anti_alias(Some(ConvAaKind::Blur))
    }

    /// The `resnetaa34d` prefab: ResNet-34-D with average-pool anti-aliasing.
    pub fn resnetaa34d() -> Self {
        Self::resnet34d().with_anti_alias(Some(ConvAaKind::Avg))
    }

    /// The `resnetaa50` prefab: ResNet-50 with average-pool anti-aliasing.
    pub fn resnetaa50() -> Self {
        Self::resnet50().with_anti_alias(Some(ConvAaKind::Avg))
    }

    /// The `resnetaa50d` prefab: ResNet-50-D with average-pool anti-aliasing.
    pub fn resnetaa50d() -> Self {
        Self::resnet50d().with_anti_alias(Some(ConvAaKind::Avg))
    }

    /// The `resnetaa101d` prefab: ResNet-101-D with average-pool anti-aliasing.
    pub fn resnetaa101d() -> Self {
        Self::resnet101d().with_anti_alias(Some(ConvAaKind::Avg))
    }

    /// The `seresnetaa50d` prefab: SE-ResNet-50-D with average-pool
    /// anti-aliasing.
    pub fn seresnetaa50d() -> Self {
        Self::resnet50d()
            .with_attention(Some(ConvAttentionKind::SE))
            .with_anti_alias(Some(ConvAaKind::Avg))
    }

    /// The `seresnextaa101d_32x8d` prefab: SE-ResNeXt-101-D 32x8d with
    /// average-pool anti-aliasing.
    pub fn seresnextaa101d_32x8d() -> Self {
        Self::seresnext101d_32x8d().with_anti_alias(Some(ConvAaKind::Avg))
    }

    /// The `seresnextaa201d_32x8d` prefab: SE-ResNeXt-201-D [3, 24, 36, 4], 32
    /// groups of 8, on a deep stem of width 64, average-pool shortcuts,
    /// squeeze-excite and average-pool anti-aliasing.
    pub fn seresnextaa201d_32x8d() -> Self {
        Self::grouped(vec![3, 24, 36, 4], 32, 8)
            .with_stem_type(ResNetStemKind::Deep)
            .with_stem_width(64)
            .with_avg_down(true)
            .with_attention(Some(ConvAttentionKind::SE))
            .with_anti_alias(Some(ConvAaKind::Avg))
    }

    /// A `ResNet-RS` of `layers`: bottlenecks on a deep stem of width 32
    /// whose max pool is a strided conv, average-pool shortcuts, and
    /// squeeze-excite of a quarter of the channels.
    fn resnet_rs(layers: Vec<usize>) -> Self {
        Self::deep(layers, true, ResNetStemKind::Deep, 32, true)
            .with_replace_stem_pool(true)
            .with_attention(Some(ConvAttentionKind::Se { rd_ratio: 0.25 }))
    }

    /// The `resnetrs50` prefab: ResNet-RS 50 [3, 4, 6, 3] Bottleneck.
    pub fn resnetrs50() -> Self {
        Self::resnet_rs(vec![3, 4, 6, 3])
    }

    /// The `resnetrs101` prefab: ResNet-RS 101 [3, 4, 23, 3] Bottleneck.
    pub fn resnetrs101() -> Self {
        Self::resnet_rs(vec![3, 4, 23, 3])
    }

    /// The `resnetrs152` prefab: ResNet-RS 152 [3, 8, 36, 3] Bottleneck.
    pub fn resnetrs152() -> Self {
        Self::resnet_rs(vec![3, 8, 36, 3])
    }

    /// The `resnetrs200` prefab: ResNet-RS 200 [3, 24, 36, 3] Bottleneck.
    pub fn resnetrs200() -> Self {
        Self::resnet_rs(vec![3, 24, 36, 3])
    }

    /// The `resnetrs270` prefab: ResNet-RS 270 [4, 29, 53, 4] Bottleneck.
    pub fn resnetrs270() -> Self {
        Self::resnet_rs(vec![4, 29, 53, 4])
    }

    /// The `resnetrs350` prefab: ResNet-RS 350 [4, 36, 72, 4] Bottleneck.
    pub fn resnetrs350() -> Self {
        Self::resnet_rs(vec![4, 36, 72, 4])
    }

    /// The `resnetrs420` prefab: ResNet-RS 420 [4, 44, 87, 4] Bottleneck.
    pub fn resnetrs420() -> Self {
        Self::resnet_rs(vec![4, 44, 87, 4])
    }

    /// A bottleneck `ResNet` of `layers` whose mixers are `cardinality`
    /// groups of `base_width` channels per 64 planes.
    fn grouped(
        layers: Vec<usize>,
        cardinality: usize,
        base_width: usize,
    ) -> Self {
        ResNetContractConfig::new(layers, 1000).with_bottleneck_policy(Some(
            BottleneckPolicyConfig::new()
                .with_cardinality(cardinality)
                .with_base_width(base_width),
        ))
    }

    /// The `resnext50_32x4d` prefab: ResNeXt-50 [3, 4, 6, 3], 32 groups of 4.
    pub fn resnext50_32x4d() -> Self {
        Self::grouped(vec![3, 4, 6, 3], 32, 4)
    }

    /// The `resnext101_32x4d` prefab: ResNeXt-101 [3, 4, 23, 3], 32 groups of
    /// 4.
    pub fn resnext101_32x4d() -> Self {
        Self::grouped(vec![3, 4, 23, 3], 32, 4)
    }

    /// The `resnext101_32x8d` prefab: ResNeXt-101 [3, 4, 23, 3], 32 groups of
    /// 8.
    pub fn resnext101_32x8d() -> Self {
        Self::grouped(vec![3, 4, 23, 3], 32, 8)
    }

    /// The `resnext101_32x16d` prefab: ResNeXt-101 [3, 4, 23, 3], 32 groups
    /// of 16.
    pub fn resnext101_32x16d() -> Self {
        Self::grouped(vec![3, 4, 23, 3], 32, 16)
    }

    /// The `resnext101_32x32d` prefab: ResNeXt-101 [3, 4, 23, 3], 32 groups
    /// of 32.
    pub fn resnext101_32x32d() -> Self {
        Self::grouped(vec![3, 4, 23, 3], 32, 32)
    }

    /// The `resnext101_64x4d` prefab: ResNeXt-101 [3, 4, 23, 3], 64 groups of
    /// 4.
    pub fn resnext101_64x4d() -> Self {
        Self::grouped(vec![3, 4, 23, 3], 64, 4)
    }

    /// The `wide_resnet50_2` prefab: Wide ResNet-50-2 [3, 4, 6, 3], its
    /// mixers twice a `resnet50`'s width.
    pub fn wide_resnet50_2() -> Self {
        Self::grouped(vec![3, 4, 6, 3], 1, 128)
    }

    /// The `wide_resnet101_2` prefab: Wide ResNet-101-2 [3, 4, 23, 3], its
    /// mixers twice a `resnet101`'s width.
    pub fn wide_resnet101_2() -> Self {
        Self::grouped(vec![3, 4, 23, 3], 1, 128)
    }
}

/// The `ResNet` geometries by name, timm's architecture names: 76
/// [`ResNetContractConfig`]s, each with 1000 classes: `resnet18` to
/// `resnet152` and `resnet50_gn`; the deep- and tiered-stem C, D, S and T
/// variants (`resnet50d`, `resnet10t`, …, `resnext50d_32x4d`); the
/// `ResNeXt`s, `resnext50_32x4d` to `resnext101_64x4d`; the wide
/// `wide_resnet50_2` and `wide_resnet101_2`; the squeeze-excite and ECA
/// nets (`seresnet50`, `ecaresnet50d`, …); and the anti-aliased nets
/// (`resnetaa50d`, `resnetblur50`, `seresnextaa101d_32x8d`, …); and the
/// `ResNet-RS`s, `resnetrs50` to `resnetrs420`.
///
/// A prefab has no weights. The catalog rows state the geometry of the
/// prefab they instantiate, and
/// [`ResNetConstructConfig`](super::ResNetConstructConfig) builds from it.
impl Prefabs for ResNetContractConfig {
    const PREFABS: PrefabTable<Self> = &[
        ("resnet18", Self::resnet18),
        ("resnet26", Self::resnet26),
        ("resnet34", Self::resnet34),
        ("resnet50", Self::resnet50),
        ("resnet50_gn", Self::resnet50_gn),
        ("resnet101", Self::resnet101),
        ("resnet152", Self::resnet152),
        ("resnet10t", Self::resnet10t),
        ("resnet14t", Self::resnet14t),
        ("resnet18d", Self::resnet18d),
        ("resnet26d", Self::resnet26d),
        ("resnet26t", Self::resnet26t),
        ("resnet34d", Self::resnet34d),
        ("resnet50c", Self::resnet50c),
        ("resnet50d", Self::resnet50d),
        ("resnet50s", Self::resnet50s),
        ("resnet50t", Self::resnet50t),
        ("resnet101c", Self::resnet101c),
        ("resnet101d", Self::resnet101d),
        ("resnet101s", Self::resnet101s),
        ("resnet152c", Self::resnet152c),
        ("resnet152d", Self::resnet152d),
        ("resnet152s", Self::resnet152s),
        ("resnet200d", Self::resnet200d),
        ("resnext50d_32x4d", Self::resnext50d_32x4d),
        ("resnext50_32x4d", Self::resnext50_32x4d),
        ("resnext101_32x4d", Self::resnext101_32x4d),
        ("resnext101_32x8d", Self::resnext101_32x8d),
        ("resnext101_32x16d", Self::resnext101_32x16d),
        ("resnext101_32x32d", Self::resnext101_32x32d),
        ("resnext101_64x4d", Self::resnext101_64x4d),
        ("wide_resnet50_2", Self::wide_resnet50_2),
        ("wide_resnet101_2", Self::wide_resnet101_2),
        ("ecaresnet26t", Self::ecaresnet26t),
        ("ecaresnet50d", Self::ecaresnet50d),
        ("ecaresnet50t", Self::ecaresnet50t),
        ("ecaresnet200d", Self::ecaresnet200d),
        ("ecaresnetlight", Self::ecaresnetlight),
        ("ecaresnet101d", Self::ecaresnet101d),
        ("ecaresnet269d", Self::ecaresnet269d),
        ("ecaresnext26t_32x4d", Self::ecaresnext26t_32x4d),
        ("ecaresnext50t_32x4d", Self::ecaresnext50t_32x4d),
        ("seresnet18", Self::seresnet18),
        ("seresnet34", Self::seresnet34),
        ("seresnet50", Self::seresnet50),
        ("seresnet50t", Self::seresnet50t),
        ("seresnet101", Self::seresnet101),
        ("seresnet152", Self::seresnet152),
        ("seresnet152d", Self::seresnet152d),
        ("seresnet200d", Self::seresnet200d),
        ("seresnet269d", Self::seresnet269d),
        ("seresnext26d_32x4d", Self::seresnext26d_32x4d),
        ("seresnext26t_32x4d", Self::seresnext26t_32x4d),
        ("seresnext50_32x4d", Self::seresnext50_32x4d),
        ("seresnext101_32x4d", Self::seresnext101_32x4d),
        ("seresnext101_32x8d", Self::seresnext101_32x8d),
        ("seresnext101d_32x8d", Self::seresnext101d_32x8d),
        ("seresnext101_64x4d", Self::seresnext101_64x4d),
        ("resnetblur18", Self::resnetblur18),
        ("resnetblur50", Self::resnetblur50),
        ("resnetblur50d", Self::resnetblur50d),
        ("resnetblur101d", Self::resnetblur101d),
        ("resnetaa34d", Self::resnetaa34d),
        ("resnetaa50", Self::resnetaa50),
        ("resnetaa50d", Self::resnetaa50d),
        ("resnetaa101d", Self::resnetaa101d),
        ("seresnetaa50d", Self::seresnetaa50d),
        ("seresnextaa101d_32x8d", Self::seresnextaa101d_32x8d),
        ("seresnextaa201d_32x8d", Self::seresnextaa201d_32x8d),
        ("resnetrs50", Self::resnetrs50),
        ("resnetrs101", Self::resnetrs101),
        ("resnetrs152", Self::resnetrs152),
        ("resnetrs200", Self::resnetrs200),
        ("resnetrs270", Self::resnetrs270),
        ("resnetrs350", Self::resnetrs350),
        ("resnetrs420", Self::resnetrs420),
    ];
    const PREFAB_TABLE: &'static str = "resnet";
}
