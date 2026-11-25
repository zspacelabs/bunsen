//! # `ResNet` prefabs

use alloc::{
    vec,
    vec::Vec,
};

use burn::nn::GroupNormConfig;

use crate::{
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

/// The `ResNet` geometries by name, timm's architecture names: 33
/// [`ResNetContractConfig`]s, each with 1000 classes: `resnet18` to
/// `resnet152` and `resnet50_gn`; the deep- and tiered-stem C, D, S and T
/// variants (`resnet50d`, `resnet10t`, …, `resnext50d_32x4d`); the
/// `ResNeXt`s, `resnext50_32x4d` to `resnext101_64x4d`; and the wide
/// `wide_resnet50_2` and `wide_resnet101_2`.
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
    ];
    const PREFAB_TABLE: &'static str = "resnet";
}
