//! # `ResNet` prefabs

use alloc::vec;

use burn::nn::GroupNormConfig;

use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::images::classification::resnet::{
        BottleneckPolicyConfig,
        ResNetContractConfig,
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

/// The `ResNet` geometries by name, timm's architecture names: fifteen
/// [`ResNetContractConfig`]s, each with 1000 classes: `resnet18` to
/// `resnet152` and `resnet50_gn`; the `ResNeXt`s, `resnext50_32x4d` to
/// `resnext101_64x4d`; and the wide `wide_resnet50_2` and
/// `wide_resnet101_2`.
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
