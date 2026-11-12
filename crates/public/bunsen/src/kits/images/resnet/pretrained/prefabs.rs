//! # `ResNet` prefabs

use alloc::vec;

use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::images::resnet::ResNetContractConfig,
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

    /// The `resnet101` prefab: ResNet-101 [3, 4, 23, 3] Bottleneck.
    pub fn resnet101() -> Self {
        ResNetContractConfig::new(vec![3, 4, 23, 3], 1000).with_bottleneck(true)
    }

    /// The `resnet152` prefab: ResNet-152 [3, 8, 36, 3] Bottleneck.
    pub fn resnet152() -> Self {
        ResNetContractConfig::new(vec![3, 8, 36, 3], 1000).with_bottleneck(true)
    }
}

/// The `ResNet` geometries by name: six [`ResNetContractConfig`]s,
/// `resnet18` to `resnet152`, each with 1000 classes.
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
        ("resnet101", Self::resnet101),
        ("resnet152", Self::resnet152),
    ];
    const PREFAB_TABLE: &'static str = "resnet";
}
