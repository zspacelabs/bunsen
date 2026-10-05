//! # `ResNet` prefabs

use alloc::vec;

use crate::{
    data::pretrained::{
        StaticPreFabConfig,
        StaticPreFabMap,
    },
    kits::images::resnet::ResNetContractConfig,
};

/// The `ResNet` geometries by name: six [`ResNetContractConfig`]s,
/// `resnet18` to `resnet152`, each with 1000 classes.
///
/// A prefab has no weights. The rows of
/// [`WELL_KNOWN_TABLE`](super::WELL_KNOWN_TABLE) name the prefab they
/// instantiate, and [`ResNetConstruct`](super::ResNetConstruct) builds from
/// it.
pub static RESNET_PREFABS: StaticPreFabMap<ResNetContractConfig> = StaticPreFabMap {
    name: "resnet",
    description: "Well-Know ResNet configs",

    items: &[
        &StaticPreFabConfig {
            name: "resnet18",
            description: "ResNet-18 [2, 2, 2, 2] BasicBlocks",
            builder: || ResNetContractConfig::new(vec![2, 2, 2, 2], 1000),
        },
        &StaticPreFabConfig {
            name: "resnet26",
            description: "ResNet-26 [2, 2, 2, 2] Bottleneck",
            builder: || ResNetContractConfig::new(vec![2, 2, 2, 2], 1000).with_bottleneck(true),
        },
        &StaticPreFabConfig {
            name: "resnet34",
            description: "ResNet-34 [3, 4, 6, 3] BasicBlocks",
            builder: || ResNetContractConfig::new(vec![3, 4, 6, 3], 1000),
        },
        &StaticPreFabConfig {
            name: "resnet50",
            description: "ResNet-50 [3, 4, 6, 3] Bottleneck",
            builder: || ResNetContractConfig::new(vec![3, 4, 6, 3], 1000).with_bottleneck(true),
        },
        &StaticPreFabConfig {
            name: "resnet101",
            description: "ResNet-101 [3, 4, 23, 3] Bottleneck",
            builder: || ResNetContractConfig::new(vec![3, 4, 23, 3], 1000).with_bottleneck(true),
        },
        &StaticPreFabConfig {
            name: "resnet152",
            description: "ResNet-152 [3, 8, 36, 3] Bottleneck",
            builder: || ResNetContractConfig::new(vec![3, 8, 36, 3], 1000).with_bottleneck(true),
        },
    ],
};
