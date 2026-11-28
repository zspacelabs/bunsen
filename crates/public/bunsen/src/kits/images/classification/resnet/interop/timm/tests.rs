use super::*;
use crate::{
    blocks::images::models::conv_net::GlobalPool2dConfig,
    data::pretrained::Prefabs,
    errors::testing::ErrorMatcher,
    kits::images::{
        classification::resnet::ResNetContractConfig,
        support::interop::timm::{
            TimmModelConfigMirror,
            timm_model_config_from_str,
        },
    },
    support::testing::assert_config_eq,
};

/// A timm config of `architecture` with `num_classes` classes.
fn config(
    architecture: &str,
    num_classes: usize,
) -> TimmModelConfigMirror {
    timm_model_config_from_str(&format!(
        r#"{{"architecture": "{architecture}", "num_classes": {num_classes},
            "num_features": 2048, "pretrained_cfg": {{"input_size": [3, 224, 224],
            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}}}}"#
    ))
    .unwrap()
}

/// Each prefab is a timm architecture of its name; the class count is
/// the config's.
#[test]
fn test_prefab_architectures_map() {
    for name in ResNetContractConfig::prefab_names() {
        let mut want = ResNetContractConfig::expect_prefab(name);
        want.num_classes = 11821;
        assert_config_eq(
            name,
            &resnet_contract_config(&config(name, 11821)).unwrap(),
            &want,
        );
    }
}

#[test]
fn test_unbuilt_architectures_are_unsupported() {
    let unsupported = || ErrorMatcher::unsupported();
    unsupported()
        .message_contains("timm's resnest50d")
        .assert_err(&resnet_contract_config(&config("resnest50d", 1000)));
    // A headless repo builds a headless model: its pooled features.
    assert_eq!(
        resnet_contract_config(&config("resnet50", 0))
            .unwrap()
            .num_classes,
        0
    );
    let mut unpooled = config("resnet50", 1000);
    unpooled.global_pool = Some("".into());
    unsupported()
        .message_contains("\"\" pool")
        .assert_err(&resnet_contract_config(&unpooled));
}

/// timm's pools are the head's.
#[test]
fn test_pools_map() {
    for (name, pool) in [
        ("avg", GlobalPool2dConfig::Avg),
        ("max", GlobalPool2dConfig::Max),
        ("avgmax", GlobalPool2dConfig::AvgMax),
        ("catavgmax", GlobalPool2dConfig::CatAvgMax),
    ] {
        let mut pooled = config("resnet50", 1000);
        pooled.global_pool = Some(name.into());
        assert_eq!(
            resnet_contract_config(&pooled).unwrap().global_pool,
            pool,
            "{name}"
        );
    }
}
