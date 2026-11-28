use super::*;
use crate::{
    data::pretrained::Prefabs,
    errors::testing::ErrorMatcher,
    kits::images::{
        classification::swin::v1::SwinV1ContractConfig,
        support::timm::interop::{
            TimmModelConfigMirror,
            timm_model_config_from_str,
        },
    },
    support::testing::assert_config_eq,
};

fn config(
    architecture: &str,
    num_classes: usize,
    size: usize,
    pool: &str,
) -> TimmModelConfigMirror {
    timm_model_config_from_str(&format!(
        r#"{{"architecture": "{architecture}", "num_classes": {num_classes},
            "num_features": 768, "global_pool": "{pool}", "pretrained_cfg": {{
            "input_size": [3, {size}, {size}], "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]}}}}"#
    ))
    .unwrap()
}

/// Each prefab is a timm architecture of its name; the classes are the
/// config's.
#[test]
fn test_prefab_architectures_map() {
    for name in SwinV1ContractConfig::prefab_names() {
        let mut want = SwinV1ContractConfig::expect_prefab(name);
        want.num_classes = 21841;
        assert_config_eq(
            name,
            &swin_v1_contract_config(&config(name, 21841, want.image_size, "avg")).unwrap(),
            &want,
        );
    }
    let unsupported = || ErrorMatcher::unsupported();
    unsupported()
        .message_contains("timm's swinv2_tiny_window8_256")
        .assert_err(&swin_v1_contract_config(&config(
            "swinv2_tiny_window8_256",
            1000,
            256,
            "avg",
        )));
    unsupported()
        .message_contains("\"token\" pool")
        .assert_err(&swin_v1_contract_config(&config(
            "swin_tiny_patch4_window7_224",
            1000,
            224,
            "token",
        )));
}
