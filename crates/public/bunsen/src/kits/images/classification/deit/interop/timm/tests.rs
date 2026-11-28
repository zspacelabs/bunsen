use super::*;
use crate::{
    data::pretrained::Prefabs,
    errors::testing::ErrorMatcher,
    kits::images::{
        classification::deit::DeiTContractConfig,
        support::interop::timm::{
            TimmModelConfigMirror,
            timm_model_config_from_str,
        },
    },
    support::testing::assert_config_eq,
};

fn config(
    architecture: &str,
    size: usize,
) -> TimmModelConfigMirror {
    timm_model_config_from_str(&format!(
        r#"{{"architecture": "{architecture}", "num_classes": 1000,
            "num_features": 768, "global_pool": "token", "pretrained_cfg": {{
            "input_size": [3, {size}, {size}], "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]}}}}"#
    ))
    .unwrap()
}

/// Each prefab is a timm architecture of its name.
#[test]
fn test_prefab_architectures_map() {
    for name in DeiTContractConfig::prefab_names() {
        let want = DeiTContractConfig::expect_prefab(name);
        assert_config_eq(
            name,
            &deit_contract_config(&config(name, want.vit.image_size)).unwrap(),
            &want,
        );
    }
    ErrorMatcher::unsupported()
        .message_contains("timm's vit_base_patch16_224: the deit kit builds")
        .assert_err(&deit_contract_config(&config("vit_base_patch16_224", 224)));
}
