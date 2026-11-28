use super::*;
use crate::{
    burner::module::testing::assert_rekeys,
    data::pretrained::Prefabs,
    errors::testing::ErrorMatcher,
    kits::images::{
        classification::resnetv2::ResNetV2ContractConfig,
        support::timm::interop::{
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
    for name in ResNetV2ContractConfig::prefab_names() {
        let mut want = ResNetV2ContractConfig::expect_prefab(name);
        want.num_classes = 21843;
        assert_config_eq(
            name,
            &resnetv2_contract_config(&config(name, 21843)).unwrap(),
            &want,
        );
    }
    ErrorMatcher::unsupported()
        .message_contains("timm's resnet50")
        .assert_err(&resnetv2_contract_config(&config("resnet50", 1000)));
}

/// timm's names land on the conv net's paths.
#[test]
fn test_resnetv2_rekey() {
    let rekey = resnetv2_rekey(1000, 2048, true);
    assert_rekeys(
        &rekey,
        [
            ("stem.conv.weight", "stem.convs.0.conv.weight"),
            ("stem.conv1.weight", "stem.convs.0.conv.weight"),
            ("stem.norm2.bias", "stem.convs.1.norm.bias"),
            ("stem.conv3.weight", "stem.convs.2.conv.weight"),
            (
                "stages.0.blocks.0.norm1.weight",
                "stages.0.units.0.pre.norm.weight",
            ),
            (
                "stages.0.blocks.0.conv1.weight",
                "stages.0.units.0.convs.0.conv.weight",
            ),
            (
                "stages.3.blocks.2.norm3.bias",
                "stages.3.units.2.convs.2.norm.bias",
            ),
            (
                "stages.1.blocks.0.downsample.conv.weight",
                "stages.1.units.0.shortcut.conv.weight",
            ),
            ("norm.weight", "final_conv.norm.weight"),
            ("head.fc.weight", "head.fc.weight"),
        ],
    );
}

/// The hybrid `ViT`'s backbone's names: a normed stem, `Bottleneck`s
/// whose `norm1` is their first conv's, normed shortcuts; a stem alone's
/// bare `conv` and `norm`.
#[test]
fn test_non_preact_rekey() {
    let rekey = resnetv2_rekey(0, 1024, false);
    assert_rekeys(
        &rekey,
        [
            ("stem.conv.weight", "stem.convs.0.conv.weight"),
            ("stem.norm.weight", "stem.convs.0.norm.weight"),
            (
                "stages.0.blocks.0.norm1.weight",
                "stages.0.units.0.convs.0.norm.weight",
            ),
            (
                "stages.2.blocks.8.conv3.weight",
                "stages.2.units.8.convs.2.conv.weight",
            ),
            (
                "stages.1.blocks.0.downsample.norm.bias",
                "stages.1.units.0.shortcut.norm.bias",
            ),
            ("conv.weight", "stem.convs.0.conv.weight"),
            ("norm.bias", "stem.convs.0.norm.bias"),
        ],
    );
}
