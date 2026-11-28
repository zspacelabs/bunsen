use super::*;
use crate::{
    burner::module::testing::assert_rekeys,
    data::pretrained::Prefabs,
    errors::testing::ErrorMatcher,
    kits::images::{
        classification::regnet::RegNetContractConfig,
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
            "num_features": 368, "pretrained_cfg": {{"input_size": [3, 224, 224],
            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}}}}"#
    ))
    .unwrap()
}

/// Each prefab is a timm architecture of its name; the class count is
/// the config's, 0 for a trunk.
#[test]
fn test_prefab_architectures_map() {
    for name in RegNetContractConfig::prefab_names() {
        for classes in [0, 11821] {
            let mut want = RegNetContractConfig::expect_prefab(name);
            want.num_classes = classes;
            assert_config_eq(
                name,
                &regnet_contract_config(&config(name, classes)).unwrap(),
                &want,
            );
        }
    }
    let unsupported = || ErrorMatcher::unsupported();
    unsupported()
        .message_contains("timm's resnet50")
        .assert_err(&regnet_contract_config(&config("resnet50", 1000)));
    let mut unpooled = config("regnetx_002", 1000);
    unpooled.global_pool = Some("".into());
    unsupported()
        .message_contains("\"\" pool")
        .assert_err(&regnet_contract_config(&unpooled));
}

/// timm's names land on the conv net's paths.
#[test]
fn test_regnet_rekey() {
    let rekey = regnet_rekey();
    assert_rekeys(
        &rekey,
        [
            ("stem.conv.weight", Some("stem.convs.0.conv.weight")),
            ("stem.bn.bias", Some("stem.convs.0.norm.bias")),
            (
                "s1.b1.conv1.conv.weight",
                Some("stages.0.units.0.convs.0.conv.weight"),
            ),
            (
                "s4.b7.conv3.bn.weight",
                Some("stages.3.units.6.convs.2.norm.weight"),
            ),
            ("s2.b1.se.fc1.bias", Some("stages.1.units.0.attn.fc1.bias")),
            (
                "s1.b1.downsample.conv.weight",
                Some("stages.0.units.0.shortcut.conv.weight"),
            ),
            (
                "s1.b1.downsample.bn.running_mean",
                Some("stages.0.units.0.shortcut.norm.running_mean"),
            ),
            ("head.fc.weight", Some("head.fc.weight")),
            ("s1.b1.conv1.bn.num_batches_tracked", None),
        ],
    );
    // `RegNetV`'s pre-activation names, and `RegNetZ`'s final conv.
    assert_rekeys(
        &rekey,
        [
            ("stem.weight", "stem.convs.0.conv.weight"),
            ("s1.b1.norm1.weight", "stages.0.units.0.pre.norm.weight"),
            ("s1.b1.conv1.weight", "stages.0.units.0.convs.0.conv.weight"),
            (
                "s1.b1.norm2.running_var",
                "stages.0.units.0.convs.1.norm.running_var",
            ),
            ("s1.b1.norm3.bias", "stages.0.units.0.convs.2.norm.bias"),
            ("s3.b2.conv3.weight", "stages.2.units.1.convs.2.conv.weight"),
            (
                "s1.b1.downsample.weight",
                "stages.0.units.0.shortcut.conv.weight",
            ),
            (
                "s1.b1.downsample.1.weight",
                "stages.0.units.0.shortcut.conv.weight",
            ),
            ("final_conv.conv.weight", "final_conv.conv.weight"),
            ("final_conv.bn.weight", "final_conv.norm.weight"),
        ],
    );
}
