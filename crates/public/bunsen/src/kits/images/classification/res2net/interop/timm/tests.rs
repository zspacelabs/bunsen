use super::*;
use crate::{
    burner::module::testing::assert_rekeys,
    data::pretrained::Prefabs,
    errors::testing::ErrorMatcher,
    kits::images::{
        classification::{
            res2net::Res2NetContractConfig,
            resnet::interop::torchvision::ResNetStateLayout,
        },
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
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}}}"#
    ))
    .unwrap()
}

/// Each prefab is a timm architecture of its name; the class count is
/// the config's.
#[test]
fn test_prefab_architectures_map() {
    for name in Res2NetContractConfig::prefab_names() {
        let mut want = Res2NetContractConfig::expect_prefab(name);
        want.resnet.num_classes = 10;
        assert_config_eq(
            name,
            &res2net_contract_config(&config(name, 10)).unwrap(),
            &want,
        );
    }
    ErrorMatcher::unsupported()
        .message_contains("timm's resnet50")
        .assert_err(&res2net_contract_config(&config("resnet50", 1000)));
}

/// The branches land on the mixer's; the rest on `ResNet`'s paths, a
/// deep stem's and `avg_down` shortcuts' too.
#[test]
fn test_res2net_rekey() {
    let rekey = res2net_rekey(ResNetStateLayout {
        stem_convs: 3,
        avg_down: true,
    });
    assert_rekeys(
        &rekey,
        [
            ("conv1.0.weight", Some("stem.convs.0.conv.weight")),
            ("bn1.bias", Some("stem.convs.2.norm.bias")),
            (
                "layer1.0.convs.0.weight",
                Some("stages.0.units.0.convs.1.convs.0.conv.weight"),
            ),
            (
                "layer4.2.bns.6.bias",
                Some("stages.3.units.2.convs.1.convs.6.norm.bias"),
            ),
            (
                "layer1.0.conv1.weight",
                Some("stages.0.units.0.convs.0.conv.weight"),
            ),
            (
                "layer3.0.downsample.2.weight",
                Some("stages.2.units.0.shortcut.norm.weight"),
            ),
            ("fc.weight", Some("head.fc.weight")),
            ("layer1.0.bns.0.num_batches_tracked", None),
        ],
    );
}
