use super::*;
use crate::{
    burner::module::testing::assert_rekeys,
    data::pretrained::Prefabs,
    errors::testing::ErrorMatcher,
    kits::images::{
        classification::{
            resnet::interop::torchvision::ResNetStateLayout,
            sknet::SKNetContractConfig,
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
    for name in SKNetContractConfig::prefab_names() {
        let mut want = SKNetContractConfig::expect_prefab(name);
        want.resnet.num_classes = 10;
        assert_config_eq(
            name,
            &sknet_contract_config(&config(name, 10)).unwrap(),
            &want,
        );
    }
    ErrorMatcher::unsupported()
        .message_contains("timm's resnet50")
        .assert_err(&sknet_contract_config(&config("resnet50", 1000)));
}

/// A basic block's selective kernel and its `ConvNormAct`, and a
/// bottleneck's, land on the path's convs; the rest on `ResNet`'s paths.
#[test]
fn test_sknet_rekey() {
    let rekey = sknet_rekey(ResNetStateLayout::default());
    assert_rekeys(
        &rekey,
        [
            ("conv1.weight", Some("stem.convs.0.conv.weight")),
            ("bn1.bias", Some("stem.convs.0.norm.bias")),
            (
                "layer1.0.conv1.paths.0.conv.weight",
                Some("stages.0.units.0.convs.0.paths.0.conv.weight"),
            ),
            (
                "layer1.0.conv1.attn.fc_reduce.weight",
                Some("stages.0.units.0.convs.0.reduce.conv.weight"),
            ),
            (
                "layer1.0.conv1.attn.bn.bias",
                Some("stages.0.units.0.convs.0.reduce.norm.bias"),
            ),
            (
                "layer4.1.conv2.attn.fc_select.weight",
                Some("stages.3.units.1.convs.1.select.weight"),
            ),
            (
                "layer3.0.conv3.conv.weight",
                Some("stages.2.units.0.convs.2.conv.weight"),
            ),
            (
                "layer2.0.downsample.1.running_mean",
                Some("stages.1.units.0.shortcut.norm.running_mean"),
            ),
            ("fc.weight", Some("head.fc.weight")),
            ("layer1.0.conv1.paths.1.bn.num_batches_tracked", None),
        ],
    );
}
