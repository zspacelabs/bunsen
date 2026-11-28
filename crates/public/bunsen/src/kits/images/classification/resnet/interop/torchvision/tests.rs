use super::*;
use crate::burner::module::testing::assert_rekeys;

/// torchvision's names onto bunsen's, one of each kind: the stem, a
/// basic block's and a bottleneck's convolutions and norms, a
/// downsample, the head; the counters dropped.
#[test]
fn test_resnet_rekey() {
    let rekey = resnet_rekey(ResNetStateLayout::default());
    assert_rekeys(
        &rekey,
        [
            ("conv1.weight", Some("stem.convs.0.conv.weight")),
            ("bn1.running_mean", Some("stem.convs.0.norm.running_mean")),
            ("bn1.num_batches_tracked", None),
            (
                "layer1.0.conv1.weight",
                Some("stages.0.units.0.convs.0.conv.weight"),
            ),
            (
                "layer1.1.bn2.bias",
                Some("stages.0.units.1.convs.1.norm.bias"),
            ),
            (
                "layer4.2.conv3.weight",
                Some("stages.3.units.2.convs.2.conv.weight"),
            ),
            (
                "layer2.0.downsample.0.weight",
                Some("stages.1.units.0.shortcut.conv.weight"),
            ),
            (
                "layer2.0.downsample.1.running_var",
                Some("stages.1.units.0.shortcut.norm.running_var"),
            ),
            ("layer3.5.bn3.num_batches_tracked", None),
            ("fc.weight", Some("head.fc.weight")),
        ],
    );
}

/// timm's deep stem's `Sequential` indices and its avg-pool shortcuts'
/// shifted ones, onto the same paths.
#[test]
fn test_deep_stem_and_avg_down_rekey() {
    let rekey = resnet_rekey(ResNetStateLayout {
        stem_convs: 3,
        avg_down: true,
    });
    assert_rekeys(
        &rekey,
        [
            ("conv1.0.weight", Some("stem.convs.0.conv.weight")),
            ("conv1.1.bias", Some("stem.convs.0.norm.bias")),
            ("conv1.3.weight", Some("stem.convs.1.conv.weight")),
            (
                "conv1.4.running_mean",
                Some("stem.convs.1.norm.running_mean"),
            ),
            ("conv1.6.weight", Some("stem.convs.2.conv.weight")),
            ("bn1.weight", Some("stem.convs.2.norm.weight")),
            ("maxpool.0.weight", Some("stem.pool.conv.weight")),
            ("maxpool.1.running_var", Some("stem.pool.norm.running_var")),
            (
                "layer2.0.downsample.1.weight",
                Some("stages.1.units.0.shortcut.conv.weight"),
            ),
            (
                "layer2.0.downsample.2.running_var",
                Some("stages.1.units.0.shortcut.norm.running_var"),
            ),
            (
                "layer3.1.se.fc1.bias",
                Some("stages.2.units.1.attn.fc1.bias"),
            ),
            (
                "layer4.0.se.conv.weight",
                Some("stages.3.units.0.attn.conv.weight"),
            ),
        ],
    );
}
