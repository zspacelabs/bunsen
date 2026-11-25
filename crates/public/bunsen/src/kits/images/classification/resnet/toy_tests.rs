//! The toy models against timm's own `ResNet`.
//!
//! `testdata/resnet/` holds nine tiny seeded timm `ResNet`s and their
//! answers, computed in float64 (`tools/gen_resnet_toy_fixtures.py`, timm's
//! `timm.models.resnet.ResNet`), each four stages of one block: on its
//! 64-wide 7x7 stem, one of basic blocks and one of bottlenecks at timm's
//! `channels=(8, 16, 32, 64)`, one of bottlenecks at 32 planes with timm's
//! group norm, and one of grouped bottlenecks (`ResNeXt`'s); then, on odd
//! stage sizes, a deep stem and a tiered one with average-pool shortcuts, and
//! the tiered one dilated past stride 8; and bottlenecks with timm's
//! squeeze-excite and with its ECA.
//! Each is the [`ResNetContractConfig`] of the same knobs. The weights, under
//! timm's names, load through [`load_resnet_tensors`]; each stage output, the
//! pooled features and the logits are checked against timm's.

use std::path::PathBuf;

use burn::{
    nn::GroupNormConfig,
    prelude::*,
};

use super::{
    BottleneckPolicyConfig,
    ResNetContractConfig,
    ResNetStemKind,
    interop::torchvision::load_resnet_tensors,
};
use crate::{
    blocks::images::models::conv_net::{
        ConvAttentionKind,
        ConvNetModel,
    },
    burner::module::ModuleInit,
    kits::images::{
        classification::driver::ImageClassifier,
        support::encoder::ImageEncoder,
    },
    support::testing::{
        ToyFixture,
        assert_io_close,
        cpu_device,
    },
};

/// The toys' geometry: four one-unit stages of `channels` planes, 10
/// classes, on the default 64-wide `7x7` stem.
fn toy_policy(
    bottleneck: bool,
    channels: [usize; 4],
) -> ResNetContractConfig {
    ResNetContractConfig::new(vec![1; 4], 10)
        .with_channels(channels.to_vec())
        .with_bottleneck(bottleneck)
}

/// The toy `name` of `policy`, its weights loaded, and its fixture's
/// answers.
fn load_toy(
    name: &str,
    policy: ResNetContractConfig,
    device: &Device,
) -> (ConvNetModel, ToyFixture) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/resnet")
        .join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, device);
    let mut model: ConvNetModel = policy.init(device);
    let applied = load_resnet_tensors(&mut model, fixture.take_weights()).expect("load the toy");
    // Only the batch norms' counters are left: the rekey drops them.
    assert!(
        applied
            .unused
            .iter()
            .all(|n| n.ends_with(".num_batches_tracked")),
        "{name}: {:?}",
        applied.unused
    );
    (model, fixture)
}

/// Every stage, the pooled features and the logits are timm's.
fn check_toy(
    name: &str,
    policy: ResNetContractConfig,
) {
    let device = cpu_device();
    let (model, fixture) = load_toy(name, policy, &device);
    let images: Tensor<4> = fixture.io("images");
    let stages = model.forward_intermediates(images.clone());
    assert_eq!(stages.len(), 5, "{name}: the stem and four stages");
    for (k, stage) in stages.into_iter().enumerate() {
        assert_io_close(
            &format!("{name} stage_{k}"),
            stage,
            fixture.io(&format!("stage_{k}")),
        );
    }
    assert_io_close(
        &format!("{name} pooled"),
        model.forward_pooled(images.clone()),
        fixture.io("pooled"),
    );
    assert_io_close(
        &format!("{name} logits"),
        model.forward_logits(images),
        fixture.io("logits"),
    );
}

#[test]
fn test_basic_toy_is_timms() {
    check_toy("toy_basic", toy_policy(false, [8, 16, 32, 64]));
}

#[test]
fn test_bottleneck_toy_is_timms() {
    check_toy("toy_bottleneck", toy_policy(true, [8, 16, 32, 64]));
}

/// timm's `groupnorm` (32 groups) in every norm, the stem's included, as
/// `resnet50_gn` has.
#[test]
fn test_group_norm_toy_is_timms() {
    check_toy(
        "toy_gn",
        toy_policy(true, [32; 4]).with_normalization(GroupNormConfig::new(32, 0).into()),
    );
}

/// Grouped mixers: 4 groups of `base_width` 16 per 64 planes.
#[test]
fn test_resnext_toy_is_timms() {
    check_toy(
        "toy_resnext",
        toy_policy(true, [8, 16, 32, 64]).with_bottleneck_policy(Some(
            BottleneckPolicyConfig::new()
                .with_cardinality(4)
                .with_base_width(16),
        )),
    );
}

/// A deep stem (16, 16, 32) and average-pool shortcuts, on odd sizes.
#[test]
fn test_deep_stem_toy_is_timms() {
    check_toy(
        "toy_deep",
        toy_policy(false, [8, 16, 32, 64])
            .with_stem_type(ResNetStemKind::Deep)
            .with_stem_width(16)
            .with_avg_down(true),
    );
}

/// A tiered stem (12, 16, 32) and average-pool shortcuts, on odd sizes.
#[test]
fn test_tiered_stem_toy_is_timms() {
    check_toy("toy_tiered", tiered());
}

/// The tiered toy at output stride 8: the dilated stages' shortcuts pool
/// at stride 1 over a padded input (timm's `AvgPool2dSame`).
#[test]
fn test_dilated_toy_is_timms() {
    check_toy("toy_dilated", tiered().with_output_stride(8));
}

/// The tiered toys' policy.
fn tiered() -> ResNetContractConfig {
    toy_policy(true, [8, 16, 32, 64])
        .with_stem_type(ResNetStemKind::Tiered)
        .with_stem_width(16)
        .with_avg_down(true)
}

/// timm's `'se'` after each unit's last norm.
#[test]
fn test_se_toy_is_timms() {
    check_toy(
        "toy_se",
        toy_policy(true, [8, 16, 32, 64]).with_attention(Some(ConvAttentionKind::SE)),
    );
}

/// timm's `'eca'` after each unit's last norm.
#[test]
fn test_eca_toy_is_timms() {
    check_toy(
        "toy_eca",
        toy_policy(true, [8, 16, 32, 64]).with_attention(Some(ConvAttentionKind::Eca)),
    );
}
