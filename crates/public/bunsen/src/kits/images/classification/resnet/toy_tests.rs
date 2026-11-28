//! The toy models against timm's own `ResNet`.
//!
//! `testdata/resnet/` holds fifteen tiny seeded timm `ResNet`s and their
//! answers, computed in float64 (`tools/gen_resnet_toy_fixtures.py`, timm's
//! `timm.models.resnet.ResNet`), each four stages of one block: on its
//! 64-wide 7x7 stem, one of basic blocks and one of bottlenecks at timm's
//! `channels=(8, 16, 32, 64)`, one of bottlenecks at 32 planes with timm's
//! group norm, and one of grouped bottlenecks (`ResNeXt`'s); then, on odd
//! stage sizes, a deep stem and a tiered one with average-pool shortcuts, and
//! the tiered one dilated past stride 8; bottlenecks with timm's
//! squeeze-excite and with its ECA; and timm's anti-aliasing, a blur (also
//! dilated past stride 16) and an average pool; a `ResNet-RS`'s conv stem
//! pool with its quarter squeeze-excite; and the `catavgmax` and `avgmax`
//! heads.
//! Each is the [`ResNetContractConfig`] of the same knobs. The weights, under
//! timm's names, load through [`load_resnet_tensors`]; each stage output, the
//! pooled features and the logits are checked against timm's.

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
        ConvAaKind,
        ConvAttentionKind,
        ConvNetModel,
        GlobalPool2dConfig,
    },
    burner::module::ModuleInit,
    kits::images::{
        classification::driver::ImageClassifier,
        support::encoder::ImageEncoder,
    },
    support::testing::{
        ToyFixture,
        cpu_device,
        testdata,
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
    let path = testdata!("resnet").join(format!("{name}.safetensors"));
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
        fixture.assert_io(name, &format!("stage_{k}"), stage);
    }
    fixture.assert_io(name, "pooled", model.forward_pooled(images.clone()));
    fixture.assert_io(name, "logits", model.forward_logits(images));
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
        toy_policy(true, [32; 4]).with_norm(GroupNormConfig::new(32, 0).into()),
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

/// timm's `BlurPool2d` anti-aliasing: the stem's stride-1 max pool and blur,
/// and each striding unit's blur after its mixer; on odd sizes.
#[test]
fn test_blur_toy_is_timms() {
    check_toy(
        "toy_blur",
        toy_policy(false, [8, 16, 32, 64]).with_anti_alias(Some(ConvAaKind::Blur)),
    );
}

/// timm's `nn.AvgPool2d` anti-aliasing on a deep stem with average-pool
/// shortcuts: the stem's `2x2` average pool, each striding unit's after its
/// mixer.
#[test]
fn test_avg_aa_toy_is_timms() {
    check_toy(
        "toy_aa",
        toy_policy(true, [8, 16, 32, 64])
            .with_stem_type(ResNetStemKind::Deep)
            .with_stem_width(16)
            .with_avg_down(true)
            .with_anti_alias(Some(ConvAaKind::Avg)),
    );
}

/// Blur anti-aliasing at output stride 16: the dilated stage's first unit
/// blurs at stride 1.
#[test]
fn test_blur_dilated_toy_is_timms() {
    check_toy(
        "toy_blur_dilated",
        toy_policy(true, [8, 16, 32, 64])
            .with_anti_alias(Some(ConvAaKind::Blur))
            .with_output_stride(16),
    );
}

/// `ResNet-RS`'s knobs: a deep stem whose max pool is a strided conv,
/// average-pool shortcuts, squeeze-excite of a quarter of the channels.
#[test]
fn test_rs_toy_is_timms() {
    check_toy(
        "toy_rs",
        toy_policy(true, [8, 16, 32, 64])
            .with_stem_type(ResNetStemKind::Deep)
            .with_stem_width(16)
            .with_replace_stem_pool(true)
            .with_avg_down(true)
            .with_attention(Some(ConvAttentionKind::Se { rd_ratio: 0.25 })),
    );
}

/// timm's `catavgmax` head: the means and the maxes, concatenated, into a
/// classifier twice as wide; the head's dropout is a no-op in eval.
#[test]
fn test_catavgmax_toy_is_timms() {
    let policy = toy_policy(true, [8, 16, 32, 64])
        .with_global_pool(GlobalPool2dConfig::CatAvgMax)
        .with_drop_rate(0.25);
    let device = cpu_device();
    let (model, _) = load_toy("toy_catavgmax", policy.clone(), &device);
    assert_eq!(model.head.d_features, 512);
    assert_eq!(ImageEncoder::d_features(&model), 512);
    check_toy("toy_catavgmax", policy);
}

/// timm's `avgmax` head: the mean of the means and the maxes.
#[test]
fn test_avgmax_toy_is_timms() {
    check_toy(
        "toy_avgmax",
        toy_policy(false, [8, 16, 32, 64]).with_global_pool(GlobalPool2dConfig::AvgMax),
    );
}
