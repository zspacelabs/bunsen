//! The toy models against timm's own `Res2Net`.
//!
//! `testdata/res2net/` holds four tiny seeded timm `Res2Net`s and their
//! answers, computed in float64 (`tools/gen_res2net_toy_fixtures.py`, timm's
//! `ResNet` of `Bottle2neck`s) at `(8, 16, 32, 64)` planes, some stages of
//! two blocks so the hierarchical branches run: `res2net50_26w_4s`'s knobs;
//! `Res2NeXt`'s grouped branches at three chunks; `res2net50d`'s deep stem
//! and average-pool shortcuts at two chunks, on odd sizes; and eight chunks.
//! Each is the [`Res2NetContractConfig`] of the same knobs. The weights,
//! under timm's names, load through [`load_res2net_tensors`]; each stage
//! output, the pooled features and the logits are checked against timm's.

use burn::prelude::*;

use super::{
    Res2NetContractConfig,
    interop::timm::load_res2net_tensors,
};
use crate::{
    blocks::images::models::conv_net::ConvNetModel,
    burner::module::ModuleInit,
    kits::images::{
        classification::{
            driver::ImageClassifier,
            resnet::{
                BottleneckPolicyConfig,
                ResNetContractConfig,
                ResNetStemKind,
            },
        },
        support::encoder::ImageEncoder,
    },
    support::testing::{
        ToyFixture,
        cpu_device,
        testdata,
    },
};

/// A toy `Res2Net`: stages of `layers` blocks at `(8, 16, 32, 64)` planes,
/// `base_width` per chunk per 64 planes in `cardinality` groups, `scale`
/// chunks, 10 classes.
fn toy_policy(
    layers: [usize; 4],
    base_width: usize,
    cardinality: usize,
    scale: usize,
) -> Res2NetContractConfig {
    Res2NetContractConfig::new(
        ResNetContractConfig::new(layers.to_vec(), 10)
            .with_channels(vec![8, 16, 32, 64])
            .with_bottleneck_policy(Some(
                BottleneckPolicyConfig::new()
                    .with_base_width(base_width)
                    .with_cardinality(cardinality),
            )),
    )
    .with_scale(scale)
}

/// Every stage, the pooled features and the logits of the toy `name`,
/// built from `policy`, are timm's.
fn check_toy(
    name: &str,
    policy: Res2NetContractConfig,
) {
    let device = cpu_device();
    let path = testdata!("res2net").join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, &device);
    let mut model: ConvNetModel = policy.init(&device);
    let applied = load_res2net_tensors(&mut model, fixture.take_weights()).expect("load the toy");
    assert!(
        applied
            .unused
            .iter()
            .all(|n| n.ends_with(".num_batches_tracked")),
        "{name}: {:?}",
        applied.unused
    );
    let images: Tensor<4> = fixture.io("images");
    let stages = model.forward_intermediates(images.clone());
    assert_eq!(stages.len(), 5, "{name}: the stem and four stages");
    for (k, stage) in stages.into_iter().enumerate() {
        fixture.assert_io(name, &format!("stage_{k}"), stage);
    }
    fixture.assert_io(name, "pooled", model.forward_pooled(images.clone()));
    fixture.assert_io(name, "logits", model.forward_logits(images));
}

/// `res2net50_26w_4s`'s knobs.
#[test]
fn test_res2net_toy_is_timms() {
    check_toy("toy_res2net", toy_policy([2, 2, 1, 1], 26, 1, 4));
}

/// `Res2NeXt`'s grouped branches, three chunks.
#[test]
fn test_res2next_toy_is_timms() {
    check_toy("toy_res2next", toy_policy([2, 1, 1, 1], 8, 2, 3));
}

/// `res2net50d`'s deep stem and average-pool shortcuts, two chunks, on odd
/// sizes.
#[test]
fn test_res2net_d_toy_is_timms() {
    let mut policy = toy_policy([1, 2, 1, 1], 48, 1, 2);
    policy.resnet = policy
        .resnet
        .with_stem_type(ResNetStemKind::Deep)
        .with_stem_width(16)
        .with_avg_down(true);
    check_toy("toy_res2net_d", policy);
}

/// Eight chunks.
#[test]
fn test_res2net_8s_toy_is_timms() {
    check_toy("toy_res2net_8s", toy_policy([2, 1, 1, 1], 14, 1, 8));
}
