//! The toy models against timm's own `SKNet`.
//!
//! `testdata/sknet/` holds three tiny seeded timm `SKNet`s and their answers,
//! computed in float64 (`tools/gen_sknet_toy_fixtures.py`, timm's `ResNet` of
//! `sknet.py`'s blocks), each on timm's 7x7 stem, its first stage of two
//! blocks: `skresnet18`'s knobs (basic blocks, the selective kernel their
//! first conv, the input split across its paths), on even and odd sizes; and
//! `skresnext50_32x4d`'s (grouped bottlenecks, every path reading the whole
//! input). Each is the [`SKNetContractConfig`] of the same knobs. The weights,
//! under timm's names, load through [`load_sknet_tensors`]; each stage
//! output, the pooled features and the logits are checked against timm's.

use burn::prelude::*;

use super::{
    SKNetContractConfig,
    interop::timm::load_sknet_tensors,
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

/// `skresnet18`'s knobs at `(16, 32, 32, 64)` planes, stages of
/// `[2, 1, 1, 1]` blocks, 10 classes.
fn skresnet() -> SKNetContractConfig {
    SKNetContractConfig::new(
        ResNetContractConfig::new(vec![2, 1, 1, 1], 10).with_channels(vec![16, 32, 32, 64]),
    )
    .with_rd_ratio(0.125)
    .with_rd_divisor(16)
}

/// Every stage, the pooled features and the logits of the toy `name`,
/// built from `policy`, are timm's.
fn check_toy(
    name: &str,
    policy: SKNetContractConfig,
) {
    let device = cpu_device();
    let path = testdata!("sknet").join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, &device);
    let mut model: ConvNetModel = policy.init(&device);
    let applied = load_sknet_tensors(&mut model, fixture.take_weights()).expect("load the toy");
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

/// `skresnet18`'s knobs.
#[test]
fn test_skresnet_toy_is_timms() {
    check_toy("toy_skresnet", skresnet());
}

/// `skresnet18`'s knobs on odd sizes.
#[test]
fn test_skresnet_odd_toy_is_timms() {
    check_toy("toy_skresnet_odd", skresnet());
}

/// `skresnext50_32x4d`'s: grouped bottlenecks, every path reading the whole
/// input.
#[test]
fn test_skresnext_toy_is_timms() {
    check_toy(
        "toy_skresnext",
        SKNetContractConfig::new(
            ResNetContractConfig::new(vec![2, 1, 1, 1], 10)
                .with_channels(vec![8, 16, 32, 64])
                .with_bottleneck_policy(Some(
                    BottleneckPolicyConfig::new()
                        .with_base_width(16)
                        .with_cardinality(4),
                )),
        )
        .with_rd_ratio(0.0625)
        .with_rd_divisor(32)
        .with_split_input(false),
    );
}
