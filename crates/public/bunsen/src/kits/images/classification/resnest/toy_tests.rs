//! The toy models against timm's own `ResNeSt`.
//!
//! `testdata/resnest/` holds four tiny seeded timm `ResNeSt`s and their
//! answers, computed in float64 (`tools/gen_resnest_toy_fixtures.py`, timm's
//! `ResNet` of `ResNestBottleneck`s), each on a deep stem with average-pool
//! shortcuts, its first stage of two blocks: `resnest50d`'s knobs (radix 2,
//! the `avd` pool after the mixer), on even and on odd sizes;
//! `resnest50d_4s2x40d`'s (radix 4 in 2 groups, the pool before the mixer);
//! and `resnest50d_1s4x24d`'s (radix 1, a sigmoid gate, in 4 groups). Each is
//! the [`ResNeStContractConfig`] of the same knobs. The weights, under timm's
//! names, load through [`load_resnest_tensors`]; each stage output, the
//! pooled features and the logits are checked against timm's.

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    ResNeStContractConfig,
    interop::timm::load_resnest_tensors,
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
        assert_io_close,
        cpu_device,
    },
};

/// A toy `ResNeSt` on a 16-wide deep stem with average-pool shortcuts:
/// stages of `[2, 1, 1, 1]` blocks at `channels` planes, `base_width` per 64
/// planes in `cardinality` groups, `radix` splits, the `avd` pool before the
/// mixer under `avd_first`; 10 classes.
fn toy_policy(
    channels: [usize; 4],
    base_width: usize,
    cardinality: usize,
    radix: usize,
    avd_first: bool,
) -> ResNeStContractConfig {
    ResNeStContractConfig::new(
        ResNetContractConfig::new(vec![2, 1, 1, 1], 10)
            .with_channels(channels.to_vec())
            .with_bottleneck_policy(Some(
                BottleneckPolicyConfig::new()
                    .with_base_width(base_width)
                    .with_cardinality(cardinality),
            ))
            .with_stem_type(ResNetStemKind::Deep)
            .with_stem_width(16)
            .with_avg_down(true),
    )
    .with_radix(radix)
    .with_avd_first(avd_first)
}

/// Every stage, the pooled features and the logits of the toy `name`,
/// built from `policy`, are timm's.
fn check_toy(
    name: &str,
    policy: ResNeStContractConfig,
) {
    let device = cpu_device();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/resnest")
        .join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, &device);
    let mut model: ConvNetModel = policy.init(&device);
    let applied = load_resnest_tensors(&mut model, fixture.take_weights()).expect("load the toy");
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

/// `resnest50d`'s knobs.
#[test]
fn test_resnest_toy_is_timms() {
    check_toy("toy_resnest", toy_policy([8, 16, 32, 64], 64, 1, 2, false));
}

/// `resnest50d`'s knobs on odd sizes.
#[test]
fn test_resnest_odd_toy_is_timms() {
    check_toy(
        "toy_resnest_odd",
        toy_policy([8, 16, 32, 64], 64, 1, 2, false),
    );
}

/// `resnest50d_4s2x40d`'s: radix 4 in 2 groups, the pool first.
#[test]
fn test_resnest_4s2x_toy_is_timms() {
    check_toy(
        "toy_resnest_4s2x",
        toy_policy([32, 32, 64, 64], 40, 2, 4, true),
    );
}

/// `resnest50d_1s4x24d`'s: radix 1, a sigmoid gate, in 4 groups, the pool
/// first.
#[test]
fn test_resnest_1s4x_toy_is_timms() {
    check_toy(
        "toy_resnest_1s4x",
        toy_policy([16, 32, 32, 64], 24, 4, 1, true),
    );
}
