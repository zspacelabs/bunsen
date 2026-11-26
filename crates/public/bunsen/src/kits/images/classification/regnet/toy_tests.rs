//! The toy models against timm's own `RegNet`.
//!
//! `testdata/regnet/` holds five tiny seeded timm `RegNet`s and their
//! answers, computed in float64 (`tools/gen_regnet_toy_fixtures.py`, timm's
//! `timm.models.regnet.RegNet`), each of a 7-unit `RegNetCfg` (`w0=8`,
//! `wa=6.0`, `wm=1.5`) that timm's width rule makes four stages: a `RegNetX`;
//! a `RegNetY`, its squeeze-excite after the mixer; a `_tv` arch's rounding
//! (groups of 12 under `group_min_ratio=0.9`, which moves the widths); a
//! headless trunk; and the `RegNetY` on odd stage sizes. Each is the
//! [`RegNetContractConfig`] of the same knobs. The weights, under timm's
//! names, load through [`load_regnet_tensors`]; each stage output, the pooled
//! features and the logits are checked against timm's.

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    RegNetContractConfig,
    interop::timm::load_regnet_tensors,
};
use crate::{
    blocks::images::models::conv_net::ConvNetModel,
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

/// The toys' geometry: 7 units of `w0=8`, `wa=6.0`, `wm=1.5`, in groups of
/// 4 channels, 10 classes.
fn toy_policy() -> RegNetContractConfig {
    RegNetContractConfig::new(7, 8, 6.0, 1.5, 4, 10)
}

/// Every stage, the pooled features and the logits of the toy `name`,
/// built from `policy`, are timm's.
fn check_toy(
    name: &str,
    policy: RegNetContractConfig,
) {
    let device = cpu_device();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/regnet")
        .join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, &device);
    let mut model: ConvNetModel = policy.init(&device);
    let applied = load_regnet_tensors(&mut model, fixture.take_weights()).expect("load the toy");
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

#[test]
fn test_x_toy_is_timms() {
    check_toy("toy_x", toy_policy());
}

/// The squeeze-excite after the mixer, a quarter of the unit's input.
#[test]
fn test_y_toy_is_timms() {
    check_toy("toy_y", toy_policy().with_se_ratio(0.25));
}

/// Groups of 12 under `group_min_ratio=0.9`: the widths round up to the
/// groups.
#[test]
fn test_tv_toy_is_timms() {
    check_toy(
        "toy_tv",
        RegNetContractConfig {
            group_size: 12,
            ..toy_policy()
        }
        .with_group_min_ratio(0.9)
        .with_se_ratio(0.25),
    );
}

/// No classifier: the output is the pooled features.
#[test]
fn test_headless_toy_is_timms() {
    check_toy(
        "toy_headless",
        RegNetContractConfig {
            num_classes: 0,
            ..toy_policy()
        }
        .with_se_ratio(0.25),
    );
}

/// Odd stage sizes: 36x36 images.
#[test]
fn test_odd_toy_is_timms() {
    check_toy("toy_odd", toy_policy().with_se_ratio(0.25));
}
