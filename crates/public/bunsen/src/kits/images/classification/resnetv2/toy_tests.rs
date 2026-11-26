//! The toy models against timm's own `ResNetV2`.
//!
//! `testdata/resnetv2/` holds six tiny seeded timm `ResNetV2`s and their
//! answers, computed in float64 (`tools/gen_resnetv2_toy_fixtures.py`, timm's
//! `timm.models.resnetv2.ResNetV2`), each four stages of one pre-activation
//! unit: a `BiT` (weight-standardized convs, 32-group norms, the `fixed`
//! stem), and the same on odd sizes; batch norms on the plain stem; basic
//! units on a deep stem with average-pool shortcuts; group norms and
//! `EvoNorm-S0` on that stem. Each is the [`ResNetV2ContractConfig`] of the
//! same knobs. The weights, under timm's names, load through
//! [`load_resnetv2_tensors`]; each stage output, the pooled features and the
//! logits are checked against timm's.

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    BIT_STD_CONV_EPSILON,
    ResNetV2ContractConfig,
    ResNetV2Norm,
    ResNetV2Stem,
    interop::timm::load_resnetv2_tensors,
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

/// Four stages of one unit, `channels` wide on a `stem_chs` stem, 10
/// classes.
fn toy_policy(
    channels: [usize; 4],
    stem_chs: usize,
) -> ResNetV2ContractConfig {
    ResNetV2ContractConfig::new(vec![1; 4], 10)
        .with_channels(channels.to_vec())
        .with_stem_chs(stem_chs)
}

/// `toy_bit`'s knobs.
fn bit() -> ResNetV2ContractConfig {
    toy_policy([128, 128, 256, 256], 32)
        .with_stem_type(ResNetV2Stem::Fixed)
        .with_std_conv(Some(BIT_STD_CONV_EPSILON))
}

/// The deep stem with average-pool shortcuts, plain convs.
fn deep(
    channels: [usize; 4],
    stem_chs: usize,
) -> ResNetV2ContractConfig {
    toy_policy(channels, stem_chs)
        .with_stem_type(ResNetV2Stem::Deep)
        .with_avg_down(true)
}

/// Every stage, the pooled features and the logits of the toy `name`,
/// built from `policy`, are timm's.
fn check_toy(
    name: &str,
    policy: ResNetV2ContractConfig,
) {
    let device = cpu_device();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/resnetv2")
        .join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, &device);
    let mut model: ConvNetModel = policy.init(&device);
    let applied = load_resnetv2_tensors(&mut model, fixture.take_weights()).expect("load the toy");
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

/// `BiT`: weight-standardized convs, group norms, the `fixed` stem.
#[test]
fn test_bit_toy_is_timms() {
    check_toy("toy_bit", bit());
}

/// `BiT` on odd sizes: the zero-padded pool's windows past the edge.
#[test]
fn test_bit_odd_toy_is_timms() {
    check_toy("toy_bit_odd", bit());
}

/// Batch norms on the plain stem.
#[test]
fn test_bn_toy_is_timms() {
    check_toy(
        "toy_bn",
        toy_policy([64, 128, 128, 256], 32).with_norm_layer(ResNetV2Norm::BatchNorm),
    );
}

/// Basic units on the deep stem, average-pool shortcuts, batch norms.
#[test]
fn test_basic_deep_toy_is_timms() {
    check_toy(
        "toy_basic_deep",
        deep([16, 32, 64, 128], 32)
            .with_basic(true)
            .with_bottle_ratio(1.0)
            .with_norm_layer(ResNetV2Norm::BatchNorm),
    );
}

/// Group norms on the deep stem.
#[test]
fn test_gn_deep_toy_is_timms() {
    check_toy("toy_gn_deep", deep([128, 128, 256, 256], 64));
}

/// `EvoNorm-S0` on the deep stem.
#[test]
fn test_evos_toy_is_timms() {
    check_toy(
        "toy_evos",
        deep([128, 128, 256, 256], 64).with_norm_layer(ResNetV2Norm::EvoNormS0),
    );
}
