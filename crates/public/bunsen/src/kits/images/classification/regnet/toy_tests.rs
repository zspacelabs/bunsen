//! The toy models against timm's own `RegNet`.
//!
//! `testdata/regnet/` holds nine tiny seeded timm `RegNet`s and their
//! answers, computed in float64 (`tools/gen_regnet_toy_fixtures.py`, timm's
//! `timm.models.regnet.RegNet`), each of a 7-unit `RegNetCfg` (`w0=8`,
//! `wa=6.0`, `wm=1.5`) that timm's width rule makes four stages: a `RegNetX`;
//! a `RegNetY`, its squeeze-excite after the mixer; a `_tv` arch's rounding
//! (groups of 12 under `group_min_ratio=0.9`, which moves the widths); a
//! headless trunk; the `RegNetY` on odd stage sizes; a `RegNetV`
//! (pre-activation), with conv and with `avg` shortcuts; and a `RegNetZ`
//! (an inverted bottleneck, no shortcut convs, linear unit outputs), with and
//! without a final conv. Each is the
//! [`RegNetContractConfig`] of the same knobs. The weights, under timm's
//! names, load through [`load_regnet_tensors`]; each stage output, the pooled
//! features and the logits are checked against timm's.

use burn::prelude::*;

use super::{
    RegNetContractConfig,
    RegNetDownsampleKind,
    interop::timm::load_regnet_tensors,
};
use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        images::models::conv_net::ConvNetModel,
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
    let path = testdata!("regnet").join(format!("{name}.safetensors"));
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
        fixture.assert_io(name, &format!("stage_{k}"), stage);
    }
    fixture.assert_io(name, "pooled", model.forward_pooled(images.clone()));
    fixture.assert_io(name, "logits", model.forward_logits(images));
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

/// `toy_v`'s knobs: pre-activation units, `SiLU`, the squeeze-excite.
fn regnet_v() -> RegNetContractConfig {
    toy_policy()
        .with_preact(true)
        .with_act_layer(BunsenActivationConfig::Silu)
        .with_se_ratio(0.25)
}

/// `toy_z`'s knobs: a bottleneck 4x its unit, no shortcut convs, linear unit
/// outputs, `SiLU`, the squeeze-excite.
fn regnet_z() -> RegNetContractConfig {
    toy_policy()
        .with_bottle_ratio(4.0)
        .with_downsample(RegNetDownsampleKind::None)
        .with_linear_out(true)
        .with_act_layer(BunsenActivationConfig::Silu)
        .with_se_ratio(0.25)
}

/// Pre-activation: a bare stem conv; each unit's input normalized and
/// activated before its path and its shortcut; an activation after the last
/// stage.
#[test]
fn test_v_toy_is_timms() {
    check_toy("toy_v", regnet_v());
}

/// Pre-activation with `avg` shortcuts: a pool, then a conv with no norm.
#[test]
fn test_v_avg_toy_is_timms() {
    check_toy(
        "toy_v_avg",
        regnet_v().with_downsample(RegNetDownsampleKind::Avg),
    );
}

/// An inverted bottleneck whose units that change shape have no residual,
/// and linear unit outputs, an activation after the last stage.
#[test]
fn test_z_toy_is_timms() {
    check_toy("toy_z", regnet_z());
}

/// `toy_z` with a final `1x1` conv, norm and activation to 48 features.
#[test]
fn test_z_h_toy_is_timms() {
    check_toy("toy_z_h", regnet_z().with_num_features(48));
}
