//! The toy models against timm's own `SwinTransformerV2`.
//!
//! `testdata/swin/` holds five tiny seeded timm `SwinTransformerV2`s and
//! their answers, computed in float64 (`tools/gen_swin_v2_toy_fixtures.py`),
//! each in 4x4 patches, 16 wide, two stages of two blocks (2 and 4 heads),
//! 10 classes: 32x32 images in 4x4 windows; the same with a 3x3 pretraining
//! window; 8x8 windows, which the second stage's grid shrinks; 28x28 images,
//! whose 7x7 grid pads to whole windows and whose odd side the merge pads;
//! and no classifier. Each is the [`SwinV2ContractConfig`] of the same
//! knobs. The weights, under timm's names, load through
//! [`load_swin_tensors`]; each stage's map, the output tokens, the pooled
//! features and the logits are checked against timm's.

use burn::prelude::*;

use super::SwinV2ContractConfig;
use crate::{
    blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel,
    burner::module::ModuleInit,
    kits::images::{
        classification::{
            driver::ImageClassifier,
            swin::interop::timm::load_swin_tensors,
        },
        support::encoder::{
            ImageEncoder,
            ImageFeatures,
        },
    },
    support::testing::{
        ToyFixture,
        cpu_device,
        testdata,
    },
};

/// The toys' geometry: 32x32 images in 4x4 patches, 16 wide, two stages of
/// two blocks (2 and 4 heads), 4x4 windows, 10 classes.
fn toy_policy() -> SwinV2ContractConfig {
    SwinV2ContractConfig::new(32, 16, vec![2, 2], vec![2, 4], 4, 10)
}

/// Each stage's map, the output tokens, the pooled features and the logits
/// of the toy `name` of `policy` are timm's.
fn check_toy(
    name: &str,
    policy: SwinV2ContractConfig,
) {
    let device = cpu_device();
    let path = testdata!("swin").join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, &device);
    let mut model: StagedVisionEncoderModel = policy.init(&device);
    let applied = load_swin_tensors(&mut model, fixture.take_weights()).expect("load the toy");
    assert!(applied.unused.is_empty(), "{name}: {:?}", applied.unused);
    let images: Tensor<4> = fixture.io("images");
    let maps = model.stage_maps(images.clone());
    assert_eq!(maps.len(), 2, "{name}: two stages");
    for (k, map) in maps.into_iter().enumerate() {
        fixture.assert_io(name, &format!("stage_{k}"), map);
    }
    let ImageFeatures::Tokens { tokens, .. } = model.forward_features(images.clone()) else {
        panic!("{name}: a Swin's features are tokens");
    };
    fixture.assert_io(name, "features", tokens);
    fixture.assert_io(name, "pooled", model.forward_pooled(images.clone()));
    fixture.assert_io(name, "logits", model.forward_logits(images));
}

#[test]
fn test_v2_toy_is_timms() {
    check_toy("toy_v2", toy_policy());
}

/// The continuous position bias's offsets scaled by a 3x3 pretraining
/// window.
#[test]
fn test_pretrained_window_toy_is_timms() {
    check_toy(
        "toy_v2_pretrained",
        toy_policy().with_pretrained_window_sizes(vec![3, 3]),
    );
}

/// 8x8 windows: the first stage's 8x8 grid is one window (no shift), the
/// second's 4x4 grid shrinks it.
#[test]
fn test_clamped_window_toy_is_timms() {
    check_toy(
        "toy_v2_clamp",
        SwinV2ContractConfig {
            window_size: 8,
            ..toy_policy()
        },
    );
}

/// 28x28 images: a 7x7 grid padded to whole windows, merged from an odd
/// side.
#[test]
fn test_padded_toy_is_timms() {
    check_toy(
        "toy_v2_pad",
        SwinV2ContractConfig {
            image_size: 28,
            ..toy_policy()
        },
    );
}

/// No classifier: the output is the pooled features.
#[test]
fn test_headless_toy_is_timms() {
    check_toy(
        "toy_v2_headless",
        SwinV2ContractConfig {
            num_classes: 0,
            ..toy_policy()
        },
    );
}
