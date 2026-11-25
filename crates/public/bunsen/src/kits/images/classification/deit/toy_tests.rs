//! The toy models against timm's own `DeiT`s.
//!
//! `testdata/deit/` holds three tiny seeded timm `DeiT`s and their answers,
//! computed in float64 (`tools/gen_deit_toy_fixtures.py`), each on 16x16
//! images in 4x4 patches, 16 wide, two layers of two heads: timm's
//! `VisionTransformerDistilled`, with its classifiers and without; and
//! `DeiT` III's `VisionTransformer`. Each is the [`DeiTContractConfig`] of the
//! same knobs. The weights, under timm's names, load through the vit kit's
//! [`load_vit_tensors`]; each layer's map, the output tokens, the pooled
//! features and the logits are checked against timm's.

use std::path::PathBuf;

use burn::prelude::*;

use super::DeiTContractConfig;
use crate::{
    blocks::transformers::models::vision_encoder::VisionEncoderModel,
    burner::module::ModuleInit,
    kits::images::{
        classification::{
            driver::ImageClassifier,
            vit::{
                ViTContractConfig,
                interop::timm::load_vit_tensors,
            },
        },
        support::encoder::{
            ImageEncoder,
            ImageFeatures,
        },
    },
    support::testing::{
        ToyFixture,
        assert_io_close,
        cpu_device,
    },
};

/// The toys' `ViT`: 16x16 images in 4x4 patches, 16 wide, two layers of two
/// heads, `classes` classes.
fn toy_vit(classes: usize) -> ViTContractConfig {
    ViTContractConfig::new(16, 4, 16, 2, 2, classes)
}

/// Every layer, the output tokens, the pooled features and the logits are
/// timm's.
fn check_toy(
    name: &str,
    policy: DeiTContractConfig,
) {
    let device = cpu_device();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/deit")
        .join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, &device);
    let mut model: VisionEncoderModel = policy.init(&device);
    let applied = load_vit_tensors(&mut model, fixture.take_weights()).expect("load the toy");
    assert!(applied.unused.is_empty(), "{name}: {:?}", applied.unused);
    let images: Tensor<4> = fixture.io("images");
    let layers = model.forward_intermediates(images.clone());
    assert_eq!(layers.len(), 2, "{name}: two layers");
    for (k, layer) in layers.into_iter().enumerate() {
        assert_io_close(
            &format!("{name} layer_{k}"),
            layer,
            fixture.io(&format!("layer_{k}")),
        );
    }
    let ImageFeatures::Tokens { tokens, .. } = model.forward_features(images.clone()) else {
        panic!("{name}: a DeiT's features are tokens");
    };
    assert_io_close(&format!("{name} features"), tokens, fixture.io("features"));
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

/// The distillation token and its classifier, averaged with the class
/// token's.
#[test]
fn test_distilled_toy_is_timms() {
    check_toy(
        "toy_distilled",
        DeiTContractConfig::new(toy_vit(10)).with_distilled(true),
    );
}

/// `DeiT` III: the position embedding over the patches alone, layer scale.
#[test]
fn test_deit3_toy_is_timms() {
    check_toy(
        "toy_deit3",
        DeiTContractConfig::new(toy_vit(10).with_no_embed_class(true).with_layer_scale(true)),
    );
}

/// No classifiers: the output is the two tokens' mean.
#[test]
fn test_distilled_headless_toy_is_timms() {
    check_toy(
        "toy_distilled_headless",
        DeiTContractConfig::new(toy_vit(0)).with_distilled(true),
    );
}
