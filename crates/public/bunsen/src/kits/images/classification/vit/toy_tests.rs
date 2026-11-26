//! The toy models against timm's own `VisionTransformer`.
//!
//! `testdata/vit/` holds fourteen tiny seeded timm `VisionTransformer`s and
//! their answers, computed in float64 (`tools/gen_vit_toy_fixtures.py`), each
//! on 16x16 images in 4x4 patches, 16 wide, two layers of two heads: timm's
//! defaults (the class token's pool); the patch tokens' mean and their
//! `avgmax`, with the norm after the pool; CLIP's pre-norm, with `GELU` and
//! with `QuickGELU`; register tokens with the position embedding over the
//! patches alone and layer scale; no class token; no `qkv` bias; no
//! classifier; CLIP's pre-norm with a projection head, its weights in
//! `OpenAI` CLIP's own layout; res-post-norm blocks; no final norm, with the
//! class token's pool and the mean's; and 18x18 images, the pixels past the
//! last whole patch dropped. Each is the [`ViTContractConfig`] of the same
//! knobs. The
//! weights, under timm's names, load through [`load_vit_tensors`]; each
//! layer's map, the output tokens, the pooled features and the logits are
//! checked against timm's.

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    ViTContractConfig,
    interop::timm::load_vit_tensors,
};
use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        transformers::{
            models::vision_encoder::{
                TokenPoolConfig,
                VisionEncoderModel,
            },
            sublayer::NormPlacement,
        },
    },
    burner::module::ModuleInit,
    kits::images::{
        classification::driver::ImageClassifier,
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

/// The toys' geometry: 16x16 images in 4x4 patches, 16 wide, two layers of
/// two heads, 10 classes.
fn toy_policy() -> ViTContractConfig {
    ViTContractConfig::new(16, 4, 16, 2, 2, 10)
}

/// The toy `name` of `policy`, its weights loaded, and its fixture's
/// answers.
fn load_toy(
    name: &str,
    policy: ViTContractConfig,
    device: &Device,
) -> (VisionEncoderModel, ToyFixture) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/vit")
        .join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, device);
    let mut model: VisionEncoderModel = policy.init(device);
    let applied = load_vit_tensors(&mut model, fixture.take_weights()).expect("load the toy");
    assert!(applied.unused.is_empty(), "{name}: {:?}", applied.unused);
    (model, fixture)
}

/// Every layer, the output tokens, the pooled features and the logits are
/// timm's.
fn check_toy(
    name: &str,
    policy: ViTContractConfig,
) {
    let device = cpu_device();
    let (model, fixture) = load_toy(name, policy, &device);
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
        panic!("{name}: a ViT's features are tokens");
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

#[test]
fn test_token_toy_is_timms() {
    check_toy("toy_token", toy_policy());
}

/// The mean pool, normed after the pool rather than before it.
#[test]
fn test_avg_toy_is_timms() {
    check_toy(
        "toy_avg",
        toy_policy().with_global_pool(TokenPoolConfig::Avg),
    );
}

#[test]
fn test_avgmax_toy_is_timms() {
    check_toy(
        "toy_avgmax",
        toy_policy().with_global_pool(TokenPoolConfig::AvgMax),
    );
}

/// CLIP's pre-norm: a bias-free patch embedding, `norm_pre`, epsilon 1e-5.
#[test]
fn test_clip_toy_is_timms() {
    check_toy("toy_clip", toy_policy().with_clip());
}

#[test]
fn test_quickgelu_toy_is_timms() {
    check_toy(
        "toy_quickgelu",
        toy_policy()
            .with_clip()
            .with_activation(BunsenActivationConfig::QuickGelu),
    );
}

/// Register tokens, the position embedding over the patches alone, and
/// layer scale.
#[test]
fn test_reg_toy_is_timms() {
    check_toy(
        "toy_reg",
        toy_policy()
            .with_reg_tokens(2)
            .with_no_embed_class(true)
            .with_layer_scale(true),
    );
}

/// No class token: the mean pool over every token.
#[test]
fn test_gap_toy_is_timms() {
    check_toy(
        "toy_gap",
        toy_policy()
            .with_class_token(false)
            .with_global_pool(TokenPoolConfig::Avg),
    );
}

#[test]
fn test_miil_toy_is_timms() {
    check_toy("toy_miil", toy_policy().with_qkv_bias(false));
}

/// No classifier: the output is the pooled features.
#[test]
fn test_headless_toy_is_timms() {
    let mut policy = toy_policy();
    policy.num_classes = 0;
    check_toy("toy_headless", policy);
}

/// `OpenAI` CLIP's own layout (`visual.*`, a text tower beside it, a
/// projection head of no bias): the text tower dropped, the tower loaded as
/// timm converts it.
#[test]
fn test_openai_toy_is_timms() {
    let mut policy = toy_policy().with_clip();
    policy.num_classes = 8;
    check_openai_toy("toy_openai", policy);
}

/// Res-post-norm blocks (each branch normed on its output), no class token,
/// the mean pool, no `qkv` bias: `vit_base_patch16_rpn_224`'s knobs.
#[test]
fn test_rpn_toy_is_timms() {
    check_toy(
        "toy_rpn",
        toy_policy()
            .with_norm_placement(NormPlacement::ResPost)
            .with_class_token(false)
            .with_global_pool(TokenPoolConfig::Avg)
            .with_qkv_bias(false),
    );
}

/// No norm after the last layer, with register tokens and layer scale:
/// C-RADIO's knobs.
#[test]
fn test_bare_toy_is_timms() {
    check_toy(
        "toy_bare",
        toy_policy()
            .with_final_norm(false)
            .with_reg_tokens(2)
            .with_no_embed_class(true)
            .with_layer_scale(true),
    );
}

/// No final norm under the mean pool: no norm before the pool or after it.
#[test]
fn test_bare_avg_toy_is_timms() {
    check_toy(
        "toy_bare_avg",
        toy_policy()
            .with_final_norm(false)
            .with_global_pool(TokenPoolConfig::Avg),
    );
}

/// 18x18 images in 4x4 patches: the 2 pixels past the last whole patch
/// dropped, as timm's conv drops them.
#[test]
fn test_remainder_toy_is_timms() {
    check_toy("toy_remainder", ViTContractConfig::new(18, 4, 16, 2, 2, 10));
}

/// [`check_toy`], for a toy whose weights are in `OpenAI`'s layout: only the
/// text tower's tensors are left over.
fn check_openai_toy(
    name: &str,
    policy: ViTContractConfig,
) {
    let device = cpu_device();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/vit")
        .join(format!("{name}.safetensors"));
    let mut fixture = ToyFixture::load(path, &device);
    let mut model: VisionEncoderModel = policy.clone().init(&device);
    let applied = load_vit_tensors(&mut model, fixture.take_weights()).expect("load the toy");
    assert_eq!(
        applied.unused,
        ["text_projection", "transformer.resblocks.0.ln_1.weight"],
        "{name}: the text tower is all that's left"
    );
    let images: Tensor<4> = fixture.io("images");
    for (k, layer) in model
        .forward_intermediates(images.clone())
        .into_iter()
        .enumerate()
    {
        assert_io_close(
            &format!("{name} layer_{k}"),
            layer,
            fixture.io(&format!("layer_{k}")),
        );
    }
    assert_io_close(
        &format!("{name} logits"),
        model.forward_logits(images),
        fixture.io("logits"),
    );
}
