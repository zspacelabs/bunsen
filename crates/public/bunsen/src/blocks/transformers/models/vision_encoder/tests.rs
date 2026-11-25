use burn::{
    Tensor,
    tensor::{
        Distribution,
        Tolerance,
    },
};

use super::*;
use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        images::patching::{
            patch_embed::PatchEmbedConfig,
            prefix_tokens::{
                PrefixTokensConfig,
                PrefixTokensMeta,
            },
        },
        mlp::BunsenMlpConfig,
        norm::{
            BunsenLayerNormConfig,
            BunsenNormalizationConfig,
        },
        transformers::{
            attention::csa::{
                AttentionMask,
                CausalSelfAttentionConfig,
            },
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::UniformStackConfig,
            },
            embedding::RotaryEmbeddingConfig,
        },
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    },
    support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    },
};

/// A `ViT` layer norm of `d` features.
fn norm(d: usize) -> BunsenNormalizationConfig {
    BunsenNormalizationConfig::Layer(BunsenLayerNormConfig::new(d).with_epsilon(1e-6))
}

/// A tiny `ViT`: 16x16 images in 4x4 patches (a 4x4 grid), 8 wide, two
/// layers of two heads, `reg` register tokens, and 5 classes.
fn tiny(
    pool: TokenPoolConfig,
    reg: usize,
) -> VisionEncoderStructureConfig {
    let d = 8;
    let fc_norm = pool.timm_fc_norm();
    let tokens = PrefixTokensConfig::new(d, 16).with_reg_tokens(reg);
    let stack = UniformStackConfig::new(
        2,
        CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(2, 2, d)
                .with_norm(None)
                .with_qkv_bias(true)
                .with_output_bias(true)
                .with_mask(AttentionMask::Bidirectional)
                .into(),
            BunsenMlpConfig::expansion(d, 4)
                .with_activation(BunsenActivationConfig::Gelu)
                .with_bias(true)
                .into(),
        )
        .with_norm(norm(d)),
        match fc_norm {
            true => BunsenNormalizationConfig::Identity,
            false => norm(d),
        },
        RotaryEmbeddingConfig::unrotated(4),
    )
    .to_structure();
    VisionEncoderStructureConfig::new(
        PatchEmbedConfig::new([16, 16], 4, 3, d).with_enable_patch_norm(false),
        tokens.clone(),
        stack,
        VisionHeadConfig::new(d, 5, tokens.n_prefix())
            .with_pool(pool)
            .with_fc_norm(fc_norm.then(|| norm(d))),
    )
}

/// The steps' shapes, and the meta of the config and the model agree.
#[test]
fn test_shapes_and_meta() {
    let device = cpu_device();
    for (pool, reg) in [(TokenPoolConfig::Token, 0), (TokenPoolConfig::Avg, 2)] {
        let config = tiny(pool, reg);
        let model: VisionEncoderModel = config.init(&device);
        for (name, a, b) in [
            ("in_channels", config.in_channels(), model.in_channels()),
            ("patch_size", config.patch_size(), model.patch_size()),
            ("d_model", config.d_model(), model.d_model()),
            ("n_layers", config.n_layers(), model.n_layers()),
            ("n_prefix", config.n_prefix(), model.n_prefix()),
            ("num_classes", config.num_classes(), model.num_classes()),
        ] {
            assert_eq!(a, b, "{name}");
        }
        assert_eq!(model.grid(), [4, 4]);
        assert_eq!(model.image_size(), [16, 16]);
        let images: Tensor<4> = seeded_tensor(1, [2, 3, 16, 16], Distribution::Default, &device);
        assert_eq!(model.forward(images.clone()).dims(), [2, 5]);
        assert_eq!(
            model.feature_tokens(images.clone()).dims(),
            [2, 1 + reg + 16, 8]
        );
        assert_eq!(model.pooled_features(images.clone()).dims(), [2, 8]);
        let maps = model.layer_maps(images);
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[1].dims(), [2, 8, 4, 4]);
    }
}

/// The last layer's map, normed, is the feature tokens' grid; the logits
/// are the head over the pooled features.
#[test]
fn test_steps_compose() {
    let device = cpu_device();
    let model: VisionEncoderModel = tiny(TokenPoolConfig::Token, 1).init(&device);
    let images: Tensor<4> = seeded_tensor(2, [2, 3, 16, 16], Distribution::Default, &device);
    let tolerance = Tolerance::rel_abs(1e-5, 1e-5);
    let (_, states) = model
        .stack
        .encoder_forward_layers(model.embed(images.clone()));
    let tokens = model.feature_tokens(images.clone());
    assert_tensors_close(
        &model.grid_map(model.stack.norm.forward(states[1].clone())),
        &model.grid_map(tokens.clone()),
        tolerance,
    );
    assert_tensors_close(
        &model
            .head
            .fc
            .as_ref()
            .unwrap()
            .forward(model.pooled_features(images.clone())),
        &model.forward(images),
        tolerance,
    );
}

/// A headless model's output is its pooled features; re-heading builds a
/// classifier.
#[test]
fn test_headless_and_with_classes() {
    let device = cpu_device();
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.head.num_classes = 0;
    let model: VisionEncoderModel = config.init(&device);
    assert_eq!(model.num_classes(), 0);
    let images: Tensor<4> = seeded_tensor(3, [1, 3, 16, 16], Distribution::Default, &device);
    assert_tensors_close(
        &model.forward(images.clone()),
        &model.pooled_features(images.clone()),
        Tolerance::rel_abs(0.0, 0.0),
    );
    let model = model.with_classes(7);
    assert_eq!(model.forward(images).dims(), [1, 7]);
}

/// The parts must agree.
#[test]
fn test_disagreeing_parts_are_illegal() {
    let device = cpu_device();
    let illegal = |config: VisionEncoderStructureConfig, message: &str| {
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains(message)
            .assert_err(&ModuleInit::<VisionEncoderModel>::try_init(
                &config, &device,
            ));
    };
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.tokens.d_model = 6;
    illegal(config, "the prefix tokens is 6 wide");
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.tokens.n_patches = 9;
    illegal(config, "cover 9 patches; the grid has 16");
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.head.n_prefix = 2;
    illegal(config, "the head skips 2 prefix tokens; there are 1");
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.patch_embed = PatchEmbedConfig::new([18, 16], 4, 3, 8);
    illegal(config, "18x16 images don't divide into 4x4 patches");
}
