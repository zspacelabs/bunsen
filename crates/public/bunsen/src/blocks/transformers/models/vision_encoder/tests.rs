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
        images::{
            models::{
                classifier_head::ClassifierHeadConfig,
                conv_net::{
                    ConvLayersConfig,
                    ConvNetContractConfig,
                    ConvStemContractConfig,
                    ConvStemPoolKind,
                },
            },
            patching::{
                patch_embed::PatchEmbedConfig,
                prefix_tokens::PrefixTokensConfig,
            },
        },
        mlp::BunsenMlpConfig,
        norm::{
            BunsenLayerNormConfig,
            BunsenNormalizationConfig,
        },
        pooling::LatentAttentionPoolConfig,
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
    errors::testing::ErrorMatcher,
    support::testing::{
        IDENTICAL,
        ROUNDING,
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
        PatchEmbedConfig::new([16, 16], 4, 3, d),
        tokens,
        stack,
        ClassifierHeadConfig::new(d, 5).with_fc_norm(fc_norm.then(|| norm(d))),
    )
    .with_pool(pool)
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
        .encoder_forward_layers(model.embed(images.clone()), None);
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
        IDENTICAL,
    );
    let model = model.with_classes(7);
    assert_eq!(model.forward(images).dims(), [1, 7]);
}

/// The parts must agree.
#[test]
fn test_disagreeing_parts_are_illegal() {
    let device = cpu_device();
    let illegal = |config: VisionEncoderStructureConfig, message: &str| {
        ErrorMatcher::illegal()
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
    config.pool = TokenPoolConfig::Distilled;
    illegal(config, "reads 2 prefix tokens: there are 1");
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.head.distilled = true;
    illegal(config, "a Token pool with a head that is distilled");
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.patch_embed = PatchEmbedConfig::new([18, 16], 4, 3, 8);
    illegal(config, "18x16 images don't divide into 4x4 patches");

    let latent =
        |d: usize| LatentAttentionPoolConfig::new(d, 2, BunsenMlpConfig::new(d, 4 * d, d), norm(d));
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.pool = TokenPoolConfig::Map;
    illegal(config, "a Map pool with no attention pool");
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.latent_pool = Some(latent(8));
    illegal(config, "a Token pool with an attention pool");
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.pool = TokenPoolConfig::Map;
    config.latent_pool = Some(latent(6));
    illegal(config, "the attention pool is 6 wide");
}

/// A [`Map`](TokenPoolConfig::Map) pool attends over the patch tokens: its
/// pool is one vector per image, the model's width.
#[test]
fn test_map_pool() {
    let device = cpu_device();
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.pool = TokenPoolConfig::Map;
    config.latent_pool = Some(LatentAttentionPoolConfig::new(
        8,
        2,
        BunsenMlpConfig::new(8, 32, 8),
        norm(8),
    ));
    let model: VisionEncoderModel = config.init(&device);
    assert!(model.pool.latent.is_some());
    let images = Tensor::<4>::random([2, 3, 16, 16], Distribution::Default, &device);
    assert_eq!(model.pooled_features(images.clone()).dims(), [2, 8]);
    assert_eq!(model.forward(images).dims(), [2, 5]);
}

/// A hybrid: a conv backbone (a SAME-padded stem alone, stride 4) whose
/// 6-channel map the patch embedding cuts, in 2x2 patches. The model takes
/// the backbone's images; a map that is not the patch embedding's input is
/// refused, as is a backbone with a head.
#[test]
fn test_backbone() {
    let net = |classes: usize| {
        ConvNetContractConfig::new(
            ConvStemContractConfig::new(vec![6]).with_pool(ConvStemPoolKind::SameMax),
            vec![],
            classes,
        )
        .with_layers(ConvLayersConfig::new().with_same_pad(true))
        .to_structure()
    };
    let mut config = tiny(TokenPoolConfig::Token, 0);
    config.patch_embed = PatchEmbedConfig::new([8, 8], 2, 6, 8);
    config.backbone = Some(VisionBackboneConfig::new(net(0), [30, 32]));
    assert_eq!(config.in_channels(), 3);
    assert_eq!(config.image_size(), [30, 32]);
    assert_eq!(config.grid(), [4, 4]);
    let device = cpu_device();
    let model: VisionEncoderModel = config.init(&device);
    assert_eq!(model.image_size(), [30, 32]);
    assert_eq!(model.in_channels(), 3);
    let images: Tensor<4> = seeded_tensor(3, [2, 3, 30, 32], Distribution::Default, &device);
    assert_eq!(model.forward(images.clone()).dims(), [2, 5]);
    // The backbone's map is the patch embedding's input.
    let backbone = model.backbone.as_ref().unwrap();
    let tokens = model.patch_embed.forward(backbone.forward(images.clone()));
    assert_tensors_close(
        &model.embed(images),
        &model.tokens.forward(tokens),
        ROUNDING,
    );

    let illegal = |config: VisionEncoderStructureConfig, message: &str| {
        let built: crate::errors::BunsenResult<VisionEncoderModel> = config.try_init(&device);
        ErrorMatcher::illegal()
            .message_contains(message)
            .assert_err(&built);
    };
    let mut wrong_size = config.clone();
    wrong_size.backbone = Some(VisionBackboneConfig::new(net(0), [16, 16]));
    illegal(
        wrong_size,
        "the backbone's map is 6 channels at [4, 4]; the patch embedding takes 6 at [8, 8]",
    );
    let mut headed = config;
    headed.backbone = Some(VisionBackboneConfig::new(net(5), [30, 32]));
    illegal(headed, "the backbone has a head of 5 classes");
}
