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
            models::classifier_head::ClassifierHeadConfig,
            patching::{
                patch_embed::PatchEmbedConfig,
                patch_merge::PatchMergeConfig,
            },
        },
        mlp::BunsenMlpConfig,
        norm::{
            BunsenLayerNormConfig,
            BunsenNormalizationConfig,
        },
        transformers::{
            attention::window::{
                RelativePositionBiasConfig,
                WindowAttentionConfig,
            },
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::UniformStackConfig,
            },
            embedding::RotaryEmbeddingConfig,
            sublayer::NormPlacement,
        },
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        ToStructureConfig,
    },
    errors::testing::ErrorMatcher,
    support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    },
};

fn norm() -> BunsenNormalizationConfig {
    BunsenNormalizationConfig::Layer(BunsenLayerNormConfig::new(0))
}

/// A stage's stack: `depth` Swin V2 blocks over `grid`, every second
/// shifted, res-post-norm.
fn stage_stack(
    d: usize,
    heads: usize,
    grid: [usize; 2],
    depth: usize,
) -> crate::blocks::transformers::decoders::stack::DecoderStackStructureConfig {
    let mut stack = UniformStackConfig::new(
        depth,
        CausalDecoderBlockConfig::new(
            WindowAttentionConfig::new(d, heads, grid, [4, 4])
                .with_key_bias(false)
                .with_cosine(true)
                .with_bias(RelativePositionBiasConfig::cpb(None))
                .into(),
            BunsenMlpConfig::expansion(d, 4)
                .with_activation(BunsenActivationConfig::Gelu)
                .with_bias(true)
                .into(),
        )
        .with_norm(norm())
        .with_norm_placement(NormPlacement::ResPost),
        BunsenNormalizationConfig::Identity,
        RotaryEmbeddingConfig::unrotated(d / heads),
    )
    .to_structure();
    for (j, layer) in stack.layers.iter_mut().enumerate() {
        if let crate::blocks::transformers::mixers::TokenMixerConfig::Window(w) = &mut layer.attn {
            w.shifted = j % 2 == 1;
        }
    }
    stack
}

/// A tiny Swin V2: 32x32 images in 4x4 patches (an 8x8 grid), 8 wide; a
/// stage of two blocks, then a merge to a 4x4 grid 16 wide and two more; 5
/// classes.
fn tiny() -> StagedVisionEncoderStructureConfig {
    StagedVisionEncoderStructureConfig::new(
        PatchEmbedConfig::new([32, 32], 4, 3, 8).with_norm(Some(norm())),
        alloc::vec![
            VisionStageConfig::new(stage_stack(8, 2, [8, 8], 2)),
            VisionStageConfig::new(stage_stack(16, 4, [4, 4], 2))
                .with_downsample(Some(PatchMergeConfig::new([8, 8], 8, 16, norm()))),
        ],
        norm(),
        ClassifierHeadConfig::new(16, 5),
    )
}

/// The steps' shapes, and the meta of the config and the model agree.
#[test]
fn test_shapes_and_meta() {
    let device = cpu_device();
    let config = tiny();
    let model: StagedVisionEncoderModel = config.init(&device);
    assert_eq!(config.stage_grids(), [[8, 8], [4, 4]]);
    assert_eq!(model.stage_grids(), config.stage_grids());
    assert_eq!(model.stage_widths(), [8, 16]);
    assert_eq!(model.d_features(), 16);
    assert_eq!(model.num_classes(), 5);
    let images: Tensor<4> = seeded_tensor(1, [2, 3, 32, 32], Distribution::Default, &device);
    assert_eq!(model.forward(images.clone()).dims(), [2, 5]);
    assert_eq!(model.feature_tokens(images.clone()).dims(), [2, 16, 16]);
    let maps = model.stage_maps(images.clone());
    assert_eq!(maps[0].dims(), [2, 8, 8, 8]);
    assert_eq!(maps[1].dims(), [2, 16, 4, 4]);
    assert_tensors_close(
        &model.head.forward(model.pooled_features(images.clone())),
        &model.forward(images),
        Tolerance::rel_abs(1e-6, 1e-6),
    );
}

/// A res-post-norm block holds its norms in the `post` slots, and its
/// canonical `attn_norm` / `mlp_norm` names land there.
#[test]
fn test_res_post_norm_blocks() {
    let device = cpu_device();
    let model: StagedVisionEncoderModel = tiny().init(&device);
    let block = &model.stages[0].stack.layers[0];
    assert!(block.attn.pre.is_none() && block.attn.out.is_none() && block.attn.post.is_some());
    assert!(block.mlp.pre.is_none() && block.mlp.post.is_some());
    let rekey = model.canonical_rekey();
    assert_eq!(
        rekey
            .apply("stages.1.stack.layers.0.attn_norm.gamma")
            .as_deref(),
        Some("stages.1.stack.layers.0.attn.post.gamma")
    );
    assert_eq!(
        rekey
            .apply("stages.0.stack.layers.1.mlp_norm.beta")
            .as_deref(),
        Some("stages.0.stack.layers.1.mlp.post.beta")
    );
}

/// The parts must agree.
#[test]
fn test_disagreeing_parts_are_illegal() {
    let device = cpu_device();
    let illegal = |config: StagedVisionEncoderStructureConfig, message: &str| {
        ErrorMatcher::illegal()
            .message_contains(message)
            .assert_err(&ModuleInit::<StagedVisionEncoderModel>::try_init(
                &config, &device,
            ));
    };
    let mut config = tiny();
    config.stages[1].downsample = Some(PatchMergeConfig::new([4, 4], 8, 16, norm()));
    illegal(config, "stage 1's merge is over a [4, 4] grid");
    let mut config = tiny();
    config.stages[1].stack = stage_stack(16, 4, [8, 8], 2);
    illegal(config, "attends over a [8, 8] grid; the stage's is [4, 4]");
    let mut config = tiny();
    config.head = ClassifierHeadConfig::new(8, 5);
    illegal(config, "the head reads 8 features");
    let mut config = tiny();
    config.stages.clear();
    illegal(config, "no stages");
}
