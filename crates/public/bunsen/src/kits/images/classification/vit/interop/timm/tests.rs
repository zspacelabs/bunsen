use alloc::format;

use super::*;
use crate::{
    blocks::transformers::models::vision_encoder::TokenPoolConfig,
    burner::module::testing::assert_rekeys,
    data::pretrained::Prefabs,
    errors::testing::ErrorMatcher,
    kits::images::{
        classification::vit::ViTContractConfig,
        support::timm::interop::{
            TimmModelConfigMirror,
            timm_model_config_from_str,
        },
    },
    support::testing::assert_config_eq,
};

/// A timm config of `architecture` with `num_classes` classes at `size`,
/// pooled by its class token.
fn config(
    architecture: &str,
    num_classes: usize,
    size: usize,
) -> TimmModelConfigMirror {
    pooled_config(architecture, num_classes, size, "token")
}

/// A timm config of `architecture` with `num_classes` classes at `size`,
/// pooled by `pool`.
fn pooled_config(
    architecture: &str,
    num_classes: usize,
    size: usize,
    pool: &str,
) -> TimmModelConfigMirror {
    timm_model_config_from_str(&format!(
        r#"{{"architecture": "{architecture}", "num_classes": {num_classes},
            "num_features": 768, "global_pool": "{pool}", "pretrained_cfg": {{
            "input_size": [3, {size}, {size}], "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]}}}}"#
    ))
    .unwrap()
}

/// Each prefab is a timm architecture of its name; the classes and the
/// size are the config's.
#[test]
fn test_prefab_architectures_map() {
    for name in ViTContractConfig::prefab_names() {
        let mut want = ViTContractConfig::expect_prefab(name);
        want.num_classes = 21843;
        let pool = match want.global_pool {
            TokenPoolConfig::Token => "token",
            TokenPoolConfig::Avg => "avg",
            TokenPoolConfig::Map => "map",
            other => panic!("{name}: a {other:?} prefab"),
        };
        assert_config_eq(
            name,
            &vit_contract_config(&pooled_config(name, 21843, want.image_size, pool)).unwrap(),
            &want,
        );
    }
    let headless = vit_contract_config(&config("vit_base_patch16_224", 0, 224)).unwrap();
    assert_eq!(headless.num_classes, 0);
    let large = vit_contract_config(&config("vit_base_patch16_clip_224", 512, 448)).unwrap();
    assert_eq!(large.image_size, 448);
}

#[test]
fn test_unbuilt_configs_are_unsupported() {
    let unsupported = || ErrorMatcher::unsupported();
    unsupported()
        .message_contains("timm's vit_giant_patch14_224")
        .assert_err(&vit_contract_config(&config(
            "vit_giant_patch14_224",
            1000,
            224,
        )));
    let mut prr = config("vit_base_patch16_224", 1000, 224);
    prr.global_pool = Some("prr".into());
    unsupported()
        .message_contains("\"prr\" pool")
        .assert_err(&vit_contract_config(&prr));
    let mut wide = config("vit_base_patch16_224", 1000, 224);
    wide.pretrained_cfg.input_size = [3, 224, 256];
    unsupported()
        .message_contains("square images")
        .assert_err(&vit_contract_config(&wide));
}

/// timm's names onto the model's canonical ones, one of each kind.
#[test]
fn test_vit_rekey() {
    let rekey = vit_rekey();
    assert_rekeys(
        &rekey,
        [
            ("patch_embed.proj.weight", "patch_embed.projection.weight"),
            (
                "patch_embed.backbone.stem.conv.weight",
                "backbone.net.stem.convs.0.conv.weight",
            ),
            (
                "patch_embed.backbone.stages.2.blocks.8.downsample.norm.bias",
                "backbone.net.stages.2.units.8.shortcut.norm.bias",
            ),
            (
                "patch_embed.backbone.norm.weight",
                "backbone.net.stem.convs.0.norm.weight",
            ),
            ("cls_token", "tokens.cls_token"),
            ("reg_token", "tokens.reg_token"),
            ("pos_embed", "tokens.pos_embed"),
            ("norm_pre.weight", "stack.input_norm.weight"),
            ("blocks.0.norm1.weight", "stack.layers.0.attn_norm.weight"),
            ("blocks.0.attn.q.bias", "stack.layers.0.attn.c_q.bias"),
            ("blocks.1.attn.v.weight", "stack.layers.1.attn.c_v.weight"),
            ("blocks.1.attn.proj.bias", "stack.layers.1.attn.c_proj.bias"),
            ("blocks.2.ls1.gamma", "stack.layers.2.attn_layer_scale"),
            ("blocks.2.ls2.gamma", "stack.layers.2.mlp_layer_scale"),
            ("blocks.11.norm2.bias", "stack.layers.11.mlp_norm.bias"),
            (
                "blocks.11.mlp.fc1.weight",
                "stack.layers.11.mlp.linear1.weight",
            ),
            ("norm.weight", "stack.norm.weight"),
            ("fc_norm.bias", "head.fc_norm.bias"),
            ("head.weight", "head.fc.weight"),
            ("dist_token", "tokens.dist_token"),
            ("head_dist.bias", "head.fc_dist.bias"),
        ],
    );
}
