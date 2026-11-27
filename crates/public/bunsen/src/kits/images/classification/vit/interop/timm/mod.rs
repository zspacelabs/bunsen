//! # `ViT` as timm publishes it
//!
//! A timm hub repo's `ViT` (`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`):
//! its `config.json` names the architecture, which is one of the kit's
//! prefabs, and its `model.safetensors` holds timm's `VisionTransformer`
//! state dict.
//!
//! - **Conversion:** [`vit_contract_config`], a timm `config.json` as the
//!   [`ViTContractConfig`] it builds.
//! - **Names:** [`vit_row_splits`] (the fused `qkv`, split) and [`vit_rekey`],
//!   onto the model's canonical names; for a gated MLP's model,
//!   [`vit_gated_row_splits`] and [`vit_gated_rekey`].
//! - **Loading** (feature `store`): `load_vit_tensors`; with `cache`,
//!   `load_vit`, a resource set's checkpoint into a model it builds, and
//!   `load_vit_into`, into one a family built.

use alloc::format;

use crate::{
    blocks::transformers::models::vision_encoder::TokenPoolConfig,
    burner::module::{
        Rekey,
        TensorLoadRowSplit,
    },
    data::pretrained::Prefabs,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::{
        classification::{
            resnetv2::interop::timm::resnetv2_rekey,
            vit::ViTContractConfig,
        },
        support::interop::timm::TimmModelConfigMirror,
    },
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// The [`ViTContractConfig`] a timm `config.json` describes: the prefab of
/// its architecture's name (`vit_base_patch16_224`, …), with the config's
/// classes, pool and size ([`with_timm_config`]).
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of; as [`with_timm_config`].
pub fn vit_contract_config(config: &TimmModelConfigMirror) -> BunsenResult<ViTContractConfig> {
    let architecture = config.architecture.as_str();
    let contract = ViTContractConfig::prefab(architecture).ok_or_else(|| {
        BunsenError::unsupported(format!(
            "timm's {architecture}: the vit kit builds {}",
            ViTContractConfig::prefab_names().join(", ")
        ))
    })?;
    with_timm_config(contract, config)
}

/// `contract`, the geometry of a timm `config.json`'s architecture, with
/// the config's class count (0 for a headless tower), its global pool, and
/// its input size, the size the position embedding is learnt for.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a pool
/// the head doesn't build (`prr`, `''`), or an input that isn't square or
/// has other channels than the architecture's.
pub fn with_timm_config(
    mut contract: ViTContractConfig,
    config: &TimmModelConfigMirror,
) -> BunsenResult<ViTContractConfig> {
    let architecture = config.architecture.as_str();
    if let Some(pool) = config.global_pool.as_deref() {
        contract.global_pool = TokenPoolConfig::from_timm(pool).ok_or_else(|| {
            BunsenError::unsupported(format!(
                "timm's {architecture} with a {pool:?} pool: the vit kit pools by token, avg, \
                 max, avgmax or map"
            ))
        })?;
    }
    let [channels, height, width] = config.pretrained_cfg.input_size;
    if height != width || channels != contract.in_channels {
        return Err(BunsenError::unsupported(format!(
            "timm's {architecture} at {channels}x{height}x{width}: the vit kit takes square \
             images of {} channels",
            contract.in_channels
        )));
    }
    contract.image_size = height;
    contract.num_classes = config.num_classes;
    Ok(contract)
}

/// Each layer's fused `qkv` projection, `[3 d, d]` (and its bias), split by
/// rows into `q`, `k` and `v`, and the attention pool's fused `kv`,
/// `[2 d, d]`, into `k` and `v`, in timm's names.
pub fn vit_row_splits() -> [TensorLoadRowSplit; 4] {
    [
        TensorLoadRowSplit::equal(
            ".attn.qkv.weight",
            [".attn.q.weight", ".attn.k.weight", ".attn.v.weight"],
        ),
        TensorLoadRowSplit::equal(
            ".attn.qkv.bias",
            [".attn.q.bias", ".attn.k.bias", ".attn.v.bias"],
        ),
        TensorLoadRowSplit::equal(
            "attn_pool.kv.weight",
            ["attn_pool.k.weight", "attn_pool.v.weight"],
        ),
        TensorLoadRowSplit::equal(
            "attn_pool.kv.bias",
            ["attn_pool.k.bias", "attn_pool.v.bias"],
        ),
    ]
}

/// [`vit_row_splits`], with a packed `SwiGLU`'s `fc1`, `[2 h, d]` (and its
/// bias), split by rows into its gate and its input, `fc1_g` and `fc1_x` (the
/// gate first, as timm's `SwiGLUPacked` takes it): the splits of a model
/// whose MLP is gated, where a `SwiGLU`'s `fc1_g` and `fc1_x` come apart
/// already.
pub fn vit_gated_row_splits() -> [TensorLoadRowSplit; 6] {
    let [qkv_weight, qkv_bias, kv_weight, kv_bias] = vit_row_splits();
    [
        qkv_weight,
        qkv_bias,
        kv_weight,
        kv_bias,
        TensorLoadRowSplit::equal(
            ".mlp.fc1.weight",
            [".mlp.fc1_g.weight", ".mlp.fc1_x.weight"],
        ),
        TensorLoadRowSplit::equal(".mlp.fc1.bias", [".mlp.fc1_g.bias", ".mlp.fc1_x.bias"]),
    ]
}

/// [`vit_rekey`] for a model whose MLP is gated (`AIMv2`'s `SwiGLU`, the
/// giant `DINOv2`'s packed one, after [`vit_gated_row_splits`]): a block's
/// `mlp.fc1_g`, `mlp.fc1_x` and `mlp.fc2` are its `mlp.gate_proj`,
/// `mlp.up_proj` and `mlp.down_proj`.
///
/// ```rust
/// use bunsen::kits::images::classification::vit::interop::timm::vit_gated_rekey;
///
/// assert_eq!(
///     vit_gated_rekey().apply("blocks.3.mlp.fc2.weight").as_deref(),
///     Some("stack.layers.3.mlp.down_proj.weight")
/// );
/// ```
pub fn vit_gated_rekey() -> Rekey {
    vit_rekey_with(&[
        ("mlp.fc1_g.", "mlp.gate_proj."),
        ("mlp.fc1_x.", "mlp.up_proj."),
        ("mlp.fc2.", "mlp.down_proj."),
    ])
}

/// timm's `VisionTransformer` names, after [`vit_row_splits`], onto a
/// [`VisionEncoderModel`](crate::blocks::transformers::models::vision_encoder::VisionEncoderModel)'s
/// canonical ones.
///
/// | timm | bunsen |
/// |---|---|
/// | `patch_embed.proj` | `patch_embed.projection` |
/// | a hybrid's `patch_embed.backbone`, a non-preact `ResNetV2` | `backbone.net`, under [`resnetv2_rekey`] |
/// | `cls_token`, `reg_token`, `pos_embed` | `tokens.cls_token`, `tokens.reg_token`, `tokens.pos_embed` |
/// | a distilled `ViT`'s `dist_token`, `head_dist` (`DeiT`'s) | `tokens.dist_token`, `head.fc_dist` |
/// | `norm_pre` | `stack.input_norm` |
/// | `blocks.{i}` | `stack.layers.{i}` |
/// | a block's `norm1`, `norm2` | its `attn_norm`, `mlp_norm` |
/// | a block's `attn.q`, `attn.k`, `attn.v`, `attn.proj` | its `attn.c_q`, `attn.c_k`, `attn.c_v`, `attn.c_proj` |
/// | a block's `ls1.gamma`, `ls2.gamma` | its `attn_layer_scale`, `mlp_layer_scale` |
/// | a block's `mlp.fc1`, `mlp.fc2` | its `mlp.linear1`, `mlp.linear2` |
/// | a block's `attn.norm`, `mlp.norm` (`BEiT3`'s sub-LN) | its `attn.output_norm`, `mlp.hidden_norm` |
/// | `norm` | `stack.norm` |
/// | `fc_norm`, `head` | `head.fc_norm`, `head.fc` |
/// | `attn_pool.latent`, `attn_pool.norm` | `pool.latent.latent`, `pool.latent.norm` |
/// | `attn_pool.q`, `attn_pool.k`, `attn_pool.v`, `attn_pool.proj` | `pool.latent.attn.query`, `.key`, `.value`, `.output` |
/// | `attn_pool.mlp.fc1`, `attn_pool.mlp.fc2` | `pool.latent.mlp.linear1`, `.linear2` |
///
/// Anything else is kept as it is, for the load to report.
///
/// ```rust
/// use bunsen::kits::images::classification::vit::interop::timm::vit_rekey;
///
/// assert_eq!(
///     vit_rekey().apply("blocks.3.mlp.fc2.bias").as_deref(),
///     Some("stack.layers.3.mlp.linear2.bias")
/// );
/// ```
pub fn vit_rekey() -> Rekey {
    vit_rekey_with(&[("mlp.fc1.", "mlp.linear1."), ("mlp.fc2.", "mlp.linear2.")])
}

/// [`vit_rekey`], a block's MLP renamed by `mlp`'s `(timm, bunsen)` prefixes.
fn vit_rekey_with(mlp: &[(&str, &str)]) -> Rekey {
    let block = Rekey::keep_rest()
        .rename("norm1.", "attn_norm.")
        .rename("norm2.", "mlp_norm.")
        .rename("attn.q.", "attn.c_q.")
        .rename("attn.k.", "attn.c_k.")
        .rename("attn.v.", "attn.c_v.")
        .rename("attn.proj.", "attn.c_proj.")
        .rename("attn.norm.", "attn.output_norm.")
        .rename("mlp.norm.", "mlp.hidden_norm.")
        .exact("ls1.gamma", "attn_layer_scale")
        .exact("ls2.gamma", "mlp_layer_scale");
    let block = mlp
        .iter()
        .fold(block, |block, &(from, to)| block.rename(from, to));
    let attn_pool = Rekey::keep_rest()
        .rename("q.", "attn.query.")
        .rename("k.", "attn.key.")
        .rename("v.", "attn.value.")
        .rename("proj.", "attn.output.")
        .rename("mlp.fc1.", "mlp.linear1.")
        .rename("mlp.fc2.", "mlp.linear2.");
    Rekey::keep_rest()
        .mount("attn_pool.", "pool.latent.", attn_pool)
        .mount(
            "patch_embed.backbone.",
            "backbone.net.",
            resnetv2_rekey(0, 0, false),
        )
        .rename("patch_embed.proj.", "patch_embed.projection.")
        .exact("cls_token", "tokens.cls_token")
        .exact("reg_token", "tokens.reg_token")
        .exact("dist_token", "tokens.dist_token")
        .rename("head_dist.", "head.fc_dist.")
        .exact("pos_embed", "tokens.pos_embed")
        .rename("norm_pre.", "stack.input_norm.")
        .rename("fc_norm.", "head.fc_norm.")
        .rename("norm.", "stack.norm.")
        .rename("head.", "head.fc.")
        .mount_indexed("blocks.", "stack.layers.", block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        kits::images::support::interop::timm::timm_model_config_from_str,
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
            assert_eq!(
                format!(
                    "{:?}",
                    vit_contract_config(&pooled_config(name, 21843, want.image_size, pool))
                        .unwrap()
                ),
                format!("{want:?}"),
                "{name}"
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
        for (timm, bunsen) in [
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
        ] {
            assert_eq!(rekey.apply(timm).as_deref(), Some(bunsen), "{timm}");
        }
    }
}
