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
//!   onto the model's canonical names.
//! - **Loading** (feature `store`): `load_vit_tensors`; with `cache`,
//!   `load_vit`, a resource set's safetensors checkpoint.

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
        classification::vit::ViTContractConfig,
        support::interop::timm::TimmModelConfigMirror,
    },
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// The [`ViTContractConfig`] a timm `config.json` describes: the prefab of
/// its architecture's name (`vit_base_patch16_224`, …), with its class count
/// (0 for a headless tower), its global pool, and its input size, the size
/// the position embedding is learnt for.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
/// architecture the kit has no prefab of, a pool the head doesn't build
/// (`map`, `''`), or an input that isn't square or has other channels than
/// the architecture's.
pub fn vit_contract_config(config: &TimmModelConfigMirror) -> BunsenResult<ViTContractConfig> {
    let architecture = config.architecture.as_str();
    let mut contract = ViTContractConfig::prefab(architecture).ok_or_else(|| {
        BunsenError::unsupported(format!(
            "timm's {architecture}: the vit kit builds {}",
            ViTContractConfig::prefab_names().join(", ")
        ))
    })?;
    if let Some(pool) = config.global_pool.as_deref() {
        contract.global_pool = TokenPoolConfig::from_timm(pool).ok_or_else(|| {
            BunsenError::unsupported(format!(
                "timm's {architecture} with a {pool:?} pool: the vit kit pools by token, avg, \
                 max or avgmax"
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
/// rows into `q`, `k` and `v`, in timm's names.
pub fn vit_row_splits() -> [TensorLoadRowSplit; 2] {
    [
        TensorLoadRowSplit::equal(
            ".attn.qkv.weight",
            [".attn.q.weight", ".attn.k.weight", ".attn.v.weight"],
        ),
        TensorLoadRowSplit::equal(
            ".attn.qkv.bias",
            [".attn.q.bias", ".attn.k.bias", ".attn.v.bias"],
        ),
    ]
}

/// timm's `VisionTransformer` names, after [`vit_row_splits`], onto a
/// [`VisionEncoderModel`](crate::blocks::transformers::models::vision_encoder::VisionEncoderModel)'s
/// canonical ones.
///
/// | timm | bunsen |
/// |---|---|
/// | `patch_embed.proj` | `patch_embed.projection` |
/// | `cls_token`, `reg_token`, `pos_embed` | `tokens.cls_token`, `tokens.reg_token`, `tokens.pos_embed` |
/// | `norm_pre` | `stack.input_norm` |
/// | `blocks.{i}` | `stack.layers.{i}` |
/// | a block's `norm1`, `norm2` | its `attn_norm`, `mlp_norm` |
/// | a block's `attn.q`, `attn.k`, `attn.v`, `attn.proj` | its `attn.c_q`, `attn.c_k`, `attn.c_v`, `attn.c_proj` |
/// | a block's `ls1.gamma`, `ls2.gamma` | its `attn_layer_scale`, `mlp_layer_scale` |
/// | a block's `mlp.fc1`, `mlp.fc2` | its `mlp.linear1`, `mlp.linear2` |
/// | `norm` | `stack.norm` |
/// | `fc_norm`, `head` | `head.fc_norm`, `head.fc` |
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
    let block = Rekey::keep_rest()
        .rename("norm1.", "attn_norm.")
        .rename("norm2.", "mlp_norm.")
        .rename("attn.q.", "attn.c_q.")
        .rename("attn.k.", "attn.c_k.")
        .rename("attn.v.", "attn.c_v.")
        .rename("attn.proj.", "attn.c_proj.")
        .exact("ls1.gamma", "attn_layer_scale")
        .exact("ls2.gamma", "mlp_layer_scale")
        .rename("mlp.fc1.", "mlp.linear1.")
        .rename("mlp.fc2.", "mlp.linear2.");
    Rekey::keep_rest()
        .rename("patch_embed.proj.", "patch_embed.projection.")
        .exact("cls_token", "tokens.cls_token")
        .exact("reg_token", "tokens.reg_token")
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
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        kits::images::support::interop::timm::timm_model_config_from_str,
    };

    /// A timm config of `architecture` with `num_classes` classes at `size`.
    fn config(
        architecture: &str,
        num_classes: usize,
        size: usize,
    ) -> TimmModelConfigMirror {
        timm_model_config_from_str(&format!(
            r#"{{"architecture": "{architecture}", "num_classes": {num_classes},
                "num_features": 768, "global_pool": "token", "pretrained_cfg": {{
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
            assert_eq!(
                format!(
                    "{:?}",
                    vit_contract_config(&config(name, 21843, want.image_size)).unwrap()
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
        let unsupported = || ErrorMatcher::kind(BunsenErrorKind::Unsupported);
        unsupported()
            .message_contains("timm's vit_giant_patch14_224")
            .assert_err(&vit_contract_config(&config(
                "vit_giant_patch14_224",
                1000,
                224,
            )));
        let mut mapped = config("vit_base_patch16_224", 1000, 224);
        mapped.global_pool = Some("map".into());
        unsupported()
            .message_contains("\"map\" pool")
            .assert_err(&vit_contract_config(&mapped));
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
        ] {
            assert_eq!(rekey.apply(timm).as_deref(), Some(bunsen), "{timm}");
        }
    }
}
