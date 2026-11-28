//! The state-dict names.

use crate::{
    burner::module::{
        Rekey,
        TensorLoadRowSplit,
    },
    kits::images::classification::resnetv2::interop::timm::resnetv2_rekey,
};

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
