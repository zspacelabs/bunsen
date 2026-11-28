//! The state-dict names.

use crate::burner::module::{
    Rekey,
    TensorLoadRowSplit,
};

/// Each block's fused `qkv` projection split by rows into `q`, `k` and `v`,
/// in timm's names: the weight, `[3 d, d]`, and V1's bias, `[3 d]` (V2
/// keeps its query and value biases apart, `q_bias` and `v_bias`).
pub fn swin_row_splits() -> [TensorLoadRowSplit; 2] {
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

/// timm's Swin names, V1's and V2's, after [`swin_row_splits`], onto a
/// [`StagedVisionEncoderModel`](crate::blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel)'s
/// canonical ones.
///
/// | timm | bunsen |
/// |---|---|
/// | `patch_embed.proj`, `patch_embed.norm` | `patch_embed.projection`, `patch_embed.norm` |
/// | `layers.{i}.downsample` | `stages.{i}.downsample` |
/// | `layers.{i}.blocks.{j}` | `stages.{i}.stack.layers.{j}` |
/// | a block's `norm1`, `norm2` (V1's pre-norms, V2's res-post-norms) | its `attn_norm`, `mlp_norm` |
/// | a block's `attn.q`, `attn.k`, `attn.v`, `attn.proj` | its `attn.c_q`, `attn.c_k`, `attn.c_v`, `attn.c_proj` |
/// | a V1 block's `attn.relative_position_bias_table` | its `attn.bias.table` |
/// | a V2 block's `attn.q_bias`, `attn.v_bias` | its `attn.c_q.bias`, `attn.c_v.bias` |
/// | a V2 block's `attn.cpb_mlp.0`, `attn.cpb_mlp.2` | its `attn.bias.linear1`, `attn.bias.linear2` |
/// | a V2 block's `attn.logit_scale` | its `attn.logit_scale` |
/// | a block's `mlp.fc1`, `mlp.fc2` | its `mlp.linear1`, `mlp.linear2` |
/// | `norm`, `head.fc` | `norm`, `head.fc` |
///
/// Anything else is kept as it is, for the load to report.
///
/// ```rust
/// use bunsen::kits::images::classification::swin::interop::timm::swin_rekey;
///
/// assert_eq!(
///     swin_rekey()
///         .apply("layers.2.blocks.5.attn.cpb_mlp.0.weight")
///         .as_deref(),
///     Some("stages.2.stack.layers.5.attn.bias.linear1.weight")
/// );
/// ```
pub fn swin_rekey() -> Rekey {
    let block = Rekey::keep_rest()
        .rename("norm1.", "attn_norm.")
        .rename("norm2.", "mlp_norm.")
        .rename("attn.q.", "attn.c_q.")
        .rename("attn.k.", "attn.c_k.")
        .rename("attn.v.", "attn.c_v.")
        .exact("attn.q_bias", "attn.c_q.bias")
        .exact("attn.v_bias", "attn.c_v.bias")
        .exact("attn.relative_position_bias_table", "attn.bias.table")
        .rename("attn.proj.", "attn.c_proj.")
        .rename("attn.cpb_mlp.0.", "attn.bias.linear1.")
        .rename("attn.cpb_mlp.2.", "attn.bias.linear2.")
        .rename("mlp.fc1.", "mlp.linear1.")
        .rename("mlp.fc2.", "mlp.linear2.");
    let stage = Rekey::keep_rest().mount_indexed("blocks.", "stack.layers.", block);
    Rekey::keep_rest()
        .rename("patch_embed.proj.", "patch_embed.projection.")
        .mount_indexed("layers.", "stages.", stage)
}
