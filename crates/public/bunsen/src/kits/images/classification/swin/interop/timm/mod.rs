//! # Swin as timm publishes it
//!
//! timm's `SwinTransformer` (V1) and `SwinTransformerV2` state dicts, each
//! a hub repo's `model.safetensors`, with the patch merges at the start of
//! the stages they open:
//!
//! - **Names:** [`swin_row_splits`] (the fused `qkv`, split) and
//!   [`swin_rekey`], onto the
//!   [`StagedVisionEncoderModel`](crate::blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel)'s
//!   canonical names.
//! - **Loading** (feature `store`): `load_swin_tensors`; with `cache`,
//!   `load_swin`, a resource set's checkpoint.

use alloc::format;

use crate::{
    blocks::transformers::models::vision_encoder::TokenPoolConfig,
    burner::module::{
        Rekey,
        TensorLoadRowSplit,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::images::support::interop::timm::TimmModelConfigMirror,
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

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

/// What a Swin kit takes from a timm `config.json` beside the architecture:
/// the side of its square input, and its pool, if it states one.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a pool
/// the head doesn't build (`token`), or an input that isn't square or has
/// other than `in_channels` channels; `kit` names the kit in the message.
pub(crate) fn swin_timm_input(
    config: &TimmModelConfigMirror,
    kit: &str,
    in_channels: usize,
) -> BunsenResult<(usize, Option<TokenPoolConfig>)> {
    let architecture = config.architecture.as_str();
    let pool = match config.global_pool.as_deref() {
        Some(pool) => Some(
            TokenPoolConfig::from_timm(pool)
                .filter(|p| *p != TokenPoolConfig::Token)
                .ok_or_else(|| {
                    BunsenError::unsupported(format!(
                        "timm's {architecture} with a {pool:?} pool: the {kit} kit pools by avg, \
                         max or avgmax"
                    ))
                })?,
        ),
        None => None,
    };
    let [channels, height, width] = config.pretrained_cfg.input_size;
    if height != width || channels != in_channels {
        return Err(BunsenError::unsupported(format!(
            "timm's {architecture} at {channels}x{height}x{width}: the {kit} kit takes square \
             images of {in_channels} channels"
        )));
    }
    Ok((height, pool))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// timm's names onto the model's canonical ones, one of each kind.
    #[test]
    fn test_swin_rekey() {
        let rekey = swin_rekey();
        for (timm, bunsen) in [
            ("patch_embed.proj.weight", "patch_embed.projection.weight"),
            ("patch_embed.norm.bias", "patch_embed.norm.bias"),
            (
                "layers.1.downsample.reduction.weight",
                "stages.1.downsample.reduction.weight",
            ),
            (
                "layers.1.downsample.norm.weight",
                "stages.1.downsample.norm.weight",
            ),
            (
                "layers.0.blocks.1.norm1.weight",
                "stages.0.stack.layers.1.attn_norm.weight",
            ),
            (
                "layers.0.blocks.1.norm2.bias",
                "stages.0.stack.layers.1.mlp_norm.bias",
            ),
            (
                "layers.0.blocks.0.attn.q.weight",
                "stages.0.stack.layers.0.attn.c_q.weight",
            ),
            (
                "layers.0.blocks.0.attn.k.bias",
                "stages.0.stack.layers.0.attn.c_k.bias",
            ),
            (
                "layers.0.blocks.0.attn.q_bias",
                "stages.0.stack.layers.0.attn.c_q.bias",
            ),
            (
                "layers.0.blocks.0.attn.v_bias",
                "stages.0.stack.layers.0.attn.c_v.bias",
            ),
            (
                "layers.2.blocks.3.attn.relative_position_bias_table",
                "stages.2.stack.layers.3.attn.bias.table",
            ),
            (
                "layers.3.blocks.0.attn.logit_scale",
                "stages.3.stack.layers.0.attn.logit_scale",
            ),
            (
                "layers.3.blocks.0.attn.cpb_mlp.2.weight",
                "stages.3.stack.layers.0.attn.bias.linear2.weight",
            ),
            (
                "layers.3.blocks.0.mlp.fc2.bias",
                "stages.3.stack.layers.0.mlp.linear2.bias",
            ),
            ("norm.weight", "norm.weight"),
            ("head.fc.weight", "head.fc.weight"),
        ] {
            assert_eq!(rekey.apply(timm).as_deref(), Some(bunsen), "{timm}");
        }
    }
}
