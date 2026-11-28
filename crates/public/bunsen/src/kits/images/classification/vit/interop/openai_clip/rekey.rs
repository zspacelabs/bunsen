//! The state-dict names.

use crate::burner::module::{
    Rekey,
    Select,
    TensorLoadRowSplit,
    TensorLoadTransform,
};

/// The prefix of the image tower's tensors in an `OpenAI` CLIP checkpoint.
pub const OPENAI_CLIP_PREFIX: &str = "visual.";

/// The tensor only an `OpenAI` CLIP checkpoint has: the image tower's class
/// embedding.
pub const OPENAI_CLIP_MARKER: &str = "visual.class_embedding";

/// Whether a checkpoint of tensors named `names` is in `OpenAI` CLIP's
/// layout, as timm's `checkpoint_filter_fn` tells: it has
/// [`OPENAI_CLIP_MARKER`].
pub fn is_openai_clip<'a>(names: impl IntoIterator<Item = &'a str>) -> bool {
    names.into_iter().any(|name| name == OPENAI_CLIP_MARKER)
}

/// Each residual block's fused attention projection, `in_proj_weight`
/// `[3 d, d]` (and `in_proj_bias`), split by rows into timm's `q`, `k` and
/// `v`, in the checkpoint's names.
pub fn openai_clip_row_splits() -> [TensorLoadRowSplit; 2] {
    [
        TensorLoadRowSplit::equal(
            ".attn.in_proj_weight",
            [".attn.q.weight", ".attn.k.weight", ".attn.v.weight"],
        ),
        TensorLoadRowSplit::equal(
            ".attn.in_proj_bias",
            [".attn.q.bias", ".attn.k.bias", ".attn.v.bias"],
        ),
    ]
}

/// `OpenAI` CLIP's image-tower names, after [`openai_clip_row_splits`], onto
/// timm's `VisionTransformer` names, for a tower `d_model` wide whose
/// position embedding has `pos_embed_len` rows: timm's
/// `_convert_openai_clip`.
///
/// | `OpenAI` CLIP | timm |
/// |---|---|
/// | `visual.conv1` | `patch_embed.proj` |
/// | `visual.class_embedding`, `[d]` | `cls_token`, `[1, 1, d]` |
/// | `visual.positional_embedding`, `[len, d]` | `pos_embed`, `[1, len, d]` |
/// | `visual.ln_pre`, `visual.ln_post` | `norm_pre`, `norm` |
/// | `visual.transformer.resblocks.{i}` | `blocks.{i}` |
/// | a block's `ln_1`, `ln_2` | its `norm1`, `norm2` |
/// | a block's `attn.out_proj` | its `attn.proj` |
/// | a block's `mlp.c_fc`, `mlp.c_proj` | its `mlp.fc1`, `mlp.fc2` |
/// | `visual.proj`, `[d, classes]` | `head.weight`, `[classes, d]` |
///
/// The text tower, everything outside `visual.`, is dropped; anything else
/// under it is kept as it is, for the load to report.
///
/// ```rust
/// use bunsen::kits::images::classification::vit::interop::openai_clip::openai_clip_rekey;
///
/// let rekey = openai_clip_rekey(768, 50);
/// assert_eq!(
///     rekey.apply("visual.transformer.resblocks.2.mlp.c_fc.bias").as_deref(),
///     Some("blocks.2.mlp.fc1.bias")
/// );
/// assert_eq!(rekey.apply("text_projection"), None);
/// ```
pub fn openai_clip_rekey(
    d_model: usize,
    pos_embed_len: usize,
) -> Rekey {
    let block = Rekey::keep_rest()
        .rename("ln_1.", "norm1.")
        .rename("ln_2.", "norm2.")
        .rename("attn.out_proj.", "attn.proj.")
        .rename("mlp.c_fc.", "mlp.fc1.")
        .rename("mlp.c_proj.", "mlp.fc2.");
    let visual = Rekey::keep_rest()
        .rename("conv1.", "patch_embed.proj.")
        .exact("class_embedding", "cls_token")
        .exact("positional_embedding", "pos_embed")
        .exact("proj", "head.weight")
        .rename("ln_pre.", "norm_pre.")
        .rename("ln_post.", "norm.")
        .mount_indexed("transformer.resblocks.", "blocks.", block)
        .transform(
            Select::Exact("cls_token".into()),
            TensorLoadTransform::reshape([1, 1, d_model]),
        )
        .transform(
            Select::Exact("pos_embed".into()),
            TensorLoadTransform::reshape([1, pos_embed_len, d_model]),
        )
        .transform(
            Select::Exact("head.weight".into()),
            TensorLoadTransform::transpose(),
        );
    Rekey::drop_rest().mount(OPENAI_CLIP_PREFIX, "", visual)
}
