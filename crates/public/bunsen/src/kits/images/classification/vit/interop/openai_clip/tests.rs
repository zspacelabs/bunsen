use super::*;
use crate::burner::module::testing::assert_rekeys;

/// `OpenAI` CLIP's names onto timm's, one of each kind; the text tower
/// dropped.
#[test]
fn test_openai_clip_rekey() {
    let rekey = openai_clip_rekey(16, 17);
    assert_rekeys(
        &rekey,
        [
            ("visual.conv1.weight", Some("patch_embed.proj.weight")),
            ("visual.class_embedding", Some("cls_token")),
            ("visual.positional_embedding", Some("pos_embed")),
            ("visual.proj", Some("head.weight")),
            ("visual.head.bias", Some("head.bias")),
            ("visual.ln_pre.bias", Some("norm_pre.bias")),
            ("visual.ln_post.weight", Some("norm.weight")),
            (
                "visual.transformer.resblocks.0.ln_1.weight",
                Some("blocks.0.norm1.weight"),
            ),
            (
                "visual.transformer.resblocks.1.attn.q.bias",
                Some("blocks.1.attn.q.bias"),
            ),
            (
                "visual.transformer.resblocks.1.attn.out_proj.weight",
                Some("blocks.1.attn.proj.weight"),
            ),
            (
                "visual.transformer.resblocks.1.mlp.c_proj.bias",
                Some("blocks.1.mlp.fc2.bias"),
            ),
            ("text_projection", None),
            ("transformer.resblocks.0.ln_1.weight", None),
        ],
    );
    assert!(is_openai_clip(["logit_scale", OPENAI_CLIP_MARKER]));
    assert!(!is_openai_clip(["cls_token"]));
}
