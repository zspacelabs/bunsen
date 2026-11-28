use super::*;
use crate::burner::module::testing::assert_rekeys;

/// timm's names onto the model's canonical ones, one of each kind.
#[test]
fn test_swin_rekey() {
    let rekey = swin_rekey();
    assert_rekeys(
        &rekey,
        [
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
        ],
    );
}
