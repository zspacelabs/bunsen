//! Loading `transformers`-named tensors into a [`Qwen3Asr`].
use burn::store::burn_pack::Tensor as PackTensor;

use super::blocks::Qwen3Asr;
use crate::{
    data::pretrained::{
        NamedTensorsApplied,
        apply_named_tensors_as,
    },
    errors::BunsenResult,
};

/// The bunsen module path of one `transformers`
/// `Qwen3ASRForConditionalGeneration` state-dict name, or `None` for an entry
/// that is not a parameter of its own (the tied `lm_head.weight`).
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`apply_named_tensors_as`] resolves to `gamma` / `beta`.
pub fn hf_to_bunsen_path(name: &str) -> Option<String> {
    let name = name.strip_prefix("model.")?;
    const RENAMES: &[(&str, &str)] = &[
        // The audio tower's layers are the shared encoder block's.
        (".self_attn.q_proj.", ".attn.query."),
        (".self_attn.k_proj.", ".attn.key."),
        (".self_attn.v_proj.", ".attn.value."),
        (".self_attn.out_proj.", ".attn.output."),
        (".self_attn_layer_norm.", ".attn_ln."),
        (".final_layer_norm.", ".mlp_ln."),
        (".fc1.", ".mlp.linear1."),
        (".fc2.", ".mlp.linear2."),
        (".conv2d1.", ".conv2d1.conv."),
        (".conv2d2.", ".conv2d2.conv."),
        (".conv2d3.", ".conv2d3.conv."),
    ];
    const TEXT_RENAMES: &[(&str, &str)] = &[
        // The text model's attention is the shared causal self-attention.
        (".self_attn.q_proj.", ".self_attn.c_q."),
        (".self_attn.k_proj.", ".self_attn.c_k."),
        (".self_attn.v_proj.", ".self_attn.c_v."),
        (".self_attn.o_proj.", ".self_attn.c_proj."),
    ];
    let renames = if name.starts_with("audio_tower.") {
        RENAMES
    } else if name.starts_with("language_model.") {
        TEXT_RENAMES
    } else {
        &[]
    };
    let mut path = name.to_string();
    for (from, to) in renames {
        if path.contains(from) {
            path = path.replacen(from, to, 1);
            break;
        }
    }
    Some(path)
}

/// Loads tensors named as `transformers`' `Qwen3ASRForConditionalGeneration`
/// state dict into `model`: [`apply_named_tensors_as`] under
/// [`hf_to_bunsen_path`], every floating tensor cast to the dtype `model` was
/// built in. The checkpoints ship `bf16`, which not every backend computes
/// in.
///
/// # Errors
/// As [`apply_named_tensors_as`].
pub fn load_hf_tensors(
    model: &mut Qwen3Asr,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let dtype = model.multi_modal_projector.linear_2.weight.val().dtype();
    apply_named_tensors_as(model, tensors, hf_to_bunsen_path, Some(dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_to_bunsen_path() {
        for (hf, bunsen) in [
            (
                "model.audio_tower.conv2d1.weight",
                Some("audio_tower.conv2d1.conv.weight"),
            ),
            (
                "model.audio_tower.conv_out.weight",
                Some("audio_tower.conv_out.weight"),
            ),
            (
                "model.audio_tower.layers.3.self_attn.k_proj.bias",
                Some("audio_tower.layers.3.attn.key.bias"),
            ),
            (
                "model.audio_tower.layers.3.self_attn.out_proj.weight",
                Some("audio_tower.layers.3.attn.output.weight"),
            ),
            (
                "model.audio_tower.layers.3.final_layer_norm.weight",
                Some("audio_tower.layers.3.mlp_ln.weight"),
            ),
            (
                "model.audio_tower.layers.3.fc2.bias",
                Some("audio_tower.layers.3.mlp.linear2.bias"),
            ),
            (
                "model.audio_tower.ln_post.weight",
                Some("audio_tower.ln_post.weight"),
            ),
            (
                "model.multi_modal_projector.linear_1.weight",
                Some("multi_modal_projector.linear_1.weight"),
            ),
            (
                "model.language_model.layers.0.self_attn.o_proj.weight",
                Some("language_model.layers.0.self_attn.c_proj.weight"),
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight",
                Some("language_model.layers.0.self_attn.q_norm.weight"),
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                Some("language_model.layers.0.mlp.gate_proj.weight"),
            ),
            (
                "model.language_model.norm.weight",
                Some("language_model.norm.weight"),
            ),
            ("lm_head.weight", None),
        ] {
            assert_eq!(hf_to_bunsen_path(hf).as_deref(), bunsen, "{hf}");
        }
    }
}
