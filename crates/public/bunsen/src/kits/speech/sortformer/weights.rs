//! Loading `transformers`-named tensors into a [`Sortformer`].
use burn::store::burn_pack::Tensor as PackTensor;

use super::blocks::Sortformer;
use crate::{
    data::pretrained::{
        NamedTensorsApplied,
        apply_named_tensors,
    },
    errors::BunsenResult,
};

/// The bunsen module path of one `transformers`
/// `Nemotron3DiarizationForAudioFrameClassification` state-dict name.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`apply_named_tensors`] resolves to `gamma` / `beta`.
pub fn nemotron_hf_to_bunsen_path(name: &str) -> Option<String> {
    if name == "silence_embeds" {
        return Some("silence".into());
    }
    if let Some(rest) = name.strip_prefix("model.audio_tower.") {
        if let Some(rest) = rest.strip_prefix("embedder.projection.") {
            return Some(format!("encoder.stacking.projection.{rest}"));
        }
        if let Some(rest) = rest.strip_prefix("input_layer_norm.") {
            return Some(format!("encoder.encoder.embed_norm.{rest}"));
        }
        if let Some(rest) = rest.strip_prefix("layer_norm.") {
            return Some(format!("encoder.encoder.final_norm.{rest}"));
        }
        let rest = rest.strip_prefix("layers.")?;
        let (layer, rest) = rest.split_once('.')?;
        const RENAMES: &[(&str, &str)] = &[
            ("layer_norm1.", "norm1."),
            ("layer_norm2.", "norm2."),
            ("self_attn.q_proj.", "attn.query."),
            ("self_attn.k_proj.", "attn.key."),
            ("self_attn.v_proj.", "attn.value."),
            ("self_attn.o_proj.", "attn.output."),
            ("mlp.fc1.", "mlp.linear1."),
            ("mlp.fc2.", "mlp.linear2."),
        ];
        let (from, to) = RENAMES.iter().find(|(from, _)| rest.starts_with(from))?;
        return Some(format!(
            "encoder.encoder.layers.{layer}.{to}{}",
            &rest[from.len()..]
        ));
    }
    const PREFIXES: &[(&str, &str)] = &[
        ("model.proj.", "projection."),
        ("model.upsampler.conv.", "upsample.conv."),
        ("classifier.dense.", "head.dense."),
        ("classifier.out_proj.", "head.out."),
    ];
    let (from, to) = PREFIXES.iter().find(|(from, _)| name.starts_with(from))?;
    Some(format!("{to}{}", &name[from.len()..]))
}

/// Loads tensors named as `transformers`'
/// `Nemotron3DiarizationForAudioFrameClassification` state dict into
/// `model`: [`apply_named_tensors`] under [`nemotron_hf_to_bunsen_path`].
///
/// # Errors
/// As [`apply_named_tensors`].
pub fn load_nemotron_hf_tensors(
    model: &mut Sortformer,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    apply_named_tensors(model, tensors, nemotron_hf_to_bunsen_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paths() {
        for (hf, bunsen) in [
            ("silence_embeds", "silence"),
            (
                "model.audio_tower.embedder.projection.weight",
                "encoder.stacking.projection.weight",
            ),
            (
                "model.audio_tower.input_layer_norm.bias",
                "encoder.encoder.embed_norm.bias",
            ),
            (
                "model.audio_tower.layer_norm.weight",
                "encoder.encoder.final_norm.weight",
            ),
            (
                "model.audio_tower.layers.30.self_attn.o_proj.bias",
                "encoder.encoder.layers.30.attn.output.bias",
            ),
            (
                "model.audio_tower.layers.0.mlp.fc2.weight",
                "encoder.encoder.layers.0.mlp.linear2.weight",
            ),
            ("model.upsampler.conv.weight", "upsample.conv.weight"),
            ("classifier.out_proj.bias", "head.out.bias"),
        ] {
            assert_eq!(
                nemotron_hf_to_bunsen_path(hf).as_deref(),
                Some(bunsen),
                "{hf}"
            );
        }
        assert_eq!(
            nemotron_hf_to_bunsen_path("model.audio_tower.rotary_emb.inv_freq"),
            None
        );
    }
}
