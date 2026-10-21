//! Loading `transformers`- and `NeMo`-named tensors into a [`Sortformer`].
use burn::store::burn_pack::Tensor as PackTensor;

use super::blocks::{
    NemotronDiarizationConfig,
    Sortformer,
    SortformerMeta,
};
use crate::{
    data::pretrained::{
        HfConfigJson,
        NamedTensorsApplied,
        apply_named_tensors,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::nemo::{
        nemo_conformer_to_bunsen_path,
        nemo_transformer_to_bunsen_path,
    },
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

/// The bunsen module path of one `NeMo` `SortformerEncLabelModel` state-dict
/// name in Sortformer v2's layout (a `FastConformer`, `encoder_proj`, a
/// `transformer_encoder`, the speaker head), or `None` for an entry the model
/// has no parameter for: `num_batches_tracked`, the featurizer's tables,
/// and the unused `hidden_to_spks`.
///
/// `n_stages` is the encoder's subsampling stage count
/// ([`nemo_conformer_to_bunsen_path`]).
pub fn sortformer_v2_nemo_to_bunsen_path(
    name: &str,
    n_stages: usize,
) -> Option<String> {
    if let Some(rest) = name.strip_prefix("encoder.") {
        return nemo_conformer_to_bunsen_path(rest, n_stages).map(|p| format!("encoder.{p}"));
    }
    if let Some(rest) = name.strip_prefix("transformer_encoder.") {
        return nemo_transformer_to_bunsen_path(rest).map(|p| format!("transformer.{p}"));
    }
    const PREFIXES: &[(&str, &str)] = &[
        ("sortformer_modules.encoder_proj.", "projection."),
        ("sortformer_modules.first_hidden_to_hidden.", "head.dense."),
        ("sortformer_modules.single_hidden_to_spks.", "head.out."),
    ];
    let (from, to) = PREFIXES.iter().find(|(from, _)| name.starts_with(from))?;
    Some(format!("{to}{}", &name[from.len()..]))
}

/// Loads tensors named as a `NeMo` Sortformer v2 state dict (a `.nemo`'s
/// `model_weights.ckpt`) into `model`: [`apply_named_tensors`] under
/// [`sortformer_v2_nemo_to_bunsen_path`], at `model`'s subsampling.
///
/// # Errors
/// As [`apply_named_tensors`].
pub fn load_sortformer_v2_nemo_tensors(
    model: &mut Sortformer,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let n_stages = model.subsampling_factor().trailing_zeros() as usize;
    apply_named_tensors(model, tensors, |name| {
        sortformer_v2_nemo_to_bunsen_path(name, n_stages)
    })
}

/// The [`NemotronDiarizationConfig`] of a `transformers`
/// `Nemotron3DiarizationForAudioFrameClassification` `config.json`: what an
/// `hf:` repo's geometry is.
///
/// The encoder's and the head's sizes come from the file. What bunsen's model
/// fixes is checked: the model type, a GELU feed-forward of a whole multiple
/// of the width, rotary embeddings over the whole head, an integer base, and
/// the head reading the encoder's width at the encoder's subsampling.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Sortformer is not.
pub fn nemotron_config_from_hf(json: &str) -> BunsenResult<NemotronDiarizationConfig> {
    let c = HfConfigJson::parse(json)?;
    c.expect_str("model_type", "nemotron3_diarization")?;
    let a = c.section("audio_config")?;
    a.expect_str("hidden_act", "gelu")?;
    let d_model = a.usize("hidden_size")?;
    let intermediate = a.usize("intermediate_size")?;
    if d_model == 0 || !intermediate.is_multiple_of(d_model) {
        return Err(BunsenError::unsupported(format!(
            "config.json: `audio_config.intermediate_size` {intermediate} is not a whole \
             multiple of `hidden_size` {d_model}"
        )));
    }
    let rope = a.section("rope_parameters")?;
    a.expect_eq(
        "partial_rotary_factor",
        a.f64("partial_rotary_factor")?,
        1.0,
    )?;
    let theta = rope.f64("rope_theta")?;
    if theta.fract() != 0.0 || theta < 1.0 {
        return Err(BunsenError::unsupported(format!(
            "config.json: `audio_config.rope_parameters.rope_theta` {theta} is not a whole number"
        )));
    }
    let factor = a.usize("subsampling_factor")?;

    let h = c.section("head_config")?;
    h.expect_eq("audio_hidden_size", h.usize("audio_hidden_size")?, d_model)?;
    h.expect_eq("subsampling_factor", h.usize("subsampling_factor")?, factor)?;

    Ok(NemotronDiarizationConfig::new()
        .with_n_mels(a.usize("num_mel_bins")?)
        .with_d_model(d_model)
        .with_n_layers(a.usize("num_hidden_layers")?)
        .with_n_heads(a.usize("num_attention_heads")?)
        .with_ff_expansion(intermediate / d_model)
        .with_subsampling_factor(factor)
        .with_rope_base(theta as usize)
        .with_d_hidden(h.usize("hidden_size")?)
        .with_n_speakers(h.usize("num_speakers")?))
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

    #[test]
    fn test_v2_paths() {
        for (nemo, bunsen) in [
            (
                "encoder.pre_encode.conv.0.weight",
                Some("encoder.subsampling.stem.conv.weight"),
            ),
            (
                "encoder.layers.16.self_attn.linear_q.bias",
                Some("encoder.layers.16.self_attn.query.bias"),
            ),
            ("encoder.layers.0.conv.batch_norm.num_batches_tracked", None),
            (
                "transformer_encoder.layers.17.first_sub_layer.out_projection.weight",
                Some("transformer.layers.17.attn.output.weight"),
            ),
            (
                "sortformer_modules.encoder_proj.weight",
                Some("projection.weight"),
            ),
            (
                "sortformer_modules.first_hidden_to_hidden.bias",
                Some("head.dense.bias"),
            ),
            (
                "sortformer_modules.single_hidden_to_spks.weight",
                Some("head.out.weight"),
            ),
            ("sortformer_modules.hidden_to_spks.weight", None),
            ("preprocessor.featurizer.fb", None),
        ] {
            assert_eq!(
                sortformer_v2_nemo_to_bunsen_path(nemo, 3).as_deref(),
                bunsen,
                "{nemo}"
            );
        }
    }

    /// `Nemotron-3-Diarization`'s own `config.json` is its geometry.
    #[test]
    fn test_nemotron_config_from_hf() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/sortformer/nemotron_config.json"
        ));
        let config = nemotron_config_from_hf(json).unwrap();
        let want = NemotronDiarizationConfig::nemotron_3_diarization();
        assert_eq!(config.n_mels, want.n_mels);
        assert_eq!(config.d_model, want.d_model);
        assert_eq!(config.n_layers, want.n_layers);
        assert_eq!(config.n_heads, want.n_heads);
        assert_eq!(config.ff_expansion, want.ff_expansion);
        assert_eq!(config.subsampling_factor, want.subsampling_factor);
        assert_eq!(config.rope_base, want.rope_base);
        assert_eq!(config.d_hidden, want.d_hidden);
        assert_eq!(config.n_speakers, want.n_speakers);

        let other = json.replace("\"nemotron3_diarization\"", "\"something_else\"");
        assert!(nemotron_config_from_hf(&other).is_err());
        let relu = json.replace("\"gelu\"", "\"relu\"");
        assert!(nemotron_config_from_hf(&relu).is_err());
    }
}
