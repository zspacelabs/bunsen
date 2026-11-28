//! The config conversion.

use std::path::Path;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::diarization::sortformer::blocks::NemotronDiarizationConfig,
        support::transformers::interop::{
            TransformersConfigJson,
            config_from_path,
            config_from_str,
        },
    },
};

/// The [`NemotronDiarizationConfig`] of a `transformers`
/// `Nemotron3DiarizationForAudioFrameClassification` `config.json`, parsed:
/// what an `hf:` repo's geometry is.
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
pub fn nemotron_diarization_config(
    c: &TransformersConfigJson
) -> BunsenResult<NemotronDiarizationConfig> {
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
        .with_rope_base(theta)
        .with_d_hidden(h.usize("hidden_size")?)
        .with_n_speakers(h.usize("num_speakers")?))
}

/// [`nemotron_diarization_config`] of a `config.json`'s text.
///
/// # Errors
/// As [`config_from_str`] and [`nemotron_diarization_config`].
pub fn nemotron_diarization_config_from_str(json: &str) -> BunsenResult<NemotronDiarizationConfig> {
    config_from_str(json, nemotron_diarization_config)
}

/// [`nemotron_diarization_config`] of the `config.json` at `path`; every
/// error names the file.
///
/// # Errors
/// As [`config_from_path`] and [`nemotron_diarization_config`].
pub fn nemotron_diarization_config_from_path(
    path: &Path
) -> BunsenResult<NemotronDiarizationConfig> {
    config_from_path(path, nemotron_diarization_config)
}
