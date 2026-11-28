//! The config conversion.

use std::path::Path;

use crate::{
    errors::BunsenResult,
    kits::{
        speech::asr::voxtral::blocks::{
            VoxtralAudioEncoderConfig,
            VoxtralRealtimeConfig,
            VoxtralTextConfig,
        },
        support::interop::transformers::{
            TransformersConfigJson,
            TransformersDecoderSection,
            config_from_path,
            config_from_str,
        },
    },
};

/// The [`VoxtralRealtimeConfig`] of a parsed `transformers`
/// `VoxtralRealtime` `config.json`.
///
/// The audio encoder's and the text model's sizes, their windows, norm
/// epsilons and rotary bases, and the downsampling come from the file. What
/// bunsen's model fixes is checked: a GELU stem and a `SiLU` feed-forward in
/// both stacks, an audio encoder with as many key/value heads as query
/// heads, a projector with a GELU, and a head tied to the embedding.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Voxtral Realtime is not.
pub fn voxtral_realtime_config(c: &TransformersConfigJson) -> BunsenResult<VoxtralRealtimeConfig> {
    c.expect_str("model_type", "voxtral_realtime")?;
    c.expect_str("projector_hidden_act", "gelu")?;

    let a = c.section("audio_config")?;
    a.expect_str("activation_function", "gelu")?;
    let ad = TransformersDecoderSection::read(&a)?;
    a.expect_eq("hidden_act", ad.hidden_act.as_str(), "silu")?;
    a.expect_eq(
        "num_key_value_heads",
        ad.num_key_value_heads,
        ad.num_attention_heads,
    )?;
    let audio = VoxtralAudioEncoderConfig::new(
        a.usize("num_mel_bins")?,
        ad.hidden_size,
        ad.num_attention_heads,
        ad.head_dim,
        ad.intermediate_size,
        ad.num_hidden_layers,
    )
    .with_rms_eps(ad.rms_norm_eps)
    .with_rope_theta(ad.rope_theta)
    .with_sliding_window(ad.sliding_window);

    let t = c.section("text_config")?;
    let td = TransformersDecoderSection::read(&t)?;
    t.expect_eq("hidden_act", td.hidden_act.as_str(), "silu")?;
    t.expect_eq(
        "tie_word_embeddings",
        t.bool_or(
            "tie_word_embeddings",
            c.bool_or("tie_word_embeddings", true)?,
        )?,
        true,
    )?;
    let text = VoxtralTextConfig::new(
        t.usize("vocab_size")?,
        td.hidden_size,
        td.num_hidden_layers,
        td.num_attention_heads,
        td.num_key_value_heads,
        td.head_dim,
        td.intermediate_size,
    )
    .with_rms_eps(td.rms_norm_eps)
    .with_rope_theta(td.rope_theta)
    .with_sliding_window(td.sliding_window);

    Ok(VoxtralRealtimeConfig::new(audio, text).with_downsample(c.usize("downsample_factor")?))
}

/// [`voxtral_realtime_config`] of a `config.json`'s text.
///
/// # Errors
/// As [`config_from_str`] and [`voxtral_realtime_config`].
pub fn voxtral_realtime_config_from_str(json: &str) -> BunsenResult<VoxtralRealtimeConfig> {
    config_from_str(json, voxtral_realtime_config)
}

/// [`voxtral_realtime_config`] of the `config.json` at `path`; every error
/// names the file.
///
/// # Errors
/// As [`config_from_path`] and [`voxtral_realtime_config`].
pub fn voxtral_realtime_config_from_path(path: &Path) -> BunsenResult<VoxtralRealtimeConfig> {
    config_from_path(path, voxtral_realtime_config)
}
