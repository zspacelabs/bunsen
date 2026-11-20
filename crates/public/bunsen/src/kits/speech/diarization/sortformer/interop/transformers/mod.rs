//! # Sortformer in `transformers`' names
//!
//! A `Nemotron3DiarizationForAudioFrameClassification` checkpoint as
//! bunsen's Nemotron
//! [`Sortformer`](crate::kits::speech::diarization::sortformer::Sortformer):
//!
//! - **Config:** [`nemotron_diarization_config`] of a parsed `config.json`, and
//!   [`nemotron_diarization_config_from_str`] /
//!   [`nemotron_diarization_config_from_path`].
//! - **Names:** [`nemotron_diarization_rekey`], from [`embedder_rekey`] and
//!   [`encoder_rekey`].
//! - **Loading** (feature `store`): `load_nemotron_diarization_tensors`; with
//!   `cache`, `load_nemotron_diarization`, a resource set's checkpoint.
use std::path::Path;

use crate::{
    blocks::signals::{
        FeatureStackingMeta,
        FeatureStackingOrder,
    },
    burner::module::Rekey,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::diarization::sortformer::blocks::NemotronDiarizationConfig,
        support::interop::transformers::{
            TransformersConfigJson,
            config_from_path,
            config_from_str,
        },
    },
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// `transformers`' `Nemotron3DiarizationEmbedder` names onto a
/// [`FeatureStacking`](crate::blocks::signals::FeatureStacking)'s, relative
/// to it: `projection.*`, its weight permuted when `stacking` stacks
/// otherwise than `transformers` does (frames outer). Anything else is
/// dropped.
pub fn embedder_rekey(stacking: &impl FeatureStackingMeta) -> Rekey {
    stacking.restack_from(
        FeatureStackingOrder::FramesFeatures,
        Rekey::drop_rest().rename("projection.", "projection."),
    )
}

/// `transformers`' `Nemotron3DiarizationEncoderLayer` names onto a
/// [`CausalDecoderBlock`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock)'s,
/// relative to it: `layer_norm{1,2}` its norms, `self_attn`'s
/// `{q,k,v,o}_proj` the attention, `mlp.fc{1,2}` the MLP. Anything else is
/// dropped.
pub fn encoder_layer_rekey() -> Rekey {
    Rekey::drop_rest()
        .rename("layer_norm1.", "attn_norm.")
        .rename("layer_norm2.", "mlp_norm.")
        .rename("self_attn.q_proj.", "attn.c_q.")
        .rename("self_attn.k_proj.", "attn.c_k.")
        .rename("self_attn.v_proj.", "attn.c_v.")
        .rename("self_attn.o_proj.", "attn.c_proj.")
        .rename("mlp.fc1.", "mlp.linear1.")
        .rename("mlp.fc2.", "mlp.linear2.")
}

/// `transformers`' `Nemotron3DiarizationEncoder` names onto a
/// [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)'s,
/// relative to it: `input_layer_norm` its input norm, `layer_norm` its final
/// one, each `layers.{n}` by [`encoder_layer_rekey`]. Anything else (the
/// rotary table's `inv_freq`, a buffer) is dropped.
pub fn encoder_rekey() -> Rekey {
    Rekey::drop_rest()
        .rename("input_layer_norm.", "input_norm.")
        .rename("layer_norm.", "norm.")
        .mount_indexed("layers.", "layers.", encoder_layer_rekey())
}

/// `transformers`' `Nemotron3DiarizationForAudioFrameClassification`
/// state-dict names onto a
/// [`Sortformer`](crate::kits::speech::diarization::sortformer::Sortformer)'s
/// module paths: the audio tower's embedder by [`embedder_rekey`], its encoder
/// by [`encoder_rekey`], the projection, upsample and head by prefix. Anything
/// else is dropped.
///
/// `stacking` is the encoder's front end, a config or a built module: its
/// stack order fixes its map.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) resolves to
/// `gamma` / `beta`.
pub fn nemotron_diarization_rekey(stacking: &impl FeatureStackingMeta) -> Rekey {
    Rekey::drop_rest()
        .exact("silence_embeds", "silence")
        .mount(
            "model.audio_tower.embedder.",
            "encoder.subsampling.",
            embedder_rekey(stacking),
        )
        .mount(
            "model.audio_tower.",
            "encoder.transformer.",
            encoder_rekey(),
        )
        .rename("model.proj.", "projection.")
        .rename("model.upsampler.conv.", "upsample.conv.")
        .rename("classifier.dense.", "head.dense.")
        .rename("classifier.out_proj.", "head.out.")
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paths() {
        for (hf, bunsen) in [
            ("silence_embeds", "silence"),
            (
                "model.audio_tower.embedder.projection.weight",
                "encoder.subsampling.projection.weight",
            ),
            (
                "model.audio_tower.input_layer_norm.bias",
                "encoder.transformer.input_norm.bias",
            ),
            (
                "model.audio_tower.layer_norm.weight",
                "encoder.transformer.norm.weight",
            ),
            (
                "model.audio_tower.layers.30.self_attn.o_proj.bias",
                "encoder.transformer.layers.30.attn.c_proj.bias",
            ),
            (
                "model.audio_tower.layers.0.mlp.fc2.weight",
                "encoder.transformer.layers.0.mlp.linear2.weight",
            ),
            ("model.upsampler.conv.weight", "upsample.conv.weight"),
            ("classifier.out_proj.bias", "head.out.bias"),
        ] {
            assert_eq!(
                nemotron_diarization_rekey(&crate::blocks::signals::FeatureStackingConfig::new(
                    128, 512
                ))
                .apply(hf)
                .as_deref(),
                Some(bunsen),
                "{hf}"
            );
        }
        assert_eq!(
            nemotron_diarization_rekey(&crate::blocks::signals::FeatureStackingConfig::new(
                128, 512
            ))
            .apply("model.audio_tower.rotary_emb.inv_freq"),
            None
        );
    }

    /// `Nemotron-3-Diarization`'s own `config.json` is its geometry.
    #[test]
    fn test_nemotron_diarization_config_from_str() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/sortformer/nemotron_config.json"
        ));
        let config = nemotron_diarization_config_from_str(json).unwrap();
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
        assert!(nemotron_diarization_config_from_str(&other).is_err());
        let relu = json.replace("\"gelu\"", "\"relu\"");
        assert!(nemotron_diarization_config_from_str(&relu).is_err());
    }
}
