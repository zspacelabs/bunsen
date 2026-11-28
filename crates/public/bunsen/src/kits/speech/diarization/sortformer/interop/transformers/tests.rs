use super::*;
use crate::{
    errors::testing::ErrorMatcher,
    kits::speech::diarization::sortformer::blocks::NemotronDiarizationConfig,
};

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
    ErrorMatcher::unsupported().assert_err(&nemotron_diarization_config_from_str(&other));
    let relu = json.replace("\"gelu\"", "\"relu\"");
    ErrorMatcher::unsupported().assert_err(&nemotron_diarization_config_from_str(&relu));
}
