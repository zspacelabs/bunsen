use super::*;
use crate::{
    burner::module::ToStructureConfig,
    data::formats::nemo::NemoArchive,
    errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    },
    kits::speech::{
        diarization::sortformer::{
            SortformerV2Config,
            blocks::SortformerStructureConfig,
        },
        support::nemo::{
            front_end::FeatureNormalization,
            interop::parse_nemo_yaml,
        },
    },
    support::testing::testdata,
};

fn toy_yaml() -> String {
    let path = testdata!("nemo/toy_sortformer_v2.nemo");
    NemoArchive::open(&path).unwrap().config_yaml().to_string()
}

/// The toy's geometry, as a policy.
fn toy_v2() -> SortformerV2Config {
    SortformerV2Config::new()
        .with_n_mels(24)
        .with_d_model(16)
        .with_n_layers(2)
        .with_n_heads(2)
        .with_ff_expansion(2)
        .with_subsampling_channels(4)
        .with_conv_kernel_size(3)
        .with_d_hidden(8)
        .with_transformer_layers(2)
        .with_transformer_heads(2)
        .with_transformer_ff_expansion(2)
        .with_n_speakers(3)
}

fn json(config: &SortformerStructureConfig) -> serde_json::Value {
    serde_json::to_value(config).unwrap()
}

/// The toy `.nemo`'s config is the structure its policy lowers to, behind
/// `NeMo`'s front end without normalization.
#[test]
fn test_toy_config() {
    let config: NemoSortformerMirror = parse_nemo_yaml(&toy_yaml()).unwrap();
    assert_eq!(config.sortformer_modules.num_spks, 3);
    assert_eq!(config.sortformer_modules.subsampling_factor, 8);
    assert_eq!(
        json(&config.sortformer_structure_config().unwrap()),
        json(&toy_v2().to_structure())
    );
    let front_end = config.mel_front_end_config().unwrap();
    assert_eq!(front_end.n_mels, 24);
    assert_eq!(front_end.normalize, FeatureNormalization::None);
    assert!(!front_end.bf16_tables);
}

/// A config of another model is refused naming its class, before the
/// sections it lacks; one that is not YAML is an invalid resource.
#[test]
fn test_from_yaml() {
    assert_eq!(
        NemoSortformerMirror::from_yaml(&toy_yaml()).unwrap(),
        parse_nemo_yaml::<NemoSortformerMirror>(&toy_yaml()).unwrap()
    );
    let err = NemoSortformerMirror::from_yaml(
        "target: nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel\n",
    )
    .unwrap_err();
    // {err}
    ErrorMatcher::unsupported()
        .display_contains("EncDecRNNTBPEModel")
        .assert(&err);
    assert_eq!(
        NemoSortformerMirror::from_yaml("target: [")
            .unwrap_err()
            .kind(),
        BunsenErrorKind::InvalidResource
    );
}

/// A model bunsen's Sortformer is not is refused, naming the field.
#[test]
fn test_unsupported_settings() {
    let yaml = toy_yaml();
    for (from, to, field) in [
        (
            "target: nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel",
            "target: nemo.collections.asr.models.EncDecCTCModel",
            "`target`",
        ),
        (
            "  fc_d_model: 16\n  tf_d_model: 8\n  chunk",
            "  fc_d_model: 32\n  tf_d_model: 8\n  chunk",
            "sortformer_modules.fc_d_model",
        ),
        (
            "  fc_d_model: 16\n  tf_d_model: 8\n  chunk",
            "  fc_d_model: 16\n  tf_d_model: 4\n  chunk",
            "sortformer_modules.tf_d_model",
        ),
        (
            "  num_spks: 3\n",
            "  num_spks: 3\n  subsampling_factor: 4\n",
            "sortformer_modules.subsampling_factor",
        ),
        (
            "  num_spks: 3\n",
            "  num_spks: 3\n  upsample_factor: 8\n",
            "sortformer_modules.upsample_factor",
        ),
        (
            "  num_spks: 3\n",
            "  num_spks: 3\n  use_learnable_sil_emb: true\n",
            "sortformer_modules.use_learnable_sil_emb",
        ),
        (
            "  num_spks: 3\n",
            "  num_spks: 3\n  use_activity_head: true\n",
            "sortformer_modules.use_activity_head",
        ),
        (
            "  features: 24\n",
            "  features: 32\n",
            "preprocessor.features",
        ),
        (
            "  pre_ln: false\n",
            "  pre_ln: true\n",
            "transformer_encoder.pre_ln_final_layer_norm",
        ),
        (
            "  subsampling: dw_striding\n",
            "  subsampling: striding\n",
            "encoder.subsampling",
        ),
    ] {
        assert_eq!(yaml.matches(from).count(), 1, "{from}");
        let config: NemoSortformerMirror = parse_nemo_yaml(&yaml.replacen(from, to, 1)).unwrap();
        let err = config.sortformer_structure_config().unwrap_err();
        // {field}: {err}
        ErrorMatcher::unsupported().assert(&err);
        assert!(err.to_string().contains(field), "{field}: {err}");
    }
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
            Some("encoder.transformer.layers.16.attn.query.bias"),
        ),
        ("encoder.layers.0.conv.batch_norm.num_batches_tracked", None),
        (
            "transformer_encoder.layers.17.first_sub_layer.out_projection.weight",
            Some("transformer.layers.17.attn.c_proj.weight"),
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
            sortformer_v2_rekey(&crate::blocks::signals::DwStridingSubsamplingConfig::new(
                128, 256, 512
            ))
            .apply(nemo)
            .as_deref(),
            bunsen,
            "{nemo}"
        );
    }
}
