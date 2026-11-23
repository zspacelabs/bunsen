//! # Sortformer as `NeMo` saves it
//!
//! A `.nemo` of a `SortformerEncLabelModel` (Sortformer v2 and v2.1) as
//! bunsen's [`Sortformer`](crate::kits::speech::diarization::sortformer::Sortformer): its `model_config.yaml`'s mirror,
//! [`NemoSortformerMirror`] (whose sections are the shared
//! [`support::interop::nemo`](crate::kits::support::interop::nemo)'s), the
//! configs it gives
//! ([`sortformer_structure_config`](NemoSortformerMirror::sortformer_structure_config),
//! [`mel_front_end_config`](NemoSortformerMirror::mel_front_end_config)),
//! and its state dict's names ([`sortformer_v2_rekey`]).
//!
//! - **Loading** (feature `store`): `load_sortformer_v2_tensors`; with `cache`,
//!   `load_sortformer_v2`, a resource set's archive. A `.nemo` carries its own
//!   config, so the loader reads it and returns the model's front end with the
//!   model.
use serde::Deserialize;

use crate::{
    blocks::{
        layout::BoundaryPermutes,
        signals::DwStridingSubsamplingMeta,
    },
    burner::module::{
        Rekey,
        ToStructureConfig,
    },
    errors::BunsenResult,
    kits::{
        speech::{
            diarization::sortformer::blocks::{
                SortformerHeadConfig,
                SortformerStructureConfig,
            },
            support::nemo::NemoMelFrontEndConfig,
        },
        support::interop::nemo::{
            NemoConformerEncoderMirror,
            NemoPreprocessorMirror,
            NemoTransformerEncoderMirror,
            conformer_encoder_rekey,
            parse_nemo_yaml,
            transformer_encoder_rekey,
            unsupported_field,
        },
    },
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// The `NeMo` class a Sortformer `.nemo` instantiates.
const SORTFORMER_TARGET: &str = "SortformerEncLabelModel";

/// The config of a `NeMo` `SortformerEncLabelModel` (a `.nemo`'s
/// `model_config.yaml`): the sections bunsen builds the model and its front
/// end from, each named as `NeMo` names it. Every other key is ignored.
///
/// [`sortformer_structure_config`](Self::sortformer_structure_config) is the
/// model, [`mel_front_end_config`](Self::mel_front_end_config) the front end
/// it was trained behind.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NemoSortformerMirror {
    /// The class `NeMo` instantiates.
    pub target: String,

    /// The mel front end.
    pub preprocessor: NemoPreprocessorMirror,

    /// The `FastConformer`.
    pub encoder: NemoConformerEncoderMirror,

    /// The transformer over the projected frames.
    pub transformer_encoder: NemoTransformerEncoderMirror,

    /// The projection, the head, and the streaming defaults.
    pub sortformer_modules: NemoSortformerModulesMirror,
}

/// Checks the model class is [`SORTFORMER_TARGET`].
fn expect_model(target: &str) -> BunsenResult<()> {
    if target == SORTFORMER_TARGET || target.ends_with(&format!(".{SORTFORMER_TARGET}")) {
        Ok(())
    } else {
        Err(unsupported_field(
            "target",
            target,
            &format!("not {SORTFORMER_TARGET}"),
        ))
    }
}

/// A `sortformer_modules` section: `SortformerModules`' arguments that shape
/// the model, named as `NeMo` names them and each defaulting as its
/// constructor defaults it. Its streaming arguments are not read: a stream's
/// geometry is a
/// [`SortformerStreamingConfig`](crate::kits::speech::diarization::sortformer::SortformerStreamingConfig) preset.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NemoSortformerModulesMirror {
    /// The class `NeMo` instantiates.
    #[serde(rename = "_target_")]
    pub target: String,

    /// The speakers.
    #[serde(default = "defaults::num_spks")]
    pub num_spks: usize,

    /// The encoder's width: what `encoder_proj` reads.
    #[serde(default = "defaults::fc_d_model")]
    pub fc_d_model: usize,

    /// The transformer's and the head's width.
    #[serde(default = "defaults::tf_d_model")]
    pub tf_d_model: usize,

    /// The mel frames per encoder frame.
    #[serde(default = "defaults::subsampling_factor")]
    pub subsampling_factor: usize,

    /// The output frames per encoder frame.
    #[serde(default = "defaults::upsample_factor")]
    pub upsample_factor: usize,

    /// Whether the speaker cache's silence slots hold a learned embedding,
    /// rather than the running mean of the silent frames.
    #[serde(default)]
    pub use_learnable_sil_emb: bool,

    /// Whether a second head classifies each frame's activity.
    #[serde(default)]
    pub use_activity_head: bool,
}

impl NemoSortformerMirror {
    /// Parses a `.nemo`'s config, its model class first: a config of another
    /// model is refused naming its class, not the first section it lacks.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the class, for another model; as [`parse_nemo_yaml`] for text that
    /// is not YAML or lacks a section.
    pub fn from_yaml(yaml: &str) -> BunsenResult<Self> {
        #[derive(Deserialize)]
        struct Target {
            target: String,
        }
        expect_model(&parse_nemo_yaml::<Target>(yaml)?.target)?;
        parse_nemo_yaml(yaml)
    }

    /// The [`SortformerStructureConfig`] the config describes: a
    /// `FastConformer` (the `encoder` section), `encoder_proj`, the
    /// transformer (the `transformer_encoder` section), and the speaker head,
    /// at the encoder's frame rate; the silence fill a running mean.
    ///
    /// # Errors
    /// As the sections' conversions
    /// ([`NemoConformerEncoderMirror::conformer_encoder_config`],
    /// [`NemoTransformerEncoderMirror::transformer_encoder_block_config`]);
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the field and its value, for another model class, sections that
    /// disagree on a width or on the subsampling, a transformer that ends
    /// with a norm, an upsampled output, a learned silence embedding, or an
    /// activity head.
    pub fn sortformer_structure_config(&self) -> BunsenResult<SortformerStructureConfig> {
        expect_model(&self.target)?;
        let modules = &self.sortformer_modules;
        if !modules.target.ends_with("SortformerModules") {
            return Err(unsupported_field(
                "sortformer_modules._target_",
                &modules.target,
                "not SortformerModules",
            ));
        }
        let (encoder, transformer) = (&self.encoder, &self.transformer_encoder);
        if self.preprocessor.features != encoder.feat_in {
            return Err(unsupported_field(
                "preprocessor.features",
                self.preprocessor.features,
                &format!("not encoder.feat_in {}", encoder.feat_in),
            ));
        }
        if modules.fc_d_model != encoder.d_model {
            return Err(unsupported_field(
                "sortformer_modules.fc_d_model",
                modules.fc_d_model,
                &format!("not encoder.d_model {}", encoder.d_model),
            ));
        }
        if modules.tf_d_model != transformer.hidden_size {
            return Err(unsupported_field(
                "sortformer_modules.tf_d_model",
                modules.tf_d_model,
                &format!(
                    "not transformer_encoder.hidden_size {}",
                    transformer.hidden_size
                ),
            ));
        }
        if modules.subsampling_factor != encoder.subsampling_factor {
            return Err(unsupported_field(
                "sortformer_modules.subsampling_factor",
                modules.subsampling_factor,
                &format!(
                    "not encoder.subsampling_factor {}",
                    encoder.subsampling_factor
                ),
            ));
        }
        if transformer.transformer_encoder_final_norm() {
            return Err(unsupported_field(
                "transformer_encoder.pre_ln_final_layer_norm",
                true,
                "a pre-norm stack's final norm is not built",
            ));
        }
        if modules.upsample_factor != 1 {
            return Err(unsupported_field(
                "sortformer_modules.upsample_factor",
                modules.upsample_factor,
                "only 1",
            ));
        }
        if modules.use_learnable_sil_emb {
            return Err(unsupported_field(
                "sortformer_modules.use_learnable_sil_emb",
                true,
                "only false",
            ));
        }
        if modules.use_activity_head {
            return Err(unsupported_field(
                "sortformer_modules.use_activity_head",
                true,
                "only false",
            ));
        }
        Ok(SortformerStructureConfig::new(
            encoder
                .conformer_encoder_config()?
                .try_to_structure()?
                .with_layout(BoundaryPermutes::SWAP_IN),
            SortformerHeadConfig::new(modules.tf_d_model, modules.num_spks),
        )
        .with_transformer(Some(transformer.transformer_encoder_config()?)))
    }

    /// The front end the model was trained behind: the `preprocessor`
    /// section's.
    ///
    /// # Errors
    /// As [`NemoPreprocessorMirror::mel_front_end_config`].
    pub fn mel_front_end_config(&self) -> BunsenResult<NemoMelFrontEndConfig> {
        self.preprocessor.mel_front_end_config()
    }
}

/// `NeMo` `SortformerEncLabelModel` state-dict names, in Sortformer v2's
/// layout, onto a
/// [`Sortformer`](crate::kits::speech::diarization::sortformer::Sortformer)'s
/// module paths: the `FastConformer` by [`conformer_encoder_rekey`], the
/// `transformer_encoder` by [`transformer_encoder_rekey`], the projection and
/// head by prefix. Anything else is dropped: `num_batches_tracked`, the
/// featurizer's tables, and the unused `hidden_to_spks`.
///
/// `striding` is the encoder's subsampling, a config or a built module: its
/// stage count and flatten order fix its map.
pub fn sortformer_v2_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    Rekey::drop_rest()
        .mount("encoder.", "encoder.", conformer_encoder_rekey(striding))
        .mount(
            "transformer_encoder.",
            "transformer.",
            transformer_encoder_rekey(),
        )
        .rename("sortformer_modules.encoder_proj.", "projection.")
        .rename("sortformer_modules.first_hidden_to_hidden.", "head.dense.")
        .rename("sortformer_modules.single_hidden_to_spks.", "head.out.")
}

/// `SortformerModules`' constructor defaults.
mod defaults {
    pub fn num_spks() -> usize {
        4
    }
    pub fn fc_d_model() -> usize {
        512
    }
    pub fn tf_d_model() -> usize {
        192
    }
    pub fn subsampling_factor() -> usize {
        8
    }
    pub fn upsample_factor() -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{
        burner::module::ToStructureConfig,
        data::formats::nemo::NemoArchive,
        errors::BunsenErrorKind,
        kits::speech::{
            diarization::sortformer::SortformerV2Config,
            support::nemo::FeatureNormalization,
        },
    };

    fn toy_yaml() -> String {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/nemo/toy_sortformer_v2.nemo");
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
        assert_eq!(err.kind(), BunsenErrorKind::Unsupported, "{err}");
        assert!(err.to_string().contains("EncDecRNNTBPEModel"), "{err}");
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
            let config: NemoSortformerMirror =
                parse_nemo_yaml(&yaml.replacen(from, to, 1)).unwrap();
            let err = config.sortformer_structure_config().unwrap_err();
            assert_eq!(err.kind(), BunsenErrorKind::Unsupported, "{field}: {err}");
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
}
