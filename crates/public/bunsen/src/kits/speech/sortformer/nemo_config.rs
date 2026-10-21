//! A Sortformer `.nemo`'s config, as bunsen's.
use serde::Deserialize;

use super::blocks::{
    SortformerEncoderConfig,
    SortformerHeadConfig,
    SortformerStructureConfig,
    SortformerTransformerConfig,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::nemo::{
        NemoConformerEncoderConfig,
        NemoMelFrontEndConfig,
        NemoPreprocessorConfig,
        NemoTransformerEncoderConfig,
    },
};

/// The `NeMo` class a Sortformer `.nemo` instantiates.
const SORTFORMER_TARGET: &str = "SortformerEncLabelModel";

/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value.
fn unsupported(
    field: &str,
    value: impl std::fmt::Debug,
    why: &str,
) -> BunsenError {
    BunsenError::unsupported(format!("NeMo config: `{field}` {value:?}: {why}"))
}

/// The config of a `NeMo` `SortformerEncLabelModel` (a `.nemo`'s
/// `model_config.yaml`): the sections bunsen builds the model and its front
/// end from, each named as `NeMo` names it. Every other key is ignored.
///
/// [`structure`](Self::structure) is the model,
/// [`front_end`](Self::front_end) the front end it was trained behind.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NemoSortformerConfig {
    /// The class `NeMo` instantiates.
    pub target: String,

    /// The mel front end.
    pub preprocessor: NemoPreprocessorConfig,

    /// The `FastConformer`.
    pub encoder: NemoConformerEncoderConfig,

    /// The transformer over the projected frames.
    pub transformer_encoder: NemoTransformerEncoderConfig,

    /// The projection, the head, and the streaming defaults.
    pub sortformer_modules: NemoSortformerModulesConfig,
}

/// A `sortformer_modules` section: `SortformerModules`' arguments that shape
/// the model, named as `NeMo` names them and each defaulting as its
/// constructor defaults it. Its streaming arguments are not read: a stream's
/// geometry is a
/// [`SortformerStreamingConfig`](super::SortformerStreamingConfig) preset.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NemoSortformerModulesConfig {
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

impl NemoSortformerConfig {
    /// The [`SortformerStructureConfig`] the config describes: a
    /// `FastConformer` (the `encoder` section), `encoder_proj`, the
    /// transformer (the `transformer_encoder` section), and the speaker head,
    /// at the encoder's frame rate; the silence fill a running mean.
    ///
    /// # Errors
    /// As the sections' conversions
    /// ([`NemoConformerEncoderConfig::conformer`],
    /// [`NemoTransformerEncoderConfig::block`]);
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the field and its value, for another model class, sections that
    /// disagree on a width or on the subsampling, a transformer that ends
    /// with a norm, an upsampled output, a learned silence embedding, or an
    /// activity head.
    pub fn structure(&self) -> BunsenResult<SortformerStructureConfig> {
        if self.target != SORTFORMER_TARGET
            && !self.target.ends_with(&format!(".{SORTFORMER_TARGET}"))
        {
            return Err(unsupported(
                "target",
                &self.target,
                &format!("not {SORTFORMER_TARGET}"),
            ));
        }
        let modules = &self.sortformer_modules;
        if !modules.target.ends_with("SortformerModules") {
            return Err(unsupported(
                "sortformer_modules._target_",
                &modules.target,
                "not SortformerModules",
            ));
        }
        let (encoder, transformer) = (&self.encoder, &self.transformer_encoder);
        if self.preprocessor.features != encoder.feat_in {
            return Err(unsupported(
                "preprocessor.features",
                self.preprocessor.features,
                &format!("not encoder.feat_in {}", encoder.feat_in),
            ));
        }
        if modules.fc_d_model != encoder.d_model {
            return Err(unsupported(
                "sortformer_modules.fc_d_model",
                modules.fc_d_model,
                &format!("not encoder.d_model {}", encoder.d_model),
            ));
        }
        if modules.tf_d_model != transformer.hidden_size {
            return Err(unsupported(
                "sortformer_modules.tf_d_model",
                modules.tf_d_model,
                &format!(
                    "not transformer_encoder.hidden_size {}",
                    transformer.hidden_size
                ),
            ));
        }
        if modules.subsampling_factor != encoder.subsampling_factor {
            return Err(unsupported(
                "sortformer_modules.subsampling_factor",
                modules.subsampling_factor,
                &format!(
                    "not encoder.subsampling_factor {}",
                    encoder.subsampling_factor
                ),
            ));
        }
        if transformer.final_norm() {
            return Err(unsupported(
                "transformer_encoder.pre_ln_final_layer_norm",
                true,
                "a pre-norm stack's final norm is not built",
            ));
        }
        if modules.upsample_factor != 1 {
            return Err(unsupported(
                "sortformer_modules.upsample_factor",
                modules.upsample_factor,
                "only 1",
            ));
        }
        if modules.use_learnable_sil_emb {
            return Err(unsupported(
                "sortformer_modules.use_learnable_sil_emb",
                true,
                "only false",
            ));
        }
        if modules.use_activity_head {
            return Err(unsupported(
                "sortformer_modules.use_activity_head",
                true,
                "only false",
            ));
        }
        Ok(SortformerStructureConfig::new(
            SortformerEncoderConfig::FastConformer(encoder.conformer()?),
            SortformerHeadConfig::new(modules.tf_d_model, modules.num_spks),
        )
        .with_transformer(Some(SortformerTransformerConfig::new(
            transformer.block()?,
            transformer.num_layers,
        ))))
    }

    /// The front end the model was trained behind: the `preprocessor`
    /// section's.
    ///
    /// # Errors
    /// As [`NemoPreprocessorConfig::front_end`].
    pub fn front_end(&self) -> BunsenResult<NemoMelFrontEndConfig> {
        self.preprocessor.front_end()
    }
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
        errors::BunsenErrorKind,
        kits::speech::{
            nemo::{
                FeatureNormalization,
                NemoArchive,
                parse_nemo_yaml,
            },
            sortformer::SortformerV2Config,
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
        let config: NemoSortformerConfig = parse_nemo_yaml(&toy_yaml()).unwrap();
        assert_eq!(config.sortformer_modules.num_spks, 3);
        assert_eq!(config.sortformer_modules.subsampling_factor, 8);
        assert_eq!(
            json(&config.structure().unwrap()),
            json(&toy_v2().to_structure())
        );
        let front_end = config.front_end().unwrap();
        assert_eq!(front_end.n_mels, 24);
        assert_eq!(front_end.normalize, FeatureNormalization::None);
        assert!(!front_end.bf16_tables);
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
            let config: NemoSortformerConfig =
                parse_nemo_yaml(&yaml.replacen(from, to, 1)).unwrap();
            let err = config.structure().unwrap_err();
            assert_eq!(err.kind(), BunsenErrorKind::Unsupported, "{field}: {err}");
            assert!(err.to_string().contains(field), "{field}: {err}");
        }
    }
}
