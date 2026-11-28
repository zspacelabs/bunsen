//! The config file's typed mirror.

use serde::Deserialize;

use crate::{
    blocks::layout::BoundaryPermutes,
    burner::module::ToStructureConfig,
    errors::BunsenResult,
    kits::speech::{
        diarization::sortformer::blocks::{
            SortformerHeadConfig,
            SortformerStructureConfig,
        },
        support::nemo::{
            front_end::NemoMelFrontEndConfig,
            interop::{
                NemoConformerEncoderMirror,
                NemoPreprocessorMirror,
                NemoTransformerEncoderMirror,
                parse_nemo_yaml,
                unsupported_field,
            },
        },
    },
};

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

/// The `NeMo` class a Sortformer `.nemo` instantiates.
const SORTFORMER_TARGET: &str = "SortformerEncLabelModel";

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
