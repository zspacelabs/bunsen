//! # Constructing a Qwen3-ASR pretrained
use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    PHONON_ARCHIVE,
    QWEN3_ASR_KIT,
    QWEN3_ASR_PREFABS,
    SAFETENSORS,
    TOKENIZER_JSON,
    VOCABULARY,
};
use crate::{
    burner::module::ModuleInit,
    data::{
        catalog::presets::ResolvedPresetSet,
        pretrained::{
            CONFIG,
            GEOMETRY,
            PretrainedRef,
            SafetensorsCheckpoint,
        },
        resources::{
            LoadedResources,
            ResourceBundle,
            SAFETENSORS_INDEX,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        speech::asr::qwen3_asr::{
            Qwen3Asr,
            Qwen3AsrConfig,
            Qwen3AsrConstruct,
            Qwen3AsrFrontEndConfig,
            Qwen3AsrPolicy,
            weights::{
                load_transformers_tensors_as,
                qwen3_asr_config_from_transformers,
            },
        },
        support::{
            construct::ConstructInit,
            model_config::resolve_ref_presets,
            tokens::ByteLevelVocabulary,
        },
    },
};

/// How a Qwen3-ASR checkpoint is read, by its resource `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3AsrReader {
    /// A Phonon-1 archive ([`PHONON_ARCHIVE`]): the packed weights and the
    /// vocabulary in one file. Needs the `store` feature.
    PhononArchive,

    /// A `transformers` `model.safetensors` ([`SAFETENSORS`]), with its
    /// vocabulary under [`VOCABULARY`] as a `tokenizer.json`.
    Safetensors,
}

impl Qwen3AsrReader {
    /// The reader for a checkpoint of `kind`.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// kind there is no reader for.
    pub fn for_kind(kind: Option<&str>) -> BunsenResult<Self> {
        match kind {
            Some(PHONON_ARCHIVE) => Ok(Self::PhononArchive),
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self::Safetensors),
            other => Err(BunsenError::unsupported(format!(
                "no Qwen3-ASR reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    /// Reads the weights and the vocabulary.
    fn read(
        self,
        loaded: &LoadedResources,
    ) -> BunsenResult<(Vec<burn::store::burn_pack::Tensor>, ByteLevelVocabulary)> {
        match self {
            Self::PhononArchive => crate::kits::speech::asr::qwen3_asr::read_phonon1_archive(
                loaded.expect(CHECKPOINT)?,
            ),
            Self::Safetensors => {
                let tokenizer = loaded.expect(VOCABULARY)?;
                let json = std::fs::read_to_string(tokenizer).map_err(sys_at("read", tokenizer))?;
                let vocab = ByteLevelVocabulary::from_tokenizer_json(&json)
                    .map_err(|e| e.context(tokenizer.display()))?;
                let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
                Ok((tensors, vocab))
            }
        }
    }
}

/// How a Qwen3-ASR pretrained is built: its checkpoint read by the
/// [`Qwen3AsrReader`] its `kind` names, into the geometry its `config.json`
/// gives ([`qwen3_asr_config_from_transformers`]), or else the prefab its row
/// names (`qwen3-asr-0.6b` when it names none), in the device's float dtype,
/// with its vocabulary, behind a front end of the model's mel count and the
/// default prompt and decoder.
///
/// The Qwen3-ASR kit's construct config: its [`ConstructInit`] hook.
/// [`default_qwen3_asr_factory`](super::default_qwen3_asr_factory) attaches
/// it; a caller never builds one.
#[derive(Clone, Debug)]
pub struct Qwen3AsrConstructConfig {
    /// The checkpoint's reader.
    pub reader: Qwen3AsrReader,
}

impl ConstructInit for Qwen3AsrConstructConfig {
    type Construct = Qwen3AsrConstruct;

    const KIT: &'static str = QWEN3_ASR_KIT;
    const RESOLVES_PRESETS: bool = true;

    /// The hook for the checkpoint's `kind`; a Safetensors checkpoint must
    /// come with a `tokenizer.json` vocabulary.
    fn for_resource_bundle(map: &ResourceBundle) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
        let reader = Qwen3AsrReader::for_kind(checkpoint.kind.as_deref())?;
        if reader == Qwen3AsrReader::Safetensors {
            let vocab = map.try_get(VOCABULARY)?;
            if vocab.kind.as_deref() != Some(TOKENIZER_JSON) {
                return Err(BunsenError::unsupported(format!(
                    "no Qwen3-ASR vocabulary reader for kind {:?}",
                    vocab.kind
                )));
            }
        }
        Ok(Self { reader })
    }

    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
        load: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<Qwen3AsrConstruct>> {
        let (tensors, vocab) = self.reader.read(loaded)?;
        let config = match loaded.get(CONFIG) {
            Some(part) => {
                let json =
                    std::fs::read_to_string(&part.path).map_err(sys_at("read", &part.path))?;
                qwen3_asr_config_from_transformers(&json)
                    .map_err(|e| e.context(part.path.display()))?
            }
            None => match model.stated(GEOMETRY)? {
                Some(config) => config,
                None => model
                    .prefab(&QWEN3_ASR_PREFABS)
                    .map_or_else(Qwen3AsrConfig::qwen3_asr_0_6b, |p| p.to_config()),
            },
        };
        let n_mels = config.audio.n_mels;
        // The stored dtype (bf16), where the policy and the device allow it;
        // never f16, which overflows its Qwen3 text model.
        let stored = crate::data::pretrained::stored_float_dtype(&tensors)
            .unwrap_or(burn::tensor::FloatDType::F32);
        let dtype = load.resolve(stored, device, |d| d != burn::tensor::FloatDType::F16)?;
        let mut asr: Qwen3Asr = config.try_init(device)?;
        load_transformers_tensors_as(&mut asr, tensors, dtype)?;
        let policy = resolve_ref_presets(
            &Qwen3AsrPolicy::default(),
            model,
            &ResolvedPresetSet::default(),
        )?;
        let mut bundle = Qwen3AsrConstruct::new(asr, vocab, device)?
            .with_policy(policy.config.clone(), policy.applied());
        if n_mels != bundle.front_end.config.n_mels {
            bundle.front_end = Qwen3AsrFrontEndConfig::new()
                .with_n_mels(n_mels)
                .try_init(device)?;
        }
        Ok(Arc::new(bundle))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_the_hook_follows_the_kind() {
        let qwen = Qwen3AsrConstructConfig::for_resource_bundle(
            &crate::data::pretrained::builtin_build_bundle("qwen/qwen3-asr-0.6b").unwrap(),
        )
        .unwrap();
        assert_eq!(qwen.reader, Qwen3AsrReader::Safetensors);

        let phonon = Qwen3AsrConstructConfig::for_resource_bundle(
            &crate::data::pretrained::builtin_build_bundle("fermion/phonon-1").unwrap(),
        );
        assert_eq!(phonon.unwrap().reader, Qwen3AsrReader::PhononArchive);
        assert!(Qwen3AsrReader::for_kind(Some("gguf")).is_err());
        assert!(Qwen3AsrReader::for_kind(None).is_err());
    }

    /// Every row resolves to the default policy, which has no named presets;
    /// the lookup's patch sets its fields, and a box finds no preset.
    #[test]
    fn test_the_policy() {
        use crate::kits::speech::asr::qwen3_asr::{
            Qwen3AsrPolicy,
            pretrained::default_qwen3_asr_factory,
        };

        let factory = default_qwen3_asr_factory().unwrap();
        let row = factory.ids()[0].clone();
        let policy = |spec: &str| factory.resolve_presets(spec, &Qwen3AsrPolicy::default());
        let default = policy(&row).unwrap();
        assert!(default.presets.is_empty());
        assert_eq!(default.config, Qwen3AsrPolicy::default());
        assert_eq!(default.config.window_seconds, 30.0);
        assert_eq!(
            policy(&format!("{row}{{window_seconds: 20}}"))
                .unwrap()
                .config
                .window_seconds,
            20.0
        );
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Lookup)
            .message_contains("no preset")
            .assert_err(&policy(&format!("{row}[fast]")));
    }
}
