//! # Constructing a Qwen3-ASR pretrained
use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    PHONON_ARCHIVE,
    QWEN3_ASR_KIT,
    SAFETENSORS,
    TOKENIZER_JSON,
    VOCABULARY,
};
use crate::{
    burner::module::ModuleInit,
    data::{
        builds::{
            BuildRef,
            CONFIG,
            ConstructInit,
            GEOMETRY,
        },
        catalog::presets::ResolvedPresetSet,
        resources::{
            LoadedResourceSet,
            ResourceSet,
            SAFETENSORS_INDEX,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::asr::qwen3_asr::{
            Qwen3AsrConfig,
            Qwen3AsrConstruct,
            Qwen3AsrFrontEndConfig,
            Qwen3AsrPolicy,
            interop::{
                phonon,
                transformers,
            },
        },
        support::model_config::resolve_ref_presets,
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
}

/// How a Qwen3-ASR pretrained is built: its checkpoint read by the
/// [`Qwen3AsrReader`] its `kind` names, into the geometry its `config.json`
/// gives ([`qwen3_asr_config_from_path`](transformers::qwen3_asr_config_from_path)), or else the prefab its row
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

    fn applied_presets(
        construct: &Self::Construct
    ) -> Option<crate::data::catalog::presets::PresetSelection> {
        Some(construct.presets.clone())
    }

    /// The hook for the checkpoint's `kind`; a Safetensors checkpoint must
    /// come with a `tokenizer.json` vocabulary.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
        let reader = Qwen3AsrReader::for_kind(checkpoint.format.as_deref())?;
        if reader == Qwen3AsrReader::Safetensors {
            let vocab = map.try_get(VOCABULARY)?;
            if vocab.format.as_deref() != Some(TOKENIZER_JSON) {
                return Err(BunsenError::unsupported(format!(
                    "no Qwen3-ASR vocabulary reader for kind {:?}",
                    vocab.format
                )));
            }
        }
        Ok(Self { reader })
    }

    fn try_init(
        &self,
        model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        load: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<Qwen3AsrConstruct>> {
        let config: Qwen3AsrConfig = match loaded.get(CONFIG) {
            Some(part) => transformers::qwen3_asr_config_from_path(&part.path)?,
            None => model.stated(GEOMETRY)?.ok_or_else(|| {
                BunsenError::invalid_resource(format!(
                    "{}: a Qwen3-ASR build with neither a `{CONFIG}` nor a stated `{GEOMETRY}`",
                    model.id()
                ))
            })?,
        };
        let n_mels = config.audio.n_mels;
        let (asr, vocab, front_end) = match self.reader {
            Qwen3AsrReader::Safetensors => {
                transformers::load_qwen3_asr(loaded, &config, device, load)?
            }
            Qwen3AsrReader::PhononArchive => phonon::load_qwen3_asr(loaded, &config, device, load)?,
        };
        let policy = resolve_ref_presets(
            &Qwen3AsrPolicy::default(),
            model,
            &ResolvedPresetSet::default(),
        )?;
        let mut bundle = Qwen3AsrConstruct::new(asr, vocab, device)?
            .with_policy(policy.config.clone(), policy.applied());
        // the front end the model's files describe; without one, the default
        // at the model's mel count
        let front_end =
            front_end.unwrap_or_else(|| Qwen3AsrFrontEndConfig::new().with_n_mels(n_mels));
        bundle.front_end = front_end.try_init(device)?;
        Ok(Arc::new(bundle))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_the_hook_follows_the_kind() {
        let qwen = Qwen3AsrConstructConfig::for_resource_set(
            &crate::data::catalog::BuiltinCatalog::get()
                .unwrap()
                .build("qwen/qwen3-asr-0.6b")
                .unwrap()
                .resource_set(),
        )
        .unwrap();
        assert_eq!(qwen.reader, Qwen3AsrReader::Safetensors);

        let phonon = Qwen3AsrConstructConfig::for_resource_set(
            &crate::data::catalog::BuiltinCatalog::get()
                .unwrap()
                .build("fermion/phonon-1")
                .unwrap()
                .resource_set(),
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
