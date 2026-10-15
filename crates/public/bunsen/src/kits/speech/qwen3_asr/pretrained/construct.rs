//! # Constructing a Qwen3-ASR pretrained
use std::sync::Arc;

use burn::{
    store::{
        ModuleStore,
        SafetensorsStore,
    },
    tensor::Device,
};

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
    data::pretrained::{
        Construct,
        LoadedResources,
        PretrainedRef,
        ResourceMap,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        speech::qwen3_asr::{
            Qwen3Asr,
            Qwen3AsrBundle,
            Qwen3AsrConfig,
            weights::load_hf_tensors,
        },
        tokens::ByteLevelVocabulary,
    },
};

/// How a Qwen3-ASR checkpoint is read, by its resource `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3AsrReader {
    /// A Phonon-1 archive ([`PHONON_ARCHIVE`]): the packed weights and the
    /// vocabulary in one file. Needs the `fermion` feature.
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
    /// kind there is no reader for, or a Phonon archive in a build without
    /// the `fermion` feature.
    pub fn for_kind(kind: Option<&str>) -> BunsenResult<Self> {
        match kind {
            Some(PHONON_ARCHIVE) if cfg!(feature = "fermion") => Ok(Self::PhononArchive),
            Some(PHONON_ARCHIVE) => Err(BunsenError::unsupported(
                "a Phonon archive needs bunsen's `fermion` feature",
            )),
            Some(SAFETENSORS) => Ok(Self::Safetensors),
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
        let checkpoint = loaded.expect(CHECKPOINT)?;
        match self {
            #[cfg(feature = "fermion")]
            Self::PhononArchive => crate::kits::speech::qwen3_asr::read_phonon1_archive(checkpoint),
            #[cfg(not(feature = "fermion"))]
            Self::PhononArchive => Err(BunsenError::unsupported(
                "a Phonon archive needs bunsen's `fermion` feature",
            )),
            Self::Safetensors => {
                let tokenizer = loaded.expect(VOCABULARY)?;
                let json = std::fs::read_to_string(tokenizer).map_err(sys_at("read", tokenizer))?;
                let vocab = ByteLevelVocabulary::from_tokenizer_json(&json)
                    .map_err(|e| e.context(tokenizer.display()))?;
                let tensors = SafetensorsStore::from_file(checkpoint)
                    .get_all_tensors()
                    .map_err(|e| {
                        BunsenError::invalid_resource("the store refused the checkpoint")
                            .with_cause(e)
                            .context(checkpoint.display())
                    })?
                    .values()
                    .cloned()
                    .collect();
                Ok((tensors, vocab))
            }
        }
    }
}

/// How a Qwen3-ASR pretrained is built: its checkpoint read by the
/// [`Qwen3AsrReader`] its `kind` names, into the geometry of the prefab its
/// row names (`qwen3-asr-0.6b` when it names none), in the device's float
/// dtype, with its vocabulary, behind the default front end, prompt and
/// decoder.
///
/// The [`Construct`] hook for the Qwen3-ASR kit.
/// [`default_qwen3_asr_factory`](super::default_qwen3_asr_factory) attaches
/// it; a caller never builds one.
#[derive(Clone, Debug)]
pub struct Qwen3AsrConstruct {
    /// The checkpoint's reader.
    pub reader: Qwen3AsrReader,
}

impl Construct for Qwen3AsrConstruct {
    type Built = Qwen3AsrBundle;

    const KIT: &'static str = QWEN3_ASR_KIT;

    /// The hook for the checkpoint's `kind`; a Safetensors checkpoint must
    /// come with a `tokenizer.json` vocabulary.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
        let checkpoint = map.try_get(CHECKPOINT)?;
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

    fn construct(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<Qwen3AsrBundle>> {
        let (tensors, vocab) = self.reader.read(loaded)?;
        let config = model
            .prefab(&QWEN3_ASR_PREFABS)
            .map_or_else(Qwen3AsrConfig::qwen3_asr_0_6b, |p| p.to_config());
        let mut asr: Qwen3Asr = config.try_init(device)?;
        load_hf_tensors(&mut asr, tensors)?;
        Ok(Arc::new(Qwen3AsrBundle::new(asr, vocab, device)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::qwen3_asr::pretrained::{
        PHONON_1_ARCHIVE,
        QWEN3_ASR_0_6B_FILES,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        let qwen = Qwen3AsrConstruct::for_map(&QWEN3_ASR_0_6B_FILES.to_map()).unwrap();
        assert_eq!(qwen.reader, Qwen3AsrReader::Safetensors);

        let phonon = Qwen3AsrConstruct::for_map(&PHONON_1_ARCHIVE.to_map());
        if cfg!(feature = "fermion") {
            assert_eq!(phonon.unwrap().reader, Qwen3AsrReader::PhononArchive);
        } else {
            assert!(phonon.is_err());
        }
        assert!(Qwen3AsrReader::for_kind(Some("gguf")).is_err());
        assert!(Qwen3AsrReader::for_kind(None).is_err());
    }
}
