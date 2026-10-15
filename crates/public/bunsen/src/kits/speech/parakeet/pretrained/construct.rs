//! # Constructing a Parakeet pretrained
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
    PARAKEET_KIT,
    PHONON_ARCHIVE,
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
    kits::speech::parakeet::{
        ParakeetBundle,
        ParakeetTdt,
        ParakeetTdtConfig,
        ParakeetVocabulary,
        weights::load_hf_tensors,
    },
};

/// How a Parakeet checkpoint is read, by its resource `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParakeetReader {
    /// A Phonon archive ([`PHONON_ARCHIVE`]): the weights and the
    /// vocabulary in one file. Needs the `fermion` feature.
    PhononArchive,

    /// A `transformers` `model.safetensors` ([`SAFETENSORS`]), with its
    /// vocabulary under [`VOCABULARY`] as a `tokenizer.json`.
    Safetensors,
}

impl ParakeetReader {
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
                "no Parakeet reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    /// Reads the weights and the vocabulary.
    fn read(
        self,
        loaded: &LoadedResources,
    ) -> BunsenResult<(Vec<burn::store::burn_pack::Tensor>, ParakeetVocabulary)> {
        let checkpoint = loaded.expect(CHECKPOINT)?;
        match self {
            #[cfg(feature = "fermion")]
            Self::PhononArchive => crate::kits::speech::parakeet::read_phonon2_archive(checkpoint),
            #[cfg(not(feature = "fermion"))]
            Self::PhononArchive => Err(BunsenError::unsupported(
                "a Phonon archive needs bunsen's `fermion` feature",
            )),
            Self::Safetensors => {
                let tokenizer = loaded.expect(VOCABULARY)?;
                let json = std::fs::read_to_string(tokenizer).map_err(sys_at("read", tokenizer))?;
                let vocab = ParakeetVocabulary::from_tokenizer_json(&json)
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

/// How a Parakeet pretrained is built: its checkpoint read by the
/// [`ParakeetReader`] its `kind` names, into a `parakeet-tdt-0.6b-v3`-shaped
/// model, with its vocabulary, behind the default front end.
///
/// The [`Construct`] hook for the Parakeet kit.
/// [`default_parakeet_factory`](super::default_parakeet_factory) attaches
/// it; a caller never builds one.
#[derive(Clone, Debug)]
pub struct ParakeetConstruct {
    /// The checkpoint's reader.
    pub reader: ParakeetReader,
}

impl Construct for ParakeetConstruct {
    type Built = ParakeetBundle;

    const KIT: &'static str = PARAKEET_KIT;

    /// The hook for the checkpoint's `kind`; a Safetensors checkpoint must
    /// come with a `tokenizer.json` vocabulary.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
        let checkpoint = map.try_get(CHECKPOINT)?;
        let reader = ParakeetReader::for_kind(checkpoint.kind.as_deref())?;
        if reader == ParakeetReader::Safetensors {
            let vocab = map.try_get(VOCABULARY)?;
            if vocab.kind.as_deref() != Some(TOKENIZER_JSON) {
                return Err(BunsenError::unsupported(format!(
                    "no Parakeet vocabulary reader for kind {:?}",
                    vocab.kind
                )));
            }
        }
        Ok(Self { reader })
    }

    fn construct(
        &self,
        _model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<ParakeetBundle>> {
        let (tensors, vocab) = self.reader.read(loaded)?;
        let mut model: ParakeetTdt = ParakeetTdtConfig::tdt_0_6b_v3().try_init(device)?;
        load_hf_tensors(&mut model, tensors)?;
        Ok(Arc::new(ParakeetBundle::new(model, vocab, device)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::parakeet::pretrained::{
        PARAKEET_TDT_0_6B_V3_FILES,
        PHONON_2_ARCHIVE,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        let nvidia = ParakeetConstruct::for_map(&PARAKEET_TDT_0_6B_V3_FILES.to_map()).unwrap();
        assert_eq!(nvidia.reader, ParakeetReader::Safetensors);

        let phonon = ParakeetConstruct::for_map(&PHONON_2_ARCHIVE.to_map());
        if cfg!(feature = "fermion") {
            assert_eq!(phonon.unwrap().reader, ParakeetReader::PhononArchive);
        } else {
            assert!(phonon.is_err());
        }
        assert!(ParakeetReader::for_kind(Some("pytorch fp16")).is_err());
        assert!(ParakeetReader::for_kind(None).is_err());
    }
}
