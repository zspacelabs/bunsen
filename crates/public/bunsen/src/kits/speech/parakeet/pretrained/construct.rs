//! # Constructing a Parakeet pretrained
use std::sync::Arc;

use burn::tensor::Device;

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
        CONFIG,
        Construct,
        LoadedResources,
        PretrainedRef,
        ResourceMap,
        SAFETENSORS_INDEX,
        SafetensorsCheckpoint,
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
        weights::{
            load_hf_tensors,
            parakeet_config_from_hf,
        },
    },
};

/// How a Parakeet checkpoint is read, by its resource `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParakeetReader {
    /// A Phonon archive ([`PHONON_ARCHIVE`]): the weights and the
    /// vocabulary in one file. Needs the `fermion` feature.
    PhononArchive,

    /// A `transformers` safetensors checkpoint ([`SAFETENSORS`], or the
    /// index of a sharded one, [`SAFETENSORS_INDEX`]), with its vocabulary
    /// under [`VOCABULARY`] as a `tokenizer.json`.
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
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self::Safetensors),
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
        match self {
            #[cfg(feature = "fermion")]
            Self::PhononArchive => {
                crate::kits::speech::parakeet::read_phonon2_archive(loaded.expect(CHECKPOINT)?)
            }
            #[cfg(not(feature = "fermion"))]
            Self::PhononArchive => Err(BunsenError::unsupported(
                "a Phonon archive needs bunsen's `fermion` feature",
            )),
            Self::Safetensors => {
                let tokenizer = loaded.expect(VOCABULARY)?;
                let json = std::fs::read_to_string(tokenizer).map_err(sys_at("read", tokenizer))?;
                let vocab = ParakeetVocabulary::from_tokenizer_json(&json)
                    .map_err(|e| e.context(tokenizer.display()))?;
                let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
                Ok((tensors, vocab))
            }
        }
    }
}

/// How a Parakeet pretrained is built: its checkpoint read by the
/// [`ParakeetReader`] its `kind` names, into the geometry its `config.json`
/// gives ([`parakeet_config_from_hf`]; `parakeet-tdt-0.6b-v3`'s when the map
/// has none), with its vocabulary, behind the default front end.
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

    /// The hook for the checkpoint's `kind`, one file or a sharded one's
    /// index; a Safetensors checkpoint must come with a `tokenizer.json`
    /// vocabulary.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
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
        let config = match loaded.get(CONFIG) {
            Some(part) => {
                let json =
                    std::fs::read_to_string(&part.path).map_err(sys_at("read", &part.path))?;
                parakeet_config_from_hf(&json).map_err(|e| e.context(part.path.display()))?
            }
            None => ParakeetTdtConfig::tdt_0_6b_v3(),
        };
        let mut model: ParakeetTdt = config.try_init(device)?;
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
