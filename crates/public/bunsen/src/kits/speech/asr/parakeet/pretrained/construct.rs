//! # Constructing a Parakeet pretrained
use alloc::sync::Arc;

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
    data::{
        builds::{
            BuildRef,
            CONFIG,
            ConstructInit,
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
        speech::asr::parakeet::{
            ParakeetConstruct,
            ParakeetPolicy,
            ParakeetTdtConfig,
            interop::{
                phonon,
                transformers,
            },
        },
        support::model_config::resolve_ref_presets,
    },
};

/// How a Parakeet checkpoint is read, by its resource `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParakeetReader {
    /// A Phonon archive ([`PHONON_ARCHIVE`]): the weights and the
    /// vocabulary in one file. Needs the `store` feature.
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
    /// kind there is no reader for.
    pub fn for_kind(kind: Option<&str>) -> BunsenResult<Self> {
        match kind {
            Some(PHONON_ARCHIVE) => Ok(Self::PhononArchive),
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self::Safetensors),
            other => Err(BunsenError::unsupported(format!(
                "no Parakeet reader for a checkpoint of kind {other:?}"
            ))),
        }
    }
}

/// How a Parakeet pretrained is built: its checkpoint read by the
/// [`ParakeetReader`] its `kind` names, into the geometry its `config.json`
/// gives
/// ([`parakeet_tdt_config_from_path`](transformers::parakeet_tdt_config_from_path);
/// `parakeet-tdt-0.6b-v3`'s when
/// the map has none), with its vocabulary, behind the default front end.
///
/// The Parakeet kit's construct config: its [`ConstructInit`] hook.
/// [`default_parakeet_factory`](super::default_parakeet_factory) attaches
/// it; a caller never builds one.
#[derive(Clone, Debug)]
pub struct ParakeetConstructConfig {
    /// The checkpoint's reader.
    pub reader: ParakeetReader,
}

impl ConstructInit for ParakeetConstructConfig {
    type Construct = ParakeetConstruct;

    const KIT: &'static str = PARAKEET_KIT;
    const RESOLVES_PRESETS: bool = true;

    fn applied_presets(
        construct: &Self::Construct
    ) -> Option<crate::data::catalog::presets::PresetSelection> {
        Some(construct.presets.clone())
    }

    /// The hook for the checkpoint's `kind`, one file or a sharded one's
    /// index; a Safetensors checkpoint must come with a `tokenizer.json`
    /// vocabulary.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
        let reader = ParakeetReader::for_kind(checkpoint.format.as_deref())?;
        if reader == ParakeetReader::Safetensors {
            let vocab = map.try_get(VOCABULARY)?;
            if vocab.format.as_deref() != Some(TOKENIZER_JSON) {
                return Err(BunsenError::unsupported(format!(
                    "no Parakeet vocabulary reader for kind {:?}",
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
    ) -> BunsenResult<Arc<ParakeetConstruct>> {
        let config = match loaded.get(CONFIG) {
            Some(part) => transformers::parakeet_tdt_config_from_path(&part.path)?,
            None => ParakeetTdtConfig::tdt_0_6b_v3(),
        };
        let (asr, vocab) = match self.reader {
            ParakeetReader::Safetensors => {
                transformers::load_parakeet_tdt(loaded, &config, device, load)?
            }
            ParakeetReader::PhononArchive => {
                phonon::load_parakeet_tdt(loaded, &config, device, load)?
            }
        };
        let policy = resolve_ref_presets(
            &ParakeetPolicy::default(),
            model,
            &ResolvedPresetSet::default(),
        )?;
        Ok(Arc::new(
            ParakeetConstruct::new(asr, vocab, device)?
                .with_policy(policy.config.clone(), policy.applied()),
        ))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::errors::testing::ErrorMatcher;

    #[test]
    fn test_the_hook_follows_the_kind() {
        let nvidia = ParakeetConstructConfig::for_resource_set(
            &crate::data::catalog::BuiltinCatalog::get()
                .unwrap()
                .build("nvidia/parakeet-tdt-0.6b-v3")
                .unwrap()
                .resource_set(),
        )
        .unwrap();
        assert_eq!(nvidia.reader, ParakeetReader::Safetensors);

        let phonon = ParakeetConstructConfig::for_resource_set(
            &crate::data::catalog::BuiltinCatalog::get()
                .unwrap()
                .build("fermion/phonon-2")
                .unwrap()
                .resource_set(),
        );
        assert_eq!(phonon.unwrap().reader, ParakeetReader::PhononArchive);
        ErrorMatcher::unsupported().assert_err(&ParakeetReader::for_kind(Some("pytorch fp16")));
        ErrorMatcher::unsupported().assert_err(&ParakeetReader::for_kind(None));
    }

    /// Every row resolves to the default policy, which has no named presets;
    /// the lookup's patch sets its fields, and a box finds no preset.
    #[test]
    fn test_the_policy() {
        use crate::kits::speech::asr::parakeet::{
            ParakeetPolicy,
            pretrained::default_parakeet_factory,
        };

        let factory = default_parakeet_factory().unwrap();
        let row = factory.ids()[0].clone();
        let policy = |spec: &str| factory.resolve_presets(spec, &ParakeetPolicy::default());
        let default = policy(&row).unwrap();
        assert!(default.presets.is_empty());
        assert_eq!(default.config, ParakeetPolicy::default());
        assert_eq!(default.config.window_seconds, 60.0);
        assert_eq!(
            policy(&format!("{row}{{window_seconds: 20}}"))
                .unwrap()
                .config
                .window_seconds,
            20.0
        );
        crate::errors::testing::ErrorMatcher::lookup()
            .message_contains("no preset")
            .assert_err(&policy(&format!("{row}[fast]")));
    }
}
