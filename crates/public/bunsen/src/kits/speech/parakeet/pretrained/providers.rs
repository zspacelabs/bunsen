//! # Parakeet pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::ParakeetConstructConfig;
use crate::{
    data::pretrained::{
        EmbeddedCatalog,
        HfProvider,
        PretrainedCache,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::{
        speech::parakeet::ParakeetConstruct,
        support::construct::PretrainedFactory,
    },
};

/// The kit segment of a Parakeet resource's path in the cache:
/// `<cache>/pretrained/parakeet/<namespace>/<sha256>/<file>`.
pub const PARAKEET_KIT: &str = "parakeet";

/// The key of the checkpoint in a Parakeet resource bundle.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the vocabulary in a Parakeet resource bundle, for a checkpoint
/// that does not carry its own.
pub const VOCABULARY: &str = "vocabulary";

/// The `kind` of a Phonon archive: a zstd tar of a `.fermion` container and
/// its `NeMo` config, which carries the vocabulary.
pub const PHONON_ARCHIVE: &str = "phonon archive";

/// The `kind` of a `transformers` `model.safetensors` checkpoint.
pub const SAFETENSORS: &str = "safetensors";

/// The `kind` of a `transformers` `tokenizer.json` vocabulary.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// The Parakeet checkpoints bunsen knows by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider: in group
/// `fermion`, `FermionResearch`'s Phonon-2 archive, and in group `nvidia`,
/// NVIDIA's Parakeet TDT 0.6B v3.
///
/// Built from `validation/catalogs/parakeet.yml` by `cargo validate catalog`.
pub static PARAKEET_CATALOG: EmbeddedCatalog =
    EmbeddedCatalog::new("parakeet", include_str!("catalog.json"));

/// The name [`default_parakeet_factory`] loads when a caller names none.
pub const DEFAULT_PARAKEET_MODEL: &str = "fermion/phonon-2";

/// Parakeet's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`' layout (its
/// safetensors checkpoint, one file or shards, its `config.json`, and its
/// `tokenizer.json` as the vocabulary).
pub fn default_parakeet_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(PARAKEET_CATALOG.table()),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(VOCABULARY, "tokenizer.json", TOKENIZER_JSON),
        ),
    ]
}

/// Parakeet's factory over [`default_parakeet_providers`], building through
/// [`ParakeetConstructConfig`].
///
/// [`load_construct`](PretrainedFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`ParakeetConstruct`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_parakeet_factory() -> BunsenResult<PretrainedFactory<ParakeetConstructConfig>> {
    PretrainedFactory::new().with_providers(default_parakeet_providers())
}

impl PretrainedFactory<ParakeetConstructConfig> {
    /// A name to a construct under `policy`: [`load`](Self::load), keeping
    /// the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_construct(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
        policy: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<ParakeetConstruct>> {
        Ok(self.load(spec, cache, device, policy)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is pinned, and answers by its qualified, short and alias
    /// names.
    #[test]
    fn test_rows_are_pinned_and_named() {
        PARAKEET_CATALOG.catalog().validate(&[]).unwrap();
        for bundle in &PARAKEET_CATALOG.catalog().bundles {
            let map = bundle.to_resource_bundle();
            map.validate().unwrap();
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert!(checkpoint.is_pinned(), "{}", map.name);
            assert_eq!(checkpoint.urls().len(), 1, "{}", map.name);
        }

        let factory = default_parakeet_factory().unwrap();
        assert_eq!(factory.kit(), PARAKEET_KIT);
        assert_eq!(
            factory.ids(),
            [
                "well-known:fermion/phonon-2",
                "well-known:nvidia/parakeet-tdt-0.6b-v3"
            ]
        );
        assert_eq!(
            factory
                .providers()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>(),
            ["well-known", "hf"]
        );
        for spec in [
            DEFAULT_PARAKEET_MODEL,
            "phonon-2",
            "phonon",
            "well-known:fermion/phonon-2",
        ] {
            assert_eq!(
                factory.lookup(spec).unwrap().1.name,
                "fermion/phonon-2",
                "{spec}"
            );
        }
        assert_eq!(
            factory.lookup("parakeet-tdt-0.6b-v3").unwrap().1.name,
            "nvidia/parakeet-tdt-0.6b-v3"
        );
    }
}
