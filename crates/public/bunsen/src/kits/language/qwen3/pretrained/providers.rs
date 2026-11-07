//! # Qwen3 pretrained providers and factory
use alloc::sync::Arc;

use burn::tensor::Device;

use super::Qwen3ConstructConfig;
use crate::{
    data::pretrained::{
        EmbeddedCatalog,
        HfProvider,
        PretrainedCache,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::{
        language::qwen3::{
            GENERATION_CONFIG,
            Qwen3Construct,
            TOKENIZER,
            TOKENIZER_CONFIG,
        },
        support::construct::PretrainedFactory,
    },
};

/// The kit segment of a Qwen3 resource's path in the cache:
/// `<cache>/pretrained/qwen3/<namespace>/<sha256>/<file>`.
pub const QWEN3_KIT: &str = "qwen3";

/// The key of the checkpoint in a Qwen3 resource bundle: one
/// `model.safetensors`, or `checkpoint.index` and its shards
/// `checkpoint.<n>`.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` of a `config.json`.
pub const JSON: &str = "json";

/// The `kind` of a `transformers` `tokenizer.json`.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// The `kind` of a `transformers` `tokenizer_config.json`.
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

/// The `kind` of a `transformers` `generation_config.json`.
pub const GENERATION_CONFIG_JSON: &str = "generation_config.json";

/// The Qwen3 checkpoints bunsen knows by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider, in group
/// `qwen`: Qwen's dense Qwen3 language models up to 8B, as `transformers`
/// ships them (`well-known:qwen/qwen3-0.6b`, `qwen/qwen3-0.6b`, or
/// `qwen3-0.6b`).
///
/// Built from `validation/catalogs/qwen3.yml` by `cargo validate catalog`.
pub static QWEN3_CATALOG: EmbeddedCatalog =
    EmbeddedCatalog::new("qwen3", include_str!("catalog.json"));

/// The name [`default_qwen3_factory`] loads when a caller names none.
pub const DEFAULT_QWEN3_MODEL: &str = "qwen/qwen3-0.6b";

/// Qwen3's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`'
/// `Qwen3ForCausalLM` layout (its safetensors checkpoint, one file or
/// shards, and its `config.json`), with its [`TOKENIZER`],
/// [`TOKENIZER_CONFIG`] and [`GENERATION_CONFIG`] files when it has them.
pub fn default_qwen3_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(QWEN3_CATALOG.table()),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(TOKENIZER, "tokenizer.json", TOKENIZER_JSON)
                .with_extra_file(
                    TOKENIZER_CONFIG,
                    "tokenizer_config.json",
                    TOKENIZER_CONFIG_JSON,
                )
                .with_extra_file(
                    GENERATION_CONFIG,
                    "generation_config.json",
                    GENERATION_CONFIG_JSON,
                ),
        ),
    ]
}

/// Qwen3's factory over [`default_qwen3_providers`], building through
/// [`Qwen3ConstructConfig`].
///
/// [`load_construct`](PretrainedFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`Qwen3Construct`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_qwen3_factory() -> BunsenResult<PretrainedFactory<Qwen3ConstructConfig>> {
    PretrainedFactory::new().with_providers(default_qwen3_providers())
}

impl PretrainedFactory<Qwen3ConstructConfig> {
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
    ) -> BunsenResult<Arc<Qwen3Construct>> {
        Ok(self.load(spec, cache, device, policy)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::CONFIG,
        kits::language::qwen3::{
            pretrained::QWEN3_PREFABS,
            weights::qwen3_config_from_transformers,
        },
    };

    /// Every file of every row is pinned to one URL, and each row's
    /// `config.json` (committed under `testdata/qwen3/`, at the row's digest)
    /// is the geometry of the prefab the row names.
    #[test]
    fn test_rows_are_pinned_and_match_their_prefabs() {
        let prefabs: Vec<&str> = QWEN3_PREFABS.iter().map(|p| p.name).collect();
        QWEN3_CATALOG.catalog().validate(&prefabs).unwrap();
        let factory = default_qwen3_factory().unwrap();
        assert_eq!(factory.kit(), QWEN3_KIT);
        assert_eq!(
            factory
                .providers()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>(),
            ["well-known", "hf"]
        );
        let names = [
            "qwen3-0.6b",
            "qwen3-0.6b-base",
            "qwen3-1.7b",
            "qwen3-1.7b-base",
            "qwen3-4b",
            "qwen3-4b-base",
            "qwen3-4b-instruct-2507",
            "qwen3-4b-thinking-2507",
            "qwen3-8b",
            "qwen3-8b-base",
            "qwen3-4b-saferl",
            "qwen3guard-gen-0.6b",
            "qwen3guard-gen-4b",
            "qwen3guard-gen-8b",
        ];
        assert_eq!(
            factory.ids(),
            names
                .iter()
                .map(|n| format!("well-known:qwen/{n}"))
                .collect::<Vec<_>>()
        );
        for name in names {
            let (_, row) = factory.lookup(name).unwrap();
            assert_eq!(row.name, format!("qwen/{name}"));
            let map = &row.resources;
            map.validate().unwrap();
            for resource in map.resources.values() {
                assert!(resource.is_pinned(), "{name}: {}", resource.key);
                assert_eq!(resource.urls().len(), 1, "{name}: {}", resource.key);
            }
            for key in [CONFIG, TOKENIZER, TOKENIZER_CONFIG, GENERATION_CONFIG] {
                assert!(map.get(key).is_some(), "{name}: {key}");
            }
            assert!(
                map.get(CHECKPOINT).is_some() || map.get("checkpoint.index").is_some(),
                "{name}"
            );

            let path = format!(
                "{}/testdata/qwen3/{name}.config.json",
                env!("CARGO_MANIFEST_DIR")
            );
            assert_eq!(
                Some(crate::data::cache::sha256_of(std::path::Path::new(&path)).unwrap()),
                map.get(CONFIG).unwrap().sha256,
                "{name}: the committed config.json is the pinned one"
            );
            let config =
                qwen3_config_from_transformers(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let prefab = QWEN3_PREFABS
                .expect_lookup_prefab(row.prefab.as_deref().unwrap())
                .to_config();
            assert_eq!(format!("{config:?}"), format!("{prefab:?}"), "{name}");
        }
        assert_eq!(
            factory.lookup(DEFAULT_QWEN3_MODEL).unwrap().1.name,
            "qwen/qwen3-0.6b"
        );
    }
}
