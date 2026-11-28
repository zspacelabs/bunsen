//! # Qwen3 pretrained providers and factory
use alloc::sync::Arc;

use super::Qwen3ConstructConfig;
use crate::{
    data::builds::{
        BuildFactory,
        BuildProvider,
    },
    errors::BunsenResult,
};

/// The Qwen3 kit's name: its rows' `kit` tag.
pub const QWEN3_KIT: &str = "qwen3";

/// The key of the checkpoint in a Qwen3 resource set: one
/// `model.safetensors`, or `checkpoint.index` and its shards
/// `checkpoint.<n>`.
pub use crate::data::builds::CHECKPOINT;
/// The `kind` of a `transformers` `generation_config.json`.
pub use crate::data::builds::GENERATION_CONFIG_JSON;
/// The `kind` of a `transformers` `tokenizer_config.json`.
pub use crate::data::builds::TOKENIZER_CONFIG_JSON;
/// The `kind` of a `transformers` `tokenizer.json`.
pub use crate::data::builds::TOKENIZER_JSON;
/// The `kind` of a `config.json`.
pub use crate::data::resources::JSON;

/// What the qwen3 kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const QWEN3_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const QWEN3_CATALOG_CONTRACT: &str = "qwen3-lm";

/// The name [`default_qwen3_factory`] loads when a caller names none.
pub const DEFAULT_QWEN3_MODEL: &str = "qwen/qwen3/qwen3-0.6b";

/// Qwen3's compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a Hugging Face repo in `transformers`'
/// `Qwen3ForCausalLM` layout (its safetensors checkpoint, one file or
/// shards, and its `config.json`), with its [`TOKENIZER`](crate::kits::support::construct::TOKENIZER),
/// [`TOKENIZER_CONFIG`](crate::kits::support::construct::TOKENIZER_CONFIG) and [`GENERATION_CONFIG`](crate::kits::support::construct::GENERATION_CONFIG) files when it has them.
pub fn default_qwen3_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            QWEN3_CATALOG_CONTRACT,
        )),
        Arc::new(crate::kits::support::pretrained::transformers_lm_hf_provider(false)),
    ]
}

/// Qwen3's factory over [`default_qwen3_providers`], building through
/// [`Qwen3ConstructConfig`].
///
/// [`load_construct`](BuildFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`Qwen3Construct`](crate::kits::language::generation::qwen3::Qwen3Construct).
///
/// # Errors
/// As [`BuildFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_qwen3_factory() -> BunsenResult<BuildFactory<Qwen3ConstructConfig>> {
    BuildFactory::new().with_providers(default_qwen3_providers())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{
        data::{
            builds::CONFIG,
            pretrained::Prefabs,
        },
        kits::{
            language::generation::qwen3::{
                Qwen3ContractConfig,
                interop::transformers::qwen3_config_from_str,
            },
            support::construct::{
                GENERATION_CONFIG,
                TOKENIZER,
                TOKENIZER_CONFIG,
            },
        },
        support::testing::testdata,
    };

    /// Every file of every row is pinned to one URL, and each row's
    /// `config.json` (committed under `testdata/qwen3/`, at the row's digest)
    /// is the geometry of a prefab.
    #[test]
    fn test_rows_are_pinned_and_match_their_prefabs() {
        let factory = default_qwen3_factory().unwrap();
        crate::kits::support::pretrained::testing::assert_catalog_then_hf(&factory);
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
                .map(|n| format!("catalog:qwen/qwen3/{n}"))
                .collect::<Vec<_>>()
        );
        crate::kits::support::pretrained::testing::assert_rows_pinned(
            &factory,
            &[CONFIG, TOKENIZER, TOKENIZER_CONFIG, GENERATION_CONFIG],
            None,
        );
        for name in names {
            let (_, row) = factory.lookup(name).unwrap();
            assert_eq!(row.name, format!("qwen/qwen3/{name}"));
            let map = &row.resource_set();

            let path = testdata!(format!("qwen3/{name}.config.json"));
            assert_eq!(
                Some(crate::data::cache::sha256_of(std::path::Path::new(&path)).unwrap()),
                map.get(CONFIG).unwrap().sha256,
                "{name}: the committed config.json is the pinned one"
            );
            let config = qwen3_config_from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            assert!(
                Qwen3ContractConfig::PREFABS
                    .iter()
                    .any(|(_, prefab)| crate::support::testing::config_eq(&prefab(), &config)),
                "{name}: its config.json is no prefab's geometry"
            );
        }
        assert_eq!(
            factory.lookup(DEFAULT_QWEN3_MODEL).unwrap().1.name,
            "qwen/qwen3/qwen3-0.6b"
        );
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::{
            data::{
                builds::CONFIG,
                resources::{
                    SAFETENSORS,
                    SAFETENSORS_INDEX,
                },
            },
            kits::support::{
                construct::{
                    GENERATION_CONFIG,
                    TOKENIZER,
                    TOKENIZER_CONFIG,
                },
                pretrained::catalog_contracts::testing::contract_lines,
            },
        };
        assert_eq!(
            contract_lines(QWEN3_CONTRACT),
            [
                format!("qwen3-lm: {CHECKPOINT} [{SAFETENSORS}, {SAFETENSORS_INDEX}]"),
                format!("qwen3-lm: {CONFIG} [{JSON}]"),
                format!("qwen3-lm: {GENERATION_CONFIG} [{GENERATION_CONFIG_JSON}]"),
                format!("qwen3-lm: {TOKENIZER} [{TOKENIZER_JSON}]"),
                format!("qwen3-lm: {TOKENIZER_CONFIG} [{TOKENIZER_CONFIG_JSON}]"),
            ]
        );
    }
}
