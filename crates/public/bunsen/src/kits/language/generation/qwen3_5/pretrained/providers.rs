//! # Qwen3.5 pretrained providers and factory
use alloc::sync::Arc;

use super::Qwen3_5ConstructConfig;
use crate::{
    data::builds::{
        BuildFactory,
        BuildProvider,
    },
    errors::BunsenResult,
};

/// The Qwen3.5 kit's name: its rows' `kit` tag.
pub const QWEN3_5_KIT: &str = "qwen3.5";

/// The `kind` of a `chat_template.jinja`.
pub use crate::data::builds::CHAT_TEMPLATE_JINJA;
/// The key of the checkpoint in a Qwen3.5 resource set: one
/// `model.safetensors`, or `checkpoint.index` and its shards
/// `checkpoint.<n>`.
pub use crate::data::builds::CHECKPOINT;
/// The `kind` of a `transformers` `generation_config.json`.
pub use crate::data::builds::GENERATION_CONFIG_JSON;
/// The `kind` of a `transformers` `tokenizer_config.json`.
pub use crate::data::builds::TOKENIZER_CONFIG_JSON;
/// The `kind` of a `transformers` `tokenizer.json`.
pub use crate::data::builds::TOKENIZER_JSON;

/// What the qwen3.5 kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::pretrained::catalog_contracts::kit_catalog_contracts).
pub const QWEN3_5_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const QWEN3_5_CATALOG_CONTRACT: &str = "qwen3.5-lm";

/// The name [`default_qwen3_5_factory`] loads when a caller names none.
pub const DEFAULT_QWEN3_5_MODEL: &str = "qwen/qwen3.5/qwen3.5-0.8b";

/// Qwen3.5's compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a Hugging Face repo in a `transformers` Qwen3.5's
/// layout (its safetensors checkpoint, one file or shards, and its
/// `config.json`), with its [`TOKENIZER`](crate::kits::support::construct::TOKENIZER), [`TOKENIZER_CONFIG`](crate::kits::support::construct::TOKENIZER_CONFIG),
/// [`GENERATION_CONFIG`](crate::kits::support::construct::GENERATION_CONFIG) and [`CHAT_TEMPLATE`](crate::kits::support::construct::CHAT_TEMPLATE) files when it has them.
pub fn default_qwen3_5_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            QWEN3_5_CATALOG_CONTRACT,
        )),
        Arc::new(crate::kits::support::pretrained::transformers_lm_hf_provider(true)),
    ]
}

/// Qwen3.5's factory over [`default_qwen3_5_providers`], building through
/// [`Qwen3_5ConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_qwen3_5_factory() -> BunsenResult<BuildFactory<Qwen3_5ConstructConfig>> {
    BuildFactory::new().with_providers(default_qwen3_5_providers())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{
        data::builds::CONFIG,
        kits::support::construct::{
            CHAT_TEMPLATE,
            TOKENIZER,
            TOKENIZER_CONFIG,
        },
    };

    /// The rows, by name, every file pinned to one URL on Qwen's repo, each
    /// row with its config and tokenizer files and its checkpoint's index.
    #[test]
    fn test_rows_are_pinned() {
        let factory = default_qwen3_5_factory().unwrap();
        crate::kits::support::pretrained::testing::assert_catalog_then_hf(&factory);
        let names = [
            "qwen3.5-0.8b",
            "qwen3.5-0.8b-base",
            "qwen3.5-2b",
            "qwen3.5-2b-base",
            "qwen3.5-4b",
            "qwen3.5-4b-base",
            "qwen3.5-9b",
            "qwen3.5-9b-base",
            "qwen3.5-27b",
        ];
        assert_eq!(
            factory.ids(),
            names
                .iter()
                .map(|n| format!("catalog:qwen/qwen3.5/{n}"))
                .collect::<Vec<_>>()
        );
        crate::kits::support::pretrained::testing::assert_rows_pinned(
            &factory,
            &[CONFIG, TOKENIZER, TOKENIZER_CONFIG, "checkpoint.index"],
            Some("/Qwen/Qwen3.5-"),
        );
        for name in names {
            let (_, row) = factory.lookup(name).unwrap();
            let map = &row.resource_set();
            assert_eq!(
                map.get(CHAT_TEMPLATE).is_some(),
                !name.ends_with("-base"),
                "{name}: a chat row's template"
            );
        }
        assert_eq!(
            factory.lookup(DEFAULT_QWEN3_5_MODEL).unwrap().1.name,
            "qwen/qwen3.5/qwen3.5-0.8b"
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
                    CHAT_TEMPLATE,
                    GENERATION_CONFIG,
                    TOKENIZER,
                    TOKENIZER_CONFIG,
                },
                pretrained::catalog_contracts::testing::contract_lines,
            },
        };
        assert_eq!(
            contract_lines(QWEN3_5_CONTRACT),
            [
                format!("qwen3.5-lm: {CHAT_TEMPLATE}? [{CHAT_TEMPLATE_JINJA}]"),
                format!("qwen3.5-lm: {CHECKPOINT} [{SAFETENSORS}, {SAFETENSORS_INDEX}]"),
                format!("qwen3.5-lm: {CONFIG} [json]"),
                format!("qwen3.5-lm: {GENERATION_CONFIG}? [{GENERATION_CONFIG_JSON}]"),
                format!("qwen3.5-lm: {TOKENIZER} [{TOKENIZER_JSON}]"),
                format!("qwen3.5-lm: {TOKENIZER_CONFIG} [{TOKENIZER_CONFIG_JSON}]"),
            ]
        );
    }
}
