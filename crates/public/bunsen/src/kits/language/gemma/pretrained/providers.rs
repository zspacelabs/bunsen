//! # Gemma pretrained providers and factory
use alloc::sync::Arc;

use super::GemmaConstructConfig;
use crate::{
    data::builds::{
        BuildFactory,
        BuildProvider,
        HfProvider,
    },
    errors::BunsenResult,
    kits::language::gemma::{
        CHAT_TEMPLATE,
        GENERATION_CONFIG,
        TOKENIZER,
        TOKENIZER_CONFIG,
    },
};

/// The Gemma kit's name: its rows' `kit` tag.
pub const GEMMA_KIT: &str = "gemma";

/// The key of the checkpoint in a Gemma resource set: one
/// `model.safetensors`, or `checkpoint.index` and its shards
/// `checkpoint.<n>`.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` of a `transformers` `tokenizer.json`.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// The `kind` of a `transformers` `tokenizer_config.json`.
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

/// The `kind` of a `transformers` `generation_config.json`.
pub const GENERATION_CONFIG_JSON: &str = "generation_config.json";

/// The `kind` of a `chat_template.jinja`.
pub const CHAT_TEMPLATE_JINJA: &str = "chat_template.jinja";

/// What the gemma kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const GEMMA_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const GEMMA_CATALOG_CONTRACT: &str = "gemma-lm";

/// The built-in catalog's template class every Gemma build is of, and an
/// `hf:` repo's presets lie over: its own chat template where it has one,
/// else bunsen's rendering of Gemma's turn format.
pub const GEMMA_TEMPLATE_CLASS: &str = "gemma/gemma/gemma";

/// The name [`default_gemma_factory`] loads when a caller names none.
pub const DEFAULT_GEMMA_MODEL: &str = "gemma/gemma/gemma-3-1b-it";

/// Gemma's compiled-in providers, in search order: the catalog provider,
/// then `hf:org/repo`, a Hugging Face repo in a `transformers` Gemma's
/// layout (its safetensors checkpoint, one file or shards, and its
/// `config.json`), with its [`TOKENIZER`], [`TOKENIZER_CONFIG`],
/// [`GENERATION_CONFIG`] and [`CHAT_TEMPLATE`] files when it has them.
pub fn default_gemma_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![
        Arc::new(crate::data::catalog::BuiltinCatalog::expect_provider(
            GEMMA_CATALOG_CONTRACT,
        )),
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
                )
                .with_extra_file(CHAT_TEMPLATE, "chat_template.jinja", CHAT_TEMPLATE_JINJA),
        ),
    ]
}

/// Gemma's factory over [`default_gemma_providers`], building through
/// [`GemmaConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_gemma_factory() -> BunsenResult<BuildFactory<GemmaConstructConfig>> {
    BuildFactory::new().with_providers(default_gemma_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::builds::CONFIG;

    /// The rows, by name, every file pinned to one URL on its mirror, each
    /// row with its config, tokenizer and generation files.
    #[test]
    fn test_rows_are_pinned() {
        let factory = default_gemma_factory().unwrap();
        assert_eq!(factory.kit(), GEMMA_KIT);
        assert_eq!(
            factory
                .providers()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>(),
            ["catalog", "hf"]
        );
        let names = [
            "gemma-3-270m",
            "gemma-3-270m-it",
            "gemma-3-1b-pt",
            "gemma-3-1b-it",
            "gemma-3-4b-pt",
            "gemma-3-4b-it",
            "gemma-2-2b",
            "gemma-2-2b-it",
            "gemma-2-9b",
            "gemma-2-9b-it",
            "gemma-2b",
            "gemma-2b-it",
            "gemma-7b",
            "gemma-7b-it",
        ];
        assert_eq!(
            factory.ids(),
            names
                .iter()
                .map(|n| format!("catalog:gemma/gemma/{n}"))
                .collect::<Vec<_>>()
        );
        for name in names {
            let (_, row) = factory.lookup(name).unwrap();
            let map = &row.resource_set();
            map.validate().unwrap();
            for resource in map.resources.values() {
                assert!(resource.is_pinned(), "{name}: {}", resource.slot);
                let urls = resource.urls();
                assert_eq!(urls.len(), 1, "{name}: {}", resource.slot);
                assert!(urls[0].contains("/unsloth/"), "{name}: {}", urls[0]);
            }
            for key in [CONFIG, TOKENIZER, TOKENIZER_CONFIG, GENERATION_CONFIG] {
                assert!(map.get(key).is_some(), "{name}: {key}");
            }
            assert!(
                map.get(CHECKPOINT).is_some() || map.get("checkpoint.index").is_some(),
                "{name}"
            );
        }
        assert_eq!(
            factory.lookup(DEFAULT_GEMMA_MODEL).unwrap().1.name,
            "gemma/gemma/gemma-3-1b-it"
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
                catalog_contracts::testing::contract_lines,
                construct::{
                    CHAT_TEMPLATE,
                    GENERATION_CONFIG,
                    TOKENIZER,
                    TOKENIZER_CONFIG,
                },
            },
        };
        assert_eq!(
            contract_lines(GEMMA_CONTRACT),
            [
                format!("gemma-lm: {CHAT_TEMPLATE}? [{CHAT_TEMPLATE_JINJA}]"),
                format!("gemma-lm: {CHECKPOINT} [{SAFETENSORS}, {SAFETENSORS_INDEX}]"),
                format!("gemma-lm: {CONFIG} [json]"),
                format!("gemma-lm: {GENERATION_CONFIG} [{GENERATION_CONFIG_JSON}]"),
                format!("gemma-lm: {TOKENIZER} [{TOKENIZER_JSON}]"),
                format!("gemma-lm: {TOKENIZER_CONFIG} [{TOKENIZER_CONFIG_JSON}]"),
            ]
        );
    }
}
