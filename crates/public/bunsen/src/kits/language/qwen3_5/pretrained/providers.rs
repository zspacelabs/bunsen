//! # Qwen3.5 pretrained providers and factory
use alloc::sync::Arc;

use super::Qwen3_5ConstructConfig;
use crate::{
    data::builds::{
        HfProvider,
        PretrainedFactory,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::language::qwen3_5::{
        CHAT_TEMPLATE,
        GENERATION_CONFIG,
        TOKENIZER,
        TOKENIZER_CONFIG,
    },
};

/// The Qwen3.5 kit's name: its rows' `kit` tag.
pub const QWEN3_5_KIT: &str = "qwen3.5";

/// The key of the checkpoint in a Qwen3.5 resource set: one
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

/// What the qwen3.5 kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const QWEN3_5_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `well-known:` rows are its builds.
pub const QWEN3_5_CATALOG_CONTRACT: &str = "qwen3.5-lm";

/// The name [`default_qwen3_5_factory`] loads when a caller names none.
pub const DEFAULT_QWEN3_5_MODEL: &str = "qwen/qwen3.5/qwen3.5-0.8b";

/// Qwen3.5's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in a `transformers` Qwen3.5's
/// layout (its safetensors checkpoint, one file or shards, and its
/// `config.json`), with its [`TOKENIZER`], [`TOKENIZER_CONFIG`],
/// [`GENERATION_CONFIG`] and [`CHAT_TEMPLATE`] files when it has them.
pub fn default_qwen3_5_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(crate::data::builds::BuildCatalogProvider::expect_builtin(
            QWEN3_5_CATALOG_CONTRACT,
            QWEN3_5_KIT,
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

/// Qwen3.5's factory over [`default_qwen3_5_providers`], building through
/// [`Qwen3_5ConstructConfig`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_qwen3_5_factory() -> BunsenResult<PretrainedFactory<Qwen3_5ConstructConfig>> {
    PretrainedFactory::new().with_providers(default_qwen3_5_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::builds::CONFIG;

    /// The rows, by name, every file pinned to one URL on Qwen's repo, each
    /// row with its config and tokenizer files and its checkpoint's index.
    #[test]
    fn test_rows_are_pinned() {
        let factory = default_qwen3_5_factory().unwrap();
        assert_eq!(factory.kit(), QWEN3_5_KIT);
        assert_eq!(
            factory
                .providers()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>(),
            ["well-known", "hf"]
        );
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
                .map(|n| format!("well-known:qwen/qwen3.5/{n}"))
                .collect::<Vec<_>>()
        );
        for name in names {
            let (_, row) = factory.lookup(name).unwrap();
            let map = &row.resources;
            map.validate().unwrap();
            for resource in map.resources.values() {
                assert!(resource.is_pinned(), "{name}: {}", resource.slot);
                let urls = resource.urls();
                assert_eq!(urls.len(), 1, "{name}: {}", resource.slot);
                assert!(urls[0].contains("/Qwen/Qwen3.5-"), "{name}: {}", urls[0]);
            }
            for key in [CONFIG, TOKENIZER, TOKENIZER_CONFIG, "checkpoint.index"] {
                assert!(map.get(key).is_some(), "{name}: {key}");
            }
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
