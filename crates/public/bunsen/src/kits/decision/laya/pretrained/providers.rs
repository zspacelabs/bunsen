//! # Laya pretrained providers and factory
use alloc::sync::Arc;

use super::LayaConstructConfig;
use crate::{
    data::builds::{
        BuildFactory,
        BuildProvider,
    },
    errors::BunsenResult,
};

/// The Laya kit's name: its rows' `kit` tag.
pub const LAYA_KIT: &str = "laya";

/// The key of the checkpoint in a Laya resource set: its
/// `model.safetensors`.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the encoder's `config.json`.
pub const ENCODER_CONFIG: &str = "encoder_config";

/// The key of the head's `rl_agent_config.json`.
pub const AGENT_CONFIG: &str = "agent_config";

/// What the laya kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const LAYA_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const LAYA_CATALOG_CONTRACT: &str = "laya-decision";

/// The name [`default_laya_factory`] loads when a caller names none.
pub const DEFAULT_LAYA_MODEL: &str = "laya/laya/laya";

/// Laya's compiled-in providers: the catalog provider.
pub fn default_laya_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![Arc::new(
        crate::data::catalog::BuiltinCatalog::expect_provider(LAYA_CATALOG_CONTRACT, LAYA_KIT),
    )]
}

/// Laya's factory over [`default_laya_providers`], building through
/// [`LayaConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::with_providers`].
pub fn default_laya_factory() -> BunsenResult<BuildFactory<LayaConstructConfig>> {
    BuildFactory::new().with_providers(default_laya_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::support::construct::{
        TOKENIZER,
        TOKENIZER_CONFIG,
    };

    /// The rows, by name, every file pinned to one URL on the repo's fixed
    /// revision, each row with all five of its files.
    #[test]
    fn test_rows_are_pinned() {
        let factory = default_laya_factory().unwrap();
        assert_eq!(factory.kit(), LAYA_KIT);
        let names = ["laya", "laya-multilingual", "laya-typed-decisions"];
        assert_eq!(
            factory.ids(),
            names
                .iter()
                .map(|n| format!("catalog:laya/laya/{n}"))
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
                assert!(
                    urls[0].starts_with("https://huggingface.co/convaiinnovations/laya/resolve/"),
                    "{name}: {}",
                    urls[0]
                );
            }
            for key in [
                CHECKPOINT,
                ENCODER_CONFIG,
                AGENT_CONFIG,
                TOKENIZER,
                TOKENIZER_CONFIG,
            ] {
                assert!(map.get(key).is_some(), "{name}: {key}");
            }
        }
        assert_eq!(factory.lookup("laya-en").unwrap().1.name, "laya/laya/laya");
        assert_eq!(
            factory.lookup(DEFAULT_LAYA_MODEL).unwrap().1.name,
            "laya/laya/laya"
        );
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::{
            data::resources::SAFETENSORS,
            kits::support::{
                catalog_contracts::testing::contract_lines,
                construct::{
                    TOKENIZER,
                    TOKENIZER_CONFIG,
                },
            },
        };
        assert_eq!(
            contract_lines(LAYA_CONTRACT),
            [
                format!("laya-decision: {AGENT_CONFIG} [json]"),
                format!("laya-decision: {CHECKPOINT} [{SAFETENSORS}]"),
                format!("laya-decision: {ENCODER_CONFIG} [json]"),
                format!("laya-decision: {TOKENIZER} [tokenizer.json]"),
                format!("laya-decision: {TOKENIZER_CONFIG} [tokenizer_config.json]"),
            ]
        );
    }
}
