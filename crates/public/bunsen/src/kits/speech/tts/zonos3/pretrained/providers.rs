//! # ZONOS3 pretrained providers

use alloc::sync::Arc;

use super::Zonos3ConstructConfig;
use crate::{
    data::builds::{
        BuildFactory,
        BuildProvider,
    },
    errors::BunsenResult,
};

/// The ZONOS3 kit's name: its rows' `kit` tag.
pub const ZONOS3_KIT: &str = "zonos3";

/// The key of the checkpoint in a ZONOS3 resource set.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of `config.json`.
pub const CONFIG: &str = "config";

/// The key of the tokenizer.
pub const TOKENIZER: &str = "tokenizer";

/// The key of the stock voices.
pub const VOICES: &str = "voices";

/// The format of a safetensors file.
pub const SAFETENSORS_FORMAT: &str = "safetensors";

/// What the zonos3 kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const ZONOS3_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const ZONOS3_CATALOG_CONTRACT: &str = "zonos3-tts";

/// The name [`default_zonos3_factory`] loads when a caller names none.
pub const DEFAULT_ZONOS3_MODEL: &str = "zyphra/zonos3/zonos3-mini";

/// ZONOS3's compiled-in providers: the catalog provider.
pub fn default_zonos3_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![Arc::new(
        crate::data::catalog::BuiltinCatalog::expect_provider(ZONOS3_CATALOG_CONTRACT),
    )]
}

/// ZONOS3's factory: [`default_zonos3_providers`] behind
/// [`Zonos3ConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_zonos3_factory() -> BunsenResult<BuildFactory<Zonos3ConstructConfig>> {
    BuildFactory::new().with_providers(default_zonos3_providers())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one row, by its qualified and short names, every file pinned to
    /// one URL on Zyphra's `-hf` repo at its revision.
    #[test]
    fn test_the_row_is_pinned_and_named() {
        let factory = default_zonos3_factory().unwrap();
        assert_eq!(factory.kit(), ZONOS3_KIT);
        assert_eq!(factory.ids(), ["catalog:zyphra/zonos3/zonos3-mini"]);
        for spec in [
            "zyphra/zonos3/zonos3-mini",
            "zyphra/zonos3-mini",
            "zonos3-mini",
        ] {
            assert_eq!(
                factory.lookup(spec).unwrap().1.name,
                DEFAULT_ZONOS3_MODEL,
                "{spec}"
            );
        }
        let map = factory
            .lookup(DEFAULT_ZONOS3_MODEL)
            .unwrap()
            .1
            .resource_set();
        map.validate().unwrap();
        for key in [CHECKPOINT, CONFIG, TOKENIZER, VOICES] {
            let resource = map.get(key).unwrap_or_else(|| panic!("{key}"));
            assert!(resource.is_pinned(), "{key}");
            let urls = resource.urls();
            assert_eq!(urls.len(), 1, "{key}");
            assert!(
                urls[0].contains(
                    "Zyphra/ZONOS3-Mini-hf/resolve/73500c5a1fc3e0c4da69908e3cffd9346cdd97c2/"
                ),
                "{}",
                urls[0]
            );
        }
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::kits::support::catalog_contracts::testing::contract_lines;
        assert_eq!(
            contract_lines(ZONOS3_CONTRACT),
            [
                format!("zonos3-tts: {CHECKPOINT} [{SAFETENSORS_FORMAT}]"),
                format!("zonos3-tts: {CONFIG} [json]"),
                format!("zonos3-tts: {TOKENIZER} [tokenizer.json]"),
                format!("zonos3-tts: {VOICES} [{SAFETENSORS_FORMAT}]"),
            ]
        );
    }
}
