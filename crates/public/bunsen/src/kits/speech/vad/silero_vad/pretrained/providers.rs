//! # Silero VAD pretrained providers

use alloc::sync::Arc;

use crate::{
    data::pretrained::PretrainedProvider,
    errors::BunsenResult,
    kits::{
        speech::vad::silero_vad::pretrained::SileroConstructConfig,
        support::construct::PretrainedFactory,
    },
};

/// The Silero kit's name: its rows' `kit` tag.
pub const SILERO_KIT: &str = "silero_vad";

/// The key of the burnpack in a Silero resource bundle.
pub const BURNPACK: &str = "burnpack";

/// The `kind` label of the burnpack resource.
pub const BURNPACK_KIND: &str = "burnpack";

/// The key of the checkpoint in a Silero resource bundle of upstream's files.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` label of a `PyTorch` state dict as safetensors: upstream's
/// `silero_vad_16k.safetensors`.
pub const SAFETENSORS_KIND: &str = "safetensors";

/// What the `silero_vad` kit's loader implements, as a build catalog's
/// contract: `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const SILERO_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `well-known:` rows are its builds.
pub const SILERO_CATALOG_CONTRACT: &str = "silero-vad";

/// Silero's compiled-in providers: the well-known table (with
/// `store`), and none without it.
///
/// One row, `well-known:silero/vad-16k`: upstream's
/// `silero_vad_16k.safetensors`, the 16 kHz model, fetched from a pinned
/// revision of `snakers4/silero-vad`.
pub fn default_silero_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    let providers: Vec<Arc<dyn PretrainedProvider>> = vec![Arc::new(
        crate::data::pretrained::BuildCatalogProvider::expect_builtin(
            SILERO_CATALOG_CONTRACT,
            SILERO_KIT,
        ),
    )];
    providers
}

/// Silero's factory: [`default_silero_providers`] behind
/// [`SileroConstructConfig`].
///
/// The index a caller holds: `load("silero/vad-16k", ..)` gives the
/// [`SileroVadCollection`](crate::kits::speech::vad::silero_vad::SileroVadCollection)
/// of the 16 kHz model, from upstream's safetensors.
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_silero_factory() -> BunsenResult<PretrainedFactory<SileroConstructConfig>> {
    PretrainedFactory::new().with_providers(default_silero_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    /// The embedded catalog hangs together.
    #[test]
    fn test_catalog() {}

    /// Upstream's row is the factory's only one, and the old bare names
    /// are not found.
    #[test]
    fn test_only_upstream_is_listed() {
        let factory = default_silero_factory().unwrap();
        assert_eq!(factory.kit(), SILERO_KIT);
        assert_eq!(factory.ids(), ["well-known:silero/silero_vad/vad-16k"]);
        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(&factory.lookup("vad"));
    }

    /// Upstream's row is pinned to one URL, and answers by its qualified and
    /// short names.
    #[test]
    fn test_the_upstream_row_is_pinned_and_named() {
        let map = crate::data::pretrained::builtin_build_bundle("vad-16k").unwrap();
        map.validate().unwrap();
        let r = map.get(CHECKPOINT).unwrap();
        assert!(r.is_pinned());
        assert_eq!(
            r.urls(),
            [
                "https://raw.githubusercontent.com/snakers4/silero-vad/fb7d7c7f5df63235f0afd037d8397593829aeb41/src/silero_vad/data/silero_vad_16k.safetensors"
            ]
        );

        let factory = default_silero_factory().unwrap();
        for spec in ["well-known:silero/vad-16k", "silero/vad-16k", "vad-16k"] {
            assert_eq!(
                factory.lookup(spec).unwrap().1.name,
                "silero/silero_vad/vad-16k",
                "{spec}"
            );
        }
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::kits::support::catalog_contracts::testing::contract_lines;
        assert_eq!(
            contract_lines(SILERO_CONTRACT),
            [format!("silero-vad: {CHECKPOINT} [{SAFETENSORS_KIND}]")]
        );
    }
}
