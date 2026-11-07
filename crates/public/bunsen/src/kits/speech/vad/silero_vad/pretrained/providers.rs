//! # Silero VAD pretrained providers

use std::sync::Arc;

use crate::{
    data::pretrained::PretrainedProvider,
    errors::BunsenResult,
    kits::{
        speech::vad::silero_vad::pretrained::SileroConstructConfig,
        support::construct::PretrainedFactory,
    },
};

/// The kit segment of a Silero resource's path in the cache:
/// `<cache>/pretrained/silero_vad/<namespace>/<sha256>/<file>`.
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

/// The namespace upstream's files are cached under.
pub const SILERO_NAMESPACE: &str = "silero";

/// The Silero checkpoints bunsen fetches by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider:
/// `well-known:silero/vad-16k`, upstream's `silero_vad_16k.safetensors` at a
/// pinned revision of `snakers4/silero-vad`.
///
/// Built from `validation/catalogs/silero.yml` by `cargo validate catalog`.
#[cfg(feature = "store_safetensors")]
pub static SILERO_CATALOG: crate::data::pretrained::EmbeddedCatalog =
    crate::data::pretrained::EmbeddedCatalog::new("silero", include_str!("catalog.json"));

/// Silero's compiled-in providers: the well-known table (with
/// `store_safetensors`), and none without it.
///
/// One row, `well-known:silero/vad-16k`: upstream's
/// `silero_vad_16k.safetensors`, the 16 kHz model, fetched from a pinned
/// revision of `snakers4/silero-vad`.
pub fn default_silero_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    let providers: Vec<Arc<dyn PretrainedProvider>> = vec![
        #[cfg(feature = "store_safetensors")]
        Arc::new(SILERO_CATALOG.table()),
    ];
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
    #[cfg(feature = "store_safetensors")]
    #[test]
    fn test_catalog() {
        SILERO_CATALOG.catalog().validate(&[]).unwrap();
    }

    /// Upstream's row is the factory's only one, and the old bare names
    /// are not found.
    #[cfg(feature = "store_safetensors")]
    #[test]
    fn test_only_upstream_is_listed() {
        let factory = default_silero_factory().unwrap();
        assert_eq!(factory.kit(), SILERO_KIT);
        assert_eq!(factory.ids(), ["well-known:silero/vad-16k"]);
        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(&factory.lookup("vad"));
    }

    /// Upstream's row is pinned to one URL, and answers by its qualified and
    /// short names.
    #[cfg(feature = "store_safetensors")]
    #[test]
    fn test_the_upstream_row_is_pinned_and_named() {
        let map = SILERO_CATALOG.resource_bundle("silero/vad-16k");
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
                "silero/vad-16k",
                "{spec}"
            );
        }
    }
}
