//! # Silero VAD pretrained providers

use std::sync::Arc;

use crate::{
    data::pretrained::{
        PretrainedFactory,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::speech::silero_vad::pretrained::SileroConstruct,
};

/// The kit segment of a Silero resource's path in the cache:
/// `<cache>/pretrained/silero_vad/<namespace>/<sha256>/<file>`.
pub const SILERO_KIT: &str = "silero_vad";

/// The key of the burnpack in a Silero resource map.
pub const BURNPACK: &str = "burnpack";

/// The `kind` label of the burnpack resource.
pub const BURNPACK_KIND: &str = "burnpack";

/// The namespace the burnpack is cached under: bunsen's, since the
/// conversion is bunsen's.
pub const BUNSEN_NAMESPACE: &str = "bunsen";

/// The key of the checkpoint in a Silero resource map of upstream's files.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` label of a `PyTorch` state dict as safetensors: upstream's
/// `silero_vad_16k.safetensors`.
pub const SAFETENSORS_KIND: &str = "safetensors";

/// The namespace upstream's files are cached under.
pub const SILERO_NAMESPACE: &str = "silero";

#[cfg(feature = "store_safetensors")]
const SILERO_ORIGIN: &str = "https://github.com/snakers4/silero-vad";

#[cfg(feature = "store_safetensors")]
mod upstream {
    use super::{
        CHECKPOINT,
        SAFETENSORS_KIND,
        SILERO_NAMESPACE,
        SILERO_ORIGIN,
    };
    use crate::data::pretrained::{
        StaticBase,
        StaticPretrained,
        StaticPretrainedGroup,
        StaticPretrainedTable,
        StaticResource,
        StaticResourceMap,
        WELL_KNOWN,
    };

    /// Upstream's `silero_vad_16k.safetensors`, the 16 kHz model's state
    /// dict, at a fixed revision of `snakers4/silero-vad`.
    pub static SILERO_VAD_16K_FILES: StaticResourceMap<'static> = StaticResourceMap {
        name: "silero/vad-16k",
        description: "Silero VAD's 16 kHz model, upstream's safetensors (f32)",
        license: Some("MIT"),
        origin: Some(SILERO_ORIGIN),
        namespace: SILERO_NAMESPACE,
        bases: &[StaticBase::Url(
            "https://raw.githubusercontent.com/snakers4/silero-vad/fb7d7c7f5df63235f0afd037d8397593829aeb41/src/silero_vad/data",
        )],
        resources: &[StaticResource {
            key: CHECKPOINT,
            file: "silero_vad_16k.safetensors",
            sha256: Some("c59271c284ae9c8335d795d60e0bfdb71aaaceec578d9bd9ffc1b8153c319ea1"),
            kind: Some(SAFETENSORS_KIND),
            sources: &[],
        }],
    };

    static VAD_16K: StaticPretrained<'static> = StaticPretrained {
        name: "vad-16k",
        aliases: &[],
        description: "Silero VAD, the 16 kHz model, from upstream's safetensors (1.2 MB)",
        license: Some("MIT"),
        origin: Some(SILERO_ORIGIN),
        prefab: None,
        maps: &[&SILERO_VAD_16K_FILES],
    };

    /// `snakers4/silero-vad`'s own files.
    pub static SILERO_UPSTREAM: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
        name: "silero",
        description: "snakers4/silero-vad, as upstream publishes it",
        license: Some("MIT"),
        origin: Some(SILERO_ORIGIN),
        items: &[&VAD_16K],
    };

    /// The Silero checkpoints bunsen fetches by name, behind the
    /// [`WELL_KNOWN`] provider: `well-known:silero/vad-16k`.
    pub static WELL_KNOWN_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
        name: WELL_KNOWN,
        description: "the Silero VAD checkpoints bunsen fetches by name",
        groups: &[&SILERO_UPSTREAM],
    };
}

#[cfg(feature = "store_safetensors")]
pub use upstream::*;

#[cfg(feature = "silero-weights")]
mod bundled {
    use super::{
        BUNSEN_NAMESPACE,
        BURNPACK,
        BURNPACK_KIND,
    };
    use crate::data::pretrained::{
        BUNDLED,
        StaticPretrained,
        StaticPretrainedGroup,
        StaticPretrainedTable,
        StaticResource,
        StaticResourceMap,
        StaticSource,
    };

    /// The burnpack linked into the binary, pinned to the digest its
    /// build computed.
    pub static BUNDLED_BURNPACK: StaticResourceMap<'static> = StaticResourceMap {
        name: "bunsen/silero_vad_op18_ifless.bpk",
        description: "the Silero VAD graph as a burnpack, both sample-rate branches",
        license: Some("MIT"),
        origin: Some("https://github.com/snakers4/silero-vad"),
        namespace: BUNSEN_NAMESPACE,
        bases: &[],
        resources: &[StaticResource {
            key: BURNPACK,
            file: bunsen_bundled_silero::BURNPACK_FILE,
            sha256: Some(bunsen_bundled_silero::BURNPACK_SHA256),
            kind: Some(BURNPACK_KIND),
            sources: &[StaticSource::Bundled(
                bunsen_bundled_silero::BURNPACK_WEIGHTS,
            )],
        }],
    };

    static VAD: StaticPretrained<'static> = StaticPretrained {
        name: "vad",
        aliases: &[],
        description: "Silero VAD, the 16 kHz and 8 kHz branches in one burnpack",
        license: Some("MIT"),
        origin: Some("https://github.com/snakers4/silero-vad"),
        prefab: None,
        maps: &[&BUNDLED_BURNPACK],
    };

    /// `snakers4/silero-vad`, as bunsen converts it.
    pub static SILERO: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
        name: "silero",
        description: "snakers4/silero-vad, as bunsen converts it",
        license: Some("MIT"),
        origin: Some("https://github.com/snakers4/silero-vad"),
        items: &[&VAD],
    };

    /// The rows `bunsen-bundled-silero` links into the binary, behind the
    /// [`BUNDLED`] provider: `bundled:silero/vad`.
    pub static BUNDLED_TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
        name: BUNDLED,
        description: "the Silero VAD burnpack bunsen-bundled-silero links into the binary",
        groups: &[&SILERO],
    };
}

#[cfg(feature = "silero-weights")]
pub use bundled::*;

/// Silero's compiled-in providers, in search order: the well-known table
/// (with `store_safetensors`), then the bundled table (with
/// `silero-weights`).
///
/// - `well-known:silero/vad-16k`: upstream's `silero_vad_16k.safetensors`, the
///   16 kHz model, fetched from a pinned revision of `snakers4/silero-vad`.
/// - `bundled:silero/vad`: the Silero VAD graph, both sample-rate branches, as
///   the burnpack `bunsen-bundled-silero` generates from the ONNX export and
///   links into the binary. It has no URL: the burnpack is bunsen's build
///   artifact, pinned to the digest that build computed, and written into the
///   cache from the binary on first use.
pub fn default_silero_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    let providers: Vec<Arc<dyn PretrainedProvider>> = vec![
        #[cfg(feature = "store_safetensors")]
        Arc::new(WELL_KNOWN_TABLE.to_table()),
        #[cfg(feature = "silero-weights")]
        Arc::new(BUNDLED_TABLE.to_table()),
    ];
    providers
}

/// Silero's factory: [`default_silero_providers`] behind
/// [`SileroConstruct`].
///
/// The index a caller holds: `load("silero/vad-16k", ..)` or
/// `load("bundled:silero/vad", ..)` gives the
/// [`SileroVadCollection`](crate::kits::speech::silero_vad::SileroVadCollection):
/// the 16 kHz branch from upstream's safetensors, or both branches from the
/// bundled burnpack.
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_silero_factory() -> BunsenResult<PretrainedFactory<SileroConstruct>> {
    PretrainedFactory::new().with_providers(default_silero_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    /// Without the bundle, only upstream's row is listed, and the bundle's
    /// name is not found.
    #[cfg(all(feature = "store_safetensors", not(feature = "silero-weights")))]
    #[test]
    fn test_without_the_bundle_only_upstream_is_listed() {
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
        let map = SILERO_VAD_16K_FILES.to_map();
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

    /// The bundled row is pinned to the build's digest, has the bytes, and
    /// is the factory's only ref, qualified or bare.
    #[cfg(feature = "silero-weights")]
    #[test]
    fn test_the_bundled_row_is_pinned_and_listed() {
        use crate::data::pretrained::BUNDLED;

        let map = BUNDLED_BURNPACK.to_map();
        map.validate().unwrap();
        let r = map.get(BURNPACK).unwrap();
        assert!(r.is_pinned());
        assert_eq!(
            r.sha256.as_deref(),
            Some(bunsen_bundled_silero::BURNPACK_SHA256)
        );
        assert_eq!(
            r.bundled().map(<[u8]>::len),
            Some(bunsen_bundled_silero::BURNPACK_WEIGHTS.len())
        );
        assert!(r.urls().is_empty());

        let factory = default_silero_factory().unwrap();
        assert_eq!(factory.kit(), SILERO_KIT);
        assert_eq!(
            factory.ids(),
            ["well-known:silero/vad-16k", "bundled:silero/vad"]
        );
        assert_eq!(factory.provider(BUNDLED).unwrap().name(), "bundled");
        for spec in ["bundled:silero/vad", "silero/vad", "vad"] {
            assert_eq!(factory.lookup(spec).unwrap().1.name, "silero/vad", "{spec}");
        }
        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(&factory.lookup("silero/v4"));
    }

    /// `bundled:silero/vad` loads through the factory from a cache that
    /// has nothing: the burnpack lands in it from the binary, is cached
    /// after, and the collection scores as the bytes loader's does.
    #[cfg(feature = "silero-weights")]
    #[test]
    #[serial_test::serial]
    fn test_the_bundled_burnpack_loads_through_the_factory() {
        use burn::tensor::{
            Distribution,
            Tensor,
        };

        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                pretrained::{
                    CacheStatus,
                    PretrainedCache,
                    PretrainedCacheOptions,
                    Provenance,
                },
            },
            kits::speech::silero_vad::{
                SileroVadCollection,
                SileroVadMeta,
            },
            support::testing::{
                DeviceMemoryGuard,
                performance_device,
            },
        };

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let dir = tempfile::tempdir().unwrap();
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let factory = default_silero_factory().unwrap();

        let model = factory.resolve("bundled:silero/vad", &cache).unwrap();
        assert_eq!(model.id(), "bundled:silero/vad");
        assert_eq!(model.status(&cache)[BURNPACK], CacheStatus::Bundled);

        let loaded = factory.load("bundled:silero/vad", &cache, &device).unwrap();
        let part = loaded.resources.get(BURNPACK).unwrap();
        assert_eq!(part.provenance, Provenance::Bundled);
        assert!(part.path.starts_with(dir.path().join("cache")));
        assert_eq!(
            std::fs::read(&part.path).unwrap(),
            bunsen_bundled_silero::BURNPACK_WEIGHTS
        );
        assert_eq!(model.status(&cache)[BURNPACK], CacheStatus::Cached);

        let again = factory.load("vad", &cache, &device).unwrap();
        assert_eq!(
            again.resources.get(BURNPACK).unwrap().provenance,
            Provenance::Cached
        );

        // The same numbers as the bytes loader's, on both branches.
        let direct = SileroVadCollection::load_pretrained(&device).unwrap();
        for rate in [16000, 8000] {
            let a = loaded.handle.expect_branch(rate);
            let b = direct.expect_branch(rate);
            let batch = 2;
            let input =
                Tensor::<2>::random([batch, 64 + a.chunk_size()], Distribution::Default, &device);
            let (pa, _) = a.forward(input.clone(), a.init_state(batch, &device));
            let (pb, _) = b.forward(input, b.init_state(batch, &device));
            let pa: Vec<f32> = pa.into_data().try_into_vec_as().unwrap();
            let pb: Vec<f32> = pb.into_data().try_into_vec_as().unwrap();
            assert_eq!(pa.len(), batch);
            for (x, y) in pa.iter().zip(&pb) {
                assert!((x - y).abs() < 1e-5, "{rate} Hz: {x} vs {y}");
            }
        }
    }
}
