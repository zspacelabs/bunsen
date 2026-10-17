//! # Silero VAD pretrained providers

use alloc::sync::Arc;

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

/// Silero's compiled-in providers: the well-known table (with
/// `store_safetensors`), and none without it.
///
/// One row, `well-known:silero/vad-16k`: upstream's
/// `silero_vad_16k.safetensors`, the 16 kHz model, fetched from a pinned
/// revision of `snakers4/silero-vad`.
pub fn default_silero_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    let providers: Vec<Arc<dyn PretrainedProvider>> = vec![
        #[cfg(feature = "store_safetensors")]
        Arc::new(WELL_KNOWN_TABLE.to_table()),
    ];
    providers
}

/// Silero's factory: [`default_silero_providers`] behind
/// [`SileroConstruct`].
///
/// The index a caller holds: `load("silero/vad-16k", ..)` gives the
/// [`SileroVadCollection`](crate::kits::speech::silero_vad::SileroVadCollection)
/// of the 16 kHz model, from upstream's safetensors.
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

    /// `silero/vad-16k` loads through the factory from the cache
    /// `bunsen-bundled-silero` filled, offline: the file is found where the
    /// build laid it out, and the model scores as the shortcut loader's does.
    #[cfg(feature = "silero-weights")]
    #[test]
    #[serial_test::serial]
    fn test_the_bundled_cache_resolves_upstream_offline() {
        use burn::tensor::{
            Distribution,
            Tensor,
        };

        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                pretrained::{
                    PretrainedCache,
                    PretrainedCacheOptions,
                    Provenance,
                },
            },
            kits::speech::silero_vad::{
                SileroVad,
                SileroVadMeta,
            },
            support::testing::{
                DeviceMemoryGuard,
                performance_device,
            },
        };

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(bunsen_bundled_silero::cache_dir().to_path_buf()))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let loaded = default_silero_factory()
            .unwrap()
            .load("silero/vad-16k", &cache, &device)
            .unwrap();
        let part = loaded.resources.get(CHECKPOINT).unwrap();
        assert_eq!(part.provenance, Provenance::Cached);
        assert_eq!(
            part.path,
            bunsen_bundled_silero::silero_vad_16k_safetensors()
        );

        let a = loaded.handle.expect_branch(16000);
        let b = SileroVad::load_16khz_pretrained(&device).unwrap();
        let batch = 2;
        let input =
            Tensor::<2>::random([batch, 64 + a.chunk_size()], Distribution::Default, &device);
        let (pa, _) = a.forward(input.clone(), a.init_state(batch, &device));
        let (pb, _) = b.forward(input, b.init_state(batch, &device));
        let pa: Vec<f32> = pa.into_data().try_into_vec_as().unwrap();
        let pb: Vec<f32> = pb.into_data().try_into_vec_as().unwrap();
        assert_eq!(pa, pb);
    }
}
