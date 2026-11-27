//! # Constructing a Sortformer pretrained
use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    NEMO,
    SAFETENSORS,
    SORTFORMER_KIT,
};
use crate::{
    data::{
        builds::{
            BuildRef,
            CONFIG,
            ConstructInit,
        },
        resources::{
            LoadedResourceSet,
            ResourceCache,
            ResourceSet,
            SAFETENSORS_INDEX,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::{
            diarization::sortformer::{
                NemotronDiarizationConfig,
                SortformerConstruct,
                SortformerFamily,
                SortformerPolicy,
                interop::{
                    nemo,
                    transformers,
                },
            },
            support::nemo::NemoMelFrontEndConfig,
        },
        support::model_config::resolve_ref_presets,
    },
};

/// How a Sortformer checkpoint is read, by its resource `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortformerReader {
    /// A `NeMo` `.nemo` ([`NEMO`]): the config and the weights in one
    /// archive, a Sortformer v2's. Needs the `store` feature.
    Nemo,

    /// A `transformers` safetensors checkpoint ([`SAFETENSORS`], or the
    /// index of a sharded one, [`SAFETENSORS_INDEX`]), in
    /// `Nemotron3Diarization`'s layout.
    Safetensors,
}

impl SortformerReader {
    /// The reader for a checkpoint of `kind`.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// kind there is no reader for.
    pub fn for_kind(kind: Option<&str>) -> BunsenResult<Self> {
        match kind {
            Some(NEMO) => Ok(Self::Nemo),
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self::Safetensors),
            other => Err(crate::kits::support::pretrained::unsupported_kind(
                "Sortformer",
                "checkpoint",
                other,
            )),
        }
    }
}

/// The bytes of a `.nemo`'s head [`plan`](ConstructInit::plan) reads its config
/// from: `NeMo` writes the config, a few kilobytes, before the weights, and
/// after a tokenizer's files when the model has them.
pub const NEMO_CONFIG_PREFIX: u64 = 1024 * 1024;

/// How a Sortformer pretrained is built, by the [`SortformerReader`] its
/// checkpoint's `kind` names:
///
/// - a `.nemo`: the model and its front end from the archive's config
///   ([`NemoSortformerMirror`](crate::kits::speech::diarization::sortformer::interop::nemo::NemoSortformerMirror)), the weights from its state
///   dict; a [`SortformerFamily::SortformerV2`].
/// - a `transformers` safetensors checkpoint (one file, or a sharded one's
///   index): the geometry its `config.json` gives (Nemotron-3-Diarization's
///   when the map has none), behind Nemotron's front end; a
///   [`SortformerFamily::Nemotron`].
///
/// Either way the config is read in [`plan`](ConstructInit::plan), before the
/// checkpoint is fetched, so a repo of another model is refused first: a
/// `config.json` whole, a `.nemo`'s from the archive's head alone.
///
/// The Sortformer kit's construct config: its [`ConstructInit`] hook.
/// [`default_sortformer_factory`](super::default_sortformer_factory) attaches
/// it; a caller never builds one.
#[derive(Clone, Debug)]
pub struct SortformerConstructConfig {
    /// The checkpoint's reader.
    pub reader: SortformerReader,
}

impl SortformerConstructConfig {
    /// A Nemotron construct from a `transformers` checkpoint.
    fn construct_safetensors(
        loaded: &LoadedResourceSet,
        device: &Device,
        load: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<SortformerConstruct> {
        let config = match loaded.get(CONFIG) {
            Some(part) => transformers::nemotron_diarization_config_from_path(&part.path)?,
            None => NemotronDiarizationConfig::nemotron_3_diarization(),
        };
        let model = transformers::load_nemotron_diarization(loaded, &config, device, load)?;
        let front_end = NemoMelFrontEndConfig::nemotron_diarization().with_n_mels(config.n_mels);
        SortformerConstruct::new(model, SortformerFamily::Nemotron, &front_end, device)
    }

    /// Checks a `.nemo`'s config from the head of the archive, read alone
    /// (a range request when it is not local), so a repo of a model bunsen
    /// does not build is refused before the archive is fetched. An archive
    /// whose config is not in its first [`NEMO_CONFIG_PREFIX`] bytes, or a
    /// gzipped one, is checked once it is fetched, in
    /// [`try_init`](ConstructInit::try_init).
    fn check_nemo_config(
        map: &ResourceSet,
        cache: &ResourceCache,
    ) -> BunsenResult<()> {
        use crate::data::formats::nemo::NemoArchive;
        let checkpoint = map.try_get(CHECKPOINT)?;
        let prefix = cache.read_prefix(checkpoint, NEMO_CONFIG_PREFIX)?;
        let Some(yaml) = NemoArchive::config_from_prefix(&prefix) else {
            return Ok(());
        };
        let in_file = |e: BunsenError| e.context(&checkpoint.file);
        let config = nemo::NemoSortformerMirror::from_yaml(&yaml).map_err(in_file)?;
        config.sortformer_structure_config().map_err(in_file)?;
        config.mel_front_end_config().map_err(in_file)?;
        Ok(())
    }

    /// A Sortformer v2 construct from a `.nemo`.
    fn construct_nemo(
        loaded: &LoadedResourceSet,
        device: &Device,
        load: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<SortformerConstruct> {
        let (model, front_end) = nemo::load_sortformer_v2(loaded, device, load)?;
        SortformerConstruct::new(model, SortformerFamily::SortformerV2, &front_end, device)
    }
}

impl ConstructInit for SortformerConstructConfig {
    type Construct = SortformerConstruct;

    const KIT: &'static str = SORTFORMER_KIT;
    const RESOLVES_PRESETS: bool = true;

    fn applied_presets(
        construct: &Self::Construct
    ) -> Option<crate::data::catalog::presets::PresetSelection> {
        Some(construct.presets.clone())
    }

    /// The hook for the checkpoint's `kind`, one file or a sharded one's
    /// index.
    fn for_resource_set(map: &ResourceSet) -> BunsenResult<Self> {
        let checkpoint = map.try_get_checkpoint(CHECKPOINT)?;
        Ok(Self {
            reader: SortformerReader::for_kind(checkpoint.format.as_deref())?,
        })
    }

    fn plan(
        &self,
        model: &BuildRef,
        cache: &ResourceCache,
    ) -> BunsenResult<ResourceSet> {
        let map = model.to_resource_set();
        match self.reader {
            SortformerReader::Safetensors => {
                if let Some(config) = map.get(CONFIG) {
                    let resolved = cache.resolve(config)?;
                    transformers::nemotron_diarization_config_from_path(&resolved.path)?;
                }
            }
            SortformerReader::Nemo => Self::check_nemo_config(&map, cache)?,
        }
        Ok(map)
    }

    /// Builds the model, then resolves its [`SortformerPolicy`] through the
    /// row's presets, or, for a model no row describes, its family's
    /// ([`SortformerFamily::template_class`]).
    fn try_init(
        &self,
        model: &BuildRef,
        loaded: &LoadedResourceSet,
        device: &Device,
        load: crate::data::formats::LoadPolicy,
    ) -> BunsenResult<Arc<SortformerConstruct>> {
        let construct = match self.reader {
            SortformerReader::Safetensors => Self::construct_safetensors(loaded, device, load)?,
            SortformerReader::Nemo => Self::construct_nemo(loaded, device, load)?,
        };
        let family = crate::data::catalog::BuiltinCatalog::get()?
            .template_presets(construct.family.template_class())?;
        let policy = resolve_ref_presets(&SortformerPolicy::default(), model, &family)?;
        Ok(Arc::new(
            construct.with_policy(policy.config.clone(), policy.applied()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::testdata,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        let hook = SortformerConstructConfig::for_resource_set(
            &crate::data::catalog::BuiltinCatalog::get()
                .unwrap()
                .build("nvidia/nemotron-3-diarization")
                .unwrap()
                .resource_set(),
        )
        .unwrap();
        assert_eq!(hook.reader, SortformerReader::Safetensors);
        let mut map = crate::data::catalog::BuiltinCatalog::get()
            .unwrap()
            .build("nvidia/nemotron-3-diarization")
            .unwrap()
            .resource_set();
        for res in map.resources.values_mut() {
            res.format = Some("pytorch".into());
        }
        ErrorMatcher::unsupported().assert_err(&SortformerConstructConfig::for_resource_set(&map));
        ErrorMatcher::unsupported().assert_err(&SortformerReader::for_kind(None));
        assert_eq!(
            SortformerReader::for_kind(Some(NEMO)).ok(),
            Some(SortformerReader::Nemo)
        );
    }

    /// A `.nemo` builds a Sortformer v2 construct: the model and the front end
    /// from its config, every parameter from its weights.
    #[test]
    fn test_a_nemo_builds_a_v2_bundle() {
        use crate::{
            data::{
                builds::DeferredBuild,
                cache::BunsenDiskCacheOptions,
                resources::ResourceCacheOptions,
            },
            kits::speech::{
                diarization::sortformer::SortformerMeta,
                support::nemo::NemoMelFrontEndMeta,
            },
            support::testing::cpu_device,
        };
        let path = testdata!("nemo/toy_sortformer_v2.nemo");
        let mut map = ResourceSet::given("toy", CHECKPOINT, &path);
        map.resources.get_mut(CHECKPOINT).unwrap().format = Some(NEMO.into());

        let dir = tempfile::tempdir().unwrap();
        let cache = ResourceCache::new(
            ResourceCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().to_path_buf()))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let construct = DeferredBuild::<SortformerConstructConfig>::from_resource_set(map)
            .unwrap()
            .load(
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::load_as_stored(),
            )
            .unwrap()
            .handle;
        assert_eq!(construct.family, SortformerFamily::SortformerV2);
        assert_eq!(
            (construct.model.n_mels(), construct.model.d_model()),
            (24, 16)
        );
        assert_eq!(construct.model.n_speakers(), 3);
        assert!(construct.model.transformer.is_some());
        assert_eq!(construct.front_end.n_mels(), 24);
        assert!(
            construct
                .to_string()
                .starts_with("Sortformer v2: 3 speakers")
        );
        // A given file has no row: it takes its family's default presets.
        assert_eq!(construct.presets.presets, ["low", "none"]);
        let streaming = construct.streaming().unwrap();
        assert_eq!(
            (streaming.latency_frames(), streaming.left_context),
            (13, 1)
        );
    }

    /// Each family's latencies are its model card's: the latency in frames
    /// each name says, a budget the model's speakers fit, and the same wall
    /// time in every family that has it; the turn presets are `NeMo`'s
    /// tuned parameters; the catalog's rows inherit their family's.
    #[test]
    fn test_family_presets() {
        use crate::{
            data::catalog::presets::PresetSelection,
            kits::{
                speech::diarization::{
                    TurnFilterConfig,
                    sortformer::SortformerStreamingConfig,
                },
                support::model_config::resolve_presets,
            },
        };

        let family_presets = |family: &str| {
            crate::data::catalog::BuiltinCatalog::get()
                .unwrap()
                .template_presets(&format!("nvidia/sortformer/{family}"))
                .unwrap()
        };
        let resolve = |family: &str, presets: &[&str]| {
            resolve_presets(
                &SortformerPolicy::default(),
                &family_presets(family),
                &PresetSelection::of(presets.iter().copied()),
            )
            .unwrap()
            .config
        };
        let streaming = |family: &str, latency: &str| -> SortformerStreamingConfig {
            resolve(family, &[latency]).streaming.unwrap()
        };

        let frames = [
            ("ultra-low", 4),
            ("very-low", 8),
            ("low", 13),
            ("high", 125),
            ("offline", 380),
        ];
        for (family, speakers, latencies) in [
            (
                "nemotron",
                8,
                &["ultra-low", "very-low", "low", "offline"][..],
            ),
            (
                "sortformer-v2",
                4,
                &["ultra-low", "low", "high", "offline"][..],
            ),
        ] {
            for latency in latencies {
                let c = streaming(family, latency);
                c.check(speakers).unwrap();
                let want = frames.iter().find(|(n, _)| n == latency).unwrap().1;
                assert_eq!(c.latency_frames(), want, "{family} {latency}");
            }
        }
        let low = streaming("nemotron", "low");
        assert_eq!((low.left_context, low.speaker_budget(8)), (0, 32));
        let v2 = streaming("sortformer-v2", "low");
        assert_eq!((v2.left_context, v2.speaker_budget(4)), (1, 44));
        assert_eq!(resolve("nemotron", &[]).streaming.unwrap(), low);

        assert_eq!(resolve("nemotron", &[]).turns, TurnFilterConfig::none());
        assert_eq!(
            resolve("sortformer-v2", &["dihard3"]).turns,
            TurnFilterConfig::new()
                .with_onset(0.56)
                .with_offset(1.0)
                .with_pad_onset(0.063)
                .with_pad_offset(0.002)
                .with_min_duration_on(0.007)
                .with_min_duration_off(0.151)
        );
        assert_eq!(
            resolve("nemotron", &["callhome"]).turns.min_duration_on,
            0.511
        );

        let provider = crate::data::catalog::BuiltinCatalog::expect_provider(
            super::super::SORTFORMER_CATALOG_CONTRACT,
        );
        for row in provider.builds() {
            let family = if row.name.to_string().contains("nemotron") {
                "nemotron"
            } else {
                "sortformer-v2"
            };
            assert_eq!(row.presets, family_presets(family), "{}", row.name);
        }
    }

    /// A `.nemo`'s config is checked in `plan` from the archive's head:
    /// a local one from its file, a remote one by a range request that
    /// fetches nothing else, a model bunsen does not build refused there.
    #[cfg(feature = "fetch")]
    #[test]
    fn test_plan_checks_a_nemo_config_from_its_head() {
        use crate::{
            data::{
                cache::{
                    BunsenDiskCacheOptions,
                    testing::serve_ranges,
                },
                formats::nemo::{
                    NEMO_CONFIG_MEMBER,
                    NEMO_WEIGHTS_MEMBER,
                    NemoArchive,
                },
                resources::{
                    CacheStatus,
                    Resource,
                    ResourceCacheOptions,
                    Source,
                },
            },
            errors::testing::ErrorMatcher,
        };
        let toy = testdata!("nemo/toy_sortformer_v2.nemo");
        let dir = tempfile::tempdir().unwrap();
        let cache = ResourceCache::new(
            ResourceCacheOptions::default().with_disk(
                BunsenDiskCacheOptions::default()
                    .with_cache_dir(Some(dir.path().join("cache")))
                    .without_transfer_observers(),
            ),
        )
        .unwrap();
        let hook = SortformerConstructConfig {
            reader: SortformerReader::Nemo,
        };

        let mut given = ResourceSet::given("toy", CHECKPOINT, &toy);
        given.resources.get_mut(CHECKPOINT).unwrap().format = Some(NEMO.into());
        hook.plan(&BuildRef::from(given), &cache).unwrap();

        // An archive of the toy's config, edited, and stand-in weights.
        let archive = |yaml: &str| -> &'static [u8] {
            let mut tar = tar::Builder::new(Vec::new());
            for (name, bytes) in [
                (NEMO_CONFIG_MEMBER, yaml.as_bytes()),
                (NEMO_WEIGHTS_MEMBER, &[0u8; 4096][..]),
            ] {
                let mut header = tar::Header::new_gnu();
                header.set_size(bytes.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                tar.append_data(&mut header, format!("./{name}"), bytes)
                    .unwrap();
            }
            Box::leak(tar.into_inner().unwrap().into_boxed_slice())
        };
        let remote = |body: &'static [u8]| {
            let mut map = ResourceSet::new("remote");
            map.insert(Resource {
                size: None,
                dtype: None,
                slot: CHECKPOINT.into(),
                file: "remote.nemo".into(),
                sha256: None,
                format: Some(NEMO.into()),
                name: "test".into(),
                sources: vec![Source::Url(serve_ranges("remote.nemo", body, 1))],
            });
            map
        };
        let yaml = NemoArchive::open(&toy).unwrap().config_yaml().to_string();

        let map = remote(archive(&yaml));
        hook.plan(&BuildRef::from(map.clone()), &cache).unwrap();
        let res = map.get(CHECKPOINT).unwrap();
        assert_eq!(cache.status(res), CacheStatus::Remote);

        let modules = "  fc_d_model: 16\n  tf_d_model: 8\n  chunk";
        assert_eq!(yaml.matches(modules).count(), 1);
        let edited = yaml.replace(modules, "  fc_d_model: 32\n  tf_d_model: 8\n  chunk");
        let map = remote(archive(&edited));
        ErrorMatcher::unsupported()
            .message_contains("sortformer_modules.fc_d_model")
            .frame_contains("remote.nemo")
            .assert_err(&hook.plan(&BuildRef::from(map.clone()), &cache));
        let res = map.get(CHECKPOINT).unwrap();
        assert_eq!(cache.status(res), CacheStatus::Remote);
    }
}
