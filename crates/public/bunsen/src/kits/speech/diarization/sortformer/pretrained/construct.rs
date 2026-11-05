//! # Constructing a Sortformer pretrained
use std::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    NEMO,
    SAFETENSORS,
    SORTFORMER_CATALOG,
    SORTFORMER_KIT,
};
use crate::{
    burner::module::ModuleInit,
    data::pretrained::{
        CONFIG,
        LoadedResources,
        PretrainedCache,
        PretrainedRef,
        ResourceBundle,
        SAFETENSORS_INDEX,
        SafetensorsCheckpoint,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        speech::{
            diarization::sortformer::{
                NemotronDiarizationConfig,
                Sortformer,
                SortformerConstruct,
                SortformerFamily,
                SortformerPolicy,
                weights::{
                    load_nemotron_hf_tensors,
                    nemotron_config_from_hf,
                },
            },
            nemo::NemoMelFrontEndConfig,
        },
        support::{
            construct::ConstructInit,
            model_config::resolve_ref_presets,
        },
    },
};

/// How a Sortformer checkpoint is read, by its resource `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortformerReader {
    /// A `NeMo` `.nemo` ([`NEMO`]): the config and the weights in one
    /// archive, a Sortformer v2's. Needs the `nemo` feature.
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
    /// kind there is no reader for, or a `.nemo` in a build without the
    /// `nemo` feature.
    pub fn for_kind(kind: Option<&str>) -> BunsenResult<Self> {
        match kind {
            Some(NEMO) if cfg!(feature = "nemo") => Ok(Self::Nemo),
            Some(NEMO) => Err(no_nemo_feature()),
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self::Safetensors),
            other => Err(BunsenError::unsupported(format!(
                "no Sortformer reader for a checkpoint of kind {other:?}"
            ))),
        }
    }
}

/// The bytes of a `.nemo`'s head [`plan`](ConstructInit::plan) reads its config
/// from: `NeMo` writes the config, a few kilobytes, before the weights, and
/// after a tokenizer's files when the model has them.
pub const NEMO_CONFIG_PREFIX: u64 = 1024 * 1024;

fn no_nemo_feature() -> BunsenError {
    BunsenError::unsupported("a .nemo checkpoint needs bunsen's `nemo` feature")
}

/// How a Sortformer pretrained is built, by the [`SortformerReader`] its
/// checkpoint's `kind` names:
///
/// - a `.nemo`: the model and its front end from the archive's config
///   (`NemoSortformerConfig`, feature `nemo`), the weights from its state dict;
///   a [`SortformerFamily::SortformerV2`].
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
    /// The geometry of a map's `config.json` at `path`.
    fn read_config(path: &std::path::Path) -> BunsenResult<NemotronDiarizationConfig> {
        let json = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
        nemotron_config_from_hf(&json).map_err(|e| e.context(path.display()))
    }

    /// A Nemotron construct from a `transformers` checkpoint.
    fn construct_safetensors(
        loaded: &LoadedResources,
        device: &Device,
        load: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<SortformerConstruct> {
        let config = match loaded.get(CONFIG) {
            Some(part) => Self::read_config(&part.path)?,
            None => NemotronDiarizationConfig::nemotron_3_diarization(),
        };
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        let mut model: Sortformer = config.try_init(device)?;
        load_nemotron_hf_tensors(&mut model, tensors)?;
        // The files' precision (f32), or wider as the policy asks.
        let model = load.place(model, device, |_| true)?;
        let front_end = NemoMelFrontEndConfig::nemotron_diarization().with_n_mels(config.n_mels);
        SortformerConstruct::new(model, SortformerFamily::Nemotron, &front_end, device)
    }

    /// Checks a `.nemo`'s config from the head of the archive, read alone
    /// (a range request when it is not local), so a repo of a model bunsen
    /// does not build is refused before the archive is fetched. An archive
    /// whose config is not in its first [`NEMO_CONFIG_PREFIX`] bytes, or a
    /// gzipped one, is checked once it is fetched, in
    /// [`try_init`](ConstructInit::try_init).
    #[cfg(feature = "nemo")]
    fn check_nemo_config(
        map: &ResourceBundle,
        cache: &PretrainedCache,
    ) -> BunsenResult<()> {
        use crate::kits::speech::{
            diarization::sortformer::NemoSortformerConfig,
            nemo::NemoArchive,
        };
        let checkpoint = map.try_get(CHECKPOINT)?;
        let prefix = cache.read_prefix(SORTFORMER_KIT, checkpoint, NEMO_CONFIG_PREFIX)?;
        let Some(yaml) = NemoArchive::config_from_prefix(&prefix) else {
            return Ok(());
        };
        let in_file = |e: BunsenError| e.context(&checkpoint.file);
        let config = NemoSortformerConfig::from_yaml(&yaml).map_err(in_file)?;
        config.structure().map_err(in_file)?;
        config.front_end().map_err(in_file)?;
        Ok(())
    }

    /// A Sortformer v2 construct from a `.nemo`.
    #[cfg(feature = "nemo")]
    fn construct_nemo(
        loaded: &LoadedResources,
        device: &Device,
        load: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<SortformerConstruct> {
        use crate::kits::speech::{
            diarization::sortformer::{
                NemoSortformerConfig,
                weights::load_sortformer_v2_nemo_tensors,
            },
            nemo::NemoArchive,
        };
        let path = loaded.expect(CHECKPOINT)?;
        let in_path = |e: BunsenError| e.context(path.display());
        let archive = NemoArchive::open(path)?;
        let config = NemoSortformerConfig::from_yaml(archive.config_yaml()).map_err(in_path)?;
        let structure = config.structure().map_err(in_path)?;
        let front_end = config.front_end().map_err(in_path)?;
        let mut model: Sortformer = structure.try_init(device)?;
        archive.with_weights(|tensors| load_sortformer_v2_nemo_tensors(&mut model, tensors))?;
        let model = load.place(model, device, |_| true)?;
        SortformerConstruct::new(model, SortformerFamily::SortformerV2, &front_end, device)
    }
}

impl ConstructInit for SortformerConstructConfig {
    type Construct = SortformerConstruct;

    const KIT: &'static str = SORTFORMER_KIT;
    const RESOLVES_PRESETS: bool = true;

    /// The hook for the checkpoint's `kind`, one file or a sharded one's
    /// index.
    fn for_resource_bundle(map: &ResourceBundle) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
        Ok(Self {
            reader: SortformerReader::for_kind(checkpoint.kind.as_deref())?,
        })
    }

    fn plan(
        &self,
        model: &PretrainedRef,
        cache: &PretrainedCache,
    ) -> BunsenResult<ResourceBundle> {
        let map = model.to_resource_bundle();
        match self.reader {
            SortformerReader::Safetensors => {
                if let Some(config) = map.get(CONFIG) {
                    let resolved = cache.resolve(SORTFORMER_KIT, config)?;
                    Self::read_config(&resolved.path)?;
                }
            }
            #[cfg(feature = "nemo")]
            SortformerReader::Nemo => Self::check_nemo_config(&map, cache)?,
            #[cfg(not(feature = "nemo"))]
            SortformerReader::Nemo => return Err(no_nemo_feature()),
        }
        Ok(map)
    }

    /// Builds the model, then resolves its [`SortformerPolicy`] through the
    /// row's presets, or, for a model no row describes, its family's
    /// ([`SortformerFamily::catalog_family`]).
    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
        load: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<SortformerConstruct>> {
        let bundle = match self.reader {
            SortformerReader::Safetensors => Self::construct_safetensors(loaded, device, load)?,
            #[cfg(feature = "nemo")]
            SortformerReader::Nemo => Self::construct_nemo(loaded, device, load)?,
            #[cfg(not(feature = "nemo"))]
            SortformerReader::Nemo => return Err(no_nemo_feature()),
        };
        let family = SORTFORMER_CATALOG.family_presets(bundle.family.catalog_family());
        let policy = resolve_ref_presets(&SortformerPolicy::default(), model, &family)?;
        Ok(Arc::new(
            bundle.with_policy(policy.config.clone(), policy.applied()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::diarization::sortformer::pretrained::SORTFORMER_CATALOG;

    #[test]
    fn test_the_hook_follows_the_kind() {
        let hook = SortformerConstructConfig::for_resource_bundle(
            &SORTFORMER_CATALOG.resource_bundle("nvidia/nemotron-3-diarization"),
        )
        .unwrap();
        assert_eq!(hook.reader, SortformerReader::Safetensors);
        let mut map = SORTFORMER_CATALOG.resource_bundle("nvidia/nemotron-3-diarization");
        for res in map.resources.values_mut() {
            res.kind = Some("pytorch".into());
        }
        assert!(SortformerConstructConfig::for_resource_bundle(&map).is_err());
        assert!(SortformerReader::for_kind(None).is_err());
        assert_eq!(
            SortformerReader::for_kind(Some(NEMO)).ok(),
            cfg!(feature = "nemo").then_some(SortformerReader::Nemo)
        );
    }

    /// A `.nemo` builds a Sortformer v2 construct: the model and the front end
    /// from its config, every parameter from its weights.
    #[cfg(feature = "nemo")]
    #[test]
    fn test_a_nemo_builds_a_v2_bundle() {
        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                pretrained::PretrainedCacheOptions,
            },
            kits::{
                speech::{
                    diarization::sortformer::SortformerMeta,
                    nemo::NemoMelFrontEndMeta,
                },
                support::construct::Deferred,
            },
            support::testing::cpu_device,
        };
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/nemo/toy_sortformer_v2.nemo");
        let mut map = ResourceBundle::given("toy", CHECKPOINT, &path);
        map.resources.get_mut(CHECKPOINT).unwrap().kind = Some(NEMO.into());

        let dir = tempfile::tempdir().unwrap();
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.path().to_path_buf()))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap();
        let bundle = Deferred::<SortformerConstructConfig>::from_resource_bundle(map)
            .unwrap()
            .load(
                &cache,
                &cpu_device(),
                crate::kits::support::construct::LoadPolicy::as_stored(),
            )
            .unwrap()
            .handle;
        assert_eq!(bundle.family, SortformerFamily::SortformerV2);
        assert_eq!((bundle.model.n_mels(), bundle.model.d_model()), (24, 16));
        assert_eq!(bundle.model.n_speakers(), 3);
        assert!(bundle.model.transformer.is_some());
        assert_eq!(bundle.front_end.n_mels(), 24);
        assert!(bundle.to_string().starts_with("Sortformer v2: 3 speakers"));
        // A given file has no row: it takes its family's default presets.
        assert_eq!(bundle.presets.presets, ["low", "none"]);
        let streaming = bundle.streaming().unwrap();
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
            data::pretrained::PresetSelection,
            kits::{
                speech::diarization::{
                    TurnFilterConfig,
                    sortformer::SortformerStreamingConfig,
                },
                support::model_config::resolve_presets,
            },
        };

        let resolve = |family: &str, presets: &[&str]| {
            resolve_presets(
                &SortformerPolicy::default(),
                &SORTFORMER_CATALOG.family_presets(family),
                &PresetSelection::new(presets.iter().copied()),
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

        let table = SORTFORMER_CATALOG.table();
        for row in table.groups.iter().flat_map(|g| &g.items) {
            let family = if row.name.contains("nemotron") {
                "nemotron"
            } else {
                "sortformer-v2"
            };
            assert_eq!(
                row.presets,
                SORTFORMER_CATALOG.family_presets(family),
                "{}",
                row.name
            );
        }
    }

    /// A `.nemo`'s config is checked in `plan` from the archive's head:
    /// a local one from its file, a remote one by a range request that
    /// fetches nothing else, a model bunsen does not build refused there.
    #[cfg(all(feature = "nemo", feature = "fetch"))]
    #[test]
    fn test_plan_checks_a_nemo_config_from_its_head() {
        use crate::{
            data::{
                cache::{
                    BunsenDiskCacheOptions,
                    testing::serve_ranges,
                },
                pretrained::{
                    CacheStatus,
                    PretrainedCacheOptions,
                    Resource,
                    Source,
                },
            },
            errors::{
                BunsenErrorKind,
                testing::ErrorMatcher,
            },
            kits::speech::nemo::{
                NEMO_CONFIG_MEMBER,
                NEMO_WEIGHTS_MEMBER,
                NemoArchive,
            },
        };
        let toy = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/nemo/toy_sortformer_v2.nemo");
        let dir = tempfile::tempdir().unwrap();
        let cache = PretrainedCache::new(
            PretrainedCacheOptions::default().with_disk(
                BunsenDiskCacheOptions::default()
                    .with_cache_dir(Some(dir.path().join("cache")))
                    .without_transfer_observers(),
            ),
        )
        .unwrap();
        let hook = SortformerConstructConfig {
            reader: SortformerReader::Nemo,
        };

        let mut given = ResourceBundle::given("toy", CHECKPOINT, &toy);
        given.resources.get_mut(CHECKPOINT).unwrap().kind = Some(NEMO.into());
        hook.plan(&PretrainedRef::from(given), &cache).unwrap();

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
            let mut map = ResourceBundle::new("remote");
            map.insert(Resource {
                size: None,
                dtype: None,
                key: CHECKPOINT.into(),
                file: "remote.nemo".into(),
                sha256: None,
                kind: Some(NEMO.into()),
                namespace: "test".into(),
                sources: vec![Source::Url(serve_ranges("remote.nemo", body, 1))],
            });
            map
        };
        let yaml = NemoArchive::open(&toy).unwrap().config_yaml().to_string();

        let map = remote(archive(&yaml));
        hook.plan(&PretrainedRef::from(map.clone()), &cache)
            .unwrap();
        let res = map.get(CHECKPOINT).unwrap();
        assert_eq!(cache.status(SORTFORMER_KIT, res), CacheStatus::Remote);

        let modules = "  fc_d_model: 16\n  tf_d_model: 8\n  chunk";
        assert_eq!(yaml.matches(modules).count(), 1);
        let edited = yaml.replace(modules, "  fc_d_model: 32\n  tf_d_model: 8\n  chunk");
        let map = remote(archive(&edited));
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("sortformer_modules.fc_d_model")
            .frame_contains("remote.nemo")
            .assert_err(&hook.plan(&PretrainedRef::from(map.clone()), &cache));
        let res = map.get(CHECKPOINT).unwrap();
        assert_eq!(cache.status(SORTFORMER_KIT, res), CacheStatus::Remote);
    }
}
