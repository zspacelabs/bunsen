//! # Constructing a Sortformer pretrained
use std::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    NEMO,
    SAFETENSORS,
    SORTFORMER_KIT,
};
use crate::{
    burner::module::ModuleInit,
    data::pretrained::{
        CONFIG,
        Construct,
        LoadedResources,
        PretrainedCache,
        PretrainedRef,
        ResourceMap,
        SAFETENSORS_INDEX,
        SafetensorsCheckpoint,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::speech::{
        nemo::NemoMelFrontEndConfig,
        sortformer::{
            NemotronDiarizationConfig,
            Sortformer,
            SortformerBundle,
            SortformerFamily,
            weights::{
                load_nemotron_hf_tensors,
                nemotron_config_from_hf,
            },
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

/// The bytes of a `.nemo`'s head [`plan`](Construct::plan) reads its config
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
///   ([`NemoSortformerConfig`](crate::kits::speech::sortformer::NemoSortformerConfig)),
///   the weights from its state dict; a [`SortformerFamily::SortformerV2`].
/// - a `transformers` safetensors checkpoint (one file, or a sharded one's
///   index): the geometry its `config.json` gives (Nemotron-3-Diarization's
///   when the map has none), behind Nemotron's front end; a
///   [`SortformerFamily::Nemotron`].
///
/// Either way the config is read in [`plan`](Construct::plan), before the
/// checkpoint is fetched, so a repo of another model is refused first: a
/// `config.json` whole, a `.nemo`'s from the archive's head alone.
///
/// The [`Construct`] hook for the Sortformer kit.
/// [`default_sortformer_factory`](super::default_sortformer_factory) attaches
/// it; a caller never builds one.
#[derive(Clone, Debug)]
pub struct SortformerConstruct {
    /// The checkpoint's reader.
    pub reader: SortformerReader,
}

impl SortformerConstruct {
    /// The geometry of a map's `config.json` at `path`.
    fn read_config(path: &std::path::Path) -> BunsenResult<NemotronDiarizationConfig> {
        let json = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
        nemotron_config_from_hf(&json).map_err(|e| e.context(path.display()))
    }

    /// A Nemotron bundle from a `transformers` checkpoint.
    fn construct_safetensors(
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<SortformerBundle> {
        let config = match loaded.get(CONFIG) {
            Some(part) => Self::read_config(&part.path)?,
            None => NemotronDiarizationConfig::nemotron_3_diarization(),
        };
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        let mut model: Sortformer = config.try_init(device)?;
        load_nemotron_hf_tensors(&mut model, tensors)?;
        let front_end = NemoMelFrontEndConfig::nemotron_diarization().with_n_mels(config.n_mels);
        SortformerBundle::new(model, SortformerFamily::Nemotron, &front_end, device)
    }

    /// Checks a `.nemo`'s config from the head of the archive, read alone
    /// (a range request when it is not local), so a repo of a model bunsen
    /// does not build is refused before the archive is fetched. An archive
    /// whose config is not in its first [`NEMO_CONFIG_PREFIX`] bytes, or a
    /// gzipped one, is checked once it is fetched, in
    /// [`construct`](Construct::construct).
    #[cfg(feature = "nemo")]
    fn check_nemo_config(
        map: &ResourceMap,
        cache: &PretrainedCache,
    ) -> BunsenResult<()> {
        use crate::kits::speech::{
            nemo::NemoArchive,
            sortformer::NemoSortformerConfig,
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

    /// A Sortformer v2 bundle from a `.nemo`.
    #[cfg(feature = "nemo")]
    fn construct_nemo(
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<SortformerBundle> {
        use crate::kits::speech::{
            nemo::NemoArchive,
            sortformer::{
                NemoSortformerConfig,
                weights::load_sortformer_v2_nemo_tensors,
            },
        };
        let path = loaded.expect(CHECKPOINT)?;
        let in_path = |e: BunsenError| e.context(path.display());
        let archive = NemoArchive::open(path)?;
        let config = NemoSortformerConfig::from_yaml(archive.config_yaml()).map_err(in_path)?;
        let structure = config.structure().map_err(in_path)?;
        let front_end = config.front_end().map_err(in_path)?;
        let mut model: Sortformer = structure.try_init(device)?;
        archive.with_weights(|tensors| load_sortformer_v2_nemo_tensors(&mut model, tensors))?;
        SortformerBundle::new(model, SortformerFamily::SortformerV2, &front_end, device)
    }
}

impl Construct for SortformerConstruct {
    type Built = SortformerBundle;

    const KIT: &'static str = SORTFORMER_KIT;

    /// The hook for the checkpoint's `kind`, one file or a sharded one's
    /// index.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
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
    ) -> BunsenResult<ResourceMap> {
        let map = model.to_map();
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

    fn construct(
        &self,
        _model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
    ) -> BunsenResult<Arc<SortformerBundle>> {
        let bundle = match self.reader {
            SortformerReader::Safetensors => Self::construct_safetensors(loaded, device)?,
            #[cfg(feature = "nemo")]
            SortformerReader::Nemo => Self::construct_nemo(loaded, device)?,
            #[cfg(not(feature = "nemo"))]
            SortformerReader::Nemo => return Err(no_nemo_feature()),
        };
        Ok(Arc::new(bundle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::sortformer::pretrained::NEMOTRON_3_DIARIZATION_FILES;

    #[test]
    fn test_the_hook_follows_the_kind() {
        let hook = SortformerConstruct::for_map(&NEMOTRON_3_DIARIZATION_FILES.to_map()).unwrap();
        assert_eq!(hook.reader, SortformerReader::Safetensors);
        let mut map = NEMOTRON_3_DIARIZATION_FILES.to_map();
        for res in map.resources.values_mut() {
            res.kind = Some("pytorch".into());
        }
        assert!(SortformerConstruct::for_map(&map).is_err());
        assert!(SortformerReader::for_kind(None).is_err());
        assert_eq!(
            SortformerReader::for_kind(Some(NEMO)).ok(),
            cfg!(feature = "nemo").then_some(SortformerReader::Nemo)
        );
    }

    /// A `.nemo` builds a Sortformer v2 bundle: the model and the front end
    /// from its config, every parameter from its weights.
    #[cfg(feature = "nemo")]
    #[test]
    fn test_a_nemo_builds_a_v2_bundle() {
        use crate::{
            data::{
                cache::BunsenDiskCacheOptions,
                pretrained::{
                    Deferred,
                    PretrainedCacheOptions,
                },
            },
            kits::speech::{
                nemo::NemoMelFrontEndMeta,
                sortformer::{
                    SortformerLatency,
                    SortformerMeta,
                },
            },
            support::testing::cpu_device,
        };
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/nemo/toy_sortformer_v2.nemo");
        let mut map = ResourceMap::given("toy", CHECKPOINT, &path);
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
        let bundle = Deferred::<SortformerConstruct>::from_map(map)
            .unwrap()
            .load(&cache, &cpu_device())
            .unwrap()
            .handle;
        assert_eq!(bundle.family, SortformerFamily::SortformerV2);
        assert_eq!((bundle.model.n_mels(), bundle.model.d_model()), (24, 16));
        assert_eq!(bundle.model.n_speakers(), 3);
        assert!(bundle.model.transformer.is_some());
        assert_eq!(bundle.front_end.n_mels(), 24);
        assert!(bundle.to_string().starts_with("Sortformer v2: 3 speakers"));
        assert_eq!(
            bundle.preset(SortformerLatency::High).unwrap(),
            crate::kits::speech::sortformer::SortformerStreamingConfig::sortformer_v2_high_latency(
            )
        );
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
        let hook = SortformerConstruct {
            reader: SortformerReader::Nemo,
        };

        let mut given = ResourceMap::given("toy", CHECKPOINT, &toy);
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
            let mut map = ResourceMap::new("remote");
            map.insert(Resource {
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
