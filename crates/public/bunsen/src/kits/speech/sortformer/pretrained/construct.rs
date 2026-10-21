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
///   [`SortformerFamily::Nemotron`]. A `config.json` is read in
///   [`plan`](Construct::plan), before the checkpoint is fetched, so a repo of
///   another model is refused first.
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
        let config: NemoSortformerConfig = archive.config().map_err(in_path)?;
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
        if self.reader == SortformerReader::Safetensors
            && let Some(config) = map.get(CONFIG)
        {
            let resolved = cache.resolve(SORTFORMER_KIT, config)?;
            Self::read_config(&resolved.path)?;
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
}
