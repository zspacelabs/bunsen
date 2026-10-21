//! # Constructing a Sortformer pretrained
use std::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
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

/// How a Sortformer pretrained is built: its `transformers` safetensors
/// checkpoint (one file, or a sharded one's index), into the geometry its
/// `config.json` gives (Nemotron-3-Diarization's when the map has none),
/// behind its front end.
///
/// A `config.json` is read in [`plan`](Construct::plan), before the
/// checkpoint is fetched, so a repo of another model is refused first.
///
/// The [`Construct`] hook for the Sortformer kit.
/// [`default_sortformer_factory`](super::default_sortformer_factory) attaches
/// it; a caller never builds one.
#[derive(Clone, Debug, Default)]
pub struct SortformerConstruct;

impl SortformerConstruct {
    /// The geometry of a map's `config.json` at `path`.
    fn read_config(path: &std::path::Path) -> BunsenResult<NemotronDiarizationConfig> {
        let json = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
        nemotron_config_from_hf(&json).map_err(|e| e.context(path.display()))
    }
}

impl Construct for SortformerConstruct {
    type Built = SortformerBundle;

    const KIT: &'static str = SORTFORMER_KIT;

    /// The hook for a safetensors checkpoint, one file or a sharded one's
    /// index.
    fn for_map(map: &ResourceMap) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
        match checkpoint.kind.as_deref() {
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self),
            other => Err(BunsenError::unsupported(format!(
                "no Sortformer reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    fn plan(
        &self,
        model: &PretrainedRef,
        cache: &PretrainedCache,
    ) -> BunsenResult<ResourceMap> {
        let map = model.to_map();
        if let Some(config) = map.get(CONFIG) {
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
        let config = match loaded.get(CONFIG) {
            Some(part) => Self::read_config(&part.path)?,
            None => NemotronDiarizationConfig::nemotron_3_diarization(),
        };
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        let mut model: Sortformer = config.try_init(device)?;
        load_nemotron_hf_tensors(&mut model, tensors)?;
        let front_end = NemoMelFrontEndConfig::nemotron_diarization().with_n_mels(config.n_mels);
        Ok(Arc::new(SortformerBundle::new(
            model,
            SortformerFamily::Nemotron,
            &front_end,
            device,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::sortformer::pretrained::NEMOTRON_3_DIARIZATION_FILES;

    #[test]
    fn test_the_hook_follows_the_kind() {
        SortformerConstruct::for_map(&NEMOTRON_3_DIARIZATION_FILES.to_map()).unwrap();
        let mut map = NEMOTRON_3_DIARIZATION_FILES.to_map();
        for res in map.resources.values_mut() {
            res.kind = Some("pytorch".into());
        }
        assert!(SortformerConstruct::for_map(&map).is_err());
    }
}
