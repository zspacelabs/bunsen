//! # Constructing a Gemma pretrained
use std::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    GEMMA_KIT,
    GEMMA_PREFABS,
};
use crate::{
    burner::module::ModuleInit,
    data::pretrained::{
        CONFIG,
        CatalogProbe,
        LoadedResources,
        PresetSet,
        PretrainedRef,
        ResourceBundle,
        SAFETENSORS,
        SAFETENSORS_INDEX,
        SafetensorsCheckpoint,
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::{
        gpts::gemma::{
            GENERATION_CONFIG,
            GemmaConstruct,
            GemmaModel,
            GemmaPolicy,
            weights::{
                gemma_config_from_hf,
                load_hf_tensors_as,
            },
        },
        support::{
            construct::ConstructInit,
            model_config::{
                HfConfigJson,
                ProbeFiles,
                apply_probes,
                resolve_ref_presets,
            },
        },
    },
};

/// How a Gemma pretrained is built: its safetensors checkpoint (one file or
/// shards) read into the geometry its `config.json` gives
/// ([`gemma_config_from_hf`]), or else the prefab its row names, in the
/// dtype the load policy resolves; bundled with the positions the
/// `config.json` says it was trained on, and every file of the checkpoint by
/// its key.
///
/// The Gemma kit's construct config: its [`ConstructInit`] hook.
/// [`default_gemma_factory`](super::default_gemma_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug)]
pub struct GemmaConstructConfig;

impl GemmaConstructConfig {
    /// The presets of a checkpoint no row describes, an `hf:` repo's: its
    /// `generation_config.json`, when it has one, probed into the base (the
    /// stop tokens and sampling), as the catalog tool probes a row's.
    fn probed(loaded: &LoadedResources) -> BunsenResult<PresetSet> {
        let Some(part) = loaded.get(GENERATION_CONFIG) else {
            return Ok(PresetSet::default());
        };
        let bytes = std::fs::read(&part.path).map_err(sys_at("read", &part.path))?;
        apply_probes(
            &PresetSet::default(),
            &PresetSet::default(),
            &[CatalogProbe {
                probe: "generation_config".to_string(),
                file: GENERATION_CONFIG.to_string(),
                into: vec!["base".to_string()],
            }],
            &ProbeFiles::new().with(GENERATION_CONFIG, bytes),
        )
    }
}

impl ConstructInit for GemmaConstructConfig {
    type Construct = GemmaConstruct;

    const KIT: &'static str = GEMMA_KIT;
    const RESOLVES_PRESETS: bool = true;

    /// The hook for a safetensors checkpoint.
    fn for_resource_bundle(map: &ResourceBundle) -> BunsenResult<Self> {
        let index = format!("{CHECKPOINT}.index");
        let checkpoint = match map.get(&index) {
            Some(index) => index,
            None => map.try_get(CHECKPOINT)?,
        };
        match checkpoint.kind.as_deref() {
            Some(SAFETENSORS | SAFETENSORS_INDEX) => Ok(Self),
            other => Err(BunsenError::unsupported(format!(
                "no Gemma reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
        policy: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<GemmaConstruct>> {
        let (config, max_positions) = match loaded.get(CONFIG) {
            Some(part) => {
                let json =
                    std::fs::read_to_string(&part.path).map_err(sys_at("read", &part.path))?;
                let config =
                    gemma_config_from_hf(&json).map_err(|e| e.context(part.path.display()))?;
                let root = HfConfigJson::parse(&json)?;
                let text = root.section("text_config").unwrap_or(root);
                let max_positions = text.usize("max_position_embeddings").ok();
                (config, max_positions)
            }
            None => match model.prefab(&GEMMA_PREFABS) {
                Some(prefab) => (prefab.to_config(), None),
                None => {
                    return Err(BunsenError::invalid_resource(
                        "a Gemma checkpoint with neither a config.json nor a prefab",
                    )
                    .context(&loaded.map.name));
                }
            },
        };
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        // The stored dtype (bf16; f32 for Gemma 2's base models), where the
        // policy and the device allow it; never f16, which overflows Gemma's
        // activations.
        let stored = crate::data::pretrained::stored_float_dtype(&tensors)
            .unwrap_or(burn::tensor::FloatDType::F32);
        let dtype = policy.resolve(stored, device, |d| d != burn::tensor::FloatDType::F16)?;
        let mut gemma: GemmaModel = config.try_init(device)?;
        load_hf_tensors_as(&mut gemma, tensors, dtype)?;
        let policy = resolve_ref_presets(&GemmaPolicy::default(), model, &Self::probed(loaded)?)?;
        let mut bundle = GemmaConstruct::new(gemma)
            .with_max_positions(max_positions)
            .with_policy(policy.config.clone(), policy.applied());
        for (key, part) in &loaded.parts {
            bundle = bundle.with_file(key, &part.path);
        }
        Ok(Arc::new(bundle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::gpts::gemma::pretrained::{
        GEMMA_CATALOG,
        default_gemma_factory,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        GemmaConstructConfig::for_resource_bundle(
            &GEMMA_CATALOG.resource_bundle("gemma/gemma-3-270m"),
        )
        .unwrap();
        GemmaConstructConfig::for_resource_bundle(
            &GEMMA_CATALOG.resource_bundle("gemma/gemma-2-9b-it"),
        )
        .unwrap();

        let mut gguf = GEMMA_CATALOG.resource_bundle("gemma/gemma-3-270m");
        gguf.resources.get_mut(CHECKPOINT).unwrap().kind = Some("gguf".to_string());
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("gguf")
            .assert_err(&GemmaConstructConfig::for_resource_bundle(&gguf));
    }

    /// Every row stops where its `generation_config.json` says, probed when
    /// the catalog was built: the instruction-tuned Gemma 3s at the end of a
    /// turn as well as at the end of the text.
    #[test]
    fn test_the_generation_presets() {
        let factory = default_gemma_factory().unwrap();
        let policy = |spec: &str| {
            factory
                .resolve_presets(spec, &GemmaPolicy::default())
                .unwrap()
        };
        assert_eq!(
            policy("gemma/gemma-3-1b-it").config.generation.stop_ids,
            [1, 106]
        );
        assert_eq!(policy("gemma/gemma-3-270m").config.generation.stop_ids, [1]);
        for row in ["gemma/gemma-2-2b-it", "gemma/gemma-7b"] {
            assert!(!policy(row).config.generation.stop_ids.is_empty(), "{row}");
        }
    }
}
