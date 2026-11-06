//! # Constructing a Qwen3 pretrained
use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    QWEN3_KIT,
    QWEN3_PREFABS,
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
        language::qwen3::{
            GENERATION_CONFIG,
            Qwen3Construct,
            Qwen3Model,
            Qwen3Policy,
            TOKENIZER_CONFIG,
            weights::{
                load_transformers_tensors_as,
                qwen3_config_from_transformers,
            },
        },
        support::{
            construct::ConstructInit,
            model_config::{
                TransformersConfigJson,
                probe_loaded,
                resolve_ref_presets,
            },
        },
    },
};

/// How a Qwen3 pretrained is built: its safetensors checkpoint (one file or
/// shards) read into the geometry its `config.json` gives
/// ([`qwen3_config_from_transformers`]), or else the prefab its row names, in
/// the device's float dtype; bundled with the positions the `config.json` says
/// it was trained on, and every file of the checkpoint by its key.
///
/// The Qwen3 kit's construct config: its [`ConstructInit`] hook.
/// [`default_qwen3_factory`](super::default_qwen3_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug)]
pub struct Qwen3ConstructConfig;

impl Qwen3ConstructConfig {
    /// The presets of a checkpoint no row describes, an `hf:` repo's, as the
    /// catalog tool probes a row's: its `generation_config.json`, when it
    /// has one (the stop tokens and sampling), and the chat template its
    /// `tokenizer_config.json` holds, probed into the base.
    fn probed(loaded: &LoadedResources) -> BunsenResult<PresetSet> {
        probe_loaded(
            loaded,
            &[
                CatalogProbe::into_base("generation_config", GENERATION_CONFIG),
                CatalogProbe::into_base("chat_template", TOKENIZER_CONFIG),
            ],
            &PresetSet::default(),
        )
    }
}

impl ConstructInit for Qwen3ConstructConfig {
    type Construct = Qwen3Construct;

    const KIT: &'static str = QWEN3_KIT;
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
                "no Qwen3 reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
        policy: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<Qwen3Construct>> {
        let (config, max_positions) = match loaded.get(CONFIG) {
            Some(part) => {
                let json =
                    std::fs::read_to_string(&part.path).map_err(sys_at("read", &part.path))?;
                let config = qwen3_config_from_transformers(&json)
                    .map_err(|e| e.context(part.path.display()))?;
                let max_positions = TransformersConfigJson::parse(&json)?
                    .usize("max_position_embeddings")
                    .ok();
                (config, max_positions)
            }
            None => match model.prefab(&QWEN3_PREFABS) {
                Some(prefab) => (prefab.to_config(), None),
                None => {
                    return Err(BunsenError::invalid_resource(
                        "a Qwen3 checkpoint with neither a config.json nor a prefab",
                    )
                    .context(&loaded.map.name));
                }
            },
        };
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        // The stored dtype (bf16), where the policy and the device allow it;
        // never f16, which overflows Qwen3's activations.
        let stored = crate::data::pretrained::stored_float_dtype(&tensors)
            .unwrap_or(burn::tensor::FloatDType::F32);
        let dtype = policy.resolve(stored, device, |d| d != burn::tensor::FloatDType::F16)?;
        let mut qwen3: Qwen3Model = config.try_init(device)?;
        load_transformers_tensors_as(&mut qwen3, tensors, dtype)?;
        let policy = resolve_ref_presets(&Qwen3Policy::default(), model, &Self::probed(loaded)?)?;
        let mut bundle = Qwen3Construct::new(qwen3)
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
    use crate::kits::language::qwen3::pretrained::QWEN3_CATALOG;

    #[test]
    fn test_the_hook_follows_the_kind() {
        Qwen3ConstructConfig::for_resource_bundle(
            &QWEN3_CATALOG.resource_bundle("qwen/qwen3-0.6b"),
        )
        .unwrap();
        Qwen3ConstructConfig::for_resource_bundle(&QWEN3_CATALOG.resource_bundle("qwen/qwen3-4b"))
            .unwrap();

        let mut gguf = QWEN3_CATALOG.resource_bundle("qwen/qwen3-0.6b");
        gguf.resources.get_mut(CHECKPOINT).unwrap().kind = Some("gguf".to_string());
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("gguf")
            .assert_err(&Qwen3ConstructConfig::for_resource_bundle(&gguf));
    }

    /// The hybrid chat rows think by default and answer at once by name,
    /// each under its card's sampling; every row stops where its
    /// `generation_config.json` says; the pretrained models and the
    /// fixed-mode 2507 builds have no `thinking-mode` axis.
    #[test]
    fn test_the_generation_presets() {
        use crate::kits::{
            language::qwen3::pretrained::default_qwen3_factory,
            support::model_config::SamplingPolicy,
        };

        let factory = default_qwen3_factory().unwrap();
        let policy = |spec: &str| factory.resolve_presets(spec, &Qwen3Policy::default());

        let thinking = policy("qwen/qwen3-0.6b").unwrap();
        assert_eq!(thinking.presets, ["thinking"]);
        let g = &thinking.config.generation;
        assert_eq!(g.stop_ids, [151645, 151643]);
        assert_eq!(g.template_kwargs["enable_thinking"], true);
        assert_eq!(
            g.sampling,
            Some(SamplingPolicy {
                temperature: 0.6,
                top_k: Some(20),
                top_p: Some(0.95),
                min_p: None,
            })
        );

        let fast = policy("qwen/qwen3-8b[no-thinking]").unwrap();
        let g = &fast.config.generation;
        assert_eq!(g.template_kwargs["enable_thinking"], false);
        assert_eq!(
            g.sampling,
            Some(SamplingPolicy {
                temperature: 0.7,
                top_k: Some(20),
                top_p: Some(0.8),
                min_p: Some(0.0),
            })
        );

        for row in [
            "qwen/qwen3-0.6b-base",
            "qwen/qwen3-4b-instruct-2507",
            "qwen/qwen3guard-gen-4b",
        ] {
            let r = policy(row).unwrap();
            assert!(r.presets.is_empty(), "{row}");
            assert!(!r.config.generation.stop_ids.is_empty(), "{row}");
            assert!(r.config.generation.template_kwargs.is_empty(), "{row}");
            crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Lookup)
                .message_contains("no preset")
                .assert_err(&policy(&format!("{row}[thinking]")));
        }
    }
}
