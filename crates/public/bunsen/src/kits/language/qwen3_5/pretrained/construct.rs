//! # Constructing a Qwen3.5 pretrained
use alloc::sync::Arc;

use burn::tensor::Device;

use super::{
    CHECKPOINT,
    QWEN3_5_KIT,
    QWEN3_5_PREFABS,
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
        language::qwen3_5::{
            CHAT_TEMPLATE,
            GENERATION_CONFIG,
            Qwen3_5Construct,
            Qwen3_5Model,
            Qwen3_5Policy,
            TOKENIZER_CONFIG,
            weights::{
                load_transformers_tensors_as,
                qwen3_5_config_from_transformers,
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

/// The end of the text, which a Qwen3.5 chat model's stop tokens hold beside
/// its `eos_token` (`<|im_end|>`, a turn's end).
pub const END_OF_TEXT: &str = "<|endoftext|>";

/// How a Qwen3.5 pretrained is built: its safetensors checkpoint (one file or
/// shards) read into the geometry its `config.json` gives
/// ([`qwen3_5_config_from_transformers`]), or else the prefab its row names,
/// in the dtype the load policy resolves; bundled with the positions the
/// `config.json` says it was trained on, and every file of the checkpoint by
/// its key.
///
/// The Qwen3.5 kit's construct config: its [`ConstructInit`] hook.
/// [`default_qwen3_5_factory`](super::default_qwen3_5_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug)]
pub struct Qwen3_5ConstructConfig;

impl Qwen3_5ConstructConfig {
    /// The presets of a checkpoint no row describes, an `hf:` repo's, as the
    /// catalog tool probes a row's: its `generation_config.json`, when it
    /// has one; its `tokenizer_config.json`'s `eos_token` and
    /// [`END_OF_TEXT`] (the stop tokens, joined) and chat template; and its
    /// `chat_template.jinja`, when it has one; probed into the base.
    fn probed(loaded: &LoadedResources) -> BunsenResult<PresetSet> {
        probe_loaded(
            loaded,
            &[
                CatalogProbe::into_base("generation_config", GENERATION_CONFIG),
                CatalogProbe::into_base("tokenizer_config", TOKENIZER_CONFIG),
                CatalogProbe::into_base("tokenizer_config", TOKENIZER_CONFIG)
                    .with_token(END_OF_TEXT),
                CatalogProbe::into_base("chat_template", TOKENIZER_CONFIG),
                CatalogProbe::into_base("chat_template", CHAT_TEMPLATE),
            ],
            &PresetSet::default(),
        )
    }
}

impl ConstructInit for Qwen3_5ConstructConfig {
    type Construct = Qwen3_5Construct;

    const KIT: &'static str = QWEN3_5_KIT;
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
                "no Qwen3.5 reader for a checkpoint of kind {other:?}"
            ))),
        }
    }

    fn try_init(
        &self,
        model: &PretrainedRef,
        loaded: &LoadedResources,
        device: &Device,
        policy: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<Qwen3_5Construct>> {
        let (config, max_positions) = match loaded.get(CONFIG) {
            Some(part) => {
                let json =
                    std::fs::read_to_string(&part.path).map_err(sys_at("read", &part.path))?;
                let config = qwen3_5_config_from_transformers(&json)
                    .map_err(|e| e.context(part.path.display()))?;
                let root = TransformersConfigJson::parse(&json)?;
                let text = root.section("text_config").unwrap_or(root);
                let max_positions = text.usize("max_position_embeddings").ok();
                (config, max_positions)
            }
            None => match model.prefab(&QWEN3_5_PREFABS) {
                Some(prefab) => (prefab.to_config(), None),
                None => {
                    return Err(BunsenError::invalid_resource(
                        "a Qwen3.5 checkpoint with neither a config.json nor a prefab",
                    )
                    .context(&loaded.map.name));
                }
            },
        };
        let tensors = SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
        // The stored dtype (bf16), where the policy and the device allow it;
        // never f16, whose range a bf16-trained model's activations exceed.
        let stored = crate::data::pretrained::stored_float_dtype(&tensors)
            .unwrap_or(burn::tensor::FloatDType::F32);
        let dtype = policy.resolve(stored, device, |d| d != burn::tensor::FloatDType::F16)?;
        let mut qwen: Qwen3_5Model = config.try_init(device)?;
        load_transformers_tensors_as(&mut qwen, tensors, dtype)?;
        let policy = resolve_ref_presets(&Qwen3_5Policy::default(), model, &Self::probed(loaded)?)?;
        let mut bundle = Qwen3_5Construct::new(qwen)
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
    use crate::kits::language::qwen3_5::pretrained::{
        QWEN3_5_CATALOG,
        default_qwen3_5_factory,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        Qwen3_5ConstructConfig::for_resource_bundle(
            &QWEN3_5_CATALOG.resource_bundle("qwen/qwen3.5-0.8b"),
        )
        .unwrap();
        Qwen3_5ConstructConfig::for_resource_bundle(
            &QWEN3_5_CATALOG.resource_bundle("qwen/qwen3.5-9b"),
        )
        .unwrap();
        let mut gguf = QWEN3_5_CATALOG.resource_bundle("qwen/qwen3.5-0.8b");
        gguf.resources.get_mut("checkpoint.index").unwrap().kind = Some("gguf".to_string());
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("gguf")
            .assert_err(&Qwen3_5ConstructConfig::for_resource_bundle(&gguf));
    }

    /// Every row stops where its `tokenizer_config.json` says: a chat row at
    /// `<|im_end|>` (248046, its `eos_token`) and [`END_OF_TEXT`] (248044), a
    /// `-base` row at its `eos_token`, `<|endoftext|>`. A chat row does by
    /// default what its own template does unset, under the card's sampling
    /// for it: the 0.8B and 2B answer at once, the 4B and 9B think; the
    /// `thinking-mode` axis switches either.
    #[test]
    fn test_the_generation_presets() {
        let factory = default_qwen3_5_factory().unwrap();
        let resolve = |spec: &str| {
            factory
                .resolve_presets(spec, &Qwen3_5Policy::default())
                .unwrap_or_else(|e| panic!("{spec}: {e}"))
                .config
                .generation
        };
        let thinks = |g: &crate::kits::support::model_config::GenerationPolicy| {
            g.template_kwargs["enable_thinking"].as_bool().unwrap()
        };
        let top_p = |g: &crate::kits::support::model_config::GenerationPolicy| {
            g.sampling.as_ref().unwrap().top_p
        };
        for (row, thinking) in [
            ("qwen3.5-0.8b", false),
            ("qwen3.5-2b", false),
            ("qwen3.5-4b", true),
            ("qwen3.5-9b", true),
        ] {
            let chat = resolve(row);
            assert_eq!(chat.stop_ids, [248_046, 248_044], "{row}");
            assert_eq!(thinks(&chat), thinking, "{row}: by default");
            assert_eq!(
                top_p(&chat),
                Some(if thinking { 0.95 } else { 1.0 }),
                "{row}"
            );
            let base = resolve(&format!("{row}-base"));
            assert_eq!(base.stop_ids, [248_044], "{row}-base");
        }
        assert!(thinks(&resolve("qwen3.5-0.8b[thinking]")));
        let quick = resolve("qwen3.5-9b[no-thinking]");
        assert!(!thinks(&quick));
        assert_eq!(top_p(&quick), Some(1.0));
    }
}
