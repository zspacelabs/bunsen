//! # Constructing a Qwen3.5 pretrained
use alloc::vec::Vec;

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::{
    CHECKPOINT,
    QWEN3_5_KIT,
    QWEN3_5_PREFABS,
};
use crate::{
    data::{
        catalog::presets::ResolvedPresetSet,
        pretrained::{
            NamedTensorsApplied,
            StaticPreFabMap,
        },
        resources::LoadedResources,
    },
    errors::BunsenResult,
    kits::{
        language::qwen3_5::{
            CHAT_TEMPLATE,
            GENERATION_CONFIG,
            Qwen3_5Config,
            Qwen3_5Model,
            TOKENIZER_CONFIG,
            weights::{
                load_transformers_tensors_as,
                qwen3_5_config_from_transformers,
            },
        },
        support::{
            construct::TransformersLmKit,
            model_config::{
                CatalogProbe,
                probe_loaded,
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
/// The Qwen3.5 kit's construct config: its
/// [`ConstructInit`](crate::kits::support::construct::ConstructInit) hook, by
/// [`TransformersLmKit`].
/// [`default_qwen3_5_factory`](super::default_qwen3_5_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug, Default)]
pub struct Qwen3_5ConstructConfig;

impl TransformersLmKit for Qwen3_5ConstructConfig {
    type Config = Qwen3_5Config;
    type Model = Qwen3_5Model;

    const CHECKPOINT: &'static str = CHECKPOINT;
    const KIT: &'static str = QWEN3_5_KIT;
    const NAME: &'static str = "Qwen3.5";

    fn prefabs() -> &'static StaticPreFabMap<Qwen3_5Config> {
        &QWEN3_5_PREFABS
    }

    fn config_from_transformers(json: &str) -> BunsenResult<Qwen3_5Config> {
        qwen3_5_config_from_transformers(json)
    }

    fn load_tensors(
        model: &mut Qwen3_5Model,
        tensors: Vec<PackTensor>,
        dtype: FloatDType,
    ) -> BunsenResult<NamedTensorsApplied> {
        load_transformers_tensors_as(model, tensors, dtype)
    }

    /// The presets of a checkpoint no row describes, an `hf:` repo's, as the
    /// catalog builder probes a row's: its `generation_config.json`, when it
    /// has one; its `tokenizer_config.json`'s `eos_token` and
    /// [`END_OF_TEXT`] (the stop tokens, joined) and chat template; and its
    /// `chat_template.jinja`, when it has one; probed into the base.
    fn probed(loaded: &LoadedResources) -> BunsenResult<ResolvedPresetSet> {
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
            &ResolvedPresetSet::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::{
        language::qwen3_5::{
            Qwen3_5Policy,
            pretrained::default_qwen3_5_factory,
        },
        support::construct::ConstructInit,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        Qwen3_5ConstructConfig::for_resource_bundle(
            &crate::data::pretrained::builtin_build_bundle("qwen/qwen3.5-0.8b").unwrap(),
        )
        .unwrap();
        Qwen3_5ConstructConfig::for_resource_bundle(
            &crate::data::pretrained::builtin_build_bundle("qwen/qwen3.5-9b").unwrap(),
        )
        .unwrap();
        let mut gguf = crate::data::pretrained::builtin_build_bundle("qwen/qwen3.5-0.8b").unwrap();
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
