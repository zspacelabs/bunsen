//! # Constructing a Qwen3 pretrained
use alloc::vec::Vec;

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::{
    CHECKPOINT,
    QWEN3_KIT,
    QWEN3_PREFABS,
};
use crate::{
    data::pretrained::{
        LoadedResources,
        NamedTensorsApplied,
        ResolvedPresetSet,
        StaticPreFabMap,
    },
    errors::BunsenResult,
    kits::{
        language::qwen3::{
            GENERATION_CONFIG,
            Qwen3Config,
            Qwen3Model,
            TOKENIZER_CONFIG,
            weights::{
                load_transformers_tensors_as,
                qwen3_config_from_transformers,
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

/// How a Qwen3 pretrained is built: its safetensors checkpoint (one file or
/// shards) read into the geometry its `config.json` gives
/// ([`qwen3_config_from_transformers`]), or else the prefab its row names, in
/// the dtype the load policy resolves; bundled with the positions the
/// `config.json` says it was trained on, and every file of the checkpoint by
/// its key.
///
/// The Qwen3 kit's construct config: its
/// [`ConstructInit`](crate::kits::support::construct::ConstructInit) hook, by
/// [`TransformersLmKit`].
/// [`default_qwen3_factory`](super::default_qwen3_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug, Default)]
pub struct Qwen3ConstructConfig;

impl TransformersLmKit for Qwen3ConstructConfig {
    type Config = Qwen3Config;
    type Model = Qwen3Model;

    const CHECKPOINT: &'static str = CHECKPOINT;
    const KIT: &'static str = QWEN3_KIT;
    const NAME: &'static str = "Qwen3";

    fn prefabs() -> &'static StaticPreFabMap<Qwen3Config> {
        &QWEN3_PREFABS
    }

    fn config_from_transformers(json: &str) -> BunsenResult<Qwen3Config> {
        qwen3_config_from_transformers(json)
    }

    fn load_tensors(
        model: &mut Qwen3Model,
        tensors: Vec<PackTensor>,
        dtype: FloatDType,
    ) -> BunsenResult<NamedTensorsApplied> {
        load_transformers_tensors_as(model, tensors, dtype)
    }

    /// The presets of a checkpoint no row describes, an `hf:` repo's, as the
    /// catalog builder probes a row's: its `generation_config.json`, when it
    /// has one (the stop tokens and sampling), and the chat template its
    /// `tokenizer_config.json` holds, probed into the base.
    fn probed(loaded: &LoadedResources) -> BunsenResult<ResolvedPresetSet> {
        probe_loaded(
            loaded,
            &[
                CatalogProbe::into_base("generation_config", GENERATION_CONFIG),
                CatalogProbe::into_base("chat_template", TOKENIZER_CONFIG),
            ],
            &ResolvedPresetSet::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::{
        language::qwen3::Qwen3Policy,
        support::construct::ConstructInit,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        Qwen3ConstructConfig::for_resource_bundle(
            &crate::data::pretrained::builtin_build_bundle("qwen/qwen3-0.6b").unwrap(),
        )
        .unwrap();
        Qwen3ConstructConfig::for_resource_bundle(
            &crate::data::pretrained::builtin_build_bundle("qwen/qwen3-4b").unwrap(),
        )
        .unwrap();

        let mut gguf = crate::data::pretrained::builtin_build_bundle("qwen/qwen3-0.6b").unwrap();
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
            // A fixed mode: `no-thinking` applies, and adds nothing; `thinking`
            // is refused, with why.
            let r = policy(row).unwrap();
            assert_eq!(r.presets, ["no-thinking"], "{row}");
            assert!(!r.config.generation.stop_ids.is_empty(), "{row}");
            assert!(r.config.generation.template_kwargs.is_empty(), "{row}");
            crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Policy)
                .message_contains("fixed at `no-thinking`")
                .assert_err(&policy(&format!("{row}[thinking]")));
        }
    }
}
