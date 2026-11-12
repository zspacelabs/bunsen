//! # Constructing a Gemma pretrained
use alloc::vec::Vec;

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::{
    CHECKPOINT,
    GEMMA_KIT,
    GEMMA_PREFABS,
    GEMMA_TEMPLATE_CLASS,
};
use crate::{
    data::{
        pretrained::{
            NamedTensorsApplied,
            ResolvedPresetSet,
            StaticPreFabMap,
        },
        resources::LoadedResources,
    },
    errors::BunsenResult,
    kits::{
        language::gemma::{
            CHAT_TEMPLATE,
            END_OF_TURN,
            GENERATION_CONFIG,
            GemmaConfig,
            GemmaModel,
            TOKENIZER_CONFIG,
            weights::{
                gemma_config_from_transformers,
                load_transformers_tensors_as,
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

/// How a Gemma pretrained is built: its safetensors checkpoint (one file or
/// shards) read into the geometry its `config.json` gives
/// ([`gemma_config_from_transformers`]), or else the prefab its row names, in
/// the dtype the load policy resolves; bundled with the positions the
/// `config.json` says it was trained on, and every file of the checkpoint by
/// its key.
///
/// The Gemma kit's construct config: its
/// [`ConstructInit`](crate::kits::support::construct::ConstructInit) hook, by
/// [`TransformersLmKit`].
/// [`default_gemma_factory`](super::default_gemma_factory) attaches it; a
/// caller never builds one.
#[derive(Clone, Debug, Default)]
pub struct GemmaConstructConfig;

impl TransformersLmKit for GemmaConstructConfig {
    type Config = GemmaConfig;
    type Model = GemmaModel;

    const CHECKPOINT: &'static str = CHECKPOINT;
    const KIT: &'static str = GEMMA_KIT;
    const NAME: &'static str = "Gemma";

    fn prefabs() -> &'static StaticPreFabMap<GemmaConfig> {
        &GEMMA_PREFABS
    }

    fn config_from_transformers(json: &str) -> BunsenResult<GemmaConfig> {
        gemma_config_from_transformers(json)
    }

    fn load_tensors(
        model: &mut GemmaModel,
        tensors: Vec<PackTensor>,
        dtype: FloatDType,
    ) -> BunsenResult<NamedTensorsApplied> {
        load_transformers_tensors_as(model, tensors, dtype)
    }

    /// The presets of a checkpoint no row describes, an `hf:` repo's, as the
    /// catalog builder probes a row's, over the [`GEMMA_TEMPLATE_CLASS`]'s: its
    /// `generation_config.json`, when it has one (the stop tokens and
    /// sampling); its `tokenizer_config.json`'s [`END_OF_TURN`] (a turn's
    /// end, which joins the stop tokens) and chat template, when it has one;
    /// and its `chat_template.jinja`, when it has one; probed into the base.
    /// Without a template of its own, it renders with the family's.
    fn probed(loaded: &LoadedResources) -> BunsenResult<ResolvedPresetSet> {
        probe_loaded(
            loaded,
            &[
                CatalogProbe::into_base("generation_config", GENERATION_CONFIG),
                CatalogProbe::into_base("tokenizer_config", TOKENIZER_CONFIG)
                    .with_token(END_OF_TURN),
                CatalogProbe::into_base("chat_template", TOKENIZER_CONFIG),
                CatalogProbe::into_base("chat_template", CHAT_TEMPLATE),
            ],
            &crate::data::pretrained::builtin_template_presets(GEMMA_TEMPLATE_CLASS)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::{
        language::gemma::{
            GemmaPolicy,
            pretrained::default_gemma_factory,
        },
        support::construct::ConstructInit,
    };

    #[test]
    fn test_the_hook_follows_the_kind() {
        GemmaConstructConfig::for_resource_bundle(
            &crate::data::pretrained::builtin_build_bundle("gemma/gemma-3-270m").unwrap(),
        )
        .unwrap();
        GemmaConstructConfig::for_resource_bundle(
            &crate::data::pretrained::builtin_build_bundle("gemma/gemma-2-9b-it").unwrap(),
        )
        .unwrap();

        let mut gguf = crate::data::pretrained::builtin_build_bundle("gemma/gemma-3-270m").unwrap();
        gguf.resources.get_mut(CHECKPOINT).unwrap().kind = Some("gguf".to_string());
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("gguf")
            .assert_err(&GemmaConstructConfig::for_resource_bundle(&gguf));
    }

    /// Every row stops where its files say: the end of the text its
    /// `generation_config.json` names, and, for an instruction-tuned row, the
    /// end of a turn its `tokenizer_config.json` gives [`END_OF_TURN`] (106 in
    /// Gemma 3, 107 in Gemma 1 and 2), the two probes' ids joined. A lookup's
    /// patch is an explicit layer over them, which replaces.
    #[test]
    fn test_the_generation_presets() {
        let factory = default_gemma_factory().unwrap();
        let stops = |spec: &str| {
            factory
                .resolve_presets(spec, &GemmaPolicy::default())
                .unwrap_or_else(|e| panic!("{spec}: {e}"))
                .config
                .generation
                .stop_ids
        };
        assert_eq!(stops("gemma/gemma-3-1b-it"), [1, 106]);
        assert_eq!(stops("gemma/gemma-3-270m"), [1]);
        for row in [
            "gemma/gemma-2-2b-it",
            "gemma/gemma-2-9b-it",
            "gemma/gemma-2b-it",
            "gemma/gemma-7b-it",
        ] {
            assert_eq!(stops(row), [1, 107], "{row}");
        }
        assert_eq!(
            stops("gemma/gemma-2-9b-it{generation: {stop_ids: [1]}}"),
            [1],
            "a lookup's patch replaces the probed stops"
        );
    }

    /// A row's chat template is its own file where it has one (Gemma 3's
    /// `chat_template.jinja`, the rest's `tokenizer_config.json`), and the
    /// [`GEMMA_TEMPLATE_CLASS`]'s own source where it has none (the pretrained
    /// models); a lookup's patch replaces either.
    #[test]
    fn test_the_chat_templates() {
        use crate::kits::support::model_config::ChatTemplateSource;

        let factory = default_gemma_factory().unwrap();
        let template = |spec: &str| {
            factory
                .resolve_presets(spec, &GemmaPolicy::default())
                .unwrap_or_else(|e| panic!("{spec}: {e}"))
                .config
                .generation
                .chat_template
        };
        let file = |key: &str| {
            Some(ChatTemplateSource::File {
                file: key.to_string(),
            })
        };
        assert_eq!(template("gemma/gemma-3-1b-it"), file("chat_template"));
        assert_eq!(template("gemma/gemma-2-9b-it"), file("tokenizer_config"));
        let family = crate::data::pretrained::builtin_template_presets(GEMMA_TEMPLATE_CLASS)
            .unwrap()
            .common["generation"]["chat_template"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(family.contains("<start_of_turn>"));
        for row in ["gemma/gemma-2b", "gemma/gemma-3-270m", "gemma/gemma-2-9b"] {
            assert_eq!(
                template(row),
                Some(ChatTemplateSource::Inline(family.clone())),
                "{row}"
            );
        }
        assert_eq!(
            template("gemma/gemma-3-1b-it{generation: {chat_template: \"{{ messages }}\"}}"),
            Some(ChatTemplateSource::Inline("{{ messages }}".to_string()))
        );
    }
}
