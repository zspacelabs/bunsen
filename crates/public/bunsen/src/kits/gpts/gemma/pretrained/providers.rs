//! # Gemma pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::GemmaConstructConfig;
use crate::{
    data::pretrained::{
        EmbeddedCatalog,
        HfProvider,
        PretrainedCache,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::{
        gpts::gemma::{
            CHAT_TEMPLATE,
            GENERATION_CONFIG,
            GemmaConstruct,
            TOKENIZER,
            TOKENIZER_CONFIG,
        },
        support::construct::PretrainedFactory,
    },
};

/// The kit segment of a Gemma resource's path in the cache:
/// `<cache>/pretrained/gemma/<namespace>/<sha256>/<file>`.
pub const GEMMA_KIT: &str = "gemma";

/// The key of the checkpoint in a Gemma resource bundle: one
/// `model.safetensors`, or `checkpoint.index` and its shards
/// `checkpoint.<n>`.
pub const CHECKPOINT: &str = "checkpoint";

/// The `kind` of a `transformers` `tokenizer.json`.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// The `kind` of a `transformers` `tokenizer_config.json`.
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

/// The `kind` of a `transformers` `generation_config.json`.
pub const GENERATION_CONFIG_JSON: &str = "generation_config.json";

/// The `kind` of a `chat_template.jinja`.
pub const CHAT_TEMPLATE_JINJA: &str = "chat_template.jinja";

/// The Gemma checkpoints bunsen knows by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider, in group
/// `gemma`: Google's Gemma 1, 2 and 3 text models
/// (`well-known:gemma/gemma-3-1b-it`, `gemma/gemma-3-1b-it`, or
/// `gemma-3-1b-it`), each fetched from its `unsloth/` mirror, whose files
/// are ungated (Google's own repos are gated).
///
/// Built from `validation/catalogs/gemma.yml` by `cargo validate catalog`.
pub static GEMMA_CATALOG: EmbeddedCatalog =
    EmbeddedCatalog::new("gemma", include_str!("catalog.json"));

/// The name [`default_gemma_factory`] loads when a caller names none.
pub const DEFAULT_GEMMA_MODEL: &str = "gemma/gemma-3-1b-it";

/// Gemma's compiled-in providers, in search order: the well-known table,
/// then `hf:org/repo`, a Hugging Face repo in a `transformers` Gemma's
/// layout (its safetensors checkpoint, one file or shards, and its
/// `config.json`), with its [`TOKENIZER`], [`TOKENIZER_CONFIG`],
/// [`GENERATION_CONFIG`] and [`CHAT_TEMPLATE`] files when it has them.
pub fn default_gemma_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(GEMMA_CATALOG.table()),
        Arc::new(
            HfProvider::new()
                .with_checkpoint_key(CHECKPOINT)
                .with_extra_file(TOKENIZER, "tokenizer.json", TOKENIZER_JSON)
                .with_extra_file(
                    TOKENIZER_CONFIG,
                    "tokenizer_config.json",
                    TOKENIZER_CONFIG_JSON,
                )
                .with_extra_file(
                    GENERATION_CONFIG,
                    "generation_config.json",
                    GENERATION_CONFIG_JSON,
                )
                .with_extra_file(CHAT_TEMPLATE, "chat_template.jinja", CHAT_TEMPLATE_JINJA),
        ),
    ]
}

/// Gemma's factory over [`default_gemma_providers`], building through
/// [`GemmaConstructConfig`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_gemma_factory() -> BunsenResult<PretrainedFactory<GemmaConstructConfig>> {
    PretrainedFactory::new().with_providers(default_gemma_providers())
}

impl PretrainedFactory<GemmaConstructConfig> {
    /// A name to a construct under `policy`: [`load`](Self::load), keeping
    /// the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_construct(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &Device,
        policy: crate::kits::support::construct::LoadPolicy,
    ) -> BunsenResult<Arc<GemmaConstruct>> {
        Ok(self.load(spec, cache, device, policy)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::CONFIG,
        kits::gpts::gemma::pretrained::GEMMA_PREFABS,
    };

    /// The rows, by name, every file pinned to one URL on its mirror, each
    /// row with its config, tokenizer and generation files and naming a
    /// prefab there is.
    #[test]
    fn test_rows_are_pinned() {
        let prefabs: Vec<&str> = GEMMA_PREFABS.iter().map(|p| p.name).collect();
        GEMMA_CATALOG.catalog().validate(&prefabs).unwrap();
        let factory = default_gemma_factory().unwrap();
        assert_eq!(factory.kit(), GEMMA_KIT);
        assert_eq!(
            factory
                .providers()
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>(),
            ["well-known", "hf"]
        );
        let names = [
            "gemma-3-270m",
            "gemma-3-270m-it",
            "gemma-3-1b-pt",
            "gemma-3-1b-it",
            "gemma-3-4b-pt",
            "gemma-3-4b-it",
            "gemma-2-2b",
            "gemma-2-2b-it",
            "gemma-2-9b",
            "gemma-2-9b-it",
            "gemma-2b",
            "gemma-2b-it",
            "gemma-7b",
            "gemma-7b-it",
        ];
        assert_eq!(
            factory.ids(),
            names
                .iter()
                .map(|n| format!("well-known:gemma/{n}"))
                .collect::<Vec<_>>()
        );
        for name in names {
            let (_, row) = factory.lookup(name).unwrap();
            let map = &row.resources;
            map.validate().unwrap();
            for resource in map.resources.values() {
                assert!(resource.is_pinned(), "{name}: {}", resource.key);
                let urls = resource.urls();
                assert_eq!(urls.len(), 1, "{name}: {}", resource.key);
                assert!(urls[0].contains("/unsloth/"), "{name}: {}", urls[0]);
            }
            for key in [CONFIG, TOKENIZER, TOKENIZER_CONFIG, GENERATION_CONFIG] {
                assert!(map.get(key).is_some(), "{name}: {key}");
            }
            assert!(
                map.get(CHECKPOINT).is_some() || map.get("checkpoint.index").is_some(),
                "{name}"
            );
        }
        assert_eq!(
            factory.lookup(DEFAULT_GEMMA_MODEL).unwrap().1.name,
            "gemma/gemma-3-1b-it"
        );
    }
}
