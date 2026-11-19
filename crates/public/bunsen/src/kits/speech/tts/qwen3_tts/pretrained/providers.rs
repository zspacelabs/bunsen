//! # Qwen3-TTS pretrained providers

use alloc::{
    sync::Arc,
    vec,
    vec::Vec,
};

use super::Qwen3TtsConstructConfig;
use crate::{
    data::builds::{
        BuildFactory,
        BuildProvider,
    },
    errors::BunsenResult,
};

/// The Qwen3-TTS kit's name: its rows' `kit` tag.
pub const QWEN3_TTS_KIT: &str = "qwen3_tts";

/// The key of the model's checkpoint in a Qwen3-TTS resource set.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the model's `config.json`.
pub const CONFIG: &str = "config";

/// The key of `generation_config.json`.
pub const GENERATION_CONFIG: &str = "generation_config";

/// The key of the tokenizer's vocabulary, `vocab.json`.
pub const VOCAB: &str = "vocab";

/// The key of the tokenizer's merges, `merges.txt`.
pub const MERGES: &str = "merges";

/// The key of the tokenizer's added tokens, `tokenizer_config.json`.
pub const TOKENIZER_CONFIG: &str = "tokenizer_config";

/// The key of the codec's checkpoint.
pub const CODEC: &str = "codec";

/// The key of the codec's `config.json`.
pub const CODEC_CONFIG: &str = "codec_config";

/// The format of a safetensors file.
pub const SAFETENSORS_FORMAT: &str = "safetensors";

/// What the `qwen3_tts` kit's loader implements, as a build catalog's
/// contract: `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const QWEN3_TTS_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `catalog:` rows are its builds.
pub const QWEN3_TTS_CATALOG_CONTRACT: &str = "qwen3-tts";

/// The name [`default_qwen3_tts_factory`] loads when a caller names none.
pub const DEFAULT_QWEN3_TTS_MODEL: &str = "qwen/qwen3-tts/qwen3-tts-12hz-0.6b-customvoice";

/// Qwen3-TTS's compiled-in providers: the catalog provider.
pub fn default_qwen3_tts_providers() -> Vec<Arc<dyn BuildProvider>> {
    vec![Arc::new(
        crate::data::catalog::BuiltinCatalog::expect_provider(QWEN3_TTS_CATALOG_CONTRACT),
    )]
}

/// Qwen3-TTS's factory: [`default_qwen3_tts_providers`] behind
/// [`Qwen3TtsConstructConfig`].
///
/// # Errors
/// As [`BuildFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_qwen3_tts_factory() -> BunsenResult<BuildFactory<Qwen3TtsConstructConfig>> {
    BuildFactory::new().with_providers(default_qwen3_tts_providers())
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use super::*;

    /// The rows, by their qualified names, every file pinned to one URL on
    /// its Qwen repository at its revision.
    #[test]
    fn test_the_rows_are_pinned_and_named() {
        let factory = default_qwen3_tts_factory().unwrap();
        assert_eq!(factory.kit(), QWEN3_TTS_KIT);
        assert_eq!(
            factory.ids(),
            [
                "catalog:qwen/qwen3-tts/qwen3-tts-12hz-0.6b-customvoice",
                "catalog:qwen/qwen3-tts/qwen3-tts-12hz-1.7b-customvoice",
                "catalog:qwen/qwen3-tts/qwen3-tts-12hz-1.7b-voicedesign",
            ]
        );
        for (name, repo) in [
            (
                "qwen3-tts-12hz-0.6b-customvoice",
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/resolve/85e237c12c027371202489a0ec509ded67b5e4b5/",
            ),
            (
                "qwen3-tts-12hz-1.7b-customvoice",
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/resolve/0c0e3051f131929182e2c023b9537f8b1c68adfe/",
            ),
            (
                "qwen3-tts-12hz-1.7b-voicedesign",
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/resolve/5ecdb67327fd37bb2e042aab12ff7391903235d3/",
            ),
        ] {
            let (_, build) = factory.lookup(name).unwrap();
            let map = build.resource_set();
            map.validate().unwrap();
            for key in [
                CHECKPOINT,
                CONFIG,
                GENERATION_CONFIG,
                VOCAB,
                MERGES,
                TOKENIZER_CONFIG,
                CODEC,
                CODEC_CONFIG,
            ] {
                let resource = map.get(key).unwrap_or_else(|| panic!("{name}: {key}"));
                assert!(resource.is_pinned(), "{name}: {key}");
                let urls = resource.urls();
                assert_eq!(urls.len(), 1, "{name}: {key}");
                assert!(urls[0].contains(repo), "{name}: {}", urls[0]);
            }
        }
        assert_eq!(
            factory
                .lookup("qwen3-tts-12hz-0.6b-customvoice")
                .unwrap()
                .1
                .name,
            DEFAULT_QWEN3_TTS_MODEL
        );
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::kits::support::catalog_contracts::testing::contract_lines;
        let line = |key: &str, format: &str| format!("qwen3-tts: {key} [{format}]");
        assert_eq!(
            contract_lines(QWEN3_TTS_CONTRACT),
            [
                line(CHECKPOINT, SAFETENSORS_FORMAT),
                line(CODEC, SAFETENSORS_FORMAT),
                line(CODEC_CONFIG, "json"),
                line(CONFIG, "json"),
                line(GENERATION_CONFIG, "generation_config.json"),
                line(MERGES, "merges.txt"),
                line(TOKENIZER_CONFIG, "tokenizer_config.json"),
                line(VOCAB, "vocab.json"),
            ]
        );
    }
}
