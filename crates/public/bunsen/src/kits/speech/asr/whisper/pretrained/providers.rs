//! # Whisper pretrained providers

use alloc::sync::Arc;

use crate::{
    data::pretrained::{
        HfProvider,
        PretrainedProvider,
    },
    kits::speech::asr::whisper::pretrained::CHECKPOINT,
};

/// The Whisper kit's name: its rows' `kit` tag.
pub const WHISPER_KIT: &str = "whisper";

/// The name of the local-dir base that is `openai-whisper`'s download
/// root; a [`PretrainedCacheOptions`](crate::data::pretrained::PretrainedCacheOptions)
/// override by this name points it elsewhere.
pub const OPENAI_LOCAL_DIR: &str = "openai-whisper";

/// What the whisper kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const WHISPER_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `well-known:` rows are its builds.
pub const WHISPER_CATALOG_CONTRACT: &str = "whisper-asr";

/// Whisper's compiled-in providers, in search order: the well-known table,
/// then Hugging Face, which answers only `hf:org/repo`, through the cache,
/// and lists nothing.
///
/// The data layer's [`HfProvider`] serves the last: a repo's
/// `transformers` safetensors checkpoint, one file or shards, under
/// [`CHECKPOINT`]. The factory over these is
/// [`default_whisper_factory`](super::default_whisper_factory).
pub fn default_whisper_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![
        Arc::new(
            crate::data::pretrained::BuildCatalogProvider::expect_builtin(
                WHISPER_CATALOG_CONTRACT,
                WHISPER_KIT,
            ),
        ),
        Arc::new(HfProvider::new().with_checkpoint_key(CHECKPOINT)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::asr::whisper::{
        driver::WhisperSpecialIds,
        pretrained::{
            VOCABULARY,
            WHISPER_PREFABS,
            WhisperVocabulary,
        },
    };

    fn layout_of(prefab: &str) -> WhisperSpecialIds {
        let cfg = WHISPER_PREFABS.expect_lookup_prefab(prefab).to_config();
        *cfg.token_layout
            .policy_for_vocab(cfg.vocab_size)
            .unwrap()
            .ids()
    }

    /// The prefab each well-known row's checkpoint has the geometry of, by
    /// the row's leaf name: `large-v1` and `large-v2` share `large`.
    const ROW_PREFABS: [(&str, &str); 12] = [
        ("large-v1", "large"),
        ("large-v2", "large"),
        ("large-v3", "large-v3"),
        ("large-v3-turbo", "large-v3-turbo"),
        ("tiny.en", "tiny.en"),
        ("tiny", "tiny"),
        ("base.en", "base.en"),
        ("base", "base"),
        ("small.en", "small.en"),
        ("small", "small"),
        ("medium.en", "medium.en"),
        ("medium", "medium"),
    ];

    fn well_known() -> crate::data::pretrained::BuildCatalogProvider {
        crate::data::pretrained::BuildCatalogProvider::expect_builtin(
            WHISPER_CATALOG_CONTRACT,
            WHISPER_KIT,
        )
    }

    fn leaf(name: &str) -> &str {
        name.rsplit('/').next().unwrap()
    }

    fn prefab_of(name: &str) -> &'static str {
        ROW_PREFABS
            .iter()
            .find(|(row, _)| *row == leaf(name))
            .unwrap_or_else(|| panic!("{name}: no prefab in ROW_PREFABS"))
            .1
    }

    /// Every row has a prefab the map has and fuses into a checkpoint and a
    /// vocabulary, the checkpoint file named after the row and pinned.
    #[test]
    fn test_every_row_has_a_prefab_and_fuses() {
        let provider = well_known();
        for row in provider.rows() {
            let prefab = prefab_of(&row.name);
            assert!(
                WHISPER_PREFABS.lookup_prefab(prefab).is_some(),
                "{}: prefab {prefab:?} is not in WHISPER_PREFABS",
                row.name,
            );
            let map = &row.resources;
            map.validate().unwrap();
            assert_eq!(map.keys(), [CHECKPOINT, VOCABULARY], "{}", row.name);
            let checkpoint = map.get(CHECKPOINT).unwrap();
            assert_eq!(checkpoint.file, format!("{}.pt", leaf(&row.name)));
            assert!(checkpoint.is_pinned());
            assert!(map.get(VOCABULARY).unwrap().is_pinned());
        }
        assert_eq!(provider.rows().len(), ROW_PREFABS.len());
    }

    /// The vocabulary a row declares is the one the rule selects for its
    /// prefab's layout, so a name and a path agree on it.
    #[test]
    fn test_the_declared_vocabulary_is_the_rules() {
        for row in well_known().rows() {
            let declared = row.resources.get(VOCABULARY).unwrap();
            let rule =
                WhisperVocabulary::for_layout(&layout_of(prefab_of(&row.name))).resource_bundle();
            let rule = rule.get(VOCABULARY).unwrap();
            assert_eq!(declared, rule, "{}", row.name);
        }
    }

    /// The well-known provider lists the twelve `openai` rows by full name
    /// and answers shorter spellings, the bare names and upstream's aliases.
    #[test]
    fn test_the_well_known_provider_lists_the_openai_rows() {
        let provider = well_known();
        assert_eq!(provider.name(), "well-known");
        let names: Vec<&str> = provider.rows().iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names.len(), 12);
        assert!(
            names.iter().all(|n| n.starts_with("openai/whisper/")),
            "{names:?}"
        );
        let name = |spec: &str| provider.lookup(spec).unwrap().map(|p| p.name);
        assert_eq!(
            name("openai/tiny.en").as_deref(),
            Some("openai/whisper/tiny.en")
        );
        assert_eq!(
            name("openai/turbo").as_deref(),
            Some("openai/whisper/large/large-v3-turbo")
        );
        assert_eq!(
            name("turbo").as_deref(),
            Some("openai/whisper/large/large-v3-turbo")
        );
        assert_eq!(
            name("large").as_deref(),
            Some("openai/whisper/large/large-v3")
        );
        assert_eq!(
            name("large-v3-turbo").as_deref(),
            Some("openai/whisper/large/large-v3-turbo")
        );
        assert_eq!(name("openai/gigantic"), None);
        assert_eq!(name("gigantic"), None);
    }

    /// No two rows share a leaf name or an alias.
    #[test]
    fn test_names_and_aliases_are_unique() {
        let provider = well_known();
        let mut names: Vec<&str> = provider
            .rows()
            .iter()
            .flat_map(|p| {
                core::iter::once(leaf(&p.name)).chain(p.aliases.iter().map(String::as_str))
            })
            .collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "a name is repeated");
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::kits::{
            speech::asr::whisper::pretrained::{
                PYTORCH_FP16,
                TIKTOKEN,
            },
            support::catalog_contracts::testing::contract_lines,
        };
        assert_eq!(
            contract_lines(WHISPER_CONTRACT),
            [
                format!("whisper-asr: {CHECKPOINT} [{PYTORCH_FP16}]"),
                format!("whisper-asr: {VOCABULARY} [{TIKTOKEN}]"),
            ]
        );
    }
}
