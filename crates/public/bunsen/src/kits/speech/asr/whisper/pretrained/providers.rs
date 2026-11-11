//! # Whisper pretrained providers

use alloc::sync::Arc;

use crate::{
    data::pretrained::{
        EmbeddedCatalog,
        HfProvider,
        PretrainedProvider,
    },
    kits::speech::asr::whisper::pretrained::CHECKPOINT,
};

/// The kit segment of a Whisper resource's path in the cache:
/// `<cache>/pretrained/whisper/<namespace>/<sha256>/<file>`.
pub const WHISPER_KIT: &str = "whisper";

/// The name of the local-dir base that is `openai-whisper`'s download
/// root; a [`PretrainedCacheOptions`](crate::data::pretrained::PretrainedCacheOptions)
/// override by this name points it elsewhere.
pub const OPENAI_LOCAL_DIR: &str = "openai-whisper";

/// The Whisper checkpoints bunsen knows by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider, in group
/// `openai`: `OpenAI`'s twelve checkpoints as `openai-whisper` names and pins
/// them (`well-known:openai/tiny`, `openai/tiny`, or `tiny`).
///
/// Each row fuses a checkpoint bundle with the vocabulary bundle its token
/// layout selects ([`WhisperVocabulary`](super::WhisperVocabulary)); the
/// tests pin that every row's declaration and the rule agree, so a path
/// model, which has only the rule, gets the same file a name does. A
/// checkpoint is found first under `openai-whisper`'s download root (the
/// local-dir base [`OPENAI_LOCAL_DIR`]: `$XDG_CACHE_HOME/whisper`, else
/// `~/.cache/whisper`, used in place), then at upstream's digest-addressed
/// URL.
///
/// Built from `validation/catalogs/whisper.yml` by `cargo validate catalog`.
pub static WHISPER_CATALOG: EmbeddedCatalog =
    EmbeddedCatalog::new("whisper", include_str!("catalog.json"));

/// What the whisper kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const WHISPER_CONTRACT: &str = include_str!("contract.yml");

/// The tag rules of the kit's rows, which the catalog tool holds its catalog
/// to: the keys every row carries, a speech recognizer's: what speech to text
/// chooses its model by.
pub const WHISPER_TAGS: crate::data::pretrained::tags::KitTags =
    crate::data::pretrained::tags::KitTags {
        keys: &[],
        required: &[
            crate::data::pretrained::tags::LANGUAGE.key,
            crate::data::pretrained::tags::TRANSLATE.key,
            crate::data::pretrained::tags::PARAMETERS.key,
        ],
    };

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
        Arc::new(WHISPER_CATALOG.table()),
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

    /// Every row names a prefab the map has and fuses into a checkpoint
    /// and a vocabulary, the checkpoint file named after the row and
    /// pinned.
    #[test]
    fn test_every_row_names_a_prefab_and_fuses() {
        let prefabs: Vec<&str> = WHISPER_PREFABS.iter().map(|p| p.name).collect();
        WHISPER_CATALOG.catalog().validate(&prefabs).unwrap();
        let table = WHISPER_CATALOG.table();
        for group in &table.groups {
            for row in &group.items {
                let prefab = row
                    .prefab
                    .as_deref()
                    .expect("every whisper row names a prefab");
                assert!(
                    WHISPER_PREFABS.lookup_prefab(prefab).is_some(),
                    "{}: prefab {prefab:?} is not in WHISPER_PREFABS",
                    group.id(row),
                );
                let map = &row.resources;
                map.validate().unwrap();
                assert_eq!(map.keys(), [CHECKPOINT, VOCABULARY], "{}", group.id(row));
                let checkpoint = map.get(CHECKPOINT).unwrap();
                assert_eq!(checkpoint.file, format!("{}.pt", row.name));
                assert!(checkpoint.is_pinned());
                assert!(map.get(VOCABULARY).unwrap().is_pinned());
            }
        }
        assert_eq!(table.groups[0].items.len(), 12);
    }

    /// The vocabulary a row declares is the one the rule selects for its
    /// prefab's layout, so a name and a path agree on it.
    #[test]
    fn test_the_declared_vocabulary_is_the_rules() {
        let table = WHISPER_CATALOG.table();
        let openai = &table.groups[0];
        for row in &openai.items {
            let declared = row.resources.get(VOCABULARY).unwrap();
            let rule = WhisperVocabulary::for_layout(&layout_of(row.prefab.as_deref().unwrap()))
                .resource_bundle();
            let rule = rule.get(VOCABULARY).unwrap();
            assert_eq!(declared, rule, "{}", openai.id(row));
        }
    }

    /// The well-known table lists the twelve `openai` rows by ref and
    /// answers the refs, the bare names and upstream's aliases.
    #[test]
    fn test_the_well_known_table_lists_the_openai_rows() {
        let table = WHISPER_CATALOG.table();
        assert_eq!(table.name(), "well-known");
        let ids = table.ids();
        assert_eq!(ids.len(), 12);
        assert_eq!(ids[0], "well-known:openai/tiny.en");
        assert_eq!(ids[11], "well-known:openai/large-v3-turbo");
        let name = |spec: &str| table.lookup(spec).unwrap().map(|p| p.name);
        assert_eq!(name("openai/tiny.en").as_deref(), Some("openai/tiny.en"));
        assert_eq!(
            name("openai/turbo").as_deref(),
            Some("openai/large-v3-turbo")
        );
        assert_eq!(name("turbo").as_deref(), Some("openai/large-v3-turbo"));
        assert_eq!(name("large").as_deref(), Some("openai/large-v3"));
        assert_eq!(
            name("large-v3-turbo").as_deref(),
            Some("openai/large-v3-turbo")
        );
        assert_eq!(name("openai/gigantic"), None);
        assert_eq!(name("gigantic"), None);
    }

    #[test]
    fn test_names_and_aliases_are_unique_within_a_group() {
        for group in &WHISPER_CATALOG.table().groups {
            let mut names: Vec<&str> = group
                .items
                .iter()
                .flat_map(|p| {
                    core::iter::once(p.name.as_str()).chain(p.aliases.iter().map(String::as_str))
                })
                .collect();
            let n = names.len();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), n, "{}: a name is repeated", group.name);
        }
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
