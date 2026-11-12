//! # nanochat pretrained providers and factory
use alloc::sync::Arc;

use super::NanoChatConstructConfig;
use crate::{
    data::builds::{
        PretrainedFactory,
        PretrainedProvider,
    },
    errors::BunsenResult,
};

/// The nanochat kit's name: its rows' `kit` tag.
pub const NANOCHAT_KIT: &str = "nanochat";

/// The `kind` of a `PyTorch` state dict (`torch.save`), as upstream writes
/// `model_<step>.pt` and `token_bytes.pt`.
pub const PYTORCH: &str = "pytorch";

/// The `kind` of a nanochat checkpoint's `meta_<step>.json`.
pub const META_JSON: &str = "nanochat-meta.json";

/// The `kind` of a pickled `tiktoken` `Encoding`, as upstream writes
/// `tokenizer.pkl`.
pub const TIKTOKEN_PICKLE: &str = "tiktoken-pickle";

/// What the nanochat kit's loader implements, as a build catalog's contract:
/// `contract.yml`, gathered by
/// [`kit_catalog_contracts`](crate::kits::support::catalog_contracts::kit_catalog_contracts).
pub const NANOCHAT_CONTRACT: &str = include_str!("contract.yml");

/// The contract the kit's loader implements, by its name in the built-in
/// catalog's `$contracts`: the kit's `well-known:` rows are its builds.
pub const NANOCHAT_CATALOG_CONTRACT: &str = "nanochat-lm";

/// The name [`default_nanochat_factory`] loads when a caller names none.
pub const DEFAULT_NANOCHAT_MODEL: &str = "nanochat-students/rl-d20";

/// nanochat's compiled-in providers: the well-known table.
pub fn default_nanochat_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![Arc::new(
        crate::data::builds::BuildCatalogProvider::expect_builtin(
            NANOCHAT_CATALOG_CONTRACT,
            NANOCHAT_KIT,
        ),
    )]
}

/// nanochat's factory over [`default_nanochat_providers`], building through
/// [`NanoChatConstructConfig`].
///
/// [`load_construct`](PretrainedFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`NanoChatConstruct`](crate::kits::language::nanochat::NanoChatConstruct).
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_nanochat_factory() -> BunsenResult<PretrainedFactory<NanoChatConstructConfig>> {
    PretrainedFactory::new().with_providers(default_nanochat_providers())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::language::nanochat::{
        CHECKPOINT,
        META,
        TOKEN_BYTES,
        TOKENIZER,
        pretrained::NANOCHAT_PREFABS,
        weights::{
            NanoChatGeneration,
            NanoChatMeta,
        },
    };

    /// Every file of every row is pinned to one URL, each row's alias finds
    /// it, and each row's meta (committed under `testdata/nanochat/meta/`, at
    /// the row's digest) is the geometry of a prefab.
    #[test]
    fn test_rows_are_pinned_and_match_their_prefabs() {
        let factory = default_nanochat_factory().unwrap();
        assert_eq!(factory.kit(), NANOCHAT_KIT);
        let rows = [
            ("nanochat-students", "base-d20", Some("nanochat-d20-base")),
            ("nanochat-students", "mid-d20", Some("nanochat-d20-mid")),
            ("nanochat-students", "rl-d20", Some("nanochat-d20-rl")),
            ("nitishpandey04", "nanochat-d18-base", None),
            ("nitishpandey04", "nanochat-d18-sft", None),
        ];
        assert_eq!(
            factory.ids(),
            rows.iter()
                .map(|(group, name, _)| format!("well-known:{group}/{name}"))
                .collect::<Vec<_>>()
        );
        for (group, name, alias) in rows {
            let (_, row) = factory.lookup(name).unwrap();
            assert_eq!(row.name, format!("{group}/{name}"));
            if let Some(alias) = alias {
                assert_eq!(factory.lookup(alias).unwrap().1.name, row.name, "{alias}");
            }
            let map = &row.resources;
            map.validate().unwrap();
            for resource in map.resources.values() {
                assert!(resource.is_pinned(), "{name}: {}", resource.slot);
                assert_eq!(resource.urls().len(), 1, "{name}: {}", resource.slot);
            }
            for key in [CHECKPOINT, META, TOKENIZER, TOKEN_BYTES] {
                assert!(map.get(key).is_some(), "{name}: {key}");
            }

            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(format!("testdata/nanochat/meta/{name}.meta.json"));
            assert_eq!(
                Some(crate::data::cache::sha256_of(&path).unwrap()),
                map.get(META).unwrap().sha256,
                "{name}: the committed meta is the pinned one"
            );
            let meta = NanoChatMeta::from_file(&path).unwrap();
            let generation = match meta.model_config.window_pattern {
                Some(_) => NanoChatGeneration::Upstream2026,
                None => NanoChatGeneration::Upstream2025,
            };
            let config = format!("{:?}", meta.gpt_config(generation));
            assert!(
                NANOCHAT_PREFABS
                    .iter()
                    .any(|p| format!("{:?}", p.to_config()) == config),
                "{name}: its meta is no prefab's geometry"
            );
        }
        assert_eq!(
            factory.lookup(DEFAULT_NANOCHAT_MODEL).unwrap().1.name,
            "nanochat-students/rl-d20"
        );
    }

    /// `contract.yml` loads as a base catalog's contracts, and its slots are
    /// the resource keys the loader reads, of the kinds it reads them as.
    #[test]
    fn test_the_contract_is_the_loaders() {
        use crate::kits::{
            language::nanochat::{
                CHECKPOINT,
                META,
                TOKEN_BYTES,
                TOKENIZER,
            },
            support::catalog_contracts::testing::contract_lines,
        };
        assert_eq!(
            contract_lines(NANOCHAT_CONTRACT),
            [
                format!("nanochat-lm: {CHECKPOINT} [{PYTORCH}]"),
                format!("nanochat-lm: {META} [{META_JSON}]"),
                format!("nanochat-lm: {TOKEN_BYTES} [{PYTORCH}]"),
                format!("nanochat-lm: {TOKENIZER} [{TIKTOKEN_PICKLE}]"),
            ]
        );
    }
}
