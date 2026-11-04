//! # nanochat pretrained providers and factory
use std::sync::Arc;

use burn::tensor::Device;

use super::NanoChatConstructConfig;
use crate::{
    data::pretrained::{
        EmbeddedCatalog,
        PretrainedCache,
        PretrainedProvider,
    },
    errors::BunsenResult,
    kits::{
        gpts::nanochat::NanoChatConstruct,
        support::construct::PretrainedFactory,
    },
};

/// The kit segment of a nanochat resource's path in the cache:
/// `<cache>/pretrained/nanochat/<namespace>/<sha256>/<file>`.
pub const NANOCHAT_KIT: &str = "nanochat";

/// The `kind` of a `PyTorch` state dict (`torch.save`), as upstream writes
/// `model_<step>.pt` and `token_bytes.pt`.
pub const PYTORCH: &str = "pytorch";

/// The `kind` of a nanochat checkpoint's `meta_<step>.json`.
pub const META_JSON: &str = "nanochat-meta.json";

/// The `kind` of a pickled `tiktoken` `Encoding`, as upstream writes
/// `tokenizer.pkl`.
pub const TIKTOKEN_PICKLE: &str = "tiktoken-pickle";

/// The nanochat checkpoints bunsen knows by name, behind the
/// [`WELL_KNOWN`](crate::data::pretrained::WELL_KNOWN) provider: in group
/// `nanochat-students`, the d20 run of upstream's 2025 model (base, mid and
/// RL; `well-known:nanochat-students/rl-d20`, or its alias
/// `nanochat-d20-rl`), and in group `nitishpandey04`, the d18 run of its
/// 2026 model (base and SFT).
///
/// Built from `validation/catalogs/nanochat.yml` by `cargo validate catalog`.
pub static NANOCHAT_CATALOG: EmbeddedCatalog =
    EmbeddedCatalog::new("nanochat", include_str!("catalog.json"));

/// The name [`default_nanochat_factory`] loads when a caller names none.
pub const DEFAULT_NANOCHAT_MODEL: &str = "nanochat-students/rl-d20";

/// nanochat's compiled-in providers: the well-known table.
pub fn default_nanochat_providers() -> Vec<Arc<dyn PretrainedProvider>> {
    vec![Arc::new(NANOCHAT_CATALOG.table())]
}

/// nanochat's factory over [`default_nanochat_providers`], building through
/// [`NanoChatConstructConfig`].
///
/// [`load_construct`](PretrainedFactory::load_construct) with
/// `"[provider:]name"` is the whole pathway from a name to a
/// [`NanoChatConstruct`].
///
/// # Errors
/// As [`PretrainedFactory::with_providers`], if two of the defaults share a
/// name, which the tests pin they do not.
pub fn default_nanochat_factory() -> BunsenResult<PretrainedFactory<NanoChatConstructConfig>> {
    PretrainedFactory::new().with_providers(default_nanochat_providers())
}

impl PretrainedFactory<NanoChatConstructConfig> {
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
    ) -> BunsenResult<Arc<NanoChatConstruct>> {
        Ok(self.load(spec, cache, device, policy)?.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::gpts::nanochat::{
        CHECKPOINT,
        META,
        TOKEN_BYTES,
        TOKENIZER,
        pretrained::NANOCHAT_PREFABS,
        weights::{
            NanoChatGeneration,
            NanoChatMeta,
            nanochat_config,
        },
    };

    /// Every file of every row is pinned to one URL, each row's alias finds
    /// it, and each row's meta (committed under `testdata/nanochat/meta/`, at
    /// the row's digest) is the geometry of the prefab the row names.
    #[test]
    fn test_rows_are_pinned_and_match_their_prefabs() {
        let prefabs: Vec<&str> = NANOCHAT_PREFABS.iter().map(|p| p.name).collect();
        NANOCHAT_CATALOG.catalog().validate(&prefabs).unwrap();
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
                assert!(resource.is_pinned(), "{name}: {}", resource.key);
                assert_eq!(resource.urls().len(), 1, "{name}: {}", resource.key);
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
            let prefab = NANOCHAT_PREFABS
                .expect_lookup_prefab(row.prefab.as_deref().unwrap())
                .to_config();
            assert_eq!(
                format!("{:?}", nanochat_config(&meta, generation)),
                format!("{prefab:?}"),
                "{name}"
            );
        }
        assert_eq!(
            factory.lookup(DEFAULT_NANOCHAT_MODEL).unwrap().1.name,
            "nanochat-students/rl-d20"
        );
    }
}
