//! Each row's checkpoint `config.json` against the kit's prefabs.
//!
//! The configs are not committed: each is fetched from the row's repo,
//! pinned, and read through the kit's reader, whose geometry must be one
//! of the prefabs'.

use bunsen::{
    data::{
        builds::CONFIG,
        pretrained::Prefabs,
        resources::ResourceSet,
    },
    kits::language::qwen3_5::{
        Qwen3_5Config,
        pretrained::default_qwen3_5_factory,
        weights::qwen3_5_config_from_transformers,
    },
};

use super::*;

#[test]
fn test_every_row_is_a_prefab() {
    let factory = default_qwen3_5_factory().expect("the qwen3.5 factory");
    let cache = weights_cache();
    let ids = factory.ids();
    assert_eq!(ids.len(), 9);
    for id in ids {
        let (_, row) = factory.lookup(&id).unwrap();
        let mut map = ResourceSet::new(row.resource_set().name.clone());
        map.insert(row.resource_set().try_get(CONFIG).unwrap().clone());
        let loaded = cache.load(&map).unwrap_or_else(|e| panic!("{id}: {e}"));
        let json = std::fs::read_to_string(loaded.expect(CONFIG).unwrap()).unwrap();
        let config =
            qwen3_5_config_from_transformers(&json).unwrap_or_else(|e| panic!("{id}: {e}"));
        let config = format!("{config:?}");
        assert!(
            Qwen3_5Config::PREFABS
                .iter()
                .any(|(_, prefab)| format!("{:?}", prefab()) == config),
            "{id}: no prefab is its geometry"
        );
    }
}
