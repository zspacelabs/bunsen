//! Each row's checkpoint `config.json` against the kit's prefabs.
//!
//! The configs are not committed (`[decided_gating]`): each is fetched from
//! the row's mirror, pinned, and read through the kit's reader, whose
//! geometry must be one of the prefabs', unsloth's edits to the configs and
//! all.

use bunsen::{
    data::pretrained::{
        CONFIG,
        ResourceBundle,
    },
    kits::language::gemma::{
        pretrained::{
            GEMMA_PREFABS,
            default_gemma_factory,
        },
        weights::gemma_config_from_transformers,
    },
};

use super::*;

#[test]
fn test_every_row_is_a_prefab() {
    let factory = default_gemma_factory().expect("the gemma factory");
    let cache = weights_cache();
    let ids = factory.ids();
    assert_eq!(ids.len(), 14);
    for id in ids {
        let (_, row) = factory.lookup(&id).unwrap();
        let mut map = ResourceBundle::new(row.resources.name.clone());
        map.insert(row.resources.try_get(CONFIG).unwrap().clone());
        let loaded = cache.load(&map).unwrap_or_else(|e| panic!("{id}: {e}"));
        let json = std::fs::read_to_string(loaded.expect(CONFIG).unwrap()).unwrap();
        let config = gemma_config_from_transformers(&json).unwrap_or_else(|e| panic!("{id}: {e}"));
        let config = format!("{config:?}");
        assert!(
            GEMMA_PREFABS
                .iter()
                .any(|p| format!("{:?}", p.to_config()) == config),
            "{id}: no prefab is its geometry"
        );
    }
}
