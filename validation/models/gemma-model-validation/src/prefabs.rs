//! Each row's prefab against its checkpoint's `config.json`.
//!
//! The configs are not committed (`[decided_gating]`): each is fetched from
//! the row's mirror, pinned, and read through the kit's reader, whose
//! geometry must be the prefab's, unsloth's edits to the configs and all.

use bunsen::{
    data::pretrained::{
        CONFIG,
        ResourceBundle,
    },
    kits::language::gemma::{
        pretrained::{
            GEMMA_KIT,
            GEMMA_PREFABS,
            default_gemma_factory,
        },
        weights::gemma_config_from_hf,
    },
};

use super::*;

#[test]
fn test_every_row_is_its_prefab() {
    let factory = default_gemma_factory().expect("the gemma factory");
    let cache = weights_cache();
    let ids = factory.ids();
    assert_eq!(ids.len(), 14);
    for id in ids {
        let (_, row) = factory.lookup(&id).unwrap();
        let mut map = ResourceBundle::new(row.resources.name.clone());
        map.insert(row.resources.try_get(CONFIG).unwrap().clone());
        let loaded = cache
            .load(GEMMA_KIT, &map)
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        let json = std::fs::read_to_string(loaded.expect(CONFIG).unwrap()).unwrap();
        let config = gemma_config_from_hf(&json).unwrap_or_else(|e| panic!("{id}: {e}"));
        let prefab = GEMMA_PREFABS
            .expect_lookup_prefab(row.prefab.as_deref().unwrap())
            .to_config();
        assert_eq!(format!("{config:?}"), format!("{prefab:?}"), "{id}");
    }
}
