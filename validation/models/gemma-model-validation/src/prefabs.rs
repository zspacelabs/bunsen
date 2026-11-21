//! Each row's checkpoint `config.json` against the kit's prefabs.
//!
//! The configs are not committed (`[decided_gating]`): each is fetched from
//! the row's mirror, pinned, and read through the kit's reader, whose
//! geometry must be one of the prefabs', unsloth's edits to the configs and
//! all.

use bunsen::{
    data::{
        builds::CONFIG,
        pretrained::Prefabs,
        resources::ResourceSet,
    },
    kits::language::generation::gemma::{
        GemmaConfig,
        interop::transformers::gemma_config_from_path,
        pretrained::default_gemma_factory,
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
        let mut map = ResourceSet::new(row.resource_set().name.clone());
        map.insert(row.resource_set().try_get(CONFIG).unwrap().clone());
        let loaded = cache.load(&map).unwrap_or_else(|e| panic!("{id}: {e}"));
        let config = gemma_config_from_path(loaded.expect(CONFIG).unwrap())
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        let config = format!("{config:?}");
        assert!(
            GemmaConfig::PREFABS
                .iter()
                .any(|(_, prefab)| format!("{:?}", prefab()) == config),
            "{id}: no prefab is its geometry"
        );
    }
}
