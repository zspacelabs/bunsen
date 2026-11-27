//! Audits of a kit's factory and its pinned rows.

use alloc::{
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

use crate::data::{
    builds::{
        BuildFactory,
        CHECKPOINT,
        ConstructInit,
    },
    catalog::resolve::ResolvedBuild,
};

/// Asserts that `factory` serves `C`'s kit through the built-in catalog,
/// then `hf:`.
///
/// # Panics
/// When it doesn't.
#[track_caller]
pub fn assert_catalog_then_hf<C: ConstructInit>(factory: &BuildFactory<C>) {
    assert_eq!(factory.kit(), C::KIT);
    let names: Vec<&str> = factory.providers().iter().map(|p| p.name()).collect();
    assert_eq!(names, ["catalog", "hf"], "{}'s providers", C::KIT);
}

/// Asserts that every row `factory` lists validates, has a checkpoint (one
/// file or a shard index) and each of `keys`, and that every resource in it
/// is pinned to one URL, which contains `url_part` when given.
///
/// # Returns
/// The rows, by name, for the kit's own further checks.
///
/// # Panics
/// When a row breaks any of that; the message names the row and the slot.
#[track_caller]
pub fn assert_rows_pinned<C: ConstructInit>(
    factory: &BuildFactory<C>,
    keys: &[&str],
    url_part: Option<&str>,
) -> Vec<(String, ResolvedBuild)> {
    let mut rows = Vec::new();
    for (_, row) in factory.list() {
        let name = row.name.to_string();
        let map = row.resource_set();
        map.validate().unwrap_or_else(|e| panic!("{name}: {e:#}"));
        map.try_get_checkpoint(CHECKPOINT)
            .unwrap_or_else(|e| panic!("{name}: {e:#}"));
        for key in keys {
            assert!(map.get(key).is_some(), "{name}: {key}");
        }
        for resource in map.resources.values() {
            assert!(resource.is_pinned(), "{name}: {} is pinned", resource.slot);
            let urls = resource.urls();
            assert_eq!(urls.len(), 1, "{name}: {}'s one URL", resource.slot);
            if let Some(part) = url_part {
                assert!(urls[0].contains(part), "{name}: {}", urls[0]);
            }
        }
        rows.push((name, row));
    }
    rows
}
