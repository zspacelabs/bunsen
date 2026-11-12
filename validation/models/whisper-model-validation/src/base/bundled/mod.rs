//! The Whisper kit's pretrained pathway on the bundle's files.

use std::{
    path::PathBuf,
    sync::Arc,
};

use bunsen::{
    data::{
        builds::{
            BuildFactory,
            CATALOG,
            CatalogBuildProvider,
        },
        cache::BunsenDiskCacheOptions,
        catalog::{
            BuiltinCatalog,
            BunsenCatalog,
            CatalogName,
            builtin_stack,
        },
        resources::{
            ResourceCache,
            ResourceCacheOptions,
            ResourceSet,
        },
    },
    kits::{
        speech::asr::whisper::{
            driver::WhisperSpecialIds,
            pretrained::{
                CHECKPOINT,
                VOCABULARY,
                WHISPER_CATALOG_CONTRACT,
                WhisperConstructConfig,
                WhisperVocabulary,
                default_whisper_providers,
            },
        },
        support::tokens::TiktokenRanks,
    },
};
use serde_json::json;

mod cache;
mod construct;
mod hf;
mod vocabulary;

/// A cache rooted at the bundle's directory, offline.
pub fn offline_cache() -> ResourceCache {
    cache_in(bind_whisper_onnx::cache_dir().to_path_buf())
}

/// An offline cache rooted at `dir`.
pub fn cache_in(dir: PathBuf) -> ResourceCache {
    ResourceCache::new(
        ResourceCacheOptions::default()
            .with_disk(
                BunsenDiskCacheOptions::default()
                    .with_cache_dir(Some(dir))
                    .without_transfer_observers(),
            )
            .with_offline(true),
    )
    .unwrap()
}

/// The bundled rank file a layout selects.
pub fn bundled_vocabulary_path(ids: &WhisperSpecialIds) -> PathBuf {
    let map = WhisperVocabulary::for_layout(ids).resource_set();
    offline_cache()
        .resolve(map.try_get(VOCABULARY).unwrap())
        .unwrap()
        .path
}

/// The bundled rank file a layout selects, parsed.
pub fn bundled_vocabulary(ids: &WhisperSpecialIds) -> TiktokenRanks {
    WhisperVocabulary::for_layout(ids)
        .load(&offline_cache())
        .unwrap()
}

/// `openai/whisper/base`'s checkpoint alone, as a resource set: the
/// built-in catalog's resource `openai/whisper/base/checkpoint`.
pub fn base_checkpoint() -> ResourceSet {
    bunsen::data::catalog::BuiltinCatalog::get()
        .unwrap()
        .resource("openai/whisper/base/checkpoint", CHECKPOINT)
        .unwrap()
}

/// The local dir the bundle layer names: what a caller would override to
/// serve the bundle's files from elsewhere.
pub const BUNDLE: &str = "whisper-bundle";

/// The extension layer that serves `openai/whisper/base`'s files from the
/// bundle, in place: an update of each resource the build binds (its
/// checkpoint, and the multilingual vocabulary), with the directory the
/// bundle keeps the file in as a local-dir source ([`BUNDLE`]), tried first.
///
/// It is how a deployment that ships its files serves them, so it is built
/// here the way such a caller would: from the catalog's own entries, which
/// an update must match in every field but `$sources`.
pub fn bundle_layer() -> BunsenCatalog {
    let stack = builtin_stack().unwrap();
    let base = BuiltinCatalog::get()
        .unwrap()
        .build("openai/whisper/base")
        .unwrap();
    let mut root = serde_json::Map::new();
    root.insert(
        "$description".to_string(),
        json!("The Whisper base files bind-whisper-onnx ships, used in place."),
    );
    for (slot, path) in [
        (CHECKPOINT, bind_whisper_onnx::base_pt()),
        (VOCABULARY, bind_whisper_onnx::multilingual_tiktoken()),
    ] {
        let bound = &base.resources[slot];
        let mut entry = serde_json::to_value(&bound.entry).unwrap();
        entry["$kind"] = json!("resource");
        entry["$merge"] = json!("update");
        entry["$sources"] = json!([{
            "$local_dir": {
                "$name": BUNDLE,
                "$default": [path.parent().unwrap().display().to_string()],
            }
        }]);
        // The scopes down to it, each an extending occurrence of its kind.
        let segments = bound.resource.segments();
        let mut node = &mut root;
        let mut scope: Option<CatalogName> = None;
        for segment in &segments[..segments.len() - 1] {
            let name = CatalogName::under(scope.as_ref(), segment).unwrap();
            let kind = stack.declaration(&name).unwrap().kind();
            node = node
                .entry(segment.clone())
                .or_insert_with(|| json!({ "$kind": kind.as_str(), "$merge": "update" }))
                .as_object_mut()
                .unwrap();
            scope = Some(name);
        }
        node.insert(segments[segments.len() - 1].clone(), entry);
    }
    BunsenCatalog::from_json(&serde_json::Value::Object(root).to_string()).unwrap()
}

/// The default Whisper factory over the built-in catalog with
/// [`bundle_layer`] imported on top: the same names, `openai/whisper/base`
/// served from the bundle.
pub fn factory_with_bundle() -> BuildFactory<WhisperConstructConfig> {
    let mut stack = builtin_stack().unwrap();
    stack.push("whisper-bundle", None, bundle_layer()).unwrap();
    let catalog = CatalogBuildProvider::new(
        CATALOG,
        "the built-in catalog, with the bundle's files",
        &stack,
        WHISPER_CATALOG_CONTRACT,
    )
    .unwrap();
    let rest = default_whisper_providers()
        .into_iter()
        .filter(|p| p.name() != CATALOG);
    BuildFactory::new()
        .with_provider(Arc::new(catalog))
        .unwrap()
        .with_providers(rest)
        .unwrap()
}
