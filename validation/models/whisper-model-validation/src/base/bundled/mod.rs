//! The Whisper kit's pretrained pathway on the bundle's files.

use std::{
    path::PathBuf,
    sync::Arc,
};

use bunsen::{
    data::{
        builds::{
            BUNDLED,
            BuildFactory,
            PretrainedGroup,
            PretrainedTable,
        },
        cache::BunsenDiskCacheOptions,
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
                WhisperConstructConfig,
                WhisperVocabulary,
                default_whisper_factory,
            },
        },
        support::tokens::TiktokenRanks,
    },
};

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

/// A [`BUNDLED`] table of one row, `bundled:openai/base`: the catalog
/// `openai/whisper/base` row with its two resources served in place from
/// the bundle's files rather than fetched.
///
/// It is how a deployment that ships its files would serve them, so it is
/// built here the way such a caller would.
pub fn bundled_table() -> PretrainedTable {
    let (_, mut base) = default_whisper_factory()
        .unwrap()
        .lookup("openai/whisper/base")
        .unwrap();
    base.name = "base".to_string();
    base.aliases.clear();
    base.resources = base
        .resources
        .with_local_files(
            BUNDLED,
            &[
                (CHECKPOINT, bind_whisper_onnx::base_pt()),
                (VOCABULARY, bind_whisper_onnx::multilingual_tiktoken()),
            ],
        )
        .unwrap();
    PretrainedTable {
        name: BUNDLED.to_string(),
        description: "the Whisper assets bind-whisper-onnx ships, used in place".to_string(),
        groups: vec![PretrainedGroup {
            name: "openai".to_string(),
            description: "OpenAI's Whisper checkpoints".to_string(),
            license: None,
            origin: None,
            items: vec![base],
        }],
        case_sensitive: false,
    }
}

/// The default Whisper factory with [`bundled_table`] registered.
pub fn factory_with_bundle() -> BuildFactory<WhisperConstructConfig> {
    default_whisper_factory()
        .unwrap()
        .with_provider(Arc::new(bundled_table()))
        .unwrap()
}
